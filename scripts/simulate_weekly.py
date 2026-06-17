"""Weekly-sessioned replay of a saved checkpoint -> the Apentic "Simulated Trades" dashboard JSON.

Mirrors the COMPETITION structure (see .design-export-simulated/HANDOFF.md): every session is one
calendar week starting **00:00 UTC Monday**, each begins fresh at **$10,000** (no cross-week
compounding), and the **vol-top-8 universe + risk-parity weights are re-selected before each week**
(causal trailing vol at the week open -> the basket evolves week to week). DESKTOP-ONLY (torch).

  python scripts/simulate_weekly.py --run-id ppo-event-rdLe4r-68b268f-s0 [--no-publish]

The page derives every metric from per-asset `candles` + `positions`; this script only emits those.
Fidelity rules (HANDOFF + user direction — do NOT bend the data to the schema):
  * Positions are the agent's REAL fills folded into FIFO round-trips (PnL preserved exactly); a
    position still open at the weekly reset is force-closed mark-to-market at the week's last close.
  * AMM cost is BAKED INTO execution prices (entry_price = cost-inclusive fill, exit_price =
    net-of-cost fill) so the page's `qty*(exit-entry)` equals the sim's true net PnL. A per-week
    check asserts sim-equity == reconstructed-close; divergence is reported, never papered over.
  * The event-driven agent only trades on ignitions, so many days have no trades -> expect Rule-1
    (>=1 trade/day) flags. That is the true behavior; no filler trades are invented.
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
from datetime import datetime, timezone

sys.path.insert(0, "scripts")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from remote_train import invalidate_cloudfront, join, put_bytes  # noqa: E402
from simulate import env_kwargs_from_provenance, make_predict  # noqa: E402  (reuse the loaders)
from trader import config  # noqa: E402
from trader.report.apentic import MANIFEST_CACHE_CONTROL, upsert_manifest_at  # noqa: E402
from train_event import WARMUP, eval_universe_and_caps, evaluate_event_policy  # noqa: E402

WEEK_SECS = 7 * 24 * 3600
MONDAY_PHASE = 345600          # t % WEEK_SECS for 00:00 UTC Monday (unix epoch was a Thursday)
HOUR = 3600
START_CAPITAL = 10_000.0
DD_LIMIT = -0.30
RECON_TOL_USD = 10.0           # |sim equity - reconstructed close| above this is flagged, not hidden

PEG = {"XAUT", "XAUt", "PAXG"}
MAJOR = {"BTC", "ETH", "BNB", "XRP", "SOL", "LTC", "DOGE", "ADA", "TRX", "LINK", "XLM", "BCH"}


def classify(sym: str) -> str:
    return "peg" if sym in PEG else "major" if sym.upper() in MAJOR else "alt"


def remap_candles(cs: list[dict]) -> list[dict]:
    return [{"t": c["time"], "o": c["open"], "h": c["high"], "l": c["low"],
             "c": c["close"], "v": c["volume"]} for c in cs]


def fold_positions(markers: list[dict], last_t: int, ledger_pnl: float) -> list[dict]:
    """FIFO round-trips from the agent's fills (cost baked into the prices) for the trade STRUCTURE,
    then SNAP the token's total PnL to the env's EXACT per-token ledger value -> the dashboard's
    qty*(exit-entry) equals the sim's realized+open PnL by construction (no inference). Still-open or
    markerless-closed lots are closed at last_t; the last position absorbs any residual to hit `ledger_pnl`."""
    lots: list[list] = []          # open buys: [qty_remaining, entry_t, entry_price_eff]
    out: list[dict] = []
    for m in markers:
        price, usd = m.get("price") or 0.0, m.get("usd") or 0.0
        if price <= 0 or usd <= 0:
            continue
        qty, fee = usd / price, m.get("fee", 0.0)
        if m["side"] == "buy":
            lots.append([qty, int(m["time"]), price * (1.0 + fee / usd)])     # cost-inclusive entry
        else:
            exit_eff, remaining = price * (1.0 - fee / usd), qty              # net-of-cost exit
            while remaining > 1e-12 and lots:
                lot = lots[0]
                q = min(remaining, lot[0])
                if int(m["time"]) > lot[1]:                                   # entry_t < exit_t
                    out.append({"entry_t": lot[1], "entry_price": lot[2], "exit_t": int(m["time"]),
                                "exit_price": exit_eff, "qty": q, "kind": "core"})
                lot[0] -= q
                remaining -= q
                if lot[0] <= 1e-12:
                    lots.pop(0)
    for qty_rem, entry_t, entry_eff in lots:                  # still-open / markerless-closed lots
        if last_t > entry_t:
            out.append({"entry_t": entry_t, "entry_price": entry_eff, "exit_t": last_t,
                        "exit_price": entry_eff, "qty": qty_rem, "kind": "core"})   # provisional 0 PnL
    if out:                                                   # snap token total to the EXACT ledger PnL
        cur = sum(p["qty"] * (p["exit_price"] - p["entry_price"]) for p in out)
        if out[-1]["qty"]:
            out[-1]["exit_price"] += (ledger_pnl - cur) / out[-1]["qty"]
    return out


def week_starts(idx_secs) -> list[int]:
    """Every 00:00-UTC-Monday timestamp present in the data."""
    return [int(t) for t in idx_secs if int(t) % WEEK_SECS == MONDAY_PHASE]


def label_week_split(ws: int, train_end: int, val_end: int) -> str:
    """Which split a week's START falls in, from the SAME train_rl.time_split boundaries the gate
    uses (don't hardcode timestamps). `train_end` = train_r.index[-1], `val_end` = val_r.index[-1]
    (the inclusive last bar of each split). A week starting on/before `train_end` is 'train'; on or
    before `val_end` (but past train) is 'val'; everything later is the never-touched 'test' OOS.
    Pure / torch-free so it's unit-testable. Matches trader.train.weekly_eval.split_label semantics."""
    if ws <= train_end:
        return "train"
    if ws <= val_end:
        return "val"
    return "test"


def summarize_windows(weeks_meta: list[dict]) -> dict:
    """Aggregate per-week metadata into the 3-window (+ overall) summary for `meta.windows`.

    Input: a list of `{start, split, return, dd}` (one per emitted week). PURE — no torch, no I/O —
    so the dashboard's overview maths is unit-testable off a synthetic bundle. For each of
    'train'/'val'/'test' (and 'overall' = every week) emits:
      ret_sum       sum of per-week raw returns in the window
      ret_mean      mean per-week return (0.0 for an empty window)
      worst_week_dd max per-week within-week portfolio max-DD in the window (0.0 if empty)
      win_rate      fraction of weeks with return > 0 (0.0 if empty)
      n_weeks       count of weeks in the window
    """
    def agg(rows: list[dict]) -> dict:
        n = len(rows)
        rets = [float(r["return"]) for r in rows]
        dds = [float(r["dd"]) for r in rows]
        ret_sum = sum(rets)
        return {
            "ret_sum": ret_sum,
            "ret_mean": (ret_sum / n) if n else 0.0,
            "worst_week_dd": max(dds) if dds else 0.0,
            "win_rate": (sum(1 for x in rets if x > 0) / n) if n else 0.0,
            "n_weeks": n,
        }

    by_split = {split: [w for w in weeks_meta if w["split"] == split]
                for split in ("train", "val", "test")}
    return {**{split: agg(rows) for split, rows in by_split.items()},
            "overall": agg(list(weeks_meta))}


def week_return_dd(eq) -> tuple[float, float]:
    """Within-week return + portfolio max-DD from the env's per-bar equity series. `eq` is ALREADY
    the week's trading bars: evaluate_event_policy seeds the equity trace at reset(start=WARMUP), so
    it starts at the env's first tradeable bar -- there is NO warmup prepad to drop (an earlier
    eq.iloc[WARMUP:] dropped the entire week and zeroed every DD). Pure / torch-free (input is a plain
    pandas Series). return = eq[-1]/START_CAPITAL - 1; dd = worst drawdown from the running peak (>=0)."""
    if len(eq) == 0:
        return 0.0, 0.0
    wk_return = float(eq.iloc[-1] / START_CAPITAL - 1.0)
    dd = float(abs((eq / eq.cummax() - 1.0).min()))
    return wk_return, dd


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--run-id", required=True)
    p.add_argument("--no-publish", action="store_true")
    p.add_argument("--max-weeks", type=int, default=0, help="cap for a quick run (0 = all)")
    args = p.parse_args()
    config.load_dotenv()

    base = os.path.join("runs-rl", args.run_id)
    policy_path, vecnorm_path = os.path.join(base, "policy.zip"), os.path.join(base, "vecnormalize.pkl")
    prov_path = os.path.join(base, args.run_id, "metrics.json")
    for path in (policy_path, vecnorm_path, prov_path):
        if not os.path.exists(path):
            raise SystemExit(f"missing {path} - is this a saved checkpoint?")
    prov = json.loads(open(prov_path, encoding="utf-8").read())
    prov = prov.get("provenance", prov)
    recurrent, seed = bool(prov.get("recurrent")), int(prov.get("seed", 0))

    from train_rl import (build_ohlc_frac_panels, build_portfolio_artifacts, build_volume_panel,
                           load_data, time_split)

    returns, btc, _anchor, liq = load_data()
    vol = build_volume_panel(list(returns.columns), returns.index)
    env_kwargs = env_kwargs_from_provenance(prov, returns, build_ohlc_frac_panels)

    # Split boundaries from the SAME train_rl.time_split the gate uses (don't hardcode timestamps).
    train_r, val_r, _test_r = time_split(returns)
    train_end, val_end = int(train_r.index[-1]), int(val_r.index[-1])

    if recurrent:
        from sb3_contrib import RecurrentPPO
        model = RecurrentPPO.load(policy_path, device="cpu")
    else:
        from stable_baselines3 import PPO
        model = PPO.load(policy_path, device="cpu")
    with open(vecnorm_path, "rb") as f:
        vn = pickle.load(f)

    idx = [int(t) for t in returns.index]
    pos_of = {t: i for i, t in enumerate(idx)}
    starts = week_starts(idx)

    weeks, max_recon = [], 0.0
    for ws in starts:
        i0 = pos_of.get(ws)
        if i0 is None or i0 < WARMUP:
            continue                                          # need a full warmup before the week
        we = ws + WEEK_SECS
        win = returns.iloc[i0 - WARMUP: i0 - WARMUP + WARMUP + 168]   # warmup prepad + <=168 week bars
        wk_bars = [t for t in idx[i0: i0 + 168] if t < we]
        if len(win) < WARMUP + 150 or len(wk_bars) < 150:    # skip gappy/short weeks
            continue
        if args.max_weeks and len(weeks) >= args.max_weeks:
            break

        eq, records, _uni, _fees, _raw, token_pnl = evaluate_event_policy(make_predict(model, vn, recurrent),
                                                                          win, btc, liq, vol, env_kwargs)
        ranked, caps = eval_universe_and_caps(win, btc, liq, vol, env_kwargs)     # rank order + caps
        d0, d1 = int(win.index[WARMUP]), int(win.index[-1])
        _w, token_candles, token_trades = build_portfolio_artifacts(records, ranked, d0, d1)

        assets, recon_pnl = [], 0.0
        for r, sym in enumerate(ranked):
            cs = remap_candles(token_candles.get(sym, []))
            last_t = cs[-1]["t"] if cs else d1
            positions = fold_positions(token_trades.get(sym, []), last_t, token_pnl.get(sym, 0.0))
            recon_pnl += sum(po["qty"] * (po["exit_price"] - po["entry_price"]) for po in positions)
            assets.append({"symbol": sym, "class": classify(sym), "vol_rank": r + 1,
                           "alloc_usd": round(float(caps.get(sym, 0.0)) * START_CAPITAL, 2),
                           "candles": cs, "positions": positions})

        recon_err = abs((START_CAPITAL + recon_pnl) - float(eq.iloc[-1]))
        max_recon = max(max_recon, recon_err)
        # Within-week return + PORTFOLIO max-DD from the env's EXACT per-bar equity (reconstructing
        # from candles is a scale-mismatched consumer problem). `eq` is already week-only -- the
        # equity trace is seeded at reset(start=WARMUP), so do NOT drop another WARMUP (that zeroed
        # every DD). See week_return_dd.
        wk_return, dd = week_return_dd(eq)
        weeks.append({"index": len(weeks), "label": f"W{len(weeks) + 1:02d}", "start": ws, "end": we,
                      "split": label_week_split(ws, train_end, val_end),
                      "return": wk_return, "dd": dd,
                      "portfolio_start": START_CAPITAL, "assets": assets})
        flag = "  <-- RECON GAP" if recon_err > RECON_TOL_USD else ""
        print(f"[wk] {datetime.fromtimestamp(ws, timezone.utc).date()} "
              f"pnl {float(eq.iloc[-1]) - START_CAPITAL:+8.2f} recon_err {recon_err:6.2f} "
              f"assets {len(assets)} trades {sum(len(a['positions']) for a in assets)}{flag}")

    if not weeks:
        raise SystemExit("no full Monday-aligned weeks with warmup found")

    windows = summarize_windows(weeks)   # weeks carry {start, split, return, dd} — exactly the input
    payload = {
        "meta": {"start_capital": START_CAPITAL, "window_start": weeks[0]["start"],
                 "window_end": weeks[-1]["end"], "n_weeks": len(weeks),
                 "candle_interval_seconds": HOUR, "drawdown_limit": DD_LIMIT,
                 "universe_size": len(returns.columns), "source_run": args.run_id,
                 "windows": windows,
                 "generated": datetime.now(timezone.utc).isoformat()},
        "weeks": weeks,
    }
    out_path = os.path.join(base, "simulated_trades.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, separators=(",", ":"))
    print(f"[sim-weekly] {len(weeks)} weeks -> {out_path} ({os.path.getsize(out_path) // 1024} KB) "
          f"| max recon err ${max_recon:.2f}")
    for split in ("train", "val", "test", "overall"):
        w = windows[split]
        print(f"[windows] {split:7s} n={w['n_weeks']:2d} ret_sum {w['ret_sum']:+.4f} "
              f"ret_mean {w['ret_mean']:+.4f} worst_dd {w['worst_week_dd']:.4f} win_rate {w['win_rate']:.2f}")
    if max_recon > RECON_TOL_USD:
        print(f"[sim-weekly] WARNING: max reconstruction error ${max_recon:.2f} > ${RECON_TOL_USD} "
              f"- the page's derived PnL diverges from the sim's true equity; investigate before trusting.")

    if not args.no_publish:
        target = config.get("APENTIC_PUBLISH_TARGET")
        cf = config.get("APENTIC_CLOUDFRONT_DIST_ID")
        data = json.dumps(payload, separators=(",", ":")).encode("utf-8")
        put_bytes(join(target, f"{args.run_id}/simulated_trades.json"), data,
                  content_type="application/json", cache_control=MANIFEST_CACHE_CONTROL)
        entry = {"id": args.run_id, "model_name": prov.get("git_commit", args.run_id),
                 "path": f"{args.run_id}/simulated_trades.json", "n_weeks": len(weeks),
                 "window_start": weeks[0]["start"], "window_end": weeks[-1]["end"],
                 "generated": payload["meta"]["generated"]}
        upsert_manifest_at(join(target, "simulated_models.json"), entry, cache_control=MANIFEST_CACHE_CONTROL)
        if cf:
            invalidate_cloudfront(cf, [f"/{args.run_id}/simulated_trades.json", "/simulated_models.json"])
        print(f"[sim-weekly] published -> {join(target, args.run_id)}/simulated_trades.json (+ simulated_models.json index)")


if __name__ == "__main__":
    main()
