"""Trade POST-MORTEM grader — dissect a published run's trades and grade the agent's craft.

For one run-id: reconstruct every ROUND-TRIP from the published per-token markers + the local
hourly panel (prepad-aware), then grade four axes the user defined (entry, exit, allocation,
frequency/risk) per-trip and in aggregate. The purpose is diagnostic: the grades localize WHICH
knowledge the agent lacks (bad entries = entry-context features missing; givebacks = exit/memory;
size-outcome corr ~0 = conviction signal missing), feeding the obs-expansion design.

Honest-grading note: per-trip entry/exit scores use the FORWARD path (hindsight) — they grade
OUTCOMES to find patterns, they are NOT a claim the agent could have known. Aggregate patterns
across many trips are the signal; a single graded trade is an anecdote.

  python scripts/trade_postmortem.py --run-id ppo-event-rdLe4-c07bda0-s0 [--json out.json]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.request

import numpy as np
import pandas as pd

sys.path.insert(0, "scripts")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

HOST = "https://data.alexlouis.dev"
WARMUP = 168
H_REGRET = 24          # post-exit regret horizon (bars)
H_ENTRY = 24           # entry-timing window (bars each side)


def fetch(url):
    with urllib.request.urlopen(url, timeout=30) as r:
        return json.load(r)


def to_secs(v):
    v = int(v)
    return v // 1000 if v > 10_000_000_000 else v


def load_run_panel(prov):
    """The eval panel exactly as the run saw it (split + prepad), close-price indexed."""
    from train_rl import load_data, time_split
    returns, btc, anchor, liq = load_data()
    train_r, val_r, test_r = time_split(returns)
    eval_r = test_r if prov["eval_split"] == "test" else val_r
    if prov.get("eval_prepad"):
        prev = train_r if prov["eval_split"] == "val" else val_r
        eval_r = pd.concat([prev.tail(WARMUP), eval_r])
    px = (1.0 + eval_r.fillna(0.0)).cumprod()
    secs = np.array([to_secs(t) for t in eval_r.index])
    return eval_r, px, secs


def round_trips(markers, secs):
    """FIFO-pair buys/sells into round-trips. A trip opens on the first buy from flat and closes
    when the held notional is (near-)fully sold; partial sells accumulate as scale-outs."""
    trips, open_trip = [], None
    for m in sorted(markers, key=lambda x: to_secs(x["time"])):
        bar = int(np.searchsorted(secs, to_secs(m["time"])))
        if m["side"] == "buy":
            if open_trip is None:
                open_trip = {"entry_bar": bar, "entry_px": m["price"], "cost": m["usd"],
                             "fees": m.get("fee", 0.0), "sells": []}
            else:                                   # scale-in: blend the basis
                t = open_trip
                t["entry_px"] = ((t["entry_px"] * t["cost"] + m["price"] * m["usd"])
                                 / (t["cost"] + m["usd"]))
                t["cost"] += m["usd"]
                t["fees"] += m.get("fee", 0.0)
        else:
            if open_trip is None:
                continue                            # sell with no tracked basis (window edge)
            open_trip["sells"].append({"bar": bar, "px": m["price"], "usd": m["usd"]})
            open_trip["fees"] += m.get("fee", 0.0)
            sold = sum(s["usd"] for s in open_trip["sells"])
            # value the remaining basis at the current price to test "fully closed"
            remain = open_trip["cost"] * (m["price"] / open_trip["entry_px"]) - sold
            if remain < max(10.0, 0.02 * open_trip["cost"]):
                open_trip["exit_bar"] = bar
                trips.append(open_trip)
                open_trip = None
    if open_trip is not None:
        open_trip["exit_bar"] = None                # still open at window end
        trips.append(open_trip)
    return trips


def grade_trip(t, px_col, n_bars):
    """Per-trip metrics over the close path. Hindsight-grading caveat in the module docstring."""
    e, xb = t["entry_bar"], t["exit_bar"]
    end = xb if xb is not None else n_bars - 1
    hold = px_col[e:end + 1] / px_col[e]            # index units: path relative to the entry bar
    mfe = float(hold.max() - 1.0)                   # max favorable excursion
    mae = float(hold.min() - 1.0)                   # max adverse excursion
    proceeds = sum(s["usd"] for s in t["sells"])
    realized = (proceeds - t["cost"]) / t["cost"] if (xb is not None and t["cost"]) else None
    # entry timing: how far above the local low (±H_ENTRY bars) did it buy?
    lo = max(0, e - H_ENTRY)
    local_low = float(px_col[lo:min(e + H_ENTRY, n_bars)].min())
    entry_vs_low = float(px_col[e] / local_low - 1.0) if local_low > 0 else None
    # exit: giveback off the in-trade peak + 24h post-exit regret
    giveback = regret = None
    if xb is not None:
        peak_px = float(px_col[e:xb + 1].max())
        giveback = float(1.0 - px_col[xb] / peak_px) if peak_px > 0 else None
        fwd = min(xb + H_REGRET, n_bars - 1)
        regret = float(px_col[fwd] / px_col[xb] - 1.0) if px_col[xb] > 0 else None
    capture = (realized / mfe) if (realized is not None and mfe > 0.02) else None
    return {"entry_bar": int(e), "exit_bar": (int(xb) if xb is not None else None),
            "hold_bars": int(end - e), "cost": round(t["cost"], 2),
            "realized_pct": realized, "mfe": mfe, "mae": mae, "capture": capture,
            "entry_vs_local_low": entry_vs_low, "exit_giveback": giveback,
            "post_exit_regret_24h": regret, "fees": round(t["fees"], 2)}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run-id", required=True)
    p.add_argument("--json", default=None)
    args = p.parse_args()

    m = fetch(f"{HOST}/{args.run_id}/metrics.json")
    prov = m["provenance"]
    info = fetch(f"{HOST}/{args.run_id}/run_info.json")
    eval_r, px, secs = load_run_panel(prov)
    cols = list(eval_r.columns)
    pxm = px.to_numpy()
    n = len(eval_r)

    all_trips, per_token = [], {}
    for u in info.get("universe", []):
        try:
            markers = fetch(f"{HOST}/{args.run_id}/tk_{u['slug']}_trades.json")
        except Exception:  # noqa: BLE001
            continue
        if not markers or u["symbol"] not in cols:
            continue
        j = cols.index(u["symbol"])
        trips = round_trips(markers, secs)
        graded = [dict(grade_trip(t, pxm[:, j], n), token=u["symbol"]) for t in trips]
        per_token[u["symbol"]] = graded
        all_trips += graded

    closed = [t for t in all_trips if t["realized_pct"] is not None]
    wins = [t for t in closed if t["realized_pct"] > 0]
    days = (secs[-1] - secs[WARMUP if prov.get("eval_prepad") else 0]) / 86400
    sizes = [t["cost"] for t in closed]
    outs = [t["realized_pct"] for t in closed]
    size_corr = None
    if len(closed) >= 5:                              # Spearman = Pearson on ranks (no scipy)
        rs = pd.Series(sizes).rank()
        ro = pd.Series(outs).rank()
        size_corr = float(np.corrcoef(rs, ro)[0, 1])
    hhi = (sum((t["cost"] / sum(sizes)) ** 2 for t in closed) if sizes else None)

    def fmean(key, rows):
        v = [r[key] for r in rows if r[key] is not None]
        return float(np.mean(v)) if v else None

    agg = {
        "run_id": args.run_id, "n_round_trips": len(all_trips), "n_closed": len(closed),
        "win_rate": (len(wins) / len(closed)) if closed else None,
        "avg_win": fmean("realized_pct", wins),
        "avg_loss": fmean("realized_pct", [t for t in closed if t["realized_pct"] <= 0]),
        "ENTRY  avg_entry_vs_local_low": fmean("entry_vs_local_low", all_trips),
        "ENTRY  avg_mae": fmean("mae", all_trips),
        "EXIT   avg_capture_of_mfe": fmean("capture", closed),
        "EXIT   avg_giveback": fmean("exit_giveback", closed),
        "EXIT   avg_post_exit_regret_24h": fmean("post_exit_regret_24h", closed),
        "ALLOC  size_outcome_spearman": size_corr,
        "ALLOC  trip_size_hhi": hhi,
        "FREQ   round_trips_per_day": len(closed) / days if days else None,
        "RISK   worst_trip_mae": (min(t["mae"] for t in all_trips) if all_trips else None),
        "RISK   total_fees": round(sum(t["fees"] for t in all_trips), 2),
    }

    print(f"\n=== POST-MORTEM {args.run_id} ===")
    for k, v in agg.items():
        if isinstance(v, float):
            print(f"  {k:36}: {v:+.3f}" if abs(v) < 10 else f"  {k:36}: {v:,.1f}")
        else:
            print(f"  {k:36}: {v}")
    print("\nper round-trip:")
    for t in sorted(all_trips, key=lambda x: x["entry_bar"]):
        r = f"{t['realized_pct']:+.1%}" if t["realized_pct"] is not None else "open"
        cap = f"cap {t['capture']:.0%}" if t["capture"] is not None else ""
        gb = f"gb {t['exit_giveback']:.0%}" if t["exit_giveback"] is not None else ""
        rg = (f"regret {t['post_exit_regret_24h']:+.0%}"
              if t["post_exit_regret_24h"] is not None else "")
        print(f"  {t['token']:10} ${t['cost']:7,.0f}  {r:>7}  mfe {t['mfe']:+.0%} mae {t['mae']:+.0%}"
              f"  hold {t['hold_bars']:4}h  {cap} {gb} {rg}")

    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump({"aggregate": agg, "trips": all_trips}, f, indent=2)
        print(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()
