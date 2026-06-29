"""Backfill-correct the published competition history (series.json + snapshots/<id>/leaderboard.json).

The bug fixes (start-block drift, daily-trade=any-swap, entry rule, $1 floor, no $25 filter) changed only
DERIVED fields. The captured `equity_usd` per (wallet, hour) was always correct (live holdings x live
prices, independent of the bugs), so we recompute everything else from it + a correct E0 + per-hour
flows/trades, then re-rank and re-publish. Read-only chain; overwrites the CDN history.

  PYTHONPATH=src python scripts/backfill_competition_history.py [--publish s3://alexlouis-apentic-data]
"""
from __future__ import annotations

import argparse
import json
import os
import time
from collections import defaultdict
from datetime import datetime, timezone

from trader.chain.rpc import BscRpc
from trader.competition import flows, publish
from trader.competition import multicall as mc
from trader.competition import pricing
from trader.competition.nodereal import NodeReal
from trader.competition.snapshot import completed_window_days, get_decimals
from trader.competition.universe import load_universe
from trader.agent.wallet_recon import build_wallet_payload

WIN_TS = 1782086400          # 2026-06-22T00:00:00Z
WIN_BLOCK = 105617727        # the CORRECT window-open block (fixed)
ARCHIVE_RPC = "https://bsc-mainnet.public.blastapi.io"
COMP = "data/competition_out/competition"


def _iso_ts(s: str) -> int:
    return int(datetime.fromisoformat(s.replace("Z", "+00:00")).timestamp())


def wallet_events(nr, wallet, *, to_block, eligible, counted, prices, bnb_now, bnb_at):
    """Timestamped events: swaps `(ts, day, acquired_eligible)` and signed capital flows `(ts, usd)` —
    the latter incl. uncounted<->counted boundary crossings. Mirrors `flows.wallet_cost_basis`."""
    inbound = nr.asset_transfers(to_address=wallet, from_block=WIN_BLOCK, to_block=to_block)
    outbound = nr.asset_transfers(from_address=wallet, from_block=WIN_BLOCK, to_block=to_block)
    for t in inbound:
        t["dir"] = "in"
    for t in outbound:
        t["dir"] = "out"
    by_hash = defaultdict(list)
    for t in inbound + outbound:
        if (t.get("qty") or 0) > 0 and t.get("hash"):
            by_hash[t["hash"]].append(t)
    swaps, caps = [], []
    for legs in by_hash.values():
        dirs = {leg["dir"] for leg in legs}
        if "in" in dirs and "out" in dirs:
            ts = max((leg.get("ts") or 0) for leg in legs)
            acquired = any((leg.get("contract") or "") in eligible and leg["dir"] == "in" for leg in legs)
            if ts:
                swaps.append((ts, flows._utc_day(ts), acquired))
            cin = cout = 0.0
            has_uncounted = False
            for leg in legs:
                cu = flows._counted_usd(leg, counted, prices, bnb_price_now=bnb_now, bnb_price_at=bnb_at)
                if cu is None:
                    has_uncounted = True
                elif leg["dir"] == "in":
                    cin += cu
                else:
                    cout += cu
            if has_uncounted and ts:
                caps.append((ts, cin - cout, "bnd"))    # boundary flow = capital, not PnL
            continue
        cap_dir = "in" if "in" in dirs else "out"
        for leg in legs:
            usd = flows._fundable_usd(leg, bnb_price_now=bnb_now, bnb_price_at=bnb_at)
            if usd is None or not leg.get("ts"):
                continue
            caps.append((leg["ts"], usd if cap_dir == "in" else -usd, "dep"))
    return sorted(swaps), sorted(caps)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--publish", default=None)
    ap.add_argument("--cloudfront", default=publish.CLOUDFRONT_DIST)
    args = ap.parse_args(argv)

    universe = load_universe()
    eligible = {(u.get("contract") or "").lower() for u in universe
                if u.get("contract") and not u.get("is_stable")}
    counted = {(u.get("contract") or "").lower() for u in universe if u.get("contract")}
    counted.add("0xbb4cdb9cbd36b01bd1cbaebf2de08d9173bc095c")   # WBNB
    rpc = BscRpc(endpoints=[ARCHIVE_RPC])
    nr = NodeReal()
    latest = nr.block_number() - 100
    bnb_now = pricing._bnb_anchor_close()
    bnb_at = (lambda ts: pricing._bnb_anchor_close(at_ts=ts)) if bnb_now else None

    # --- load the archived hourly snapshots (source of the CORRECT per-(wallet,hour) equity) ---
    idx = json.load(open(os.path.join(COMP, "snapshots", "index.json"), encoding="utf-8"))
    snaps = sorted(idx["snapshots"], key=lambda s: s["id"])
    archive = {}   # id -> {wallet -> stored row}
    gen_of = {}    # id -> generated iso
    for s in snaps:
        lb = json.load(open(os.path.join(COMP, "snapshots", s["id"], "leaderboard.json"), encoding="utf-8"))
        archive[s["id"]] = {r["wallet"]: r for r in lb["rows"]}
        gen_of[s["id"]] = lb["generated"]
    wallets = sorted({w for rows in archive.values() for w in rows})
    print(f"snapshots={len(snaps)}  wallets={len(wallets)}  range {snaps[0]['id']}..{snaps[-1]['id']}")

    # --- E0 (open equity) at the CORRECT block + per-wallet event timelines (one pass) ---
    decimals = get_decimals(rpc.call, universe)
    start_h = {}
    held_start = set()
    for i, w in enumerate(wallets, 1):
        start_h[w] = mc.read_holdings(rpc.call, w, universe, decimals, block=hex(WIN_BLOCK))
        held_start |= {s for s, q in start_h[w].items() if q and q > 0}
        if i % 25 == 0 or i == len(wallets):
            print(f"  E0 holdings {i}/{len(wallets)}")
    cur_px = pricing.current_prices(held_start, universe)
    start_px, _ = pricing.start_prices(held_start, universe, WIN_TS, current=cur_px)

    e0, e0_elig, ev = {}, {}, {}
    for i, w in enumerate(wallets, 1):
        pay = build_wallet_payload({s: q for s, q in start_h[w].items() if q > 0}, start_px,
                                   baseline_usd=None, address=w)
        e0[w] = pay["equity_usd"]
        e0_elig[w] = sum((h["value_usd"] or 0.0) for h in pay["holdings"] if h["token"] != "BNB")
        time.sleep(0.3)   # pace NodeReal so the per-wallet event scan isn't truncated under load
        ev[w] = wallet_events(nr, w, to_block=latest, eligible=eligible, counted=counted,
                              prices=cur_px, bnb_now=bnb_now, bnb_at=bnb_at)
        if i % 10 == 0 or i == len(wallets):
            print(f"  E0+events {i}/{len(wallets)}")

    # --- recompute every hour, in order (running equity-min per wallet for the $1 floor) ---
    run_min = {w: e0[w] for w in wallets}         # min equity seen so far (incl open)
    new_series_pts = defaultdict(list)
    new_snapshots = {}
    for s in snaps:
        sid, gen = s["id"], gen_of[s["id"]]
        cutoff = _iso_ts(gen)
        comp_days = completed_window_days(WIN_TS, cutoff)
        rows = []
        for w in wallets:
            stored = archive[sid].get(w)
            if stored is None:
                continue
            eq = stored["equity_usd"]
            run_min[w] = min(run_min[w], eq)
            swaps, caps = ev[w]
            flows_asof = round(sum(u for ts, u, _ in caps if ts <= cutoff), 2)
            boundary_asof = round(sum(u for ts, u, k in caps if ts <= cutoff and k == "bnd"), 2)
            capital_basis = round(e0[w] + flows_asof, 2)
            pnl_usd = round(eq - capital_basis, 2)
            pnl_pct = round(pnl_usd / capital_basis * 100, 2) if capital_basis > 0 else None
            trade_days = sorted({d for ts, d, _ in swaps if ts <= cutoff})
            n_swaps = sum(1 for ts, _, _ in swaps if ts <= cutoff)
            n_buys = sum(1 for ts, _, a in swaps if ts <= cutoff and a)
            traded = n_buys > 0
            entered = e0_elig[w] > 0.01
            floor_min = min(run_min[w], e0[w])
            missing = [d for d in comp_days if d not in trade_days]
            reasons = []
            if not entered:
                reasons.append("No eligible token held at window open (Jun 22 00:00)")
            if missing:
                reasons.append(f"No trade on {missing[0]}")
            if floor_min < 1.0:
                reasons.append(f"Wallet value fell below $1 (min ${floor_min:.2f})")
            dq = bool(reasons)
            ranked = (not dq) and pnl_pct is not None
            rows.append({**stored, "equity_open_usd": round(e0[w], 2),
                         "window_flows_usd": flows_asof, "boundary_flow_usd": boundary_asof, "capital_basis_usd": capital_basis,
                         "pnl_usd": pnl_usd, "pnl_pct": pnl_pct, "traded_in_window": traded,
                         "n_eligible_buys": int(n_buys), "n_swaps": n_swaps, "trade_days": trade_days,
                         "entered": entered, "e0_eligible_usd": round(e0_elig[w], 2),
                         "floor_min_usd": round(floor_min, 2), "ranked": ranked,
                         "disqualified": dq, "dq_reason": "; ".join(reasons) if dq else None,
                         "dq_risk": False})
        rows.sort(key=lambda r: (not r["disqualified"], r["ranked"],
                                 r["pnl_pct"] if (r["ranked"] and r["pnl_pct"] is not None) else -1e18,
                                 r["equity_usd"]), reverse=True)
        for rank, r in enumerate(rows, 1):
            r["rank"] = rank
        new_snapshots[sid] = {
            "generated": gen, "metric": "window_pnl_vs_capital_basis",
            "window": {"start_block": WIN_BLOCK, "start_ts": WIN_TS,
                       "start_utc": datetime.fromtimestamp(WIN_TS, timezone.utc).isoformat(),
                       "end_utc": datetime.fromtimestamp(WIN_TS + 7 * 86400, timezone.utc).isoformat(),
                       "completed_days": comp_days},
            "dd_gate": 0.30, "floor_usd": 1.0, "n_participants": len(rows),
            "n_traded_in_window": sum(1 for r in rows if r["traded_in_window"]),
            "n_entered": sum(1 for r in rows if r["entered"]),
            "n_ranked": sum(1 for r in rows if r["ranked"]),
            "n_disqualified": sum(1 for r in rows if r["disqualified"]),
            "n_dq_risk": 0, "total_equity_usd": round(sum(r["equity_usd"] for r in rows), 2),
            "rows": rows, "backfilled": True}
        for r in rows:
            new_series_pts[r["wallet"]].append({
                "id": sid, "rank": r["rank"], "equity_usd": r["equity_usd"], "pnl_pct": r["pnl_pct"],
                "capital_basis_usd": r["capital_basis_usd"], "ranked": r["ranked"],
                "disqualified": r["disqualified"], "traded_in_window": r["traded_in_window"]})
        print(f"  recomputed {sid}: ranked={new_snapshots[sid]['n_ranked']} dq={new_snapshots[sid]['n_disqualified']}")

    # --- write corrected files locally ---
    newest_gen = gen_of[snaps[-1]["id"]]
    series = {"generated": newest_gen, "snapshots": [{"id": s["id"], "generated": gen_of[s["id"]]} for s in snaps],
              "wallets": dict(new_series_pts), "backfilled": True}
    with open(os.path.join(COMP, "series.json"), "w", encoding="utf-8") as f:
        json.dump(series, f, separators=(",", ":"))
    rels = ["series.json"]
    for sid, lb in new_snapshots.items():
        p = os.path.join(COMP, "snapshots", sid, "leaderboard.json")
        with open(p, "w", encoding="utf-8") as f:
            json.dump(lb, f, separators=(",", ":"))
        rels.append(f"snapshots/{sid}/leaderboard.json")
    print(f"wrote {len(rels)} corrected files locally")

    if args.publish:
        publish.mirror_to_cdn(COMP, args.publish, rels)
        publish.invalidate_cdn(args.cloudfront, paths=["/competition/series.json", "/competition/snapshots/*"])
        print("published + invalidated")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
