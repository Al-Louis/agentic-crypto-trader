"""Reconstruct hourly captures that were NEVER taken (the NodeReal-quota outage on 2026-06-26 06-14Z
crash-looped the scheduler, leaving holes in the series).

EQUITY is INTERPOLATED between the bracketing live captures, NOT re-priced from chain. Every gap sits
between two trusted, live-priced snapshots; equity moves slowly hour-to-hour, so a linear interpolation
of each wallet's equity is far more honest than re-pricing holdings from the local OHLCV store (which
has thin-pool spikes for the long tail — a first attempt produced a +214% phantom). The DERIVED fields
that DO change discretely within a gap (flows, trade-days, DQ) are computed AS OF each hour from the
populated transfer cache (timestamp-filtered) + the per-wallet window-open equity carried in the
existing snapshots. Every reconstructed hour is marked `backfilled: true` + `backfill_method: interp`.

  PYTHONPATH=src python scripts/backfill_missing_hours.py [--hours 2026-06-26T06Z,...] \
      [--publish s3://alexlouis-apentic-data]

With no --hours it auto-detects every gap between the first and last archived snapshot.
"""
from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timedelta, timezone

from trader.chain.rpc import BscRpc
from trader.competition import history, publish
from trader.competition import pricing
from trader.competition.nodereal import CachedNodeReal, NodeReal
from trader.competition.snapshot import completed_window_days
from trader.competition.universe import load_universe
from backfill_competition_history import ARCHIVE_RPC, COMP, WIN_BLOCK, WIN_TS, wallet_events

NR_CACHE = "data/competition_cache/nr_transfers.json"
WINDOW_DAYS = 7


def _sid_to_ts(sid: str) -> int:
    return int(datetime.strptime(sid, "%Y-%m-%dT%HZ").replace(tzinfo=timezone.utc).timestamp())


def _interp(points: list[tuple[int, float]], t: int) -> float:
    """Linear-interpolate equity at time `t` from sorted (ts, equity) live marks; clamp at the ends."""
    if not points:
        return 0.0
    if t <= points[0][0]:
        return points[0][1]
    if t >= points[-1][0]:
        return points[-1][1]
    for (t0, v0), (t1, v1) in zip(points, points[1:]):
        if t0 <= t <= t1:
            return v0 if t1 == t0 else v0 + (v1 - v0) * (t - t0) / (t1 - t0)
    return points[-1][1]


def _load_archive(comp_dir: str):
    """Existing snapshots: per-hour rows + the set with REAL captured equity (everything except our own
    interpolated fills). NOTE the rules/boundary backfill flagged its snapshots `backfilled:true` but
    KEPT each hour's originally-captured `equity_usd` — only derived fields were recomputed — so those
    are valid interpolation marks. Only `backfill_method == "interp"` snapshots have synthetic equity."""
    idx = json.load(open(os.path.join(comp_dir, "snapshots", "index.json"), encoding="utf-8"))
    by_id, gen_of, real_eq_ids = {}, {}, set()
    for s in sorted(idx["snapshots"], key=lambda s: s["id"]):
        sid = s["id"]
        lb = json.load(open(os.path.join(comp_dir, "snapshots", sid, "leaderboard.json"), encoding="utf-8"))
        by_id[sid] = {r["wallet"]: r for r in lb["rows"]}
        gen_of[sid] = lb["generated"]
        if lb.get("backfill_method") != "interp":
            real_eq_ids.add(sid)
    return by_id, gen_of, real_eq_ids


def _detect_missing(comp_dir: str) -> list[str]:
    ser = json.load(open(os.path.join(comp_dir, "series.json"), encoding="utf-8"))
    ids = sorted(s["id"] for s in ser["snapshots"])
    have = set(ids)
    t0 = datetime.strptime(ids[0], "%Y-%m-%dT%HZ").replace(tzinfo=timezone.utc)
    t1 = datetime.strptime(ids[-1], "%Y-%m-%dT%HZ").replace(tzinfo=timezone.utc)
    out, t = [], t0
    while t <= t1:
        sid = t.strftime("%Y-%m-%dT%HZ")
        if sid not in have:
            out.append(sid)
        t += timedelta(hours=1)
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hours", default=None, help="comma-separated hour-ids; default = auto-detect gaps")
    ap.add_argument("--publish", default=None)
    ap.add_argument("--cloudfront", default=publish.CLOUDFRONT_DIST)
    args = ap.parse_args(argv)

    missing = ([h.strip() for h in args.hours.split(",") if h.strip()]
               if args.hours else _detect_missing(COMP))
    if not missing:
        print("no missing hours — nothing to backfill")
        return 0
    print(f"missing hours ({len(missing)}): {', '.join(missing)}")

    universe = load_universe()
    eligible = {(u.get("contract") or "").lower() for u in universe
                if u.get("contract") and not u.get("is_stable")}
    counted = {(u.get("contract") or "").lower() for u in universe if u.get("contract")}
    counted.add("0xbb4cdb9cbd36b01bd1cbaebf2de08d9173bc095c")   # WBNB

    by_id, gen_of, real_eq_ids = _load_archive(COMP)
    # interpolate only from real captured-equity hours, never from an hour we're rebuilding this run
    live_ids = real_eq_ids - set(missing)
    latest_live = max(live_ids)
    wallets = sorted(by_id[latest_live])
    # window-open statics (constant across hours) + the live equity marks to interpolate between
    e0, e0_elig, reg = {}, {}, {}
    live_pts: dict[str, list[tuple[int, float]]] = {w: [] for w in wallets}
    for w in wallets:
        r = by_id[latest_live][w]
        e0[w] = r.get("equity_open_usd") or 0.0
        e0_elig[w] = r.get("e0_eligible_usd") or 0.0
        reg[w] = r.get("registered_ts")
    for sid in sorted(live_ids):
        ts = _sid_to_ts(sid)
        for w in wallets:
            r = by_id[sid].get(w)
            if r and r.get("equity_usd") is not None:
                live_pts[w].append((ts, r["equity_usd"]))
    print(f"  {len(wallets)} wallets, {len(live_ids)} live marks "
          f"({min(live_ids)}..{latest_live}) to interpolate across")

    # flows AS OF each hour come from the populated transfer cache (timestamp-filtered, ~one warm run)
    rpc = BscRpc(endpoints=[ARCHIVE_RPC])  # noqa: F841 — kept for parity; flows use the cache
    bnb_now = pricing._bnb_anchor_close()
    bnb_at = (lambda ts: pricing._bnb_anchor_close(at_ts=ts)) if bnb_now else None
    cur_px = pricing.current_prices({u["symbol"] for u in universe}, universe)
    nr = CachedNodeReal(NodeReal(), NR_CACHE, WIN_BLOCK)
    latest = nr.block_number() - 100
    ev = {w: wallet_events(nr, w, to_block=latest, eligible=eligible, counted=counted,
                           prices=cur_px, bnb_now=bnb_now, bnb_at=bnb_at) for w in wallets}
    print(f"  cached event timelines ready ({nr.nr.n_calls} NodeReal requests, "
          f"{nr.blocks_scanned:,} blocks scanned)")

    # --- reconstruct each missing hour (interpolated equity + as-of derived fields) ---
    all_rels: list[str] = []
    for sid in sorted(missing):
        h_ts = _sid_to_ts(sid)
        comp_days = completed_window_days(WIN_TS, h_ts)
        rows = []
        for w in wallets:
            eq = round(_interp(live_pts[w], h_ts), 2)
            swaps, caps = ev[w]
            flows_asof = round(sum(u for ts, u, _ in caps if ts <= h_ts), 2)
            boundary_asof = round(sum(u for ts, u, k in caps if ts <= h_ts and k == "bnd"), 2)
            capital_basis = round(e0[w] + flows_asof, 2)
            pnl_usd = round(eq - capital_basis, 2)
            pnl_pct = round(pnl_usd / capital_basis * 100, 2) if capital_basis > 0 else None
            trade_days = sorted({d for ts, d, _ in swaps if ts <= h_ts})
            n_swaps = sum(1 for ts, _, _ in swaps if ts <= h_ts)
            n_buys = sum(1 for ts, _, a in swaps if ts <= h_ts and a)
            entered = e0_elig[w] > 0.01
            floor_min = min([e0[w], eq] + [v for ts, v in live_pts[w] if ts <= h_ts])
            miss = [d for d in comp_days if d not in trade_days]
            reasons = []
            if not entered:
                reasons.append("No eligible token held at window open (Jun 22 00:00)")
            if miss:
                reasons.append(f"No trade on {miss[0]}")
            if floor_min < 1.0:
                reasons.append(f"Wallet value fell below $1 (min ${floor_min:.2f})")
            dq = bool(reasons)
            rows.append({
                "wallet": w, "equity_usd": eq, "equity_open_usd": round(e0[w], 2),
                "window_flows_usd": flows_asof, "boundary_flow_usd": boundary_asof,
                "capital_basis_usd": capital_basis, "pnl_usd": pnl_usd, "pnl_pct": pnl_pct,
                "traded_in_window": n_buys > 0, "n_eligible_buys": int(n_buys), "n_swaps": n_swaps,
                "trade_days": trade_days, "entered": entered, "e0_eligible_usd": round(e0_elig[w], 2),
                "floor_min_usd": round(floor_min, 2), "ranked": (not dq) and pnl_pct is not None,
                "disqualified": dq, "dq_reason": "; ".join(reasons) if dq else None, "dq_risk": False,
                "n_holdings": by_id[latest_live][w].get("n_holdings"),
                "stale": False, "registered_ts": reg[w], "interpolated": True})
        rows.sort(key=lambda r: (not r["disqualified"], r["ranked"],
                                 r["pnl_pct"] if (r["ranked"] and r["pnl_pct"] is not None) else -1e18,
                                 r["equity_usd"]), reverse=True)
        for rank, r in enumerate(rows, 1):
            r["rank"] = rank
        lb = {
            "generated": datetime.fromtimestamp(h_ts, timezone.utc).isoformat(),
            "metric": "window_pnl_vs_capital_basis",
            "window": {"start_block": WIN_BLOCK, "start_ts": WIN_TS,
                       "start_utc": datetime.fromtimestamp(WIN_TS, timezone.utc).isoformat(),
                       "end_utc": datetime.fromtimestamp(WIN_TS + WINDOW_DAYS * 86400, timezone.utc).isoformat(),
                       "completed_days": comp_days},
            "dd_gate": 0.30, "floor_usd": 1.0, "n_participants": len(rows),
            "n_traded_in_window": sum(1 for r in rows if r["traded_in_window"]),
            "n_entered": sum(1 for r in rows if r["entered"]),
            "n_ranked": sum(1 for r in rows if r["ranked"]),
            "n_disqualified": sum(1 for r in rows if r["disqualified"]),
            "n_dq_risk": 0, "total_equity_usd": round(sum(r["equity_usd"] for r in rows), 2),
            "rows": rows, "backfilled": True, "backfill_method": "interp"}
        rels = history.update_history(lb, COMP)
        all_rels += rels
        top = next((r for r in rows if r["ranked"]), None)
        print(f"  {sid}: ranked={lb['n_ranked']} dq={lb['n_disqualified']} "
              f"equity=${lb['total_equity_usd']:,.0f}" + (f" #1={top['wallet'][:10]} {top['pnl_pct']:+.1f}%"
                                                          if top else ""))

    rels = sorted(set(all_rels))
    print(f"wrote {len(rels)} files locally for {len(missing)} hours")
    if args.publish:
        publish.mirror_to_cdn(COMP, args.publish, rels)
        publish.invalidate_cdn(args.cloudfront,
                               paths=["/competition/series.json", "/competition/snapshots/*"])
        print("published + invalidated")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
