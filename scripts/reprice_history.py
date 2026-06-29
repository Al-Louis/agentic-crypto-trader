"""Re-price the TRX & ZEC contribution across the PRE-FIX competition history so the timeline is
self-consistent (no step at the hour the base-token pricing fix went live).

`pricing._deepest_price_usd` used to take the deepest BSC pool's `priceUsd` without checking our token
was that pool's BASE token. TRX's deepest pool is HTX-based (priceUsd ~$1.7e-6) and ZEC's is AAVE-based
(priceUsd ~$95 = AAVE's), so every capture before the fix undervalued TRX (~$0 vs $0.32) and ZEC
(~$95 vs ~$420). Snapshots from `--cutoff` onward were captured WITH the fix, so they're already
correct and are left untouched (re-adding the delta would double-count).

SURGICAL: only the TRX/ZEC value is adjusted — every other holding keeps its captured-at-the-time
price (no full historical re-price, so no thin-pool artifacts). For each pre-cutoff snapshot we read
each wallet's TRX/ZEC balance at that hour's block (archive Multicall) and add `qty * (correct-buggy)`
to the stored equity, then recompute capital-basis / PnL / the $1-floor / DQ / rank and re-publish.

  PYTHONPATH=src python scripts/reprice_history.py [--cutoff 2026-06-27T00Z] \
      [--publish s3://alexlouis-apentic-data]
"""
from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone

from trader.chain.rpc import BscRpc
from trader.competition import history, publish
from trader.competition import multicall as mc
from trader.competition import pricing
from trader.competition.snapshot import completed_window_days
from trader.competition.universe import load_universe
from trader.data import dexscreener
from backfill_competition_history import ARCHIVE_RPC, COMP, WIN_BLOCK, WIN_TS

WINDOW_DAYS = 7
BUGGED_SYMBOLS = ("TRX", "ZEC")   # tokens whose deepest BSC pool is cross-quoted (base != our token)


def _iso_ts(s: str) -> int:
    return int(datetime.fromisoformat(s.replace("Z", "+00:00")).timestamp())


def _per_unit_correction(universe, call) -> dict[str, tuple[str, float, int]]:
    """For each bugged token: `(contract, correct_minus_buggy_usd, decimals)`. The delta is what every
    pre-fix capture missed per token unit — base-matched price (correct) minus deepest-pool price (buggy).
    Decimals are read on-chain (TRX is 6, not the EVM-default 18 — hardcoding 18 zeroes the balance)."""
    by_sym = {u["symbol"]: u for u in universe}
    entries = [by_sym[s] for s in BUGGED_SYMBOLS]
    dec = mc.read_decimals(call, entries)
    out = {}
    for sym in BUGGED_SYMBOLS:
        u = by_sym[sym]
        addr = u["contract"].lower()
        pairs = dexscreener.token_pairs(u["contract"])
        buggy = pricing._deepest_price_usd(pairs) or 0.0          # old behavior (cross-quoted pool)
        correct = pricing._deepest_price_usd(pairs, u["contract"]) or 0.0   # base-matched (the real price)
        out[sym] = (addr, correct - buggy, dec.get(addr, 18))
        print(f"  {sym}: buggy=${buggy:.6g} correct=${correct:.4f} -> +${correct - buggy:.4f}/unit "
              f"(decimals={dec.get(addr, 18)})")
    return out


def _read_qty(call, wallets, corr, block: str) -> dict[str, dict[str, float]]:
    """`{wallet: {sym: qty}}` for the bugged tokens at `block` (one batched Multicall over all wallets)."""
    syms = list(corr)
    calls = [(corr[s][0], mc.calldata_balance_of(w)) for w in wallets for s in syms]
    res = mc.multicall_values(call, calls, block=block)
    out: dict[str, dict[str, float]] = {}
    for i, w in enumerate(wallets):
        d = {}
        for j, s in enumerate(syms):
            ok, ret = res[i * len(syms) + j]
            raw = mc._to_int(ret) if ok else None
            d[s] = (raw / (10 ** corr[s][2])) if raw else 0.0
        out[w] = d
    return out


def _delta_usd(qty: dict[str, float], corr) -> float:
    return sum(qty.get(s, 0.0) * corr[s][1] for s in corr)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cutoff", default="2026-06-27T00Z",
                    help="snapshots with id >= this already use the fix; left untouched")
    ap.add_argument("--publish", default=None)
    ap.add_argument("--cloudfront", default=publish.CLOUDFRONT_DIST)
    args = ap.parse_args(argv)

    universe = load_universe()
    rpc = BscRpc(endpoints=[ARCHIVE_RPC])
    corr = _per_unit_correction(universe, rpc.call)

    idx = json.load(open(os.path.join(COMP, "snapshots", "index.json"), encoding="utf-8"))
    snaps = [s for s in sorted(idx["snapshots"], key=lambda s: s["id"]) if s["id"] < args.cutoff]
    by_id, gen_of = {}, {}
    for s in snaps:
        lb = json.load(open(os.path.join(COMP, "snapshots", s["id"], "leaderboard.json"), encoding="utf-8"))
        by_id[s["id"]] = {r["wallet"]: r for r in lb["rows"]}
        gen_of[s["id"]] = lb["generated"]
    wallets = sorted({w for rows in by_id.values() for w in rows})
    print(f"repricing {len(snaps)} pre-{args.cutoff} snapshots over {len(wallets)} wallets")

    # E0 correction (constant): TRX/ZEC at the window-open block
    open_qty = _read_qty(rpc.call, wallets, corr, hex(WIN_BLOCK))
    e0_delta = {w: _delta_usd(open_qty[w], corr) for w in wallets}

    # E1 correction per hour: TRX/ZEC at each snapshot's block (lo bounded by the prior block, monotonic)
    eq_delta: dict[str, dict[str, float]] = {}
    prev_block = WIN_BLOCK
    for s in snaps:
        blk = rpc.block_at_timestamp(_iso_ts(gen_of[s["id"]]), lo=prev_block)
        prev_block = blk
        q = _read_qty(rpc.call, wallets, corr, hex(blk))
        eq_delta[s["id"]] = {w: _delta_usd(q[w], corr) for w in wallets}
        print(f"  {s['id']} block={blk}")

    # recompute each snapshot in order (running corrected-equity min per wallet for the $1 floor),
    # seeded with each wallet's corrected window-open equity
    run_min = {w: (by_id[snaps[0]["id"]].get(w, {}).get("equity_open_usd") or 0.0) + e0_delta[w]
               for w in wallets}
    all_rels: list[str] = []
    for s in snaps:
        sid = s["id"]
        cutoff_ts = _iso_ts(gen_of[sid])
        comp_days = completed_window_days(WIN_TS, cutoff_ts)
        rows = []
        for w in wallets:
            r = by_id[sid].get(w)
            if r is None:
                continue
            eq = round((r["equity_usd"] or 0.0) + eq_delta[sid][w], 2)
            e0 = round((r.get("equity_open_usd") or 0.0) + e0_delta[w], 2)
            run_min[w] = min(run_min[w], eq)
            window_flows = r.get("window_flows_usd") or 0.0
            capital_basis = round(e0 + window_flows, 2)
            pnl_usd = round(eq - capital_basis, 2)
            pnl_pct = round(pnl_usd / capital_basis * 100, 2) if capital_basis > 0 else None
            e0_elig = round((r.get("e0_eligible_usd") or 0.0) + e0_delta[w], 2)  # TRX/ZEC are eligible alts
            entered = e0_elig > 0.01
            trade_days = r.get("trade_days") or []
            floor_min = min(run_min[w], e0)
            miss = [d for d in comp_days if d not in trade_days]
            reasons = []
            if not entered:
                reasons.append("No eligible token held at window open (Jun 22 00:00)")
            if miss:
                reasons.append(f"No trade on {miss[0]}")
            if floor_min < 1.0:
                reasons.append(f"Wallet value fell below $1 (min ${floor_min:.2f})")
            dq = bool(reasons)
            rows.append({**r, "equity_usd": eq, "equity_open_usd": e0, "window_flows_usd": window_flows,
                         "capital_basis_usd": capital_basis, "pnl_usd": pnl_usd, "pnl_pct": pnl_pct,
                         "e0_eligible_usd": e0_elig, "entered": entered, "floor_min_usd": round(floor_min, 2),
                         "ranked": (not dq) and pnl_pct is not None, "disqualified": dq,
                         "dq_reason": "; ".join(reasons) if dq else None})
        rows.sort(key=lambda r: (not r["disqualified"], r["ranked"],
                                 r["pnl_pct"] if (r["ranked"] and r["pnl_pct"] is not None) else -1e18,
                                 r["equity_usd"]), reverse=True)
        for rank, r in enumerate(rows, 1):
            r["rank"] = rank
        lb = {
            "generated": gen_of[sid], "metric": "window_pnl_vs_capital_basis",
            "window": {"start_block": WIN_BLOCK, "start_ts": WIN_TS,
                       "start_utc": datetime.fromtimestamp(WIN_TS, timezone.utc).isoformat(),
                       "end_utc": datetime.fromtimestamp(WIN_TS + WINDOW_DAYS * 86400, timezone.utc).isoformat(),
                       "completed_days": comp_days},
            "dd_gate": 0.30, "floor_usd": 1.0, "n_participants": len(rows),
            "n_traded_in_window": sum(1 for r in rows if r.get("traded_in_window")),
            "n_entered": sum(1 for r in rows if r["entered"]),
            "n_ranked": sum(1 for r in rows if r["ranked"]),
            "n_disqualified": sum(1 for r in rows if r["disqualified"]),
            "n_dq_risk": sum(1 for r in rows if r.get("dq_risk")),
            "total_equity_usd": round(sum(r["equity_usd"] for r in rows), 2),
            "rows": rows, "backfilled": True, "reprice_fix": "trx_zec_base_token"}
        all_rels += history.update_history(lb, COMP)
        top = next((r for r in rows if r["ranked"]), None)
        print(f"  repriced {sid}: equity=${lb['total_equity_usd']:,.0f} ranked={lb['n_ranked']} "
              f"dq={lb['n_disqualified']}" + (f" #1={top['wallet'][:10]} {top['pnl_pct']:+.1f}%" if top else ""))

    rels = sorted(set(all_rels))
    print(f"wrote {len(rels)} files for {len(snaps)} snapshots")
    if args.publish:
        publish.mirror_to_cdn(COMP, args.publish, rels)
        publish.invalidate_cdn(args.cloudfront,
                               paths=["/competition/series.json", "/competition/snapshots/*"])
        print("published + invalidated")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
