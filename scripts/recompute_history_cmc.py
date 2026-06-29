"""Rebuild the ENTIRE competition history on CMC pricing so the timeline is one consistent methodology
end-to-end (no DexScreener base-token bug, no intermittency, real per-hour USD marks by address).

For every archived snapshot we re-read each wallet's on-chain holdings at that hour's block and value
them with the CMC k-line bar covering that hour (via `pricing.start_prices`, now CMC-backed). The
window-open equity (E0) and in-window flows are recomputed the same way, then capital-basis / PnL /
the $1-floor / DQ / rank follow. Each snapshot is written as it completes (idempotent), so a long run
is resumable: re-running skips nothing but overwrites with identical results.

  PYTHONPATH=src python scripts/recompute_history_cmc.py [--only 2026-06-26T16Z,...] \
      [--publish s3://alexlouis-apentic-data]

Token set is restricted to what any wallet actually touched (open holdings ∪ cache-transfer contracts)
to keep the per-block Multicall affordable; --publish mirrors the rewritten files + invalidates.
"""
from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from datetime import datetime, timezone

from trader.agent.wallet_recon import build_wallet_payload
from trader.chain.rpc import BscRpc
from trader.competition import history, publish
from trader.competition import multicall as mc
from trader.competition import pricing
from trader.competition.nodereal import CachedNodeReal, NodeReal
from trader.competition.participants import load_participants
from trader.competition.snapshot import completed_window_days, get_decimals
from trader.competition.universe import load_universe
from backfill_competition_history import ARCHIVE_RPC, COMP, WIN_BLOCK, WIN_TS, wallet_events

NR_CACHE = "data/competition_cache/nr_transfers.json"
WINDOW_DAYS = 7


def _iso_ts(s: str) -> int:
    return int(datetime.fromisoformat(s.replace("Z", "+00:00")).timestamp())


def _read_all_holdings(call, wallets, tokens, decimals, *, block):
    """`{wallet: {symbol: qty}}` for all wallets over `tokens` (+ native BNB) at `block`, one batched
    Multicall set. `tokens` is a universe subset (the contracts any wallet actually touched)."""
    calls, idx = [], []
    for w in wallets:
        for t in tokens:
            calls.append((t["contract"], mc.calldata_balance_of(w)))
            idx.append((w, t["symbol"], (t["contract"] or "").lower()))
        calls.append((mc.MULTICALL3, mc.calldata_get_eth_balance(w)))
        idx.append((w, "BNB", None))
    res = mc.multicall_values(call, calls, block=block, chunk=120)
    out: dict[str, dict[str, float]] = defaultdict(dict)
    for (w, sym, contract), (ok, ret) in zip(idx, res):
        raw = mc._to_int(ret) if ok else None
        dec = 18 if sym == "BNB" else decimals.get(contract, 18)
        out[w][sym] = (raw / (10 ** dec)) if raw else 0.0
    return out


def _touched_tokens(universe, wallets):
    """Universe subset = tokens any wallet received/sent in-window (from the transfer cache). Open-held
    tokens are added by the caller from the open-block read, so transient mid-window holdings are covered."""
    by_contract = {(u.get("contract") or "").lower(): u for u in universe}
    touched = set()
    try:
        cache = json.load(open(NR_CACHE, encoding="utf-8"))
        for w, dirs in cache.get("wallets", {}).items():
            for leg in dirs.get("in", []) + dirs.get("out", []):
                c = (leg.get("contract") or "").lower()
                if c in by_contract:
                    touched.add(c)
    except (FileNotFoundError, json.JSONDecodeError):
        pass
    return touched


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default=None, help="comma-separated snapshot ids (default = all archived)")
    ap.add_argument("--csv", default="data/bnb_hackathon_participants.csv")
    ap.add_argument("--publish", default=None)
    ap.add_argument("--cloudfront", default=publish.CLOUDFRONT_DIST)
    args = ap.parse_args(argv)

    participants = load_participants(args.csv)
    wallets = [p["wallet"] for p in participants]
    reg = {p["wallet"]: p.get("registered_ts") for p in participants}
    universe = load_universe()
    eligible = {(u.get("contract") or "").lower() for u in universe
                if u.get("contract") and not u.get("is_stable")}
    counted = {(u.get("contract") or "").lower() for u in universe if u.get("contract")}
    counted.add("0xbb4cdb9cbd36b01bd1cbaebf2de08d9173bc095c")   # WBNB

    rpc = BscRpc(endpoints=[ARCHIVE_RPC])
    decimals = get_decimals(rpc.call, universe)
    bnb_now = pricing._bnb_anchor_close()
    bnb_at = (lambda ts: pricing._bnb_anchor_close(at_ts=ts)) if bnb_now else None

    idx = json.load(open(os.path.join(COMP, "snapshots", "index.json"), encoding="utf-8"))
    snaps = sorted((s["id"] for s in idx["snapshots"]))
    if args.only:
        want = {h.strip() for h in args.only.split(",")}
        snaps = [s for s in snaps if s in want]
    gen_of = {s: json.load(open(os.path.join(COMP, "snapshots", s, "leaderboard.json"),
                                encoding="utf-8"))["generated"] for s in snaps}
    print(f"recomputing {len(snaps)} snapshots over {len(wallets)} wallets on CMC pricing")

    # --- relevant token subset: open-held ∪ transfer-touched (keeps per-block Multicall affordable) ---
    by_contract = {(u.get("contract") or "").lower(): u for u in universe}
    touched = _touched_tokens(universe, wallets)
    open_full = _read_all_holdings(rpc.call, wallets, universe, decimals, block=hex(WIN_BLOCK))
    for w in wallets:
        for sym, q in open_full[w].items():
            if q and q > 0 and sym != "BNB":
                u = next((x for x in universe if x["symbol"] == sym), None)
                if u and u.get("contract"):
                    touched.add(u["contract"].lower())
    tokens = [by_contract[c] for c in touched if c in by_contract]
    print(f"  relevant tokens: {len(tokens)} (of {len(universe)})")

    # --- E0 (window-open equity) + e0_eligible, on CMC start-of-window prices ---
    held_open = {s for w in wallets for s, q in open_full[w].items() if q and q > 0}
    cur_px = pricing.current_prices({u["symbol"] for u in universe}, universe)
    open_px, _ = pricing.start_prices(held_open, universe, WIN_TS, current=cur_px)
    e0, e0_elig = {}, {}
    for w in wallets:
        pay = build_wallet_payload({s: q for s, q in open_full[w].items() if q > 0}, open_px,
                                   baseline_usd=None, address=w)
        e0[w] = pay["equity_usd"]
        e0_elig[w] = sum((h["value_usd"] or 0.0) for h in pay["holdings"] if h["token"] != "BNB")

    # --- in-window flows (NodeReal cache) valued on CMC current prices ---
    nr = CachedNodeReal(NodeReal(), NR_CACHE, WIN_BLOCK)
    latest = nr.block_number() - 100
    ev = {w: wallet_events(nr, w, to_block=latest, eligible=eligible, counted=counted,
                           prices=cur_px, bnb_now=bnb_now, bnb_at=bnb_at) for w in wallets}
    print(f"  E0 + flows ready ({nr.nr.n_calls} NodeReal requests)")

    # --- per-snapshot: re-read holdings at the hour block, value on CMC hour prices, rebuild the board ---
    run_min = {w: e0[w] for w in wallets}
    all_rels: list[str] = []
    prev_block = WIN_BLOCK
    for sid in snaps:
        h_ts = _iso_ts(gen_of[sid])
        blk = rpc.block_at_timestamp(h_ts, lo=prev_block)
        prev_block = blk
        hold = _read_all_holdings(rpc.call, wallets, tokens, decimals, block=hex(blk))
        held = {s for w in wallets for s, q in hold[w].items() if q and q > 0}
        px, _ = pricing.start_prices(held, universe, h_ts, current=cur_px)
        comp_days = completed_window_days(WIN_TS, h_ts)
        rows = []
        for w in wallets:
            pay = build_wallet_payload({s: q for s, q in hold[w].items() if q > 0}, px,
                                       baseline_usd=None, address=w)
            eq = pay["equity_usd"]
            run_min[w] = min(run_min[w], eq)
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
            floor_min = min(run_min[w], e0[w])
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
                "n_holdings": sum(1 for h in pay["holdings"] if (h["value_usd"] or 0) > 0),
                "stale": bool(pay["stale"]), "registered_ts": reg[w]})
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
            "n_traded_in_window": sum(1 for r in rows if r["traded_in_window"]),
            "n_entered": sum(1 for r in rows if r["entered"]),
            "n_ranked": sum(1 for r in rows if r["ranked"]),
            "n_disqualified": sum(1 for r in rows if r["disqualified"]),
            "n_dq_risk": 0, "total_equity_usd": round(sum(r["equity_usd"] for r in rows), 2),
            "rows": rows, "repriced": "cmc"}
        all_rels += history.update_history(lb, COMP)
        top = next((r for r in rows if r["ranked"]), None)
        print(f"  {sid} blk={blk}: equity=${lb['total_equity_usd']:,.0f} ranked={lb['n_ranked']} "
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
