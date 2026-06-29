"""CLI: one participant-leaderboard snapshot (deposit-proof PnL).

  # dry run to a local dir (default), first 5 wallets:
  python -m trader.competition --limit 5
  # full run, publish to the CDN bucket root (writes under competition/):
  python -m trader.competition --target s3://alexlouis-apentic-data

Read-only and isolated from the live agent. PnL = current on-chain equity − net deposited capital
(NodeReal `nr_getAssetTransfers`). Set NODEREAL_API_KEY in .env — the public demo key rate-limits
above a handful of wallets.
"""

from __future__ import annotations

import argparse
import os
from datetime import datetime, timezone

from trader.chain.rpc import BscRpc
from trader.competition import history, publish, snapshot
from trader.competition.nodereal import CachedNodeReal, NodeReal
from trader.competition.participants import load_participants
from trader.competition.universe import load_universe

ARCHIVE_RPC = "https://bsc-mainnet.public.blastapi.io"
WINDOW_START = "2026-06-22T00:00:00Z"   # competition live window opens


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="BNB-hackathon participant window-PnL leaderboard")
    ap.add_argument("--csv", default="data/bnb_hackathon_participants.csv")
    ap.add_argument("--resolved", default="data/resolved.json")
    ap.add_argument("--out", default="data/competition_out",
                    help="local canonical dir; output + retained history go under <out>/competition/")
    ap.add_argument("--publish", default=None,
                    help="S3 root to mirror to, e.g. s3://alexlouis-apentic-data (writes competition/)")
    ap.add_argument("--cloudfront", default=publish.CLOUDFRONT_DIST, help="CloudFront distribution id")
    ap.add_argument("--no-invalidate", action="store_true", help="skip CloudFront invalidation")
    ap.add_argument("--endpoint", default=ARCHIVE_RPC, help="archive RPC for Multicall balance reads")
    ap.add_argument("--window-start", default=WINDOW_START, help="live-window open (ISO or unix secs)")
    ap.add_argument("--limit", type=int, default=0, help="only the first N wallets (testing)")
    ap.add_argument("--no-flows", action="store_true", help="skip PnL/flows (equity-only board)")
    ap.add_argument("--nr-cache", default="data/competition_cache/nr_transfers.json",
                    help="incremental NodeReal transfer cache (fetches only new blocks each run)")
    ap.add_argument("--ohlcv-root", default=None)
    args = ap.parse_args(argv)

    participants = load_participants(args.csv)
    if args.limit:
        participants = participants[:args.limit]
    universe = load_universe(args.resolved)
    win_ts = _parse_ts(args.window_start)
    print(f"participants={len(participants)} universe_tokens={len(universe)} "
          f"window_open={datetime.fromtimestamp(win_ts, timezone.utc).isoformat()}")

    rpc = BscRpc(endpoints=[args.endpoint])
    # Full binary search from genesis — do NOT bound `lo` by (head - N): as the chain grows that lower
    # bound creeps PAST the fixed window-open block and silently returns a too-late block, dropping all
    # early-window trades (mass false DQ) and mis-anchoring E0/PnL. The window-open block is fixed; find it.
    win_block = rpc.block_at_timestamp(win_ts)
    nr = None
    if not args.no_flows:
        # Incremental transfer cache: persist each wallet's transfers + the last block scanned, so each
        # hourly run only fetches the ~1h of NEW blocks instead of rescanning the whole window per wallet
        # (NodeReal quota is the binding constraint). First run populates it; every run after is cheap.
        nr = CachedNodeReal(NodeReal(), args.nr_cache, win_block)
    print(f"window_open_block={win_block}"
          + (f"  flows via NodeReal cache (start_scan={nr.scanned or win_block})"
             if nr is not None else "  (equity-only)"))

    # prior hourly equity per wallet (for the $1-floor DQ) from the retained series
    comp_dir = os.path.join(args.out, publish.PREFIX)
    equity_history = _load_equity_history(comp_dir)

    leaderboard, payloads = snapshot.build_leaderboard(
        participants, universe, rpc_call=rpc.call, nr=nr,
        window_start_block=win_block, window_start_ts=win_ts,
        equity_history=equity_history, ohlcv_root=args.ohlcv_root)

    if nr is not None:
        nr.save()   # persist the merged transfers + new scanned_to_block so the next run starts there
        print(f"  nr-cache: {nr.n_fetches} live fetches / {nr.nr.n_calls} requests / "
              f"{nr.blocks_scanned:,} blocks scanned (CU-proportional), "
              f"scanned_to_block={nr.scanned} -> {args.nr_cache}")

    # local canonical write + retained-history update
    publish.write_outputs(leaderboard, payloads, args.out)
    hist_rels = history.update_history(leaderboard, comp_dir)
    sid = history.snapshot_id(leaderboard["generated"])
    print(f"  archived snapshot {sid}; history + series updated")

    # mirror to CDN (latest board + wallets + new snapshot + rolling indexes), then invalidate
    if args.publish:
        rels = (["leaderboard.json", "manifest.json"]
                + [f"wallets/{w}.json" for w in payloads] + hist_rels)
        publish.mirror_to_cdn(comp_dir, args.publish, rels)
        if not args.no_invalidate:
            publish.invalidate_cdn(args.cloudfront)

    _print_top(leaderboard)
    return 0


def _parse_ts(s: str) -> int:
    s = s.strip()
    return int(s) if s.isdigit() else int(datetime.fromisoformat(s.replace("Z", "+00:00")).timestamp())


def _load_equity_history(comp_dir: str) -> dict:
    """Prior hourly equity per wallet `{wallet: [equity_usd, ...]}` from the retained series.json,
    so the $1-floor DQ can see every captured snapshot (not just open/now). Empty on the first run."""
    import json  # noqa: PLC0415
    try:
        with open(os.path.join(comp_dir, "series.json"), encoding="utf-8") as f:
            ser = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}
    return {w: [pt["equity_usd"] for pt in pts if pt.get("equity_usd") is not None]
            for w, pts in ser.get("wallets", {}).items()}


def _print_top(lb: dict, n: int = 15) -> None:
    cd = lb["window"].get("completed_days") or []
    print(f"\n  WINDOW board ({lb['n_ranked']} active, ranked by window PnL%; rules: hold eligible@open, "
          f"trade daily, equity>=$1):")
    print(f"  completed days (daily-trade rule applies): {cd or '(none yet — window just opened)'}")
    print(f"  rank  wallet           equity$    PnL%   buys/swaps  flags")
    for r in [x for x in lb["rows"] if x["ranked"]][:n]:
        pct = f"{r['pnl_pct']:+.1f}%" if r["pnl_pct"] is not None else "   n/a"
        flag = "AT-RISK" if r["dq_risk"] else ""
        print(f"  {r['rank']:>4}  {r['wallet'][:10]}…  {r['equity_usd']:>8.2f}  {pct:>7}  "
              f"{r['n_eligible_buys']:>2}/{r['n_swaps']:<3}     {flag}")
    dq = [x for x in lb["rows"] if x["disqualified"]]
    if dq:
        print(f"\n  DISQUALIFIED ({len(dq)}):")
        for r in dq[:10]:
            print(f"    {r['wallet'][:10]}…  equity ${r['equity_usd']:>8.2f}  — {r['dq_reason']}")
    print(f"\n  {lb['n_ranked']} ranked / {lb['n_traded_in_window']} traded-in-window / "
          f"{lb['n_disqualified']} disqualified / {lb['n_dq_risk']} at-risk-today / "
          f"{lb['n_participants']} total; total equity ${lb['total_equity_usd']:,.0f}")


if __name__ == "__main__":
    raise SystemExit(main())
