"""Pool-event backfill / tail driver — the trader.chain collector CLI.

Backfill (resumable; safe to re-run, it continues at the manifest cursor):
    python scripts/chain_backfill.py --init          # build registry + set range
    python scripts/chain_backfill.py                 # run / resume the scan
    python scripts/chain_backfill.py --tail          # extend to current head
    python scripts/chain_backfill.py --panels        # build hourly panels

The scan range defaults to [min OHLCV oldest_ts - 1d, OHLCV newest_ts + 1d]
across the recorded universe — the same historical window every prior probe
ran on (that alignment is the point of the backfill).
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

ROOT = os.path.join("data", "chain")


def ohlcv_window() -> tuple[int, int]:
    man = json.load(open(os.path.join("data", "ohlcv", "_manifest.json"), encoding="utf-8"))
    sel = {s["symbol"] for s in json.load(open(os.path.join("data", "selection.json"), encoding="utf-8"))}
    ents = [v for k, v in man.items() if k.endswith("|hour_1") and v["symbol"] in sel]
    return (min(e["oldest_ts"] for e in ents) - 86_400,
            max(e["newest_ts"] for e in ents) + 86_400)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--init", action="store_true", help="build pool registry + scan range")
    ap.add_argument("--tail", action="store_true", help="extend scan to current head")
    ap.add_argument("--panels", action="store_true", help="build hourly panels")
    ap.add_argument("--max-chunks", type=int, default=None, help="stop after N chunks (smoke)")
    args = ap.parse_args()

    from trader.chain.collector import LogCollector
    from trader.chain.registry import build_registry
    from trader.chain.rpc import BscRpc

    if args.init:
        rpc = BscRpc()
        print("building pool registry...")
        build_registry(rpc=rpc)
        ts_from, ts_to = ohlcv_window()
        print(f"resolving block range for [{ts_from}, {ts_to}]...")
        b_from = rpc.block_at_timestamp(ts_from)
        b_to = rpc.block_at_timestamp(ts_to)
        print(f"  blocks {b_from:,} -> {b_to:,} ({b_to - b_from:,} blocks)")
        c = LogCollector(root=ROOT, rpc=rpc)
        if "scan" not in c.manifest:
            c.manifest["scan"] = {"from_block": b_from, "to_block": b_to,
                                  "cursor": b_from - 1}
            c._save_manifest()
            print("manifest initialized")
        else:
            print("manifest already exists — range unchanged:", c.manifest["scan"])
        return

    if args.panels:
        from trader.chain.panels import build_all
        build_all(root=ROOT)
        return

    c = LogCollector(root=ROOT)
    if args.tail:
        head = c.rpc.block_number()
        print(f"extending scan to head {head:,}")
        c.scan(to_block=head, max_chunks=args.max_chunks)
    else:
        c.scan(max_chunks=args.max_chunks)
    print("scan state:", c.manifest["scan"])


if __name__ == "__main__":
    main()
