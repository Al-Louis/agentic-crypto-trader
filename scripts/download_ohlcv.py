"""CLI for the resumable OHLCV downloader (see trader.data.downloader).

Reads a selection of {symbol, pair_address} (from the screen artifacts or ad-hoc
--pairs) and backfills GeckoTerminal OHLCV into the Parquet cache. Safe to Ctrl-C
and re-run: it resumes from the manifest.

Examples:
  # daily+hourly for the first 2 tokens in the proposed set (plumbing test)
  .venv/Scripts/python.exe scripts/download_ohlcv.py --limit-tokens 2

  # ad-hoc pairs, add 1-minute, gentler pacing
  .venv/Scripts/python.exe scripts/download_ohlcv.py \
      --pairs "CAKE:0x0eD7e5...,ASTER:0x..." --timeframes day,hour,minute --min-interval 4
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from trader.data.downloader import OHLCVDownloader, load_ohlcv  # noqa: E402


def _selections_from_file(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    rows = data.get("proposed", data) if isinstance(data, dict) else data
    out = []
    for r in rows:
        if r.get("pair_address") and r.get("symbol"):
            out.append({"symbol": r["symbol"], "pair_address": r["pair_address"]})
    return out


def _selections_from_pairs(spec: str) -> list[dict]:
    out = []
    for tok in spec.split(","):
        if ":" in tok:
            sym, pool = tok.split(":", 1)
            out.append({"symbol": sym.strip(), "pair_address": pool.strip()})
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Resumable GeckoTerminal OHLCV -> Parquet")
    ap.add_argument("--selection", default="data/proposed20.json",
                    help="JSON with proposed/list of {symbol, pair_address}")
    ap.add_argument("--pairs", default=None, help='ad-hoc "SYM:pool,SYM:pool"')
    ap.add_argument("--limit-tokens", type=int, default=0, help="cap number of tokens (0=all)")
    ap.add_argument("--timeframes", default="day,hour", help="comma list of day|hour|minute")
    ap.add_argument("--aggregate", type=int, default=1)
    ap.add_argument("--max-days", type=int, default=190)
    ap.add_argument("--min-interval", type=float, default=3.0)
    ap.add_argument("--root", default="data/ohlcv")
    args = ap.parse_args()

    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:  # noqa: BLE001
        pass

    selections = (_selections_from_pairs(args.pairs) if args.pairs
                  else _selections_from_file(args.selection))
    if args.limit_tokens:
        selections = selections[:args.limit_tokens]
    timeframes = [(tf.strip(), args.aggregate) for tf in args.timeframes.split(",")]

    if not selections:
        print("no selections — provide --pairs or a valid --selection file")
        sys.exit(1)

    print(f"downloading {len(selections)} tokens x {len(timeframes)} timeframes "
          f"-> {args.root}  (min_interval={args.min_interval}s, max_days={args.max_days})")
    dl = OHLCVDownloader(root=args.root, min_interval=args.min_interval, max_days=args.max_days)
    dl.download_many(selections, timeframes)

    print("\n=== cache summary ===")
    for sel in selections:
        for tf, agg in timeframes:
            df = load_ohlcv(sel["symbol"], sel["pair_address"], tf, agg, root=args.root)
            if len(df):
                span = (df["datetime"].iloc[-1] - df["datetime"].iloc[0]).days
                print(f"  {sel['symbol']:12} {tf:6}/{agg}  {len(df):6} candles  {span:4}d  "
                      f"[{df['datetime'].iloc[0].date()} .. {df['datetime'].iloc[-1].date()}]")
            else:
                print(f"  {sel['symbol']:12} {tf:6}/{agg}  (empty)")


if __name__ == "__main__":
    main()
