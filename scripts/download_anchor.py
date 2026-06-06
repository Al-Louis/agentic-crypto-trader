"""Pull BTC + BNB factor-anchor OHLCV via ccxt (Binance.US) into data/anchor/.

The "Bitcoin-is-King" factor series the alts are regressed against. Incremental and
keyless. Run:
  .venv/Scripts/python.exe scripts/download_anchor.py                 # 1d,1h,1m for 240d
  .venv/Scripts/python.exe scripts/download_anchor.py --days 365 --timeframes 1d,1h
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from trader.data.anchor import download_anchor, load_anchor  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description="BTC/BNB anchor OHLCV via ccxt")
    ap.add_argument("--symbols", default="BTC/USDT,BNB/USDT")
    ap.add_argument("--timeframes", default="1d,1h,1m")
    ap.add_argument("--days", type=int, default=240, help="history depth (covers the alt window)")
    ap.add_argument("--exchange", default="binanceus")
    ap.add_argument("--root", default="data/anchor")
    args = ap.parse_args()

    symbols = [s.strip() for s in args.symbols.split(",") if s.strip()]
    timeframes = [t.strip() for t in args.timeframes.split(",") if t.strip()]

    print(f"anchor download: {symbols} x {timeframes}, last {args.days}d via {args.exchange}")
    download_anchor(symbols, timeframes, args.days, root=args.root, exchange_id=args.exchange)

    print("\n=== anchor cache ===")
    for sym in symbols:
        for tf in timeframes:
            df = load_anchor(sym, tf, root=args.root)
            if len(df):
                gaps = ""
                if tf == "1m":  # flag missing-minute gaps (thin-trading artifacts)
                    span_min = (df["timestamp"].iloc[-1] - df["timestamp"].iloc[0]) // 60_000 + 1
                    missing = span_min - len(df)
                    gaps = f"  gaps={missing} ({missing / span_min:.1%})"
                print(f"  {sym:9} {tf:3} {len(df):>7} candles  "
                      f"[{df['datetime'].iloc[0]:%Y-%m-%d} .. {df['datetime'].iloc[-1]:%Y-%m-%d}]{gaps}")
            else:
                print(f"  {sym:9} {tf:3} (empty)")


if __name__ == "__main__":
    main()
