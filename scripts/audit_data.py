"""Source-data integrity gate (invariant #2): the env's per-token return series (`r_alt`, via
`load_data`) must match the OHLCV candle returns the frontend shows. They diverged silently for 5
tokens — ZEC by +173pt — because the feature pipeline computed a spurious opening-bar return against
a non-existent prior price. `load_data` now zeros each token's first return; this gate proves it and
guards against the class recurring.

Run BEFORE trusting a training run. Exits non-zero if any token diverges beyond --tol.

    python scripts/audit_data.py [--tol 0.02]
"""
from __future__ import annotations

import argparse
import glob
import sys

sys.path.insert(0, "scripts")
sys.path.insert(0, "src")

import pandas as pd  # noqa: E402

from train_rl import load_data  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tol", type=float, default=0.02, help="max |cum r_alt - cum candle| per token")
    args = p.parse_args()

    returns, *_ = load_data()
    fs = lambda i: i // 1000 if i.max() > 1e12 else i
    print(f"invariant #2 — r_alt vs candle cumulative return, per token (tol {args.tol:.0%}):\n")
    print(f"  {'token':12}{'r_alt':>11}{'candle':>11}{'diff':>9}")
    flagged = []
    for tok in returns.columns:
        ds = glob.glob(f"data/ohlcv/hour_1/{tok}_*/*.parquet")
        if not ds:
            continue
        oh = (pd.concat([pd.read_parquet(f) for f in ds]).drop_duplicates("timestamp")
              .sort_values("timestamp").set_index("timestamp"))
        oh.index = fs(oh.index)
        ral = returns[tok].dropna()
        idx = ral.index.intersection(oh.index)
        if len(idx) < 50:
            continue
        rcum = (1 + ral.reindex(idx).fillna(0)).prod() - 1
        ccum = (oh["close"].reindex(idx).pct_change().fillna(0) + 1).prod() - 1
        d = rcum - ccum
        if abs(d) > args.tol:
            flagged.append((tok, d))
        mark = "  FLAG" if abs(d) > args.tol else ""
        print(f"  {tok:12}{rcum*100:>+10.1f}%{ccum*100:>+10.1f}%{d*100:>+8.1f}%{mark}")
    print()
    if flagged:
        print(f"FAIL: {len(flagged)} token(s) diverge > {args.tol:.0%}: "
              f"{', '.join(f'{t} ({d*100:+.0f}pt)' for t, d in flagged)}")
        raise SystemExit(1)
    print("OK: every token's r_alt reconciles with its candles")


if __name__ == "__main__":
    main()
