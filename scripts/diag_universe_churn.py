"""Quantify how fast the vol-top-k universe goes stale — i.e. how much the env is asking the agent
to absorb when it fixes the traded set once per episode.

The env picks `self.tokens = trailing-warmup vol top-k` ONCE at episode start and holds it for the
whole window. In bursty, rotational microcap markets the optimal set drifts fast. This rolls the
same selection daily across the series and reports, at several lags:
  - avg # of names that ENTER the top-k (out of k) — set churn
  - avg rank correlation of the names that stay — how much the positional slot-map reshuffles
  - per-token persistence — how often each token sits in the top-k at all

    python scripts/diag_universe_churn.py [--k 8] [--warmup 168] [--step 24]
"""
from __future__ import annotations

import argparse
import sys
from collections import Counter

sys.path.insert(0, "scripts")
sys.path.insert(0, "src")

import numpy as np  # noqa: E402

from train_rl import load_data  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--k", type=int, default=8)
    p.add_argument("--warmup", type=int, default=168)
    p.add_argument("--step", type=int, default=24, help="re-pick cadence in bars (24 = daily)")
    args = p.parse_args()

    returns, *_ = load_data()
    n = len(returns)
    picks = []
    for start in range(args.warmup, n - 1, args.step):
        vol = returns.iloc[start - args.warmup:start].std().sort_values(ascending=False)
        picks.append(list(vol.head(args.k).index))
    print(f"{len(picks)} daily vol-top{args.k} picks across the series "
          f"({returns.shape[1]} candidate tokens)\n")

    print(f"  {'lag (days)':>11}{'avg names changed':>19}{'worst':>7}{'avg rank corr':>15}")
    print("  " + "-" * 50)
    for lag in [1, 2, 3, 7, 14, 30]:
        changed, corrs = [], []
        for i in range(len(picks) - lag):
            a, b = picks[i], picks[i + lag]
            sb = set(b)
            changed.append(sum(t not in sb for t in a))      # how many of a's names left
            common = [t for t in a if t in sb]
            if len(common) >= 3:
                ra = [a.index(t) for t in common]
                rb = [b.index(t) for t in common]
                if np.std(ra) > 0 and np.std(rb) > 0:
                    corrs.append(np.corrcoef(ra, rb)[0, 1])
        rc = np.mean(corrs) if corrs else float("nan")
        print(f"  {lag:>11}{np.mean(changed):>17.2f}/{args.k}{max(changed):>7}{rc:>15.2f}")

    print("\n  per-token persistence (share of windows in the top-k):")
    counts = Counter(t for p in picks for t in p)
    for tok, c in counts.most_common():
        bar = "#" * round(40 * c / len(picks))
        print(f"    {tok:12}{c / len(picks):>6.0%} {bar}")
    print(f"\n  {len(counts)}/{returns.shape[1]} tokens appear in the top-{args.k} at least once")


if __name__ == "__main__":
    main()
