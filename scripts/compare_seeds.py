"""Average a seed-sweep's published bundles (laptop-side). Pulls each `<prefix>-s<seed>/metrics.json`
from the Apentic data host and reports per-seed return / maxDD / Sharpe plus the across-seed AVERAGE
and spread — single-seed RL is unstable, so the mean (and how tight the seeds cluster) is the read,
not one lucky run. Companion to the mode-based scripts/compare_sweep.py.

Thin CLI over `trader.experiment.diagnostics.compare_seeds` (same logic the rl_compare MCP tool uses).

Usage:  python scripts/compare_seeds.py --prefix ppo-rung0feat --seeds "0 1 2 3"
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from trader.experiment.diagnostics import compare_seeds  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--host", default="https://data.alexlouis.dev")
    p.add_argument("--prefix", default="ppo-rung0feat")
    p.add_argument("--seeds", default="0 1 2 3")
    args = p.parse_args()

    r = compare_seeds(args.prefix, args.seeds.split(), host=args.host)

    print(f"\n  {'run':24}{'return':>9}{'maxDD':>8}{'Sharpe':>8}{'trades':>7}{'vs base':>9}")
    for row in r["per_seed"]:
        if "skip" in row:
            print(f"  {row['run_id']:24}  (skip: {row['skip']})")
            continue
        sh = row["sharpe"] if row["sharpe"] is not None else 0.0
        delta = (f"{row['vs_baseline'] * 100:>+8.1f}%" if row["vs_baseline"] is not None
                 else f"{'—':>9}")
        print(f"  {row['run_id']:24}{row['return'] * 100:>+8.1f}%{row['maxdd'] * 100:>7.1f}%"
              f"{sh:>8.2f}{row.get('trades') or 0:>7}{delta}")

    if r["n"]:
        print(f"\n  AVERAGE across {r['n']} seeds: return {r['mean_return'] * 100:+.1f}%  "
              f"(spread +-{r['spread'] * 100:.1f}%, worst {r['worst_return'] * 100:+.1f}%, "
              f"best {r['best_return'] * 100:+.1f}%), maxDD {(r['mean_maxdd'] or 0) * 100:.1f}%")
        if r["baseline"] is not None:
            print(f"  baseline on the same window: {r['baseline'] * 100:+.1f}%")
            print(f"  -> RL {'BEATS' if r['beats_baseline'] else 'loses to'} the baseline on average")
    print()


if __name__ == "__main__":
    main()
