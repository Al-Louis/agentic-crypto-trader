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

    print(f"\n  {'run':24}{'return':>9}{'maxDD':>8}{'Sharpe':>8}{'trades':>7}{'vs B&H':>9}{'gate':>6}")
    for row in r["per_seed"]:
        if "skip" in row:
            print(f"  {row['run_id']:24}  (skip: {row['skip']})")
            continue
        sh = row["sharpe"] if row["sharpe"] is not None else 0.0
        vbh = row.get("vs_buyhold")
        delta = (f"{vbh * 100:>+8.1f}%" if vbh is not None else f"{'-':>9}")
        gate = ("PASS" if row.get("gate_pass") else "FAIL") if row.get("gate_pass") is not None else "-"
        print(f"  {row['run_id']:24}{row['return'] * 100:>+8.1f}%{row['maxdd'] * 100:>7.1f}%"
              f"{sh:>8.2f}{row.get('trades') or 0:>7}{delta}{gate:>6}")

    if r["n"]:
        if r.get("regime") is not None:
            rg = r["regime"]
            print(f"\n  regime: BTC {rg.get('btc_return', 0) * 100:+.1f}%  "
                  f"universe-EW {rg.get('universe_ew_return', 0) * 100:+.1f}%  ({rg.get('label', '?')})")
        print(f"  AVERAGE across {r['n']} seeds: return {r['mean_return'] * 100:+.1f}%  "
              f"(spread +-{r['spread'] * 100:.1f}%, worst {r['worst_return'] * 100:+.1f}%, "
              f"best {r['best_return'] * 100:+.1f}%), maxDD {(r['mean_maxdd'] or 0) * 100:.1f}%")
        bh = r.get("buyhold")
        rnd = r.get("random")
        base = r.get("baseline")
        parts = []
        if bh is not None:
            parts.append(f"Buy&Hold {bh * 100:+.1f}%")
        if rnd is not None:
            parts.append(f"Random {rnd * 100:+.1f}%")
        if base is not None:
            parts.append(f"rung-0 {base * 100:+.1f}%")
        if parts:
            print(f"  baselines on the same window: {'  '.join(parts)}")
        if "gate_pass_mean" in r:
            verdict = ("PASS - seed-mean beats all honest baselines" if r["gate_pass_mean"]
                       else f"FAIL - seed-mean must beat Buy&Hold + Random + rung-0; "
                            f"binding: {r.get('gate_binding')}")
            print(f"  -> HONEST GATE: {verdict}")
    print()


if __name__ == "__main__":
    main()
