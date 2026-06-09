"""Average a seed-sweep's published bundles (laptop-side). Pulls each `<prefix>-s<seed>/metrics.json`
from the Apentic data host and reports per-seed return / maxDD / Sharpe plus the across-seed AVERAGE
and spread — single-seed RL is unstable, so the mean (and how tight the seeds cluster) is the read,
not one lucky run. Companion to the mode-based scripts/compare_sweep.py.

Usage:  python scripts/compare_seeds.py --prefix ppo-rung0feat --seeds "0 1 2 3"
"""
from __future__ import annotations

import argparse
import json
import statistics
import urllib.request


def fetch(url):
    with urllib.request.urlopen(url, timeout=30) as r:
        return json.load(r)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--host", default="https://data.alexlouis.dev")
    p.add_argument("--prefix", default="ppo-rung0feat")
    p.add_argument("--seeds", default="0 1 2 3")
    args = p.parse_args()
    seeds = args.seeds.split()

    print(f"\n  {'run':24}{'return':>9}{'maxDD':>8}{'Sharpe':>8}{'trades':>7}{'vs base':>9}")
    rets, dds, base = [], [], None
    for s in seeds:
        rid = f"{args.prefix}-s{s}"
        try:
            m = fetch(f"{args.host}/{rid}/metrics.json")
        except Exception as e:  # noqa: BLE001 — not-yet-published / missing run
            print(f"  {rid:24}  (skip: {e})")
            continue
        r, d, sh = m.get("total_return_pct"), m.get("max_drawdown_pct"), m.get("sharpe_ratio")
        b = m.get("baseline_return")
        base = b if b is not None else base
        rets.append(r)
        dds.append(d)
        delta = f"{(r - b) * 100:>+8.1f}%" if (r is not None and b is not None) else f"{'—':>9}"
        print(f"  {rid:24}{r * 100:>+8.1f}%{d * 100:>7.1f}%{sh:>8.2f}{m.get('total_trades', 0):>7}{delta}")

    if rets:
        avg = sum(rets) / len(rets)
        sd = statistics.pstdev(rets) if len(rets) > 1 else 0.0
        print(f"\n  AVERAGE across {len(rets)} seeds: return {avg * 100:+.1f}%  "
              f"(spread +-{sd * 100:.1f}%, worst {min(rets) * 100:+.1f}%, best {max(rets) * 100:+.1f}%), "
              f"maxDD {sum(dds) / len(dds) * 100:.1f}%")
        if base is not None:
            print(f"  vol-tilt(trend50) baseline on the same val window: {base * 100:+.1f}%")
            print(f"  -> rung-0-features RL {'BEATS' if avg > base else 'loses to'} the baseline on average")
    print()


if __name__ == "__main__":
    main()
