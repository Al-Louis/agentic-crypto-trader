"""Tabulate a reward-shaping sweep (laptop-side). Pulls every `ppo-<mode>-s<seed>` bundle's
metrics.json from the Apentic data host and reports per-mode averages across seeds, so the
reward mechanisms are compared on the *mean* behavior — not one lucky seed.

Usage:  python scripts/compare_sweep.py [--host https://data.alexlouis.dev] [--steps 500000]
"""
from __future__ import annotations

import argparse
import json
import urllib.request

MODES = ["sharpe", "giveback", "realized", "turnover"]
COLS = [  # (metric key, header, format, scale-to-% ?)
    ("total_return_pct", "return", True),
    ("sharpe_ratio", "Sharpe", False),
    ("max_drawdown_pct", "maxDD", True),
    ("profit_factor", "PF", False),
    ("win_rate", "win", True),
    ("total_trades", "trades", False),
    ("eval_turnover_usd", "turnover$", False),
    ("eval_realized_usd", "realized$", False),
    ("eval_giveback", "giveback", False),
]


def fetch(url):
    with urllib.request.urlopen(url, timeout=30) as r:
        return json.load(r)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--host", default="https://data.alexlouis.dev")
    p.add_argument("--seeds", default="0 1 2")
    args = p.parse_args()
    seeds = args.seeds.split()

    rows = {}  # mode -> list of metrics dicts
    for mode in MODES:
        got = []
        for s in seeds:
            rid = f"ppo-{mode}-s{s}"
            try:
                got.append(fetch(f"{args.host}/{rid}/metrics.json"))
            except Exception as e:  # noqa: BLE001 — missing/not-yet-published run
                print(f"  (skip {rid}: {e})")
        rows[mode] = got

    hdr = f"{'mode':10}" + "".join(f"{h:>11}" for _, h, _ in COLS) + f"{'n':>4}"
    print("\n" + hdr)
    print("-" * len(hdr))
    base_ret = None
    for mode in MODES:
        got = rows[mode]
        if not got:
            print(f"{mode:10}{'— no runs —':>40}")
            continue
        line = f"{mode:10}"
        for key, _h, is_pct in COLS:
            vals = [m.get(key) for m in got if m.get(key) is not None]
            if not vals:
                line += f"{'—':>11}"
                continue
            avg = sum(vals) / len(vals)
            if key == "total_return_pct" and mode == "sharpe":
                base_ret = avg
            line += (f"{avg*100:>+10.1f}%" if is_pct else
                     (f"{avg:>11,.0f}" if abs(avg) >= 100 else f"{avg:>11.2f}"))
        line += f"{len(got):>4}"
        print(line)

    # baseline yardstick (vol-tilt is recorded on each bundle as baseline_return)
    base = next((m.get("baseline_return") for got in rows.values() for m in got
                 if m.get("baseline_return") is not None), None)
    if base is not None:
        print(f"\n  vol-tilt(trend50) baseline on same window: {base*100:+.1f}%")
    if base_ret is not None:
        print(f"  sharpe-control mean return: {base_ret*100:+.1f}%  "
              "(the reward modes must beat THIS to justify the shaping)")
    print()


if __name__ == "__main__":
    main()
