"""Deviation-alpha diagnostic — is the event-rung RL reward-bound or capacity-bound?

For every entry the agent EXECUTED (across the published seed bundles), measure how much it
over/under-sized vs the rung-0 rule's fixed 0.20, and correlate that deviation with the token's
forward-24h return. If the agent's bigger bets don't land on bigger moves (corr ~0), it is
deviating WITHOUT skill -> the REWARD isn't teaching discrimination (reward-bound, fix the reward).
A clearly positive correlation would mean it discriminates but is capped (capacity-bound).

Thin CLI over `trader.experiment.diagnostics.deviation_alpha` (same logic the rl_diagnose MCP tool uses).

Usage:  python scripts/diag_deviation_alpha.py --prefix ppo-event-rel-test --seeds "0 1 2 3"
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from trader.experiment.diagnostics import deviation_alpha  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="https://data.alexlouis.dev")
    ap.add_argument("--prefix", default="ppo-event-rel-test")
    ap.add_argument("--seeds", default="0 1 2 3")
    args = ap.parse_args()

    d = deviation_alpha(args.prefix, args.seeds.split(), host=args.host)
    print(f"\nexecuted entries analyzed: {d['n_entries']}")
    if "corr" not in d:
        print(d.get("verdict", "inconclusive"))
        return
    print(f"corr(entry over/under-size vs rule 0.20, fwd-24h return) = {d['corr']:+.3f}")
    om = f"{d['over_mean'] * 100:+.2f}% (n={d['over_n']})" if d["over_mean"] is not None else "- (n=0)"
    um = f"{d['under_mean'] * 100:+.2f}% (n={d['under_n']})" if d["under_mean"] is not None else "- (n=0)"
    print(f"mean fwd-24h return: OVERSIZED {om}  |  UNDERSIZED {um}")
    print(f"entry-size range: {d['entry_size_min']:.2f}..{d['entry_size_max']:.2f}")
    print(f"=> {d['verdict']}")


if __name__ == "__main__":
    main()
