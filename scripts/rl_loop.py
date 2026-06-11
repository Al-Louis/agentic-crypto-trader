"""The RL-loop driver CLI — one tick of the autonomous iterate loop (laptop-side).

    python scripts/rl_loop.py status            # loop state, no side effects
    python scripts/rl_loop.py step              # advance one tick (launch / poll / verdict+decide)
    python scripts/rl_loop.py propose --config '{"reward_mode":"relative",...}' \
        [--timesteps 1000000] [--seeds "0 1 2 3"] [--prefix ppo-event-x] [--sha <hex>] [--note ...]
    python scripts/rl_loop.py reset [--hard]    # clear a halt (or wipe state entirely)

The judgment step (designing the next config) belongs to the driving agent (/rl-loop skill);
this CLI is the mechanical state machine (vault "MCP Server" §driver).
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from trader.experiment import driver  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("status")
    sub.add_parser("step")
    pp = sub.add_parser("propose")
    pp.add_argument("--config", required=True, help="reward_config JSON (the full knob dict)")
    pp.add_argument("--timesteps", type=int, default=driver.DEFAULT_TIMESTEPS)
    pp.add_argument("--seeds", default=driver.DEFAULT_SEEDS)
    pp.add_argument("--prefix", default=None)
    pp.add_argument("--sha", default=None, help="sync the desktop to this commit before launch")
    pp.add_argument("--note", default="", help="one line: the hypothesis this config tests")
    pr = sub.add_parser("reset")
    pr.add_argument("--hard", action="store_true")
    args = p.parse_args()

    if args.cmd == "status":
        out = driver.load_state()
    elif args.cmd == "step":
        out = driver.step()
    elif args.cmd == "propose":
        out = driver.propose(json.loads(args.config), note=args.note, timesteps=args.timesteps,
                             seeds=args.seeds, prefix=args.prefix, sha=args.sha)
    else:
        out = driver.reset(hard=args.hard)
    print(json.dumps(out, indent=2, default=str))


if __name__ == "__main__":
    main()
