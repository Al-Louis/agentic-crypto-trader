"""Gate A for rung-1b: the event skeleton's ORACLE CEILING — how much can PERFECT discretion
extract from rung-0's event set, through the real env (g2b config: broad k=12, risk-parity caps,
real AMM costs, cooldown/dead-zone/rotation intact)?

A hindsight-greedy scripted agent answers every prompt with the future in hand:
  entry prompt -> max size iff the token's forward-H return is positive, else skip;
  exit prompt  -> hold (override) iff forward-H is up, else cut in full.
Decomposed: entry-only (exits = rule's cut), exit-only (entries = always max), both.
This BOUNDS what any learned discretion can do inside the skeleton. Kill criterion
([[AI Training]] rung-1b): if oracle-val < Buy&Hold, no policy in this skeleton can pass the
honest gate -> pivot the substrate instead of building rule-default.

  python scripts/preflight_skeleton_ceiling.py [--horizons 24 48]
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, "scripts")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

WARMUP = 168

# the g2b env config (provenance of ppo-event-g2b-*): the substrate under test
G2B = dict(k=12, warmup=WARMUP, max_entry_frac=0.34, stop_k=0.25, cooldown=48,
           reward_mode="relative", action_mode="discrete", n_action_levels=4,
           universe_mode="broad", vol_target=0.005, cap_floor=0.02, dd_lambda=0.5, dd_soft=0.15)


def run_oracle(eval_r, btc, liq, vol, *, horizon: int, mode: str, seed: int = 0):
    """One full-window episode with hindsight answers. mode: both|entry|exit|none.
    'none' = all-default (entry skip is NOT default here — current env default is do-nothing,
    so 'none' answers cut/skip everywhere = the passivity floor)."""
    from trader.train.event_env import EventRungEnv
    env = EventRungEnv(eval_r, btc, liq, volume=vol, episode_bars=len(eval_r) - WARMUP - 1,
                       seed=seed, **G2B)
    env.reset(start=WARMUP)
    n = env.n_bars
    done, trades = False, 0
    while not done:
        etype, tok = env._pending
        if tok is None:
            break
        j = env.col_ix[tok]
        b = env.bar
        fwd = env._px[min(b + horizon, n - 1), j] / env._px[b, j] - 1.0
        if etype == "entry":
            a = 3 if (mode in ("both", "entry") and fwd > 0.0) else 0   # max size vs skip
        else:
            a = 3 if (mode in ("both", "exit") and fwd > 0.0) else 0    # hold-through vs cut
        if a == 3 and etype == "entry":
            trades += 1
        _, _, done, info = env.step(a)
    eq = env._equity()
    # max drawdown over the recorded trace is only kept with record_trace; recompute cheaply
    return eq / env.capital - 1.0, trades


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--horizons", nargs="*", type=int, default=[24, 48])
    args = p.parse_args()

    from train_rl import build_volume_panel, load_data, time_split
    from trader.train.event_env import EventRungEnv

    returns, btc, anchor, liq = load_data()
    train_r, val_r, test_r = time_split(returns)
    vol = build_volume_panel(list(returns.columns), returns.index)

    for name, eval_r in (("val", val_r), ("test", test_r)):
        # context: the rule mirror on the same universe (computed by the env itself)
        env = EventRungEnv(eval_r, btc, liq, volume=vol, episode_bars=len(eval_r) - WARMUP - 1,
                           seed=0, **G2B)
        env.reset(start=WARMUP)
        rule_eq, _ = env._rule_equity_curve(WARMUP, env.end)
        rule_ret = rule_eq[-1] / rule_eq[0] - 1.0
        print(f"\n[{name}] rule mirror (uncapped ef, agent's {G2B['k']}-token universe): {rule_ret:+.1%}")
        for H in args.horizons:
            for mode in ("both", "entry", "exit", "none"):
                ret, trades = run_oracle(eval_r, btc, liq, vol, horizon=H, mode=mode)
                print(f"[{name}] oracle H={H:2d} {mode:5}: return {ret:+8.1%}  entries {trades}")


if __name__ == "__main__":
    main()
