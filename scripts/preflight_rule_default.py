"""Gates B + C for rung-1b (rule-default discretion) — run BEFORE any desktop compute.

Gate B (parity): an all-default scripted policy (action idx 0 at every prompt = EXECUTE rung-0)
through the rd env, vs the uncapped rule mirror. The residual gap is the risk-parity caps working
(the agent's "1x rule" on a monster is capped at vol_target/vol) plus queue-order effects; report
both a capped (vol_target on) and uncapped (vol_target=0) parity so the cap effect is isolated.

Gate C (in-env reward landscape, the exp5 lesson): scripted agents through the REAL env on TOTAL
REWARD (the exact signal PPO maximizes). PASS = oracle-lite (hindsight fwd-24h discretion) is the
unique argmax AND both corners (skip-all, all-max) do not beat rule-mimic.

  python scripts/preflight_rule_default.py
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, "scripts")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

WARMUP = 168
RD = dict(k=12, warmup=WARMUP, max_entry_frac=0.34, stop_k=0.25, cooldown=48,
          reward_mode="relative", action_mode="discrete", n_action_levels=4,
          universe_mode="broad", vol_target=0.005, cap_floor=0.02, dd_lambda=0.5, dd_soft=0.15,
          rule_default=True, exit_commit=12, dust_usd=10.0)


def run_scripted(eval_r, btc, liq, vol, kw, decide):
    """One full-window episode; `decide(env, etype, tok) -> action idx`. Returns (ret, total_reward,
    entries, maxDD)."""
    from trader.train.event_env import EventRungEnv
    env = EventRungEnv(eval_r, btc, liq, volume=vol, episode_bars=len(eval_r) - WARMUP - 1,
                       seed=0, **kw)
    env.reset(start=WARMUP)
    total_r, entries, peak, maxdd = 0.0, 0, env.capital, 0.0
    done = False
    while not done:
        etype, tok = env._pending
        if tok is None:
            break
        a = decide(env, etype, tok)
        if etype == "entry" and a in (0, 1, 3):
            entries += 1
        _, r, done, info = env.step([a])
        total_r += r
        eq = info["equity"]
        peak = max(peak, eq)
        maxdd = max(maxdd, (peak - eq) / peak if peak > 0 else 0.0)
    return env._equity() / env.capital - 1.0, total_r, entries, maxdd


def main():
    from train_rl import build_volume_panel, load_data, time_split
    from trader.train.event_env import EventRungEnv

    returns, btc, anchor, liq = load_data()
    train_r, val_r, test_r = time_split(returns)
    vol = build_volume_panel(list(returns.columns), returns.index)

    def rule_mimic(env, etype, tok):
        return 0

    def skip_all(env, etype, tok):
        return 2 if etype == "entry" else 0            # never enter; cut anything (there is nothing)

    def all_max(env, etype, tok):
        return 3                                       # 2x every entry, hold through every exit

    def oracle24(env, etype, tok):
        j = env.col_ix[tok]
        b = env.bar
        fwd = env._px[min(b + 24, env.n_bars - 1), j] / env._px[b, j] - 1.0
        if etype == "entry":
            return 3 if fwd > 0.0 else 2               # 2x winners, skip losers
        return 3 if fwd > 0.0 else 0                   # hold winners, cut losers

    agents = [("rule-mimic", rule_mimic), ("skip-all", skip_all),
              ("all-max", all_max), ("oracle-24h", oracle24)]

    for name, eval_r in (("val", val_r), ("test", test_r)):
        env = EventRungEnv(eval_r, btc, liq, volume=vol, episode_bars=len(eval_r) - WARMUP - 1,
                           seed=0, **RD)
        env.reset(start=WARMUP)
        rule_eq, _ = env._rule_equity_curve(WARMUP, env.end)
        mirror = rule_eq[-1] / rule_eq[0] - 1.0

        print(f"\n=== {name} ===  uncapped rule mirror {mirror:+.1%}")
        # Gate B: parity, capped + uncapped
        ret_c, _, n_c, dd_c = run_scripted(eval_r, btc, liq, vol, RD, rule_mimic)
        kw_u = {**RD, "vol_target": 0.0}
        ret_u, _, n_u, dd_u = run_scripted(eval_r, btc, liq, vol, kw_u, rule_mimic)
        print(f"[B parity] all-default capped:   {ret_c:+.1%} (maxDD {dd_c:.1%}, {n_c} entries)  "
              f"gap-to-mirror {100 * (ret_c - mirror):+.1f}pt (the caps)")
        print(f"[B parity] all-default uncapped: {ret_u:+.1%} (maxDD {dd_u:.1%}, {n_u} entries)  "
              f"gap-to-mirror {100 * (ret_u - mirror):+.1f}pt (should be ~0)")

        # Gate C: in-env reward landscape
        rows = {}
        for an, fn in agents:
            ret, tr, n, dd = run_scripted(eval_r, btc, liq, vol, RD, fn)
            rows[an] = tr
            print(f"[C reward ] {an:10}: total reward {tr:+8.3f}  (return {ret:+7.1%}, "
                  f"maxDD {dd:5.1%}, entries {n})")
        # PASS = the skilled discriminator is the UNIQUE argmax by a clear margin, and neither
        # corner is the argmax. (Skip-all MAY out-reward rule-mimic on a window where the rule
        # loses — that is the reward being honest, not a corner trap; the trap is a corner WINNING.)
        corner_best = max(rows["skip-all"], rows["all-max"])
        margin = rows["oracle-24h"] - max(corner_best, rows["rule-mimic"])
        ok = margin > 0.1
        print(f"[C verdict] {'PASS' if ok else 'FAIL'} — oracle argmax margin {margin:+.3f} "
              f"(needs > +0.1); best corner {corner_best:+.3f} vs rule-mimic {rows['rule-mimic']:+.3f}")


if __name__ == "__main__":
    main()
