"""exp5 IN-ENV landscape gate — the structural fix (no proxy preflight ever again).

The free-pool preflight false-PASSed exp3 AND exp4: it scored all ignitions with free sizing while
the env scores the agent's realized entries. So this gate runs scripted agents THROUGH the real env
(`reward_mode="entry_forward"`, `ungate=True` -> the agent decides over the full ~960-event
in-universe pool, not rung-0's 39 gated decisions) and sums the exact reward PPO maximizes. It gates:
the **correct-selector** (size by an OOS-fit forward-return prediction) must be the **unique argmax**,
with both corners (all-big / all-small) <= rule-mimic. If it fails, RL on this skeleton has no edge —
do NOT sweep; the honest move is to ship rung-0.

  python scripts/preflight_selector.py [--horizon 24 --gamma 0.05]
"""
from __future__ import annotations

import argparse
import sys

sys.path.insert(0, "scripts")
sys.path.insert(0, "src")

import numpy as np  # noqa: E402

from train_rl import build_volume_panel, load_data, time_split  # noqa: E402
from trader.train.event_env import EventRungEnv  # noqa: E402

WARMUP = 168
A_RULE = 0.20 / 0.34 * 2 - 1.0      # action that sizes exactly the rule's 0.20


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--horizon", type=int, default=24)
    ap.add_argument("--gamma", type=float, default=0.05)
    ap.add_argument("--kappa", type=float, default=12.0, help="correct-selector tilt gain")
    args = ap.parse_args()
    H = args.horizon

    returns, btc, anchor, liq = load_data()
    train_r, _, _ = time_split(returns)
    vol = build_volume_panel(list(returns.columns), returns.index)
    kw = dict(volume=vol, k=8, warmup=WARMUP, episode_bars=len(train_r) - WARMUP - 1,
              reward_mode="entry_forward", fwd_horizon=H, res_gamma=args.gamma, dd_lambda=0.0,
              ungate=True, seed=0)
    env = EventRungEnv(train_r, btc, liq, **kw)
    env.reset(start=WARMUP)                                    # sets env.universe (causal vol-top-k)
    bt = env.btc.to_numpy() / env.btc_ema.to_numpy() - 1.0

    # OOS-fit the selector's scoring model on the EARLY in-universe ignitions (causal), test it later
    feats, ys, times = [], [], []
    for t in env.universe:
        j = env.col_ix[t]
        for bar in range(WARMUP, env.n_bars - H):
            if env._ignite[bar, j] and env._px[bar, j] > 0:
                feats.append([1.0, env._cush[bar, j], env._surge[bar, j], float(bt[bar])])
                ys.append(env._px[bar + H, j] / env._px[bar, j] - 1.0)
                times.append(bar)
    X, y = np.array(feats), np.array(ys)
    order = np.argsort(times)
    X, y = X[order], y[order]
    cut = int(len(y) * 0.6)
    coef, *_ = np.linalg.lstsq(X[:cut], y[:cut], rcond=None)
    print(f"in-universe ignitions: {len(y)}   OOS coef [1,cush,surge,btcT] = "
          f"[{', '.join(f'{c:+.3f}' for c in coef)}]")

    def predict(o):                                            # o = obs: cush=o[1], surge=o[2], btcT=o[10]
        return float(coef[0] + coef[1] * o[1] + coef[2] * o[2] + coef[3] * o[10])

    def run(action_fn):
        e = EventRungEnv(train_r, btc, liq, **kw)
        e.reset(start=WARMUP)
        s, n, preds = 0.0, 0, []
        while n < 50000:
            et = e._pending[0]
            if et == "none":
                break
            o = e._obs()
            a = action_fn(et, o, preds)
            _, r, d, _ = e.step([a])
            s += r
            n += 1
            if d:
                break
        return s

    def mimic(et, o, preds):
        return -1.0 if et == "exit" else A_RULE
    def big(et, o, preds):
        return -1.0 if et == "exit" else 1.0
    def small(et, o, preds):
        return -1.0 if et == "exit" else -1.0
    def correct(et, o, preds):                                 # size by rank of the OOS prediction
        if et == "exit":
            return -1.0
        p = predict(o)
        med = float(np.median(preds)) if preds else 0.0
        preds.append(p)
        return float(np.clip(args.kappa * (p - med), -1.0, 1.0))

    agents = {"rule-mimic": mimic, "all-big": big, "all-small": small, "correct-selector": correct}
    rs = {nm: run(fn) for nm, fn in agents.items()}
    print(f"\n  IN-ENV total reward (entry_forward, ungate, dd off, gamma={args.gamma}):")
    for nm in agents:
        print(f"    {nm:18} {rs[nm]:+.4f}")
    win = max(rs, key=rs.get)
    passed = (win == "correct-selector" and rs["all-big"] <= rs["rule-mimic"] + 1e-9
              and rs["all-small"] <= rs["rule-mimic"] + 1e-9)
    print(f"\n  GATE: {'PASS' if passed else 'FAIL'} — "
          + ("correct-selector is the unique in-env argmax, corners <= rule-mimic -> train exp5."
             if passed else "the env still favors a corner -> RL has no edge here; ship rung-0."))


if __name__ == "__main__":
    main()
