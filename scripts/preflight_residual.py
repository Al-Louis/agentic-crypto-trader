"""exp3 preflight — prove the demeaned-ranked residual reward landscape BEFORE training.

The corner-solution lesson: every prior reward let a scripted corner agent (all-big / all-small) tie
or beat a skilled one, so RL found the corner. This checks the reward LANDSCAPE directly on real
train-ignition (cush, forward-return) data: it scores scripted sizing strategies on
`R = Σ dev·(ret − ret_bar) − γ·Σ dev²` and requires the **correct-discriminator** (size ∝ −cush, per
the probe's IC) to be the **unique argmax**, with both corners ≤ the rule-mimic (=0) and a one-shot
IC-hacker unable to win. It also picks `res_gamma`. If this gate fails, do NOT spend a sweep.

(Landscape proxy: demeans returns globally rather than per-interval — enough to verify the FORM
makes conditional sizing win; the env uses the finer per-interval cross-sectional mean.)

  python scripts/preflight_residual.py [--horizon 24]
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
DEV_HI = 0.34 - 0.20      # max over-size (max_entry_frac - rule_entry_frac)
DEV_LO = 0.0 - 0.20       # max under-size (skip the rule's 0.20)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--horizon", type=int, default=24)
    args = ap.parse_args()
    H = args.horizon

    returns, btc, anchor, liq = load_data()
    train_r, _, _ = time_split(returns)
    vol = build_volume_panel(list(returns.columns), returns.index)
    env = EventRungEnv(train_r, btc, liq, volume=vol, k=8, warmup=WARMUP,
                       episode_bars=len(train_r) - WARMUP - 1, reward_mode="absolute", seed=0)
    n = env.n_bars
    cush, ret = [], []
    for j in range(len(env.cols)):
        for bar in range(WARMUP, n - H):
            if env._ignite[bar, j] and env._px[bar, j] > 0:
                cush.append(env._cush[bar, j])
                ret.append(env._px[bar + H, j] / env._px[bar, j] - 1.0)
    cush, ret = np.array(cush), np.array(ret)
    retc = ret - ret.mean()                                  # demean (remove the unconditional drift)
    cz = (cush - cush.mean()) / (cush.std() + 1e-9)
    print(f"train ignitions: {len(ret)}   corr(cush, ret) = {np.corrcoef(cush, ret)[0, 1]:+.3f}")

    # scripted deviation vectors
    dev_correct = np.clip(-DEV_HI * cz, DEV_LO, DEV_HI)      # size up low-cush, down high-cush
    hacker = np.zeros_like(retc)
    k = int(np.argmax(np.abs(retc)))                         # one big correct-signed bet, 0 elsewhere
    hacker[k] = DEV_HI * np.sign(retc[k])
    strategies = {
        "rule-mimic (dev=0)": np.zeros_like(retc),
        "all-big (oversize)": np.full_like(retc, DEV_HI),
        "all-small (under)": np.full_like(retc, DEV_LO),
        "correct-disc (∝-cush)": dev_correct,
        "IC-hacker (1 bet)": hacker,
    }

    def reward(dev, g):
        return float(np.sum(dev * retc) - g * np.sum(dev * dev))

    gammas = [0.0, 0.05, 0.1, 0.2, 0.4, 0.8]
    print(f"\n  reward R = sum dev*(ret-mean) - g*sum dev^2   (per-strategy total, {len(ret)} ignitions)")
    print("  " + "g".rjust(6) + "".join(f"{nm.split()[0]:>16}" for nm in strategies) + "   gate")
    best_g = None
    for g in gammas:
        rs = {nm: reward(dev, g) for nm, dev in strategies.items()}
        win = max(rs, key=rs.get)
        corners_ok = rs["all-big (oversize)"] <= 1e-9 and rs["all-small (under)"] <= 1e-9
        hacker_ok = rs["correct-disc (∝-cush)"] > rs["IC-hacker (1 bet)"]
        passed = win.startswith("correct-disc") and corners_ok and hacker_ok
        if passed and best_g is None:
            best_g = g
        print("  " + f"{g:>6.2f}" + "".join(f"{rs[nm]:>16.3f}" for nm in strategies)
              + ("   PASS" if passed else "   --"))
    print(f"\n  corr(dev, ret) for correct-disc = {np.corrcoef(dev_correct, ret)[0, 1]:+.3f} "
          "(positive = it sizes winners bigger)")
    if best_g is not None:
        print(f"\n  GATE PASSES at g = {best_g}: correct-discriminator is the unique argmax, both corners <= 0,")
        print(f"  IC-hacker loses. -> train residual_ranked with --res-gamma {best_g}.")
    else:
        print("\n  GATE FAILS at every g — the reward landscape still lets a corner/hacker win. Do NOT sweep.")


if __name__ == "__main__":
    main()
