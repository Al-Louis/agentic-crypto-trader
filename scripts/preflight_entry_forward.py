"""exp4 preflight — prove the entry-forward reward landscape, FAITHFULLY.

The exp3 preflight gave a false PASS: it scored a proxy reward while the env implemented a different
one (per-interval universe demean), so the gate meant nothing. This version scores scripted agents on
the **exact** `trader.train.event_reward.entry_forward_reward` the env trains on, with `mu_base`
computed identically to `EventRungEnv._ignition_base_rate` — single source of reward truth. A PASS is
a real guarantee.

Require the **correct-discriminator** (`dev ∝ −cush`) to be the unique argmax, both corners ≤ 0, the
IC-hacker to lose, and `corr(dev_correct, fwd_ret) > +0.10` (the scripted optimum would clear the live
success gate). If any fails, do NOT sweep.

  python scripts/preflight_entry_forward.py [--horizon 24]
"""
from __future__ import annotations

import argparse
import sys

sys.path.insert(0, "scripts")
sys.path.insert(0, "src")

import numpy as np  # noqa: E402

from train_rl import build_volume_panel, load_data, time_split  # noqa: E402
from trader.train.event_env import EventRungEnv  # noqa: E402
from trader.train.event_reward import entry_forward_reward  # noqa: E402  (the SHARED reward fn)

WARMUP = 168
DEV_HI = 0.34 - 0.20
DEV_LO = 0.0 - 0.20


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--horizon", type=int, default=24)
    args = ap.parse_args()
    H = args.horizon

    returns, btc, anchor, liq = load_data()
    train_r, _, _ = time_split(returns)
    vol = build_volume_panel(list(returns.columns), returns.index)
    env = EventRungEnv(train_r, btc, liq, volume=vol, k=8, warmup=WARMUP,
                       episode_bars=len(train_r) - WARMUP - 1, reward_mode="entry_forward",
                       fwd_horizon=H, seed=0)
    mu_base = env._mu_base                                       # IDENTICAL to what the env credits

    cush, fwd = [], []
    for j in range(len(env.cols)):
        for bar in range(WARMUP, env.n_bars - H):
            if env._ignite[bar, j] and env._px[bar, j] > 0:
                cush.append(env._cush[bar, j])
                fwd.append(env._px[bar + H, j] / env._px[bar, j] - 1.0)
    cush, fwd = np.array(cush), np.array(fwd)
    cz = (cush - cush.mean()) / (cush.std() + 1e-9)
    print(f"train ignitions: {len(fwd)}   mu_base (typical-ignition fwd-{H}b return) = {mu_base * 100:+.2f}%")
    print(f"corr(cush, fwd return) = {np.corrcoef(cush, fwd)[0, 1]:+.3f}")

    dev_correct = np.clip(-DEV_HI * cz, DEV_LO, DEV_HI)
    hacker = np.zeros_like(fwd)
    k = int(np.argmax(np.abs(fwd - mu_base)))
    hacker[k] = DEV_HI * np.sign(fwd[k] - mu_base)
    strategies = {
        "rule-mimic": np.zeros_like(fwd),
        "all-big": np.full_like(fwd, DEV_HI),
        "all-small": np.full_like(fwd, DEV_LO),
        "correct-disc": dev_correct,
        "IC-hacker": hacker,
    }

    def total(dev, g):
        return float(sum(entry_forward_reward(d, f, mu_base, g) for d, f in zip(dev, fwd)))

    gammas = [0.0, 0.02, 0.05, 0.1, 0.2]
    print(f"\n  R via the SHARED entry_forward_reward()   (per-strategy total, {len(fwd)} ignitions)")
    print("  " + "g".rjust(6) + "".join(f"{nm:>14}" for nm in strategies) + "   gate")
    best_g = None
    for g in gammas:
        rs = {nm: total(dev, g) for nm, dev in strategies.items()}
        win = max(rs, key=rs.get)
        ok = (win == "correct-disc" and rs["all-big"] <= 1e-9 and rs["all-small"] <= 1e-9
              and rs["correct-disc"] > rs["IC-hacker"])
        if ok and best_g is None:
            best_g = g
        print("  " + f"{g:>6.2f}" + "".join(f"{rs[nm]:>14.3f}" for nm in strategies)
              + ("   PASS" if ok else "   --"))
    corr_correct = float(np.corrcoef(dev_correct, fwd)[0, 1])
    print(f"\n  corr(dev_correct, fwd) = {corr_correct:+.3f}  (must be > +0.10: the optimum clears the live gate)")
    if best_g is not None and corr_correct > 0.10:
        print(f"\n  GATE PASSES at g = {best_g}: correct-disc unique argmax, corners <= 0, hacker loses,")
        print(f"  and the optimum would pass the corr gate. -> train entry_forward --res-gamma {best_g}.")
    else:
        print("\n  GATE FAILS — do NOT sweep.")


if __name__ == "__main__":
    main()
