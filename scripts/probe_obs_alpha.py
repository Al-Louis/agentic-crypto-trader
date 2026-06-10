"""Discrimination-headroom probe — is the event-rung gap reward-bound or capacity/input-bound?

At every rung-0 ignition on the TRAIN split, take the obs features the policy actually sees for the
entry-sizing decision (cush = price/EMA-1, surge = volume ratio, btc_trend) and ask: do they predict
the token's forward return, OUT-OF-SAMPLE (temporal holdout within train)? This is the upper bound on
what ANY policy can extract from the current obs.

  - Real OOS signal (IC clearly > 0)  -> the alpha IS in the inputs -> reward-bound -> R4 / reward fix.
  - Flat (IC ~ 0)                      -> the alpha is NOT in the obs -> no reward fix helps -> upgrade
                                          the features (and only then consider an LSTM).

No training, no torch — reuses the env's precomputed causal signals. Run:
  python scripts/probe_obs_alpha.py [--horizon 24]
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


def _ic(pred, y):
    if len(y) < 3 or np.std(pred) < 1e-12 or np.std(y) < 1e-12:
        return 0.0
    return float(np.corrcoef(pred, y)[0, 1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--horizon", type=int, default=24, help="forward bars for the target return")
    ap.add_argument("--holdout", type=float, default=0.3, help="temporal OOS fraction (last X of ignitions)")
    args = ap.parse_args()
    H = args.horizon

    returns, btc, anchor, liq = load_data()
    train_r, _, _ = time_split(returns)
    vol = build_volume_panel(list(returns.columns), returns.index)
    env = EventRungEnv(train_r, btc, liq, volume=vol, k=8, warmup=WARMUP,
                       episode_bars=len(train_r) - WARMUP - 1, reward_mode="absolute", seed=0)

    bt = (env.btc.to_numpy() / env.btc_ema.to_numpy() - 1.0)     # btc_trend per bar
    feats, ys, times = [], [], []
    n = env.n_bars
    for j in range(len(env.cols)):                              # every token, every ignition bar on train
        for bar in range(WARMUP, n - H):
            if env._ignite[bar, j] and env._px[bar, j] > 0:
                fwd = env._px[bar + H, j] / env._px[bar, j] - 1.0
                feats.append([env._cush[bar, j], env._surge[bar, j], float(bt[bar])])
                ys.append(fwd)
                times.append(bar)
    X = np.array(feats, float)
    y = np.array(ys, float)
    order = np.argsort(times)                                  # temporal order for an honest OOS split
    X, y = X[order], y[order]
    print(f"ignition events on TRAIN: {len(y)}   |   forward horizon: {H} bars")
    if len(y) < 20:
        print("too few ignitions to probe")
        return

    print(f"unconditional mean fwd-{H}b return of an ignition: {y.mean() * 100:+.2f}% "
          f"(the rung-0 'take the ignition' base rate)")
    names = ["cush(px/EMA-1)", "surge(vol)", "btc_trend"]
    print("\nunivariate corr(feature, fwd return) [full sample]:")
    for k, nm in enumerate(names):
        print(f"  {nm:16} {np.corrcoef(X[:, k], y)[0, 1]:+.3f}")

    cut = int(len(y) * (1 - args.holdout))                    # fit on the earlier ignitions, test on later
    Xtr, ytr, Xte, yte = X[:cut], y[:cut], X[cut:], y[cut:]
    A = np.column_stack([np.ones(len(Xtr)), Xtr])             # OLS y ~ 1 + features
    coef, *_ = np.linalg.lstsq(A, ytr, rcond=None)
    pred = np.column_stack([np.ones(len(Xte)), Xte]) @ coef
    ic = _ic(pred, yte)
    sign_acc = float(np.mean((pred > pred.mean()) == (yte > 0)))   # does high pred -> up?
    top, bot = yte[pred >= np.median(pred)], yte[pred < np.median(pred)]
    print(f"\nOOS linear probe (fit {len(ytr)} / test {len(yte)}):")
    print(f"  OOS IC (corr pred vs realized fwd return) = {ic:+.3f}")
    print(f"  fwd return, top-half predicted {top.mean() * 100:+.2f}%  vs  bottom-half {bot.mean() * 100:+.2f}%"
          f"  (spread {(top.mean() - bot.mean()) * 100:+.2f}pt)")
    print(f"\n=> REWARD-BOUND (run R4) if OOS IC clearly > 0 and the top/bottom spread is positive;")
    print(f"   CAPACITY/INPUT-BOUND (upgrade obs) if IC ~ 0 — the obs can't discriminate winners.")


if __name__ == "__main__":
    main()
