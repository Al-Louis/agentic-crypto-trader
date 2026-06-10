"""Lever-2 GATE: do the HARVEST features carry incremental-over-`cush` OOS IC on the ungated
in-universe ignition pool? If yes → build the 13→15+ obs and A/B vs the GATE-2 champion. If NO →
drop them, don't spend a sweep (the exp4 lesson: a day lost on features with no headroom).

Features (causal, [[Trading Strategies]] §"Intraday breakout-reversal", market-indicator-expert):
  bkout_short = px[bar] / max(px[bar-23 : bar+1]) - 1      (24h breakout-distance — the signal)
  bkout_med   = px[bar] / max(px[bar-71 : bar+1]) - 1      (72h / 3-day breakout-distance)
  r24, r3d, r7d = px[bar]/px[bar-{24,72,168}] - 1          (short-horizon momentum)
Baseline = [cush, surge, btc_trend] — the exp5 selector predictor (in-env gate passed at γ=0.10).
The test: extended-OLS OOS IC minus baseline OOS IC. Net-of-cost adjudication is the later sweep;
this is only the go/no-go for RUNNING it.

  python scripts/probe_harvest_ic.py [--horizon 24] [--holdout 0.3] [--k 8]
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
HARV = ["bkout_short", "bkout_med", "r24", "r3d", "r7d"]


def _oos_combined_ic(X, y, cut):
    """OLS fit on the early (pre-`cut`) events, combined-prediction IC on the late holdout."""
    Xtr, ytr, Xte, yte = X[:cut], y[:cut], X[cut:], y[cut:]
    A = np.column_stack([np.ones(len(Xtr)), Xtr])
    coef, *_ = np.linalg.lstsq(A, ytr, rcond=None)
    pred = np.column_stack([np.ones(len(Xte)), Xte]) @ coef
    if np.std(pred) < 1e-12 or np.std(yte) < 1e-12:
        return 0.0
    return float(np.corrcoef(pred, yte)[0, 1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--horizon", type=int, default=24)
    ap.add_argument("--holdout", type=float, default=0.3)
    ap.add_argument("--k", type=int, default=8)
    args = ap.parse_args()
    H = args.horizon

    returns, btc, _, liq = load_data()
    train_r, _, _ = time_split(returns)
    vol = build_volume_panel(list(returns.columns), returns.index)
    env = EventRungEnv(train_r, btc, liq, volume=vol, k=args.k, warmup=WARMUP,
                       episode_bars=len(train_r) - WARMUP - 1, reward_mode="absolute",
                       fwd_horizon=H, seed=0)
    env.reset(start=env._min_start)
    uni = set(env.universe)
    px = env._px
    bt = env.btc.to_numpy() / env.btc_ema.to_numpy() - 1.0

    base_rows, harv_rows, ys, ts = [], [], [], []
    for j in range(len(env.cols)):
        if env.cols[j] not in uni:
            continue
        for bar in range(WARMUP, env.n_bars - H):
            if not (env._ignite[bar, j] and px[bar, j] > 0):
                continue
            p = px[bar, j]
            bshort = p / np.max(px[max(bar - 23, 0):bar + 1, j]) - 1.0
            bmed = p / np.max(px[max(bar - 71, 0):bar + 1, j]) - 1.0
            r24 = p / px[bar - 24, j] - 1.0 if px[bar - 24, j] > 0 else 0.0
            r3d = p / px[bar - 72, j] - 1.0 if px[bar - 72, j] > 0 else 0.0
            r7d = p / px[bar - 168, j] - 1.0 if px[bar - 168, j] > 0 else 0.0
            base_rows.append([env._cush[bar, j], env._surge[bar, j], float(bt[bar])])
            harv_rows.append([bshort, bmed, r24, r3d, r7d])
            ys.append(px[bar + H, j] / p - 1.0)
            ts.append(bar)

    B = np.array(base_rows, float)
    Hh = np.nan_to_num(np.array(harv_rows, float))
    y = np.array(ys, float)
    order = np.argsort(np.array(ts, float))
    B, Hh, y = B[order], Hh[order], y[order]
    cut = int(len(y) * (1 - args.holdout))

    base_ic = _oos_combined_ic(B, y, cut)
    ext_ic = _oos_combined_ic(np.column_stack([B, Hh]), y, cut)
    incr = ext_ic - base_ic

    print(f"pool: {len(y)} in-universe ignitions (k={args.k}), holdout {args.holdout:.0%}, H={H}b\n")
    print(f"  baseline [cush, surge, btcT]            OOS IC = {base_ic:+.3f}")
    print(f"  + harvest [+bkout_s/m, r24/r3d/r7d]     OOS IC = {ext_ic:+.3f}   "
          f"(incremental {incr:+.3f})\n")
    yte = y[cut:]
    for k2, nm in enumerate(HARV):
        col = Hh[cut:, k2]
        ic = 0.0 if np.std(col) < 1e-12 else float(np.corrcoef(col, yte)[0, 1])
        print(f"    univariate {nm:12} OOS IC = {ic:+.3f}")

    bucket = (Hh[:, 3] > 0) & (Hh[:, 4] < 0)               # breakout-reversal bucket: r3d>0 & r7d<0
    if bucket.sum() > 5:
        print(f"\n  breakout bucket (r3d>0 & r7d<0): {int(bucket.sum())}/{len(y)} events, "
              f"mean fwd {y[bucket].mean() * 100:+.2f}% vs pool {y.mean() * 100:+.2f}% "
              f"({'momentum-continuation' if y[bucket].mean() > y.mean() else 'no edge'})")

    ok = incr > 0.02
    verdict = ("PASS - harvest features add OOS IC -> build lever-2 obs (13->15+), A/B vs GATE-2" if ok
               else "FAIL - no incremental headroom over cush -> drop, do not sweep (exp4 lesson)")
    print(f"\nGATE: {verdict}  [incremental {incr:+.3f}, threshold +0.02]")


if __name__ == "__main__":
    main()
