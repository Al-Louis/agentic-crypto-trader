"""Probe the FALSE-FLAG hypothesis (the Q token, Mar 28): an ignition with huge volume surge but
weak price progress (`rising` barely > 0) is distribution — someone unloading into the pump — and
its forward returns should be poison, while high-`rising` ignitions are the real runners.

Buckets every in-universe ignition on the TRAIN split (causal voltop8 universe, no eval leakage)
by `rising` (the 24-bar price change at trigger), reports forward-24/48h mean return and win rate
per bucket, plus the explicit rule-candidate cut `rising >= MIN_RISE`.

  python scripts/probe_false_flag.py [--min-rise 0.15] [--split train]
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, "scripts")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

WARMUP = 168


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--min-rise", type=float, default=0.15)
    p.add_argument("--split", default="train", choices=["train", "val", "test"])
    p.add_argument("--k", type=int, default=8)
    args = p.parse_args()

    from train_rl import build_volume_panel, load_data, time_split
    from trader.train.event_env import EventRungEnv

    returns, btc, anchor, liq = load_data()
    splits = dict(zip(("train", "val", "test"), time_split(returns)))
    r = splits[args.split]
    vol = build_volume_panel(list(returns.columns), returns.index)
    env = EventRungEnv(r, btc, liq, volume=vol, episode_bars=len(r) - WARMUP - 1,
                       k=args.k, warmup=WARMUP, universe_mode="voltopk", seed=0)
    env.reset(start=WARMUP)
    uni_ix = [env.col_ix[t] for t in env.universe]
    px, ig = env._px, env._ignite

    # recompute `rising` and `surge` exactly as the env's ignition does
    v = vol.reindex(r.index).fillna(0.0)
    pxf = (1.0 + r.fillna(0.0)).cumprod()
    rising = (pxf / pxf.shift(24) - 1.0).to_numpy()
    vrec = v.rolling(4, min_periods=1).mean()
    vbase = v.shift(4).rolling(164, min_periods=1).mean()
    surge = (vrec / vbase.replace(0.0, np.nan)).fillna(0.0).to_numpy()

    rows = []  # (rising, surge, fwd24, fwd48)
    n = len(r)
    for j in uni_ix:
        for b in range(WARMUP, n - 48):
            if ig[b, j] and px[b, j] > 0:
                rows.append((rising[b, j], surge[b, j],
                             px[b + 24, j] / px[b, j] - 1.0, px[b + 48, j] / px[b, j] - 1.0))
    rows = np.array(rows)
    print(f"[{args.split}] in-universe ignitions: {len(rows)}")

    qs = np.quantile(rows[:, 0], [0.25, 0.5, 0.75])
    print(f"\nrising quartiles: {qs[0]:+.1%} / {qs[1]:+.1%} / {qs[2]:+.1%}")
    edges = [-np.inf, *qs, np.inf]
    for lo, hi, nm in [(edges[i], edges[i + 1], f"Q{i + 1}") for i in range(4)]:
        m = (rows[:, 0] > lo) & (rows[:, 0] <= hi)
        f24, f48 = rows[m, 2], rows[m, 3]
        print(f"  {nm} rising ({lo:+.0%}..{hi:+.0%}]: n={m.sum():4d}  "
              f"fwd24 {f24.mean():+6.2%} (win {np.mean(f24 > 0):4.0%})  "
              f"fwd48 {f48.mean():+6.2%} (win {np.mean(f48 > 0):4.0%})")

    cut = rows[:, 0] >= args.min_rise
    for nm, m in (("KEPT  rising >= cut", cut), ("KILLED rising <  cut", ~cut)):
        f24, f48 = rows[m, 2], rows[m, 3]
        print(f"\n{nm} ({args.min_rise:+.0%}): n={m.sum()}  "
              f"fwd24 {f24.mean():+.2%} (win {np.mean(f24 > 0):.0%})  "
              f"fwd48 {f48.mean():+.2%} (win {np.mean(f48 > 0):.0%})")

    # the false-flag corner specifically: huge surge, weak rise
    ff = (rows[:, 1] >= 8.0) & (rows[:, 0] < args.min_rise)
    if ff.sum():
        print(f"\nFALSE-FLAG corner (surge>=8x AND rising<{args.min_rise:+.0%}): n={ff.sum()}  "
              f"fwd24 {rows[ff, 2].mean():+.2%} (win {np.mean(rows[ff, 2] > 0):.0%})  "
              f"fwd48 {rows[ff, 3].mean():+.2%} (win {np.mean(rows[ff, 3] > 0):.0%})")


if __name__ == "__main__":
    main()
