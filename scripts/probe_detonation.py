"""Probe the DETONATION-BLACKLIST hypothesis (the Q pattern, 2nd pass): after a token shows an
extreme untradeable-volatility event — a DETONATION (massive volume surge while price collapses,
e.g. Q Mar 28 04:00: surge 35x, rising −25.5%) — are that token's SUBSEQUENT ignitions poison?

User read: all 4 rd8h0c1 seeds skipped Q's spikes but bled ~6-8% of equity each on its late-window
chop; proposal = blacklist a token for the remainder of the window after a detonation. This probe
buckets every in-universe ignition by time-since-last-detonation on its own token and reports
forward returns, so the blacklist (and its horizon) is set by population data, not one token.

  python scripts/probe_detonation.py [--det-surge 8] [--det-drop -0.15]
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, "scripts")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

WARMUP = 168


def run_split(name, r, btc, liq, vol, det_surge, det_drop, k=8):
    from trader.train.event_env import EventRungEnv
    env = EventRungEnv(r, btc, liq, volume=vol, episode_bars=len(r) - WARMUP - 1,
                       k=k, warmup=WARMUP, universe_mode="voltopk", seed=0)
    env.reset(start=WARMUP)
    uni_ix = [env.col_ix[t] for t in env.universe]
    px, ig = env._px, env._ignite

    v = vol.reindex(r.index).fillna(0.0)
    pxf = (1.0 + r.fillna(0.0)).cumprod()
    rising = (pxf / pxf.shift(24) - 1.0).to_numpy()
    vrec = v.rolling(4, min_periods=1).mean()
    vbase = v.shift(4).rolling(164, min_periods=1).mean()
    surge = (vrec / vbase.replace(0.0, np.nan)).fillna(0.0).to_numpy()
    det = (surge >= det_surge) & (rising <= det_drop)            # the detonation signature

    n = len(r)
    rows = []                                                    # (bars_since_det, fwd24, fwd48)
    n_det = 0
    for j in uni_ix:
        det_bars = np.where(det[WARMUP:, j])[0] + WARMUP
        n_det += len(det_bars)
        for b in range(WARMUP, n - 48):
            if not (ig[b, j] and px[b, j] > 0):
                continue
            prior = det_bars[det_bars < b]
            since = (b - prior[-1]) if len(prior) else 10 ** 9
            rows.append((since, px[b + 24, j] / px[b, j] - 1.0, px[b + 48, j] / px[b, j] - 1.0))
    rows = np.array(rows)
    print(f"\n=== {name} ===  detonations on the universe: {n_det}   ignitions: {len(rows)}")
    buckets = [("clean (no prior det)", lambda s: s >= 10 ** 9),
               ("det <=1wk ago", lambda s: s <= 168),
               ("det 1-2wk ago", lambda s: (s > 168) & (s <= 336)),
               ("det 2-4wk ago", lambda s: (s > 336) & (s <= 672)),
               ("det >4wk ago", lambda s: (s > 672) & (s < 10 ** 9))]
    for nm, f in buckets:
        m = f(rows[:, 0])
        if not m.sum():
            print(f"  {nm:22}: n=  0")
            continue
        f24, f48 = rows[m, 1], rows[m, 2]
        print(f"  {nm:22}: n={int(m.sum()):4d}  fwd24 {f24.mean():+6.2%} (win {np.mean(f24 > 0):4.0%})  "
              f"fwd48 {f48.mean():+6.2%} (win {np.mean(f48 > 0):4.0%})")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--det-surge", type=float, default=8.0)
    p.add_argument("--det-drop", type=float, default=-0.15)
    args = p.parse_args()

    from train_rl import build_volume_panel, load_data, time_split
    returns, btc, anchor, liq = load_data()
    train_r, val_r, test_r = time_split(returns)
    vol = build_volume_panel(list(returns.columns), returns.index)
    for name, r in (("train", train_r), ("val", val_r)):
        run_split(name, r, btc, liq, vol, args.det_surge, args.det_drop)


if __name__ == "__main__":
    main()
