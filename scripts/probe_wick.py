"""Probe the WICK-REJECTION entry condition (user idea, the Q Mar 28 00:00 bar): buy on a volume
ignition ONLY if the trigger bar CLOSES near its intra-bar HIGH (`close >= (1-tol)*high`).

A spike that closes at its high = demand absorbed supply; a spike with a large upper wick =
distribution started inside the bar (the close the agent buys is already the rejection). This is
NOT the refuted low-rising filter (24h price progress) — it is the trigger bar's internal shape.

Buckets every in-universe ignition by close/high on the trigger bar; reports fwd-24/48h returns,
the explicit keep/kill cut at `--tol`, and where Q's Mar 28 00:00 bar lands.

  python scripts/probe_wick.py [--tol 0.10]
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "scripts")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

WARMUP = 168


def build_closehigh_panel(tokens, index):
    """Per-token close/high per bar (1.0 = closed at the high; small = big upper wick)."""
    from train_rl import _load_token_ohlcv
    cols = {}
    for t in tokens:
        oh = _load_token_ohlcv(t)
        if oh is None:
            continue
        ts = oh["timestamp"].to_numpy()
        ts = (ts // 1000) if ts.max() > 1e12 else ts
        close, high = oh["close"].to_numpy(), oh["high"].to_numpy()
        with np.errstate(divide="ignore", invalid="ignore"):
            frac = np.where(high > 0, close / high, 1.0)
        cols[t] = pd.Series(frac, index=ts).reindex(index).fillna(1.0).clip(0.0, 1.0)
    return pd.DataFrame(cols, index=index)


def run_split(name, r, btc, liq, vol, ch, tol, k=8):
    from trader.train.event_env import EventRungEnv
    env = EventRungEnv(r, btc, liq, volume=vol, episode_bars=len(r) - WARMUP - 1,
                       k=k, warmup=WARMUP, universe_mode="voltopk", seed=0)
    env.reset(start=WARMUP)
    uni_ix = [env.col_ix[t] for t in env.universe]
    px, ig = env._px, env._ignite
    chm = ch.reindex(r.index).fillna(1.0).to_numpy()

    rows = []                                   # (close/high, fwd24, fwd48)
    n = len(r)
    for j, col in ((env.col_ix[t], t) for t in env.universe):
        for b in range(WARMUP, n - 48):
            if ig[b, j] and px[b, j] > 0:
                jj = list(r.columns).index(col)
                rows.append((chm[b, jj], px[b + 24, j] / px[b, j] - 1.0,
                             px[b + 48, j] / px[b, j] - 1.0))
    rows = np.array(rows)
    print(f"\n=== {name} ===  ignitions: {len(rows)}")
    edges = [(0.97, 1.01, "closed AT high (>=0.97)"), (1 - tol, 0.97, f"strong ({1 - tol:.2f}-0.97)"),
             (0.70, 1 - tol, f"wicked (0.70-{1 - tol:.2f})"), (-0.01, 0.70, "big rejection (<0.70)")]
    for lo, hi, nm in edges:
        m = (rows[:, 0] >= lo) & (rows[:, 0] < hi)
        if not m.sum():
            print(f"  {nm:26}: n=  0")
            continue
        f24, f48 = rows[m, 1], rows[m, 2]
        print(f"  {nm:26}: n={int(m.sum()):4d}  fwd24 {f24.mean():+6.2%} (win {np.mean(f24 > 0):4.0%})  "
              f"fwd48 {f48.mean():+6.2%} (win {np.mean(f48 > 0):4.0%})")
    keep = rows[:, 0] >= (1 - tol)
    for nm, m in ((f"KEPT  close>= {1 - tol:.2f}*high", keep), ("KILLED (wick rejection)", ~keep)):
        f24, f48 = rows[m, 1], rows[m, 2]
        print(f"{nm}: n={int(m.sum())}  fwd24 {f24.mean():+.2%} (win {np.mean(f24 > 0):.0%})  "
              f"fwd48 {f48.mean():+.2%} (win {np.mean(f48 > 0):.0%})")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tol", type=float, default=0.10)
    args = p.parse_args()
    from train_rl import build_volume_panel, load_data, time_split
    returns, btc, anchor, liq = load_data()
    train_r, val_r, test_r = time_split(returns)
    vol = build_volume_panel(list(returns.columns), returns.index)
    ch = build_closehigh_panel(list(returns.columns), returns.index)

    # where does Q's Mar 28 00:00 trigger bar land?
    t = int(pd.Timestamp("2026-03-28T00:00", tz="UTC").timestamp())
    idx = returns.index.to_numpy()
    idx_s = idx // 1000 if idx.max() > 1e12 else idx
    b = int(np.searchsorted(idx_s, t))
    if "Q" in ch.columns and b < len(ch):
        print(f"Q 2026-03-28 00:00 trigger bar close/high = {ch['Q'].iloc[b]:.3f} "
              f"(filter keeps it: {ch['Q'].iloc[b] >= 1 - args.tol})")

    for name, r in (("train", train_r), ("val", val_r)):
        run_split(name, r, btc, liq, vol, ch, args.tol)


if __name__ == "__main__":
    main()
