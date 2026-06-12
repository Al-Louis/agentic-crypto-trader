"""PROBE 1 — LP-PULL -> DETONATION LEAD (pre-registered, 2026-06-12).

Hypothesis (the order-book-depth addendum in [[Trading Strategies]]): LP
liquidity withdrawal (v2/v3 Burn events) PRECEDES the detonation bars
(surge>=8x while rising<=-15%, the det-blacklist signature) and the
post-detonation poison window — i.e. the pool-event stream upgrades our best
guardrail from reactive to predictive.

GRADED ON DRAWDOWN (the DQ-relevant target), not just returns:
  (a) lead time — for each detonation, hours since the last BIG PULL
      (trailing-24h LP removal >= --pull-frac of the pool), vs how often big
      pulls fire at all (a precursor that fires weekly is no precursor);
  (b) conditional forward damage — bucket EVERY bar by "big pull within
      trailing H hours" vs clean, report fwd 24/48h worst trough
      (min future px / px - 1) and mean return;
  (c) detonation precision/recall — P(detonation within 48h | big pull) vs
      base rate.

Splits: train/val only (test frozen). Population-level across the universe.

  python scripts/probe_lp_pull.py [--pull-frac 0.10] [--lookback 24]
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
DET_SURGE, DET_DROP = 8.0, -0.15           # the det-blacklist signature (probe-calibrated)


def detonation_mask(r: pd.DataFrame, vol: pd.DataFrame) -> np.ndarray:
    """Same construction as probe_detonation.py / the env's det_blacklist."""
    v = vol.reindex(r.index).fillna(0.0)
    pxf = (1.0 + r.fillna(0.0)).cumprod()
    rising = (pxf / pxf.shift(24) - 1.0).to_numpy()
    vrec = v.rolling(4, min_periods=1).mean()
    vbase = v.shift(4).rolling(164, min_periods=1).mean()
    surge = (vrec / vbase.replace(0.0, np.nan)).fillna(0.0).to_numpy()
    return (surge >= DET_SURGE) & (rising <= DET_DROP)


def pull_intensity(panel: pd.DataFrame, index: pd.Index, lookback: int) -> pd.Series:
    """Trailing-`lookback`h LP removal as a fraction of the pool's quote side."""
    p = panel.reindex(index)
    removed = p["lp_remove_quote"].fillna(0.0).rolling(lookback, min_periods=1).sum()
    base = p["reserve_quote_end"].ffill().shift(lookback)
    return (removed / base.replace(0.0, np.nan)).fillna(0.0)


def run_split(name: str, r: pd.DataFrame, vol: pd.DataFrame,
              panels: dict[str, pd.DataFrame], pull_frac: float, lookback: int):
    det = detonation_mask(r, vol)
    pxf = (1.0 + r.fillna(0.0)).cumprod()
    n = len(r)
    syms = [s for s in r.columns if s in panels]

    leads, n_det, n_det_led = [], 0, 0
    rows = []          # (pulled_recently, fwd24_worst, fwd48_worst, fwd24_ret, fwd48_ret, det_next48)
    pull_rate_hours = 0
    total_hours = 0

    for s in syms:
        j = r.columns.get_loc(s)
        z = pull_intensity(panels[s], r.index, lookback).to_numpy()
        big = z >= pull_frac
        px = pxf[s].to_numpy()
        det_bars = np.where(det[WARMUP:, j])[0] + WARMUP
        n_det += len(det_bars)
        big_bars = np.where(big[WARMUP:])[0] + WARMUP
        pull_rate_hours += int(big[WARMUP:].sum())
        total_hours += n - WARMUP
        for b in det_bars:
            prior = big_bars[big_bars < b]
            if len(prior) and b - prior[-1] <= 168:
                n_det_led += 1
                leads.append(b - prior[-1])
        for b in range(WARMUP, n - 48):
            if not (px[b] > 0):
                continue
            w24 = px[b + 1: b + 25].min() / px[b] - 1.0
            w48 = px[b + 1: b + 49].min() / px[b] - 1.0
            f24 = px[b + 24] / px[b] - 1.0
            f48 = px[b + 48] / px[b] - 1.0
            rows.append((big[b], w24, w48, f24, f48, det[b + 1: b + 49, j].any()))

    rows = np.array(rows, dtype=float)
    print(f"\n=== {name} ===  tokens with panels: {len(syms)}/{len(r.columns)}   "
          f"detonations: {n_det}   big-pull hours: {pull_rate_hours} "
          f"({pull_rate_hours / max(total_hours, 1):.2%} of all hours)")
    if n_det:
        led = n_det_led / n_det
        print(f"  (a) LEAD: {n_det_led}/{n_det} detonations had a big pull "
              f"<=168h before ({led:.0%})"
              + (f"; lead-time median {np.median(leads):.0f}h "
                 f"p25 {np.percentile(leads, 25):.0f}h p75 {np.percentile(leads, 75):.0f}h"
                 if leads else ""))
    for label, m in (("pulled (trailing big pull)", rows[:, 0] == 1),
                     ("clean", rows[:, 0] == 0)):
        if not m.sum():
            print(f"  (b) {label:28}: n=0")
            continue
        sub = rows[m]
        print(f"  (b) {label:28}: n={int(m.sum()):7d}  "
              f"fwd24 worst {sub[:, 1].mean():+7.2%} | fwd48 worst {sub[:, 2].mean():+7.2%}  "
              f"fwd24 ret {sub[:, 3].mean():+7.2%} | fwd48 ret {sub[:, 4].mean():+7.2%}  "
              f"P(det<=48h) {sub[:, 5].mean():6.2%}")
    base = rows[:, 5].mean()
    pulled = rows[rows[:, 0] == 1]
    if len(pulled):
        print(f"  (c) detonation within 48h: base {base:.2%} -> given big pull "
              f"{pulled[:, 5].mean():.2%}  (lift x{pulled[:, 5].mean() / max(base, 1e-9):.1f})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pull-frac", type=float, default=0.10,
                    help="trailing LP removal >= this fraction of pool = BIG PULL")
    ap.add_argument("--lookback", type=int, default=24)
    args = ap.parse_args()

    from train_rl import build_volume_panel, load_data, time_split
    from trader.chain.panels import load_panel
    from trader.chain.registry import load_registry

    returns, _btc, _anchor, _liq = load_data()
    vol = build_volume_panel(list(returns.columns), returns.index)
    panels = {}
    for p in load_registry():
        try:
            panels[p["symbol"]] = load_panel(p["symbol"])
        except FileNotFoundError:
            pass
    train_r, val_r, _test = time_split(returns)
    for name, rr in (("train", train_r), ("val", val_r)):
        run_split(name, rr, vol.loc[rr.index], panels, args.pull_frac, args.lookback)


if __name__ == "__main__":
    main()
