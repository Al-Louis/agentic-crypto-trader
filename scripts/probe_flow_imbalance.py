"""PROBE 2 — FLOW-IMBALANCE -> REVERSION (pre-registered, 2026-06-12).

Hypothesis (the AMM analog of CLOB book imbalance, [[Trading Strategies]]
addendum): trailing net swap-flow imbalance predicts forward returns /
drawdown at COST-SURVIVABLE horizons (>=24h — our round-trip is ~0.7-1%;
sub-minute is dead on arrival).

Measure: imb = rolling-`window`h net_quote_in / rolling vol_quote, bounded
[-1, 1] (+1 = pure one-sided buying pressure INTO the pool). At every bar,
bucket by imbalance quintile and report fwd {24,48,72}h return + worst
trough; plus Spearman IC per horizon. Train/val splits only (test frozen).

  python scripts/probe_flow_imbalance.py [--window 24]
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
HORIZONS = (24, 48, 72)


def imbalance(panel: pd.DataFrame, index: pd.Index, window: int) -> pd.Series:
    p = panel.reindex(index)
    net = p["net_quote_in"].fillna(0.0).rolling(window, min_periods=1).sum()
    vol = p["vol_quote"].fillna(0.0).rolling(window, min_periods=1).sum()
    return (net / vol.replace(0.0, np.nan)).clip(-1, 1)


def run_split(name: str, r: pd.DataFrame, panels: dict[str, pd.DataFrame], window: int):
    pxf = (1.0 + r.fillna(0.0)).cumprod()
    n = len(r)
    syms = [s for s in r.columns if s in panels]
    recs = []          # (imb, fwd24, fwd48, fwd72, worst24, worst48)
    for s in syms:
        imb = imbalance(panels[s], r.index, window).to_numpy()
        px = pxf[s].to_numpy()
        active = panels[s].reindex(r.index)["n_swaps"].fillna(0.0)\
            .rolling(window, min_periods=1).sum().to_numpy()
        for b in range(WARMUP, n - max(HORIZONS)):
            if not (px[b] > 0) or np.isnan(imb[b]) or active[b] < 10:
                continue          # dead pool hours carry no flow information
            recs.append((imb[b],
                         px[b + 24] / px[b] - 1.0,
                         px[b + 48] / px[b] - 1.0,
                         px[b + 72] / px[b] - 1.0,
                         px[b + 1: b + 25].min() / px[b] - 1.0,
                         px[b + 1: b + 49].min() / px[b] - 1.0))
    recs = np.array(recs)
    print(f"\n=== {name} ===  tokens: {len(syms)}   obs: {len(recs):,}")
    if not len(recs):
        return
    # Spearman IC per horizon (pandas-native; ~2/sqrt(n) is the noise bar)
    for h, col in zip(HORIZONS, (1, 2, 3)):
        ic = pd.Series(recs[:, 0]).corr(pd.Series(recs[:, col]), method="spearman")
        print(f"  IC(imb -> fwd{h}) = {ic:+.4f}  (noise ~{2 / np.sqrt(len(recs)):.4f})")
    # quintiles
    q = np.quantile(recs[:, 0], [0.2, 0.4, 0.6, 0.8])
    edges = [-np.inf, *q, np.inf]
    print(f"  quintile edges: {[f'{e:+.3f}' for e in q]}")
    for i in range(5):
        m = (recs[:, 0] > edges[i]) & (recs[:, 0] <= edges[i + 1])
        sub = recs[m]
        print(f"  Q{i + 1} (n={int(m.sum()):7d}): "
              f"fwd24 {sub[:, 1].mean():+7.2%}  fwd48 {sub[:, 2].mean():+7.2%}  "
              f"fwd72 {sub[:, 3].mean():+7.2%}  worst24 {sub[:, 4].mean():+7.2%}  "
              f"worst48 {sub[:, 5].mean():+7.2%}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--window", type=int, default=24)
    args = ap.parse_args()

    from train_rl import load_data, time_split
    from trader.chain.panels import load_panel
    from trader.chain.registry import load_registry

    returns, _btc, _anchor, _liq = load_data()
    panels = {}
    for p in load_registry():
        try:
            panels[p["symbol"]] = load_panel(p["symbol"])
        except FileNotFoundError:
            pass
    train_r, val_r, _test = time_split(returns)
    for name, rr in (("train", train_r), ("val", val_r)):
        run_split(name, rr, panels, args.window)


if __name__ == "__main__":
    main()
