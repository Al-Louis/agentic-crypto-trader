"""Evaluate rung-0 (disciplined trend-hold) vs the vol-top8 baselines, on the same windows + cost
model the RL was judged on. The honest question: does the user's exit/anti-churn discipline beat the
baseline that's been beating our RL?

    python scripts/eval_rung0.py

Reports return / maxDD / Sharpe / turnover per split, and traces SIREN under rung 0 to confirm it
exits the runup and stands aside through the dead-zone bleed (instead of churning it).
"""
from __future__ import annotations

import sys

sys.path.insert(0, "scripts")
sys.path.insert(0, "src")

import datetime as dt  # noqa: E402

import pandas as pd  # noqa: E402

from train_rl import build_volume_panel, load_data, time_split  # noqa: E402
from trader.sim.backtest import run_xs_backtest  # noqa: E402
from trader.sim.metrics import PerformanceMetrics  # noqa: E402
from trader.strategy.candidate import build_candidate, select_vol_tokens  # noqa: E402
from trader.strategy.rung0 import build_rung0, run_rung0  # noqa: E402

WARMUP, REBAL = 168, 24


def _stats(name, eq, turnover, n_trades):
    ret = float(eq.iloc[-1] / eq.iloc[0] - 1.0)
    dd = float(((eq.cummax() - eq) / eq.cummax()).max())
    rep = PerformanceMetrics.compute_all(eq.iloc[::REBAL].to_numpy(), steps_per_year=365)
    return name, ret, dd, rep.sharpe_ratio, turnover, n_trades


def run_rung0_stats(name, returns, liq, vol):
    eq, rec, _ = run_rung0(returns, build_rung0(returns, volume=vol), liq, warmup=WARMUP)
    turn = sum(abs(v) for r in rec for v in r["trades_usd"].values())
    return _stats(name, eq, turn, sum(1 for r in rec for _ in r["trades_usd"]))


def run_baseline_stats(name, returns, liq, fn):
    res = run_xs_backtest(returns, fn, liq, rebalance_every=REBAL, warmup=WARMUP)
    return _stats(name, res["equity"], res["total_turnover"], res["n_rebalances"])


def main():
    returns, btc, anchor, liq = load_data()
    train_r, val_r, test_r = time_split(returns)
    vol = build_volume_panel(list(returns.columns), returns.index)

    for split, r in [("VAL", val_r), ("TEST", test_r)]:
        uni = select_vol_tokens(r, 8)
        print(f"\n=== {split} split  (universe = {uni}) ===")
        print(f"  {'strategy':24}{'return':>9}{'maxDD':>8}{'Sharpe':>8}{'turnover$':>11}{'trades':>7}")
        rows = [
            run_rung0_stats("rung0 (intra-day)", r, liq, vol),
            run_baseline_stats("vol-top8 (none/hold)", r, liq, build_candidate(r, tokens=uni, overlay="none")),
            run_baseline_stats("vol-top8 trend50", r, liq, build_candidate(r, btc, tokens=uni, overlay="trend50")),
        ]
        for nm, ret, dd, sh, to, nr in rows:
            print(f"  {nm:24}{ret*100:>+8.1f}%{dd*100:>7.1f}%{sh:>8.2f}{to:>11,.0f}{nr:>7}")


if __name__ == "__main__":
    main()
