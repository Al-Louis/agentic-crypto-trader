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
from trader.strategy.rung0 import build_rung0  # noqa: E402

WARMUP, REBAL = 168, 24


def stats(name, res):
    eq = res["equity"]
    ret = float(eq.iloc[-1] / eq.iloc[0] - 1.0)
    dd = float(((eq.cummax() - eq) / eq.cummax()).max())
    daily = eq.iloc[::REBAL].to_numpy()
    rep = PerformanceMetrics.compute_all(daily, steps_per_year=365)
    return name, ret, dd, rep.sharpe_ratio, res["total_turnover"], res["n_rebalances"]


def run(name, returns, btc, liq, fn):
    return stats(name, run_xs_backtest(returns, fn, liq, rebalance_every=REBAL, warmup=WARMUP))


def main():
    returns, btc, anchor, liq = load_data()
    train_r, val_r, test_r = time_split(returns)
    vol = build_volume_panel(list(returns.columns), returns.index)

    for split, r in [("VAL", val_r), ("TEST", test_r)]:
        print(f"\n=== {split} split  (universe = {select_vol_tokens(r, 8)}) ===")
        print(f"  {'strategy':24}{'return':>9}{'maxDD':>8}{'Sharpe':>8}{'turnover$':>11}{'rebals':>7}")
        rows = [
            run("rung0 (disciplined)", r, btc, liq, build_rung0(r, volume=vol)),
            run("vol-top8 (none/hold)", r, btc, liq, build_candidate(r, btc, overlay="none")),
            run("vol-top8 trend50", r, btc, liq, build_candidate(r, btc, overlay="trend50")),
        ]
        for nm, ret, dd, sh, to, nr in rows:
            print(f"  {nm:24}{ret*100:>+8.1f}%{dd*100:>7.1f}%{sh:>8.2f}{to:>11,.0f}{nr:>7}")

    # SIREN trace under rung 0 (test split) — does it stand aside through the dead-zone?
    print("\n=== SIREN under rung-0 on TEST (held intervals; expect: ride runup, exit, stand aside) ===")
    fn = build_rung0(test_r, volume=vol)
    def d(t): return dt.datetime.fromtimestamp(int(t), dt.timezone.utc).strftime("%b %d")
    prev = 0.0
    for i in range(WARMUP, len(test_r), REBAL):
        w = fn(test_r.iloc[: i + 1])
        sw = float(w.get("SIREN", 0.0)) if len(w) else 0.0
        if (sw > 0) != (prev > 0):
            print(f"  {d(test_r.index[i])}  SIREN -> {'HOLD' if sw > 0 else 'CASH'}")
        prev = sw


if __name__ == "__main__":
    main()
