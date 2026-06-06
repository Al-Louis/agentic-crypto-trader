"""Honest baseline backtest — does any cross-sectional tilt beat Buy&Hold after AMM costs?

Loads the factor returns panel, runs Buy&Hold / rebalanced equal-weight / cross-sectional
momentum / cross-sectional reversal (the IC-suggested tilt) through the same cost-aware
broker, and reports return / Sharpe / maxDD / Calmar + turnover and cost drag.

Run:  .venv/Scripts/python.exe scripts/run_backtest.py [--capital 10000]
"""

from __future__ import annotations

import argparse
import functools
import glob
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from trader.sim import strategies as S  # noqa: E402
from trader.sim.backtest import NEVER, run_xs_backtest  # noqa: E402
from trader.sim.metrics import PerformanceMetrics  # noqa: E402

HOURS_PER_YEAR = 24 * 365


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--capital", type=float, default=10_000.0)
    ap.add_argument("--rebalance", type=int, default=24, help="bars between rebalances (hourly)")
    ap.add_argument("--warmup", type=int, default=168)
    args = ap.parse_args()
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:  # noqa: BLE001
        pass

    ret = {}
    for f in sorted(glob.glob("data/features/*_factor.parquet")):
        sym = os.path.basename(f)[:-len("_factor.parquet")]
        ret[sym] = pd.read_parquet(f).set_index("timestamp")["r_alt"]
    if not ret:
        print("no factor features — run scripts/build_factor_features.py first")
        return
    returns = np.expm1(pd.DataFrame(ret).sort_index())             # log -> simple

    liq = {s["symbol"]: (s.get("liq_usd") or 0.0)
           for s in json.load(open("data/selection.json", encoding="utf-8"))}

    strategies = [
        ("Buy&Hold (eq-wt)", S.equal_weight, NEVER),
        (f"Equal-wt {args.rebalance}h", S.equal_weight, args.rebalance),
        ("XS momentum k5", functools.partial(S.xs_momentum, lookback=24, k=5), args.rebalance),
        ("XS reversal k5", functools.partial(S.xs_reversal, lookback=24, k=5), args.rebalance),
    ]

    print(f"backtest: {returns.shape[1]} alts, {returns.shape[0]} hourly bars, "
          f"capital ${args.capital:,.0f}\n")
    print(f"  {'strategy':18} {'return':>9} {'Sharpe':>7} {'maxDD':>7} {'Calmar':>7} "
          f"{'turnover':>9} {'costDrag':>9}")
    for name, fn, every in strategies:
        out = run_xs_backtest(returns, fn, liq, capital=args.capital,
                              rebalance_every=every, warmup=args.warmup)
        eq = out["equity"].to_numpy()
        m = PerformanceMetrics.compute_all(eq, steps_per_year=HOURS_PER_YEAR)
        cost_drag = out["total_cost"] / args.capital
        turn = out["total_turnover"] / args.capital
        print(f"  {name:18} {m.total_return_pct:>+8.1%} {m.sharpe_ratio:>7.2f} "
              f"{m.max_drawdown_pct:>6.1%} {m.calmar_ratio:>7.2f} "
              f"{turn:>8.1f}x {cost_drag:>8.1%}")

    print("\n  return = total over the ~7-month sample; costDrag = cumulative AMM cost / capital;")
    print("  turnover = cumulative traded / capital. Beat Buy&Hold on Sharpe AND stay under the")
    print("  ~30% drawdown gate to be worth anything. (Liquidity = current snapshot, constant.)")


if __name__ == "__main__":
    main()
