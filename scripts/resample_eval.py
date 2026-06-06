"""Competition-relevant evaluation: the distribution of 7-day outcomes per strategy.

Samples many random 7-day windows through the cost-aware backtester and reports, per
strategy, the spread of weekly return / max-drawdown and — most importantly — P(breach the
30% DQ gate). Answers "which low-turnover approach actually survives a week?"

Run:  .venv/Scripts/python.exe scripts/resample_eval.py [--samples 400]
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
from trader.sim.backtest import NEVER  # noqa: E402
from trader.sim.resample import evaluate_windows, summarize  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", type=int, default=400)
    ap.add_argument("--capital", type=float, default=10_000.0)
    ap.add_argument("--rebalance", type=int, default=24)
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
    returns = np.expm1(pd.DataFrame(ret).sort_index())
    liq = {s["symbol"]: (s.get("liq_usd") or 0.0)
           for s in json.load(open("data/selection.json", encoding="utf-8"))}

    strategies = [
        ("Buy&Hold (eq-wt)", S.equal_weight, NEVER),
        (f"Equal-wt {args.rebalance}h", S.equal_weight, args.rebalance),
        ("XS momentum k5", functools.partial(S.xs_momentum, lookback=24, k=5), args.rebalance),
        ("XS reversal k5", functools.partial(S.xs_reversal, lookback=24, k=5), args.rebalance),
    ]

    print(f"7-day-window resampling: {returns.shape[1]} alts, {returns.shape[0]} hourly bars, "
          f"{args.samples} windows, capital ${args.capital:,.0f}\n")
    print(f"  {'strategy':18} {'ret_med':>8} {'ret_p5':>8} {'ret_p95':>8} {'mdd_med':>8} "
          f"{'mdd_p95':>8} {'P(DQ)':>6} {'P(win)':>7} {'ret|surv':>9}")
    for name, fn, every in strategies:
        df = evaluate_windows(returns, fn, liq, rebalance_every=every,
                              n_samples=args.samples, capital=args.capital, seed=42)
        s = summarize(df, name)
        print(f"  {name:18} {s['ret_med']:>+7.1%} {s['ret_p5']:>+7.1%} {s['ret_p95']:>+7.1%} "
              f"{s['maxdd_med']:>7.1%} {s['maxdd_p95']:>7.1%} {s['p_dq']:>5.0%} "
              f"{s['p_profit']:>6.0%} {s['ret_med_survived']:>+8.1%}")

    print("\n  P(DQ) = fraction of weeks breaching the 30% drawdown gate (disqualification risk);")
    print("  ret|surv = median weekly return among non-DQ'd weeks. Windows overlap (~30 independent")
    print("  weeks in the sample) — read the spread as a shape. Liquidity = current snapshot, constant.")


if __name__ == "__main__":
    main()
