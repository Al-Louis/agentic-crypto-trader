"""Cross-sectional portfolio backtester — dollar-based, next-bar execution, AMM costs.

A strategy emits target weights each rebalance using only data ≤ the decision bar; those
weights are exposed to the **next** bar's returns. Turnover is charged the AMM cost
(`trader.sim.broker`). Honest by construction: the same cost model applies to every
strategy and baseline, and weights can never see the bar they're judged on.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from trader.sim.broker import DEFAULT_GAS_USD, DEFAULT_LP_FEE_BPS, amm_cost_usd

NEVER = 10 ** 9  # rebalance_every for buy-and-hold (rebalance once at warmup)


def run_xs_backtest(returns: pd.DataFrame, weights_fn, liquidity: dict,
                    capital: float = 10_000.0, rebalance_every: int = 24, warmup: int = 168,
                    min_trade_usd: float = 5.0, lp_fee_bps: float = DEFAULT_LP_FEE_BPS,
                    gas_usd: float = DEFAULT_GAS_USD) -> dict:
    """Run a cross-sectional strategy over a returns panel.

    Args:
        returns: `[timestamp × symbol]` of **simple** per-bar returns.
        weights_fn: `hist_returns -> Series(index=symbol)` target weights (≥0, sum≤1; rest cash).
        liquidity: `symbol -> liquidity_usd` (constant pool-depth proxy for the AMM cost).
    Returns: `{equity, total_cost, total_turnover, n_rebalances}`.
    """
    syms = list(returns.columns)
    pos = pd.Series(0.0, index=syms)            # USD held per asset
    cash = float(capital)
    eq = np.empty(len(returns))
    total_cost = total_turnover = 0.0
    n_rebal = 0
    next_rebal = warmup

    for i in range(len(returns)):
        r = returns.iloc[i].reindex(syms).fillna(0.0).to_numpy()
        pos = pd.Series(pos.to_numpy() * (1.0 + r), index=syms)   # held positions move with bar i
        equity = float(pos.sum() + cash)

        if i >= warmup and i >= next_rebal and equity > 1.0:
            w = weights_fn(returns.iloc[: i + 1]).reindex(syms).fillna(0.0).clip(lower=0.0)
            if w.sum() > 1.0:
                w = w / w.sum()
            target = w * equity
            for sym in syms:
                trade = float(target[sym] - pos[sym])
                if abs(trade) < min_trade_usd:
                    continue
                cost = amm_cost_usd(trade, liquidity.get(sym, 0.0), lp_fee_bps, gas_usd)
                cash -= trade + cost
                pos[sym] += trade
                total_cost += cost
                total_turnover += abs(trade)
            n_rebal += 1
            next_rebal = i + rebalance_every

        eq[i] = float(pos.sum() + cash)

    return {"equity": pd.Series(eq, index=returns.index), "total_cost": total_cost,
            "total_turnover": total_turnover, "n_rebalances": n_rebal}
