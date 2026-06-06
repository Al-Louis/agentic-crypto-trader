"""AMM execution-cost model — constant-product price impact for thin BSC pools.

The CEX→on-chain slippage rebuild the TradeSim post-mortem demands (vault "Simulated
Market"): TradeSim's volume-based slippage was harmful on sparse data and a fixed spread
suited liquid BTC; our **thin AMM pools have real price impact**. For a constant-product
pool (x·y=k) with one-sided quote reserve `Q ≈ liquidity_usd/2`, a trade of size `S` has
average execution-price impact ≈ `S/Q`, plus the LP fee and gas. Big trades in small pools
cost a lot — exactly the reality that should kill thin-token churn in a backtest.
"""

from __future__ import annotations

DEFAULT_LP_FEE_BPS = 25.0   # PancakeSwap v2 ≈ 0.25%
DEFAULT_GAS_USD = 0.20      # BSC swap gas, approx


def amm_cost_usd(trade_usd: float, liquidity_usd: float,
                 lp_fee_bps: float = DEFAULT_LP_FEE_BPS,
                 gas_usd: float = DEFAULT_GAS_USD) -> float:
    """Total USD cost to execute a `trade_usd` (absolute) swap against a pool.

    `cost = LP fee + constant-product price impact (S/Q, Q≈liquidity/2) + flat gas`.
    Zero/unknown liquidity → impact of 100% (effectively untradeable).
    """
    s = abs(float(trade_usd))
    if s == 0.0:
        return 0.0
    q = (liquidity_usd or 0.0) / 2.0
    impact_frac = s / q if q > 0 else 1.0
    return s * (lp_fee_bps / 1e4 + impact_frac) + gas_usd


def amm_cost_bps(trade_usd: float, liquidity_usd: float, **kw) -> float:
    """Execution cost as bps of the trade size (for reporting)."""
    s = abs(float(trade_usd))
    return (amm_cost_usd(s, liquidity_usd, **kw) / s * 1e4) if s > 0 else 0.0
