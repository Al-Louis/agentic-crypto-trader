"""Paper broker — a first-class fill simulator, not a mock.

Paper mode fills a guardrail-passed intent against the **live price read this tick**,
charging the same AMM execution cost the backtest charges (`sim.broker.amm_cost_usd`:
LP fee + constant-product price impact + gas — the cost model the [[Simulated Market]]
post-mortem demands for thin BSC pools). So a paper forward-run pays realistic costs and
its equity/PnL curve is honest, not a frictionless toy. The same cost model the eventual
live fills will face means paper-vs-live divergence ([[Real-time Monitoring]]) measures
*execution* drift, not a modelling gap.

A `PaperFill` is the analogue of a live `execute_trade` result: the token-unit deltas,
the realized USD cost, and the effective price. The loop applies it to the portfolio and
records it to the agent ledger exactly as it would record a confirmed live trade — so the
crash-recovery and PnL paths are identical across modes.

Liquidity per token is injectable (`liquidity_usd`); absent a real pool-liquidity feed in
the loop's read step, a conservative default is used so the impact charge is never *under*
-stated (an unknown-liquidity token costs *more* to trade here, never free).
"""

from __future__ import annotations

from dataclasses import dataclass

from trader.sim.broker import DEFAULT_GAS_USD, DEFAULT_LP_FEE_BPS, amm_cost_usd

# Conservative default pool depth when the loop has no per-token liquidity read. Thin enough
# that large paper trades pay visible impact (honest), deep enough that dust trades are ~free.
DEFAULT_LIQUIDITY_USD = 250_000.0


@dataclass(frozen=True)
class PaperFill:
    """The simulated outcome of one paper swap — the paper analogue of a tx result."""

    from_asset: str
    to_asset: str
    usd_in: float            # USD notional sent (spend that counts against the daily/lifetime caps)
    usd_out: float           # USD value received after execution cost
    cost_usd: float          # total execution cost (lp fee + impact + gas)
    units_from: float        # token units of `from_asset` removed
    units_to: float          # token units of `to_asset` added
    price_from: float        # USD price used for the from leg this tick
    price_to: float          # USD price used for the to leg this tick


def fill(from_asset: str, to_asset: str, usd: float, prices: dict[str, float], *,
         liquidity_usd: float = DEFAULT_LIQUIDITY_USD,
         lp_fee_bps: float = DEFAULT_LP_FEE_BPS, gas_usd: float = DEFAULT_GAS_USD) -> PaperFill:
    """Simulate filling `usd` of `from_asset` into `to_asset` at this tick's prices.

    Both legs must have a live price this tick (a missing price means the loop must not
    propose the trade — fail closed at the call site by raising). The execution cost is
    charged against the proceeds: `usd_out = usd - amm_cost_usd(usd, liquidity)`, then
    converted to `to_asset` units at its live price. Costs can exceed notional for a tiny
    trade in a dust pool -> `usd_out` floors at 0 (a fully-eaten trade, not negative units).
    """
    pf = prices.get(from_asset.upper())
    pt = prices.get(to_asset.upper())
    if not pf or not pt or pf <= 0 or pt <= 0:
        raise ValueError(f"paper fill needs live prices for {from_asset} and {to_asset} "
                         f"(got from={pf!r}, to={pt!r})")
    usd = float(usd)
    cost = amm_cost_usd(usd, liquidity_usd, lp_fee_bps=lp_fee_bps, gas_usd=gas_usd)
    usd_out = max(0.0, usd - cost)
    return PaperFill(
        from_asset=from_asset.upper(), to_asset=to_asset.upper(),
        usd_in=usd, usd_out=usd_out, cost_usd=cost,
        units_from=usd / pf, units_to=usd_out / pt,
        price_from=pf, price_to=pt,
    )
