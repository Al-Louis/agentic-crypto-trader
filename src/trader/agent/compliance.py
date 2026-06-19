"""Daily >=1-trade/day COMPLIANCE overlay (competition Rule-1).

The event-driven champion is selective — it sits in cash between ignitions, so several cold weeks miss
the competition's >=1-trade/EVERY-day floor (a hard DQ axis; `daily_floor_ok` in status.json flags it).
This overlay guarantees the floor with a tiny, market-neutral-ish round-trip that is NOT a strategy
signal: each UTC day BUY a small slice (default 3% of equity) of BNB with USDT at 01:00 and SELL it back
to USDT at 23:00 — two recorded trades/day, flat overnight. BNB<->USDT is the deep, already-allowlisted
pair (the Phase-2 spike-trade policy).

This is a deploy GUARDRAIL, not part of the decision core — kept OUTSIDE the EventRungEnv book (the env
must stay at exactly $10k for fill/obs-parity), as a separate small sleeve the runner accounts for. Pure
scheduler + sizer/cost here; the runner records the fills through the risk guardrails and tracks the leg.
"""
from __future__ import annotations

from datetime import datetime, timezone

COMPLIANCE_TOKEN = "BNB"        # the round-trip leg: deep liquidity, native chain token, allowlisted
CASH_LEG = "USDT"
BUY_HOUR = 1                    # 01:00 UTC buy
SELL_HOUR = 23                 # 23:00 UTC sell (round-trip closed same day -> no overnight exposure)
DEFAULT_FRAC = 0.03            # 3% of equity — small enough to be immaterial to PnL, just clears Rule-1
BUY_REASON = "COMPLIANCE_BUY"
SELL_REASON = "COMPLIANCE_SELL"


def compliance_action(now_ts: int, *, buy_hour: int = BUY_HOUR, sell_hour: int = SELL_HOUR) -> str | None:
    """`'buy'` at the buy hour, `'sell'` at the sell hour, else `None` — keyed on the UTC hour of
    `now_ts`. Pure (the runner ticks ~HH:03, so the 01:0x tick buys, the 23:0x tick sells)."""
    h = datetime.fromtimestamp(int(now_ts), timezone.utc).hour
    return "buy" if h == buy_hour else "sell" if h == sell_hour else None


def compliance_cost(usd: float) -> float:
    """AMM round-leg cost for the BNB<->USDT swap via the same broker the strategy uses. BNB is a deep
    pool, so this is ~LP fee + gas on the small notional (a few dollars/day, the price of Rule-1)."""
    from trader.sim.broker import DEFAULT_GAS_USD, DEFAULT_LP_FEE_BPS, amm_cost_usd
    return amm_cost_usd(abs(float(usd)), 5.0e8, DEFAULT_LP_FEE_BPS, DEFAULT_GAS_USD)   # deep BNB liquidity
