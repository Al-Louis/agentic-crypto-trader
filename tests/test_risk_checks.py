"""Refusal matrix for the pure guardrail core (vault "TWAK Spike Runbook" §Step 4).

Every cap must refuse at its boundary with the right code, in-policy intents must pass,
and untrustworthy state/numbers must fail CLOSED. No I/O, no network.
"""

import math

from trader.risk import SPIKE_POLICY, RiskState, TradeIntent, check_trade
from trader.risk.checks import (
    CHAIN_NOT_BSC,
    DAILY_CAP,
    DRAWDOWN_STOP,
    LIFETIME_CEILING,
    NOT_ALLOWLISTED,
    PER_TRADE_CAP,
    SLIPPAGE_BOUND,
    STATE_UNAVAILABLE,
)

FRESH = RiskState()  # zero spend, no equity yet — the post-Step-4, pre-funding state


def intent(**kw) -> TradeIntent:
    base = dict(from_asset="BNB", to_asset="USDT", usd=1.0, chain="bsc", slippage_pct=1.0)
    base.update(kw)
    return TradeIntent(**base)


def codes(i, state=FRESH):
    return check_trade(SPIKE_POLICY, i, state).codes


def test_in_policy_intent_passes():
    v = check_trade(SPIKE_POLICY, intent(), FRESH)
    assert v.allowed and v.refusals == ()


def test_per_trade_cap_boundary():
    assert check_trade(SPIKE_POLICY, intent(usd=2.0), FRESH).allowed     # spend UP TO the cap
    assert codes(intent(usd=2.01)) == [PER_TRADE_CAP]


def test_daily_cap_from_state():
    at_cap = RiskState(spent_today_usd=4.0, spent_lifetime_usd=4.0)
    assert check_trade(SPIKE_POLICY, intent(usd=2.0), at_cap).allowed    # 4 + 2 == 6 passes
    over = RiskState(spent_today_usd=4.5, spent_lifetime_usd=4.5)
    assert codes(intent(usd=2.0), over) == [DAILY_CAP]


def test_lifetime_ceiling_outlives_the_day():
    # $9.5 spent across PREVIOUS days: today's budget is clear but the spike ceiling binds.
    state = RiskState(spent_today_usd=0.0, spent_lifetime_usd=9.5)
    assert codes(intent(usd=1.0), state) == [LIFETIME_CEILING]
    assert check_trade(SPIKE_POLICY, intent(usd=0.5), state).allowed     # 9.5 + 0.5 == 10


def test_slippage_bound():
    assert check_trade(SPIKE_POLICY, intent(slippage_pct=1.0), FRESH).allowed
    assert codes(intent(slippage_pct=1.2)) == [SLIPPAGE_BOUND]


def test_allowlist():
    assert codes(intent(to_asset="CAKE")) == [NOT_ALLOWLISTED]
    assert check_trade(SPIKE_POLICY, intent(from_asset="usdt", to_asset="bnb"), FRESH).allowed


def test_chain_pin():
    assert codes(intent(chain="ethereum")) == [CHAIN_NOT_BSC]


def test_drawdown_stop_at_threshold():
    # Halt AT 30% below high-water (the DQ gate is ~30% — stop on the line, not past it).
    stopped = RiskState(equity_usd=7.0, high_water_usd=10.0)
    assert codes(intent(), stopped) == [DRAWDOWN_STOP]
    trading = RiskState(equity_usd=7.5, high_water_usd=10.0)
    assert check_trade(SPIKE_POLICY, intent(), trading).allowed


def test_no_equity_recorded_means_no_drawdown_check():
    # Pre-funding there is no equity and nothing at risk; the spend caps still bind.
    assert check_trade(SPIKE_POLICY, intent(), RiskState(equity_usd=None)).allowed


def test_state_unavailable_fails_closed():
    for state in (None, RiskState(available=False, detail="ledger unreadable")):
        v = check_trade(SPIKE_POLICY, intent(), state)
        assert not v.allowed and v.codes == [STATE_UNAVAILABLE]


def test_invalid_intent_numbers_fail_closed():
    for bad in (intent(usd=0.0), intent(usd=-1.0), intent(usd=math.nan),
                intent(usd=math.inf), intent(slippage_pct=math.nan)):
        assert codes(bad) == [STATE_UNAVAILABLE]


def test_refusals_accumulate():
    state = RiskState(spent_today_usd=5.0, spent_lifetime_usd=9.0)
    v = check_trade(SPIKE_POLICY, intent(usd=5.0, to_asset="CAKE", chain="ethereum",
                                         slippage_pct=3.0), state)
    assert set(v.codes) == {CHAIN_NOT_BSC, NOT_ALLOWLISTED, PER_TRADE_CAP, DAILY_CAP,
                            LIFETIME_CEILING, SLIPPAGE_BOUND}


def test_refusal_details_are_coded_dicts():
    v = check_trade(SPIKE_POLICY, intent(usd=5.0), FRESH)
    assert v.refusals[0]["code"] == PER_TRADE_CAP and "$5.00" in v.refusals[0]["detail"]
