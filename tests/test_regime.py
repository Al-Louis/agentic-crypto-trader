"""Tests for the regime signal + the regime-gating wrapper."""

import numpy as np
import pandas as pd

from trader.features.regime import btc_risk_on
from trader.sim import strategies as S


def test_risk_on_uptrend_true_downtrend_false():
    up = pd.Series(np.linspace(100, 200, 300))
    down = pd.Series(np.linspace(200, 100, 300))
    assert bool(btc_risk_on(up, ema_span=50).iloc[-1]) is True
    assert bool(btc_risk_on(down, ema_span=50).iloc[-1]) is False


def test_risk_on_warmup_defaults_off():
    s = pd.Series([100.0] * 10)
    assert bool(btc_risk_on(s, ema_span=50).iloc[0]) is False


def test_regime_gated_holds_in_risk_on_and_goes_to_cash_off():
    risk_on = pd.Series([True, False], index=[10, 20])
    base = lambda h: pd.Series({"A": 1.0})        # noqa: E731
    g = S.regime_gated(base, risk_on)
    assert len(g(pd.DataFrame({"A": [1.0]}, index=[10]))) == 1     # risk-on -> base weights
    assert len(g(pd.DataFrame({"A": [1.0]}, index=[20]))) == 0     # risk-off -> cash


def test_regime_gated_unknown_timestamp_is_risk_off():
    g = S.regime_gated(lambda h: pd.Series({"A": 1.0}), pd.Series([True], index=[10]))
    assert len(g(pd.DataFrame({"A": [1.0]}, index=[999]))) == 0    # not in series -> off
