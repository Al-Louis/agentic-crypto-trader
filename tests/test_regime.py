"""Tests for the regime signal + the regime-gating wrapper."""

import numpy as np
import pandas as pd

from trader.features.regime import btc_risk_on, stress_exposure, trend_exposure
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


def test_trend_exposure_full_above_off_below():
    up = pd.Series(np.linspace(100, 200, 300))
    down = pd.Series(np.linspace(200, 100, 300))
    assert trend_exposure(up, ema_span=50, off=0.5).iloc[-1] == 1.0
    assert trend_exposure(down, ema_span=50, off=0.5).iloc[-1] == 0.5


def test_stress_exposure_derisks_only_on_deep_drop():
    close = pd.Series(list(np.full(100, 100.0)) + list(np.linspace(100, 70, 20)))  # -30% over 20
    se = stress_exposure(close, window=20, drop=-0.10, off=0.0)
    assert se.iloc[-1] == 0.0      # stressed -> de-risked
    assert se.iloc[50] == 1.0      # flat region -> fully invested


def test_regime_scaled_scales_base_weights():
    g = S.regime_scaled(lambda h: pd.Series({"A": 1.0}), pd.Series([0.5], index=[10]))
    w = g(pd.DataFrame({"A": [1.0]}, index=[10]))
    assert abs(w["A"] - 0.5) < 1e-12
