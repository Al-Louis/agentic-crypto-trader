"""Tests for the ported indicator pipeline + look-ahead guard."""

import numpy as np
import pandas as pd

from trader.features.indicators import IndicatorComputer, IndicatorConfig


def _ohlcv(n=320, seed=0):
    rng = np.random.default_rng(seed)
    price = np.maximum(100 + np.cumsum(rng.normal(0, 0.5, n)), 1.0)
    open_ = np.concatenate([[price[0]], price[:-1]])
    high = np.maximum(open_, price) + rng.uniform(0, 0.3, n)
    low = np.minimum(open_, price) - rng.uniform(0, 0.3, n)
    ts = np.arange(n, dtype=np.int64) * 60_000 + 1_700_000_000_000
    return pd.DataFrame({"timestamp": ts, "open": open_, "high": high, "low": low,
                         "close": price, "volume": rng.uniform(10, 100, n)})


def test_config_defaults():
    c = IndicatorConfig()
    assert c.sma_periods == [20, 50]
    assert c.return_periods == [1, 5, 15, 60]
    assert c.rsi_period == 14 and c.macd_fast == 12 and c.macd_slow == 26


def test_compute_all_adds_expected_columns():
    df = IndicatorComputer().compute_all(_ohlcv())
    expected = [
        "sma_20", "ema_12", "macd", "macd_hist", "adx", "rsi_14", "stoch_k", "willr",
        "roc_12", "bb_pctb", "bb_width", "atr_14", "hvol_20", "obv", "vwap", "vol_sma_ratio",
        "candle_body_ratio", "upper_wick", "ret_1", "ret_60", "div_rsi_10", "div_macd_hist_20",
        "div_obv_10", "macd_hist_slope", "stoch_cross", "pressure_ratio", "rvol_60",
        "sr_pivot", "sr_support1", "sr_position", "regime",
    ]
    for col in expected:
        assert col in df.columns, f"missing {col}"
    assert len(df) == len(_ohlcv())


def test_too_few_rows_returns_unchanged():
    df = pd.DataFrame({"timestamp": [0], "open": [1.0], "high": [1.0],
                       "low": [1.0], "close": [1.0], "volume": [1.0]})
    out = IndicatorComputer().compute_all(df)
    assert "rsi_14" not in out.columns


def test_regime_label_is_ternary():
    df = IndicatorComputer().compute_all(_ohlcv())
    assert set(df["regime"].unique()).issubset({-1, 0, 1})


def test_no_lookahead_holds():
    """The crown-jewel discipline: recomputing on truncated history must match."""
    df = IndicatorComputer().compute_all(_ohlcv(n=340, seed=3))
    assert IndicatorComputer().validate_no_lookahead(df, sample_indices=[200, 280, 339]) is True
