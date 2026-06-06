"""Tests for the factor / residual model."""

import numpy as np
import pandas as pd

from trader.features import factor as F


def _series(returns, start_ts=0, step=3600):
    """Build a [timestamp, close] frame from a log-return path."""
    price = 100.0 * np.exp(np.cumsum(returns))
    ts = np.arange(len(price)) * step + start_ts
    return pd.DataFrame({"timestamp": ts, "close": price})


def test_log_returns():
    r = F.log_returns(pd.Series([100.0, 110.0, 121.0]))
    assert np.isnan(r.iloc[0])
    assert abs(r.iloc[1] - np.log(1.1)) < 1e-12


def test_rolling_factor_recovers_known_betas():
    rng = np.random.default_rng(0)
    n = 1500
    r_btc = rng.normal(0, 0.01, n)
    r_bnb = rng.normal(0, 0.01, n)
    r_alt = 1.5 * r_btc + 0.5 * r_bnb + rng.normal(0, 0.001, n)
    fac = F.compute_factor_features(_series(r_alt), _series(r_btc), _series(r_bnb),
                                    window=250, mom_span=24)
    assert abs(fac["beta_btc"].dropna().mean() - 1.5) < 0.1
    assert abs(fac["beta_bnb"].dropna().mean() - 0.5) < 0.1
    assert fac["r2"].dropna().mean() > 0.9
    assert abs(fac["residual"].dropna().mean()) < 0.005


def test_rolling_factor_is_causal():
    rng = np.random.default_rng(1)
    n = 600
    ret = pd.DataFrame({"timestamp": np.arange(n) * 3600,
                        "r_alt": rng.normal(0, 0.01, n),
                        "r_btc": rng.normal(0, 0.01, n),
                        "r_bnb": rng.normal(0, 0.01, n)})
    full = F.rolling_factor(ret, 100)
    trunc = F.rolling_factor(ret.iloc[:400].copy(), 100)
    # a value at i=300 (>= window, < truncation point) must not change when later data is removed
    assert not np.isnan(full["residual"].iloc[300])
    assert np.isclose(full["residual"].iloc[300], trunc["residual"].iloc[300])
    assert np.isclose(full["beta_btc"].iloc[300], trunc["beta_btc"].iloc[300])


def test_align_reindexes_sparse_alt_onto_dense_grid():
    n = 100
    btc = _series(np.zeros(n))
    bnb = _series(np.zeros(n))
    alt_sparse = _series(np.linspace(0, 0.1, n)).iloc[::2].reset_index(drop=True)  # half the bars
    out = F.align_returns(alt_sparse, btc, bnb)
    assert {"timestamp", "r_alt", "r_btc", "r_bnb"}.issubset(out.columns)
    assert len(out) >= n // 2                       # ffilled onto the dense grid
    assert out["r_btc"].abs().max() < 1e-12         # flat anchor -> zero returns


def test_cross_sectional_zscore_standardizes_per_row():
    ts = [1, 2, 3]
    panel = {"A": pd.Series([1.0, 2.0, 3.0], index=ts),
             "B": pd.Series([2.0, 2.0, 2.0], index=ts),
             "C": pd.Series([3.0, 2.0, 1.0], index=ts)}
    z = F.cross_sectional_zscore(panel)
    assert abs(z.iloc[0].mean()) < 1e-9             # each row centered
    assert z.loc[1, "A"] < 0 < z.loc[1, "C"]        # A weakest, C strongest at ts=1
