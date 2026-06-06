"""Tests for the 7-day-window resampling evaluation."""

import numpy as np
import pandas as pd

from trader.sim import strategies as S
from trader.sim.resample import evaluate_windows, sample_window_starts, summarize


def test_sample_window_starts_in_range():
    rng = np.random.default_rng(0)
    starts = sample_window_starts(1000, window=168, warmup=168, n_samples=50, rng=rng)
    assert len(starts) == 50
    assert starts.min() >= 168 and starts.max() < 1000 - 168


def test_sample_window_starts_empty_when_too_short():
    rng = np.random.default_rng(0)
    assert len(sample_window_starts(200, window=168, warmup=168, n_samples=50, rng=rng)) == 0


def test_steady_uptrend_never_dq_and_always_profits():
    ret = pd.DataFrame(0.002, index=range(1200), columns=["A", "B"])   # +0.2%/bar, no drawdown
    df = evaluate_windows(ret, S.equal_weight, {"A": 1e12, "B": 1e12},
                          window=168, warmup=168, rebalance_every=24, n_samples=40,
                          capital=10_000, seed=1)
    assert len(df) == 40
    assert df["dq"].mean() == 0.0          # never breaches the gate
    assert df["profit"].mean() == 1.0      # always profitable
    assert (df["maxdd"] < 0.01).all()      # only the ~0.25% entry-fee dip, not a real drawdown


def test_deep_selloff_triggers_dq():
    ret = pd.DataFrame(-0.01, index=range(1200), columns=["A"])        # -1%/bar -> big drawdown
    df = evaluate_windows(ret, S.equal_weight, {"A": 1e12},
                          window=168, warmup=168, n_samples=20, seed=2)
    assert df["dq"].mean() > 0.9           # a -1%/bar week blows the 30% gate
    assert df["profit"].mean() == 0.0


def test_summarize_fields():
    ret = pd.DataFrame(0.001, index=range(800), columns=["A", "B"])
    df = evaluate_windows(ret, S.equal_weight, {"A": 1e12, "B": 1e12}, n_samples=30, seed=3)
    s = summarize(df, "eq")
    for k in ["strategy", "n", "ret_med", "p_dq", "p_profit", "ret_med_survived"]:
        assert k in s
    assert s["n"] == 30
