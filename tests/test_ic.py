"""Tests for the Information Coefficient analysis."""

import numpy as np
import pandas as pd

from trader.sim.ic import cross_sectional_ic, forward_return, ic_summary


def test_forward_return_sums_next_k():
    r = pd.Series([0.1, 0.2, 0.3, 0.4, 0.5])
    f = forward_return(r, 2)
    assert abs(f.iloc[0] - 0.5) < 1e-12     # r1+r2
    assert abs(f.iloc[2] - 0.9) < 1e-12     # r3+r4
    assert np.isnan(f.iloc[3]) and np.isnan(f.iloc[4])


def test_perfect_signal_gives_ic_one():
    rng = np.random.default_rng(0)
    fwd = pd.DataFrame(rng.normal(size=(50, 10)))
    ic = cross_sectional_ic(fwd, fwd, min_names=5)     # signal == forward
    assert (ic > 0.999).all()


def test_inverse_signal_gives_negative_ic():
    rng = np.random.default_rng(1)
    fwd = pd.DataFrame(rng.normal(size=(50, 10)))
    ic = cross_sectional_ic(-fwd, fwd, min_names=5)
    assert (ic < -0.999).all()


def test_random_signal_has_near_zero_ic():
    rng = np.random.default_rng(2)
    fwd = pd.DataFrame(rng.normal(size=(400, 12)))
    sig = pd.DataFrame(rng.normal(size=(400, 12)))
    s = ic_summary(cross_sectional_ic(sig, fwd, min_names=5), horizon_bars=1)
    assert abs(s["mean_ic"]) < 0.1
    assert abs(s["t_stat"]) < 3             # not spuriously significant


def test_min_names_filter_skips_thin_rows():
    fwd = pd.DataFrame({"A": [1.0, 2.0], "B": [2.0, 1.0]})
    assert len(cross_sectional_ic(fwd, fwd, min_names=8)) == 0


def test_ic_summary_empty_is_safe():
    s = ic_summary(pd.Series([], dtype=float), horizon_bars=24)
    assert s["n"] == 0 and np.isnan(s["mean_ic"])
