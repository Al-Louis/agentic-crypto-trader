"""Tests for the synthetic-crash builders."""

import numpy as np
import pandas as pd

from trader.sim.crash import (crash_path, inject_crash, inject_random_crashes,
                              simulate_crash_panel)


def _calm(n=300, k=4, seed=0):
    idx = pd.RangeIndex(n) * 3600
    rng = np.random.default_rng(seed)
    return pd.DataFrame({f"T{i}": pd.Series(rng.normal(0, 0.005, n), index=idx) for i in range(k)})


def test_crash_path_linear_reaches_total_drop():
    r = crash_path(100, -0.30, "linear")
    assert abs(np.prod(1 + r) - 0.70) < 1e-9
    assert (r < 0).all()


def test_crash_path_sharp_is_front_loaded():
    r = crash_path(70, -0.30, "sharp")
    assert abs(np.prod(1 + r) - 0.70) < 1e-6
    assert (r[:10] < 0).all()
    assert (r[20:] == 0).all()          # flat after the first day


def test_simulate_crash_panel_tracks_beta_without_noise():
    btc = np.full(50, -0.01)
    df = simulate_crash_panel(["A", "B"], btc, beta=2.0, resid_std={"A": 0.0, "B": 0.0}, seed=1)
    assert np.allclose(df["A"].to_numpy(), 2.0 * btc)
    assert np.allclose(df["B"].to_numpy(), 2.0 * btc)


def test_simulate_crash_panel_adds_noise():
    btc = np.full(200, -0.005)
    df = simulate_crash_panel(["A"], btc, beta=1.0, resid_std={"A": 0.02}, seed=3)
    assert df["A"].std() > 0.01         # idiosyncratic noise present


def test_inject_crash_produces_deep_correlated_drawdown():
    r = _calm()
    crashed = inject_crash(r, at=150, duration=8, total_drop=-0.6, beta=1.4)
    cum = (1 + crashed.iloc[150:158]).prod() - 1            # per-token cumulative over the window
    assert (cum < -0.5).all()                              # ~ beta*total_drop ~ -84%, SIREN-scale
    win = crashed.iloc[150:158]
    assert win.corr().to_numpy()[np.triu_indices(4, 1)].mean() > 0.9   # sell off TOGETHER


def test_inject_crash_is_a_copy_and_only_touches_the_window():
    r = _calm()
    crashed = inject_crash(r, at=150, duration=8)
    pd.testing.assert_frame_equal(crashed.iloc[:150], r.iloc[:150])
    pd.testing.assert_frame_equal(crashed.iloc[158:], r.iloc[158:])
    assert not r.iloc[150:158].equals(crashed.iloc[150:158])           # original untouched


def test_inject_crash_can_target_a_token_subset():
    r = _calm()
    crashed = inject_crash(r, at=100, duration=6, tokens=["T0", "T1"])
    pd.testing.assert_series_equal(crashed["T2"], r["T2"])             # untouched token unchanged
    assert (1 + crashed["T0"].iloc[100:106]).prod() - 1 < -0.5


def test_inject_random_crashes_places_n_nonoverlapping():
    r = _calm(n=1200)
    out, placed = inject_random_crashes(r, n_crashes=3, rng=np.random.default_rng(1),
                                        min_gap=100, duration=8)
    assert len(placed) == 3
    assert all(b - a > 100 for a, b in zip(placed, placed[1:]))       # spaced and sorted
    assert not out.equals(r)
