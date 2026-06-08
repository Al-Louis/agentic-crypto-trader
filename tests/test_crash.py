"""Tests for the synthetic-crash builders."""

import numpy as np

from trader.sim.crash import crash_path, simulate_crash_panel


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
