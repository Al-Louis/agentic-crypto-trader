"""Tests for the ported performance-metrics suite."""

import numpy as np

from trader.sim.metrics import PerformanceMetrics, Trade


def test_total_return_and_max_drawdown():
    r = PerformanceMetrics.compute_all(np.array([100.0, 120.0, 90.0, 100.0]))
    assert abs(r.total_return_pct - 0.0) < 1e-9          # 100 -> 100
    assert abs(r.max_drawdown_pct - 0.25) < 1e-9         # (120-90)/120


def test_monotonic_curve_has_positive_sharpe_and_no_drawdown():
    r = PerformanceMetrics.compute_all(np.array([100.0, 101.0, 102.0, 103.0, 104.0]))
    assert r.total_return_pct > 0
    assert r.sharpe_ratio > 0
    assert r.max_drawdown_pct == 0.0


def test_fifo_win_rate_and_profit_factor():
    trades = [Trade("buy", 1, 100), Trade("sell", 1, 110),   # +10
              Trade("buy", 1, 100), Trade("sell", 1, 90)]     # -10
    r = PerformanceMetrics.compute_all(np.array([100.0, 100.0]), trades)
    assert r.total_trades == 4
    assert abs(r.win_rate - 0.5) < 1e-9
    assert abs(r.profit_factor - 1.0) < 1e-6
    assert r.max_consecutive_losses == 1


def test_fees_tracked_and_reduce_round_trip_pnl():
    trades = [Trade("buy", 1, 100, fee=1.0), Trade("sell", 1, 110, fee=1.0)]
    r = PerformanceMetrics.compute_all(np.array([100.0, 108.0]), trades)
    assert abs(r.total_fees_paid - 2.0) < 1e-9
    assert r.win_rate == 1.0          # 10 gross - 2 fees = +8, still a win


def test_empty_inputs_are_safe():
    r = PerformanceMetrics.compute_all(np.array([100.0]))
    assert r.total_trades == 0
    assert r.win_rate == 0.0
    assert r.max_drawdown_pct == 0.0
    assert "Sharpe" in r.summary()
