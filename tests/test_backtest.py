"""Tests for the AMM broker, cross-sectional backtester, and baseline strategies."""

import numpy as np
import pandas as pd

from trader.sim import strategies as S
from trader.sim.backtest import NEVER, run_xs_backtest
from trader.sim.broker import amm_cost_usd


# --- AMM cost model -------------------------------------------------------

def test_amm_cost_zero_trade():
    assert amm_cost_usd(0, 1_000_000) == 0.0


def test_amm_cost_scales_with_size_and_inverse_with_liquidity():
    # impact term dominates: bigger trade and thinner pool both cost more
    assert amm_cost_usd(1000, 1_000_000) < amm_cost_usd(5000, 1_000_000)
    assert amm_cost_usd(1000, 100_000) > amm_cost_usd(1000, 10_000_000)


def test_amm_cost_components():
    # $1000 into a $1M pool: Q=500k, impact=1000/500k=0.2% + 0.25% fee + $0.20 gas
    c = amm_cost_usd(1000, 1_000_000, lp_fee_bps=25, gas_usd=0.20)
    assert abs(c - (1000 * (0.0025 + 0.002) + 0.20)) < 1e-9


def test_amm_zero_liquidity_is_untradeable():
    assert amm_cost_usd(100, 0) >= 100  # ~100% impact


# --- backtester -----------------------------------------------------------

DEEP = {"A": 1e12, "B": 1e12}


def test_no_exposure_before_warmup_then_grows():
    ret = pd.DataFrame(0.05, index=range(100), columns=["A", "B"])   # +5%/bar
    out = run_xs_backtest(ret, S.equal_weight, DEEP, warmup=20, rebalance_every=NEVER, capital=10_000)
    assert abs(out["equity"].iloc[19] - 10_000) < 1e-6              # flat (in cash) pre-warmup
    assert out["equity"].iloc[-1] > 10_000                          # invested after
    assert out["n_rebalances"] == 1                                 # buy & hold


def test_single_asset_compounds_correctly():
    ret = pd.DataFrame({"A": [0.01] * 50})
    out = run_xs_backtest(ret, S.equal_weight, {"A": 1e12}, warmup=5, rebalance_every=NEVER,
                          capital=1000.0, gas_usd=0.0, lp_fee_bps=0.0)
    # all-in A at bar 5, then 44 more bars of +1%
    expected = 1000.0 * (1.01 ** 44)
    assert abs(out["equity"].iloc[-1] - expected) / expected < 1e-6


def test_thin_pool_costs_more_than_deep():
    ret = pd.DataFrame(0.0, index=range(60), columns=["A"])
    deep = run_xs_backtest(ret, S.equal_weight, {"A": 1e12}, warmup=5, rebalance_every=NEVER)
    thin = run_xs_backtest(ret, S.equal_weight, {"A": 1_000_000.0}, warmup=5, rebalance_every=NEVER)
    assert thin["total_cost"] > deep["total_cost"]
    assert thin["equity"].iloc[-1] < deep["equity"].iloc[-1] <= 10_000


# --- strategies -----------------------------------------------------------

def test_equal_weight_sums_to_one():
    hist = pd.DataFrame(np.random.default_rng(0).normal(size=(30, 4)), columns=list("ABCD"))
    w = S.equal_weight(hist)
    assert abs(w.sum() - 1.0) < 1e-12 and len(w) == 4


def test_momentum_and_reversal_pick_opposite_ends():
    # A best trailing return, D worst
    hist = pd.DataFrame({"A": [0.1] * 30, "B": [0.05] * 30, "C": [-0.05] * 30, "D": [-0.1] * 30})
    mom = S.xs_momentum(hist, lookback=10, k=2)
    rev = S.xs_reversal(hist, lookback=10, k=2)
    assert set(mom.index) == {"A", "B"}
    assert set(rev.index) == {"C", "D"}
