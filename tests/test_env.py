"""Tests for the cross-sectional portfolio RL env (plain mechanics, no torch)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from trader.train.env import N_OBS, PortfolioEnv

IDX0 = 1_700_000_000


def _panel(n_bars=1200, n_tokens=10, drift=0.0, base_vol=0.01, seed=0):
    """Random panel with monotonically increasing per-token vol (so vol-top-k is well-defined)."""
    rng = np.random.default_rng(seed)
    idx = pd.RangeIndex(n_bars) * 3600 + IDX0
    cols = [f"T{i}" for i in range(n_tokens)]
    vols = np.linspace(base_vol, base_vol * 3, n_tokens)
    data = rng.normal(drift, 1.0, (n_bars, n_tokens)) * vols
    returns = pd.DataFrame(data, index=idx, columns=cols)
    btc = pd.Series(np.cumprod(1 + rng.normal(0.0, base_vol, n_bars)) * 50_000.0, index=idx)
    return returns, btc


def _const_panel(per_bar, n_bars=1200, n_tokens=10):
    """Constant per-bar return on every token (deterministic up/down paths)."""
    idx = pd.RangeIndex(n_bars) * 3600 + IDX0
    cols = [f"T{i}" for i in range(n_tokens)]
    returns = pd.DataFrame(per_bar, index=idx, columns=cols, dtype=float)
    btc = pd.Series(np.cumprod(np.full(n_bars, 1 + per_bar)) * 50_000.0, index=idx)
    return returns, btc


def _deep_liq(returns):
    return {t: 1e9 for t in returns.columns}


def test_reset_obs_shape():
    returns, btc = _panel()
    env = PortfolioEnv(returns, btc, _deep_liq(returns), episode_steps=10)
    obs = env.reset(start=300)
    assert obs.shape == (N_OBS,) and obs.dtype == np.float32 and np.all(np.isfinite(obs))


def test_universe_is_vol_top_k():
    returns, btc = _panel()
    env = PortfolioEnv(returns, btc, _deep_liq(returns), k=3, episode_steps=5)
    env.reset(start=300)
    expected = list(returns.iloc[300 - 168:300].std().sort_values(ascending=False).head(3).index)
    assert env.tokens == expected


def test_exposure_zero_keeps_capital():
    returns, btc = _panel()
    env = PortfolioEnv(returns, btc, _deep_liq(returns), episode_steps=5)
    env.reset(start=300)
    _, _, _, info = env.step(0.0)
    assert abs(info["equity"] - env.capital) < 1e-6   # no position taken
    assert info["cost"] == 0.0


def test_exposure_one_invests_grows_on_uptrend_and_costs():
    returns, btc = _const_panel(0.001)                # +0.1%/bar everywhere
    liq = {t: 5_000_000.0 for t in returns.columns}   # real liquidity → real cost
    env = PortfolioEnv(returns, btc, liq, episode_steps=5)
    env.reset(start=300)
    _, _, _, info = env.step(1.0)
    assert info["exposure"] == 1.0
    assert info["cost"] > 0.0                          # rebalancing in costs something
    assert info["equity"] > env.capital               # +0.1%/bar × 24 bars beats the cost


def test_episode_terminates_after_episode_steps():
    returns, btc = _panel()
    env = PortfolioEnv(returns, btc, _deep_liq(returns), episode_steps=4)
    env.reset(start=300)
    steps, done = 0, False
    while not done:
        _, _, done, _ = env.step(0.5)
        steps += 1
    assert steps == 4


def test_deterministic_with_seed():
    returns, btc = _panel()
    a = PortfolioEnv(returns, btc, _deep_liq(returns), episode_steps=5)
    b = PortfolioEnv(returns, btc, _deep_liq(returns), episode_steps=5)
    a.reset(seed=0)
    b.reset(seed=0)
    assert a.start == b.start and a.tokens == b.tokens


def test_drawdown_penalty_on_crash_keeps_reward_finite():
    returns, btc = _const_panel(-0.01)                # −1%/bar → deep drawdown
    env = PortfolioEnv(returns, btc, _deep_liq(returns), episode_steps=10, dd_lambda=2.0)
    env.reset(start=300)
    rewards, info, done = [], {}, False
    while not done:
        _, r, done, info = env.step(1.0)
        rewards.append(r)
    assert all(np.isfinite(r) for r in rewards)
    assert info["drawdown"] > 0.15                    # penalty zone reached
    assert min(rewards) < 0.0                          # the penalty bit


def test_too_short_series_raises():
    returns, btc = _panel(n_bars=200)
    try:
        PortfolioEnv(returns, btc, _deep_liq(returns), episode_steps=30, step_bars=24, warmup=168)
        raise AssertionError("expected ValueError for too-short series")
    except ValueError:
        pass
