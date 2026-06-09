"""Tests for the cross-sectional portfolio RL env (plain mechanics, no torch)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

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


def test_gym_adapter_conforms_and_steps():
    pytest.importorskip("gymnasium")        # adapter test only where gymnasium is installed
    from gymnasium.utils.env_checker import check_env

    from trader.train.gym_env import GymPortfolioEnv
    returns, btc = _panel()
    env = GymPortfolioEnv(returns, btc, _deep_liq(returns), episode_steps=8)
    check_env(env, skip_render_check=True)   # gymnasium's API conformance check

    obs, info = env.reset(seed=0)
    assert obs.shape == (N_OBS,) and isinstance(info, dict)
    obs, reward, terminated, truncated, info = env.step(np.array([0.5], dtype=np.float32))
    assert obs.shape == (N_OBS,) and np.isfinite(reward)
    assert isinstance(terminated, bool) and truncated is False


def test_reward_bounded_on_high_vol_returns():
    returns, btc = _panel(base_vol=0.2)               # huge returns → raw DSR would explode
    env = PortfolioEnv(returns, btc, _deep_liq(returns), episode_steps=20)
    env.reset(start=300)
    rewards, done = [], False
    while not done:
        _, r, done, _ = env.step(1.0)
        rewards.append(r)
    assert all(abs(x) <= 12.0 for x in rewards)        # DSR clipped to ±10, dd penalty up to ~2


# ---- action B (weights mode) ------------------------------------------------
def test_weights_mode_dims():
    returns, btc = _panel()
    env = PortfolioEnv(returns, btc, _deep_liq(returns), k=8, action_mode="weights", episode_steps=5)
    assert env.action_dim == 8 and env.obs_dim == 3 * 8 + 2
    obs = env.reset(start=300)
    assert obs.shape == (3 * 8 + 2,)


def test_weights_mode_allocates_and_normalizes():
    returns, btc = _const_panel(0.001)
    liq = {t: 5_000_000.0 for t in returns.columns}
    env = PortfolioEnv(returns, btc, liq, k=4, action_mode="weights", episode_steps=5)

    env.reset(start=300)                                  # all-0.5 sums to 2 → normalized to Σ=1
    _, _, _, info = env.step(np.full(4, 0.5, dtype=np.float32))
    assert info["cost"] > 0 and abs(info["exposure"] - 1.0) < 1e-6

    env.reset(start=300)                                  # one token at 0.3 → 30% invested, rest cash
    a = np.zeros(4, dtype=np.float32)
    a[0] = 0.3
    _, _, _, info = env.step(a)
    assert abs(info["exposure"] - 0.3) < 1e-6


def test_weights_mode_gym_adapter_conforms():
    pytest.importorskip("gymnasium")
    from gymnasium.utils.env_checker import check_env

    from trader.train.gym_env import GymPortfolioEnv
    returns, btc = _panel()
    env = GymPortfolioEnv(returns, btc, _deep_liq(returns), k=6, action_mode="weights", episode_steps=8)
    assert env.action_space.shape == (6,) and env.observation_space.shape == (3 * 6 + 2,)
    check_env(env, skip_render_check=True)


def test_too_short_series_raises():
    returns, btc = _panel(n_bars=200)
    try:
        PortfolioEnv(returns, btc, _deep_liq(returns), episode_steps=30, step_bars=24, warmup=168)
        raise AssertionError("expected ValueError for too-short series")
    except ValueError:
        pass


def _rotating_panel(n_bars=1500, n_tokens=12, seed=0):
    """Panel whose per-token vol leadership rotates over time, so the vol-top-k changes mid-episode."""
    rng = np.random.default_rng(seed)
    idx = pd.RangeIndex(n_bars) * 3600 + IDX0
    cols = [f"T{i}" for i in range(n_tokens)]
    t = np.arange(n_bars)
    scale = np.empty((n_bars, n_tokens))
    for i in range(n_tokens):                            # token i's vol peaks at phase i → rotation
        scale[:, i] = 0.01 + 0.07 * np.exp(-((t - (i / n_tokens) * n_bars) ** 2) / (2 * (n_bars / 7) ** 2))
    returns = pd.DataFrame(rng.normal(0, 1, (n_bars, n_tokens)) * scale, index=idx, columns=cols)
    btc = pd.Series(np.cumprod(1 + rng.normal(0, 0.005, n_bars)) * 50_000.0, index=idx)
    return returns, btc


def test_rerank_rotates_universe_and_never_mints_money():
    """rerank_every>0 re-picks the vol-top-k mid-episode: the traded set rotates, departed names are
    liquidated to cash (no orphaned positions), and rotation never creates equity. Default (0) is fixed."""
    returns, btc = _rotating_panel()
    rng = np.random.default_rng(1)

    def run(rerank_every):
        env = PortfolioEnv(returns, btc, _deep_liq(returns), k=8, action_mode="weights",
                           rich_obs=True, episode_steps=40, rerank_every=rerank_every, seed=1)
        env.reset(start=env._min_start, seed=1)
        seen = {tuple(env.tokens)}
        for _ in range(40):
            obs, reward, done, info = env.step(rng.random(8))
            assert np.all(np.isfinite(obs)) and np.isfinite(reward)
            assert set(env.pos.index) == set(env.tokens)         # no orphaned positions on rotation
            assert env.equity <= info["equity"] + 1e-6           # rotation never mints money
            seen.add(tuple(env.tokens))
            if done:
                break
        return len(seen)

    assert run(0) == 1                                            # fixed universe by default
    assert run(1) > 1                                            # re-ranking rotates the traded set
