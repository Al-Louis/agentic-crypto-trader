"""rung-0-as-features in PortfolioEnv: correct obs width, causal (no look-ahead), and the signal
actually fires on a volume-ignited runup. Pure-numpy env, so this runs on the laptop."""

from __future__ import annotations

import numpy as np
import pandas as pd

from trader.train.env import PortfolioEnv


def _panel(n=600):
    idx = pd.RangeIndex(n) * 3600
    rng = np.random.default_rng(0)
    # one igniting token (runup with a volume spike) + fillers
    px = np.ones(n)
    for i in range(300, 340):
        px[i] = px[i - 1] * 1.03
    for i in range(340, n):
        px[i] = px[i - 1] * (1.001 if i % 2 else 0.999)
    cols = {"RUN": pd.Series(px, index=idx).pct_change().fillna(0.0)}
    for j in range(8):                                   # fillers so vol-top-k has a universe
        cols[f"T{j}"] = pd.Series(rng.normal(0, 0.01, n), index=idx)
    returns = pd.DataFrame(cols)
    btc = pd.Series(np.cumprod(1 + rng.normal(0, 0.005, n)) * 1e4, index=idx)
    vol = pd.DataFrame({c: pd.Series(100.0, index=idx) for c in returns.columns})
    vol.loc[idx[298:308], "RUN"] = 500.0                 # 5x volume spike igniting the runup
    liq = {c: 1e9 for c in returns.columns}
    return returns, btc, vol, liq


def _kw(volume, rung0_obs):
    return dict(k=8, action_mode="weights", step_bars=24, episode_steps=7, warmup=168,
                volume=volume, rung0_obs=rung0_obs, seed=0)


def test_obs_width_grows_by_three_per_token():
    returns, btc, vol, liq = _panel()
    base = PortfolioEnv(returns, btc, liq, **_kw(None, False))
    aug = PortfolioEnv(returns, btc, liq, **_kw(vol, True))
    assert aug.obs_dim == base.obs_dim + 3 * aug.k, "rung-0 obs adds [ignite, surge, cushion] per token"
    obs = aug.reset(start=200)
    assert obs.shape == (aug.obs_dim,)
    assert np.isfinite(obs).all(), "no NaN/inf leaking into the observation"


def test_rung0_features_are_causal():
    """The obs at bar i must not change if FUTURE returns/volume are altered (no look-ahead)."""
    returns, btc, vol, liq = _panel()
    env = PortfolioEnv(returns, btc, liq, **_kw(vol, True))
    env.reset(start=250)
    o1 = env._obs().copy()
    # corrupt everything strictly after the current bar, rebuild, same decision bar
    fut = returns.copy()
    fut.iloc[env.i + 1:] *= -3.0
    vfut = vol.copy()
    vfut.iloc[env.i + 1:] = 0.0
    env2 = PortfolioEnv(fut, btc, liq, **_kw(vfut, True))
    env2.reset(start=250)
    env2.i = env.i
    assert np.allclose(o1, env2._obs(), atol=1e-9), "future data must not affect the current obs"


def test_ignite_fires_on_the_volume_runup():
    """RUN's ignite feature must be 1 somewhere during its volume-spiked runup, 0 in the calm tail."""
    returns, btc, vol, liq = _panel()
    env = PortfolioEnv(returns, btc, liq, **_kw(vol, True))
    j = env._r0_col["RUN"]
    ig = env._r0_ignite[:, j]
    assert ig[300:345].max() == 1.0, "must ignite during the volume-driven runup"
    assert ig[420:].max() == 0.0, "no spike in the calm tail → never ignites"
