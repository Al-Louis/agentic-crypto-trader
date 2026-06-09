"""Gymnasium adapter for `PortfolioEnv` — the bridge to stable-baselines3 (desktop trainer).

Kept separate so `trader.train.env` stays gymnasium/torch-free and testable on the laptop. This
module imports gymnasium and is only used where the `training` extra is installed (the desktop).
"""

from __future__ import annotations

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from trader.train.env import PortfolioEnv


class GymPortfolioEnv(gym.Env):
    """Wrap the plain `PortfolioEnv` in the Gymnasium API (Box action/obs, terminated/truncated).

    Action/obs dimensions come from the core env's `action_mode` — scalar exposure (C) or a
    k-vector of weights (B).
    """

    metadata = {"render_modes": []}

    def __init__(self, returns, btc_close, liquidity, **env_kwargs):
        super().__init__()
        self.core = PortfolioEnv(returns, btc_close, liquidity, **env_kwargs)
        self.action_space = spaces.Box(low=0.0, high=1.0, shape=(self.core.action_dim,),
                                       dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.core.obs_dim,),
                                            dtype=np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        start = (options or {}).get("start")
        obs = self.core.reset(start=start, seed=seed)
        return obs.astype(np.float32), {}

    def step(self, action):
        obs, reward, done, info = self.core.step(action)
        # episode end is a natural terminal here (not a time-limit truncation)
        return obs.astype(np.float32), reward, done, False, info
