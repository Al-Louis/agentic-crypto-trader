"""Cross-sectional portfolio RL environment — exposure overlay on the vol-tilt (action C).

A plain (numpy/pandas) env so it's testable without torch/gymnasium — the laptop's Python
3.14 has no torch wheel, and the policy is the only part that needs torch. A thin gymnasium
adapter wraps this for stable-baselines3 on the desktop.

The agent's job (design: vault "AI Training"): learn the **exposure dial** on the validated
equal-weight vol-top8 — the regime risk-management we hand-tuned as the trend50/severity
overlay — starting from a baseline it can't underperform by construction. The action widens to
full weights (B) later; the env, eval, and baseline stay identical.

- **Step** = one rebalance (daily by default, `step_bars=24`); **episode** = a sampled window.
- **Action** = exposure ∈ [0,1]; target = `exposure/k` on each vol-top8 token, rest cash.
- **Reward** = differential (online) Sharpe increment − a drawdown-proximity penalty ramping
  toward the ~30% DQ. AMM cost (`trader.sim.broker`) is netted into equity (so the reward
  already pays for churn).
- Next-bar execution, causal features — no look-ahead.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from trader.sim.broker import DEFAULT_GAS_USD, DEFAULT_LP_FEE_BPS, amm_cost_usd

N_OBS = 6  # [btc_trend, btc_recent_return, drawdown, exposure, last_step_return, realized_vol]


class PortfolioEnv:
    """Exposure-overlay env over a returns panel. Plain reset/step (numpy in/out)."""

    def __init__(self, returns: pd.DataFrame, btc_close: pd.Series, liquidity: dict, *,
                 k: int = 8, step_bars: int = 24, episode_steps: int = 30, warmup: int = 168,
                 capital: float = 10_000.0, lp_fee_bps: float = DEFAULT_LP_FEE_BPS,
                 gas_usd: float = DEFAULT_GAS_USD, ema_span: int = 72, dd_soft: float = 0.15,
                 dd_gate: float = 0.30, dd_lambda: float = 2.0, sharpe_eta: float = 0.04,
                 seed: int | None = None):
        self.returns = returns.sort_index()
        self.btc = btc_close.reindex(self.returns.index).ffill()
        self.btc_ema = self.btc.ewm(span=ema_span, adjust=False).mean()
        self.liquidity = liquidity
        self.k, self.step_bars, self.episode_steps = k, step_bars, episode_steps
        self.warmup, self.capital = warmup, float(capital)
        self.lp_fee_bps, self.gas_usd = lp_fee_bps, gas_usd
        self.dd_soft, self.dd_gate, self.dd_lambda = dd_soft, dd_gate, dd_lambda
        self.eta = sharpe_eta
        self.n_bars = len(self.returns)
        self.obs_dim = N_OBS
        self.rng = np.random.default_rng(seed)

        self._min_start = warmup
        self._max_start = self.n_bars - episode_steps * step_bars - 1
        if self._max_start <= self._min_start:
            raise ValueError("series too short for the episode_steps/step_bars/warmup config")

    # -- lifecycle ----------------------------------------------------------
    def reset(self, *, start: int | None = None, seed: int | None = None) -> np.ndarray:
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.start = int(start) if start is not None else int(
            self.rng.integers(self._min_start, self._max_start))
        self.i = self.start
        win = self.returns.iloc[self.start - self.warmup:self.start]   # causal universe pick
        self.tokens = list(win.std().sort_values(ascending=False).head(self.k).index)
        self.pos = pd.Series(0.0, index=self.tokens)
        self.cash = self.equity = self.peak = self.capital
        self.exposure = 0.0
        self.step_count = 0
        self._last_return = 0.0
        self.A = self.B = 0.0
        self.equity_curve = [self.capital]
        return self._obs()

    def step(self, action) -> tuple[np.ndarray, float, bool, dict]:
        exposure = float(np.clip(np.asarray(action).reshape(-1)[0], 0.0, 1.0))
        eq_start = float(self.pos.sum() + self.cash)

        # rebalance to exposure/k on each vol-top8 token (charge AMM cost on turnover)
        target = (exposure / self.k) * eq_start
        cost = 0.0
        for t in self.tokens:
            trade = target - float(self.pos[t])
            if abs(trade) >= 1.0:
                c = amm_cost_usd(trade, self.liquidity.get(t, 0.0), self.lp_fee_bps, self.gas_usd)
                self.cash -= trade + c
                self.pos[t] += trade
                cost += c
        self.exposure = exposure

        # advance step_bars bars; capture the intra-step equity path for an honest drawdown
        end = min(self.i + self.step_bars, self.n_bars - 1)
        seg = self.returns.iloc[self.i + 1:end + 1].reindex(columns=self.tokens).fillna(0.0).to_numpy()
        if len(seg):
            vals = self.pos.to_numpy() * np.cumprod(1.0 + seg, axis=0)   # [bars × k]
            eq_path = vals.sum(axis=1) + self.cash
            self.pos = pd.Series(vals[-1], index=self.tokens)
            eq_new = float(eq_path[-1])
            self.peak = max(self.peak, float(eq_path.max()))
            trough = float(eq_path.min())
        else:
            eq_new = float(self.pos.sum() + self.cash)
            self.peak = max(self.peak, eq_new)
            trough = eq_new
        self.i = end

        step_ret = eq_new / eq_start - 1.0 if eq_start > 0 else 0.0
        self.equity = eq_new
        dd = (self.peak - trough) / self.peak if self.peak > 0 else 0.0
        self._last_return = step_ret
        self.equity_curve.append(eq_new)

        reward = self._dsr(step_ret) - self.dd_lambda * self._dd_penalty(dd)
        self.step_count += 1
        done = (self.step_count >= self.episode_steps or self.i >= self.n_bars - 1 or eq_new <= 0)
        info = {"equity": eq_new, "drawdown": dd, "exposure": exposure, "cost": cost,
                "step_return": step_ret}
        return self._obs(), float(reward), bool(done), info

    # -- pieces -------------------------------------------------------------
    def _dsr(self, r: float) -> float:
        """Differential (online) Sharpe increment — Moody & Saffell. 0 until variance exists."""
        da, db = r - self.A, r * r - self.B
        denom = self.B - self.A * self.A
        d = (self.B * da - 0.5 * self.A * db) / denom ** 1.5 if denom > 1e-12 else 0.0
        self.A += self.eta * da
        self.B += self.eta * db
        return float(d)

    def _dd_penalty(self, dd: float) -> float:
        """0 below `dd_soft`, ramps² to 1 at the `dd_gate` (the DQ) — ruin is treated as ruin."""
        ramp = float(np.clip((dd - self.dd_soft) / (self.dd_gate - self.dd_soft), 0.0, 1.0))
        return ramp * ramp

    def _obs(self) -> np.ndarray:
        i = self.i
        ema = float(self.btc_ema.iloc[i])
        btc_trend = float(self.btc.iloc[i]) / ema - 1.0 if ema else 0.0
        j = max(i - self.step_bars, 0)
        prev = float(self.btc.iloc[j])
        btc_ret = float(self.btc.iloc[i]) / prev - 1.0 if prev else 0.0
        dd = (self.peak - self.equity) / self.peak if self.peak > 0 else 0.0
        port = self.returns.iloc[max(i - self.warmup, 0):i + 1][self.tokens].mean(axis=1)
        vol = float(port.std()) if len(port) > 1 else 0.0
        return np.array([btc_trend, btc_ret, dd, self.exposure, self._last_return, vol],
                        dtype=np.float32)
