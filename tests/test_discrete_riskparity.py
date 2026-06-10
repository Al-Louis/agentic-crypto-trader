"""Discrete action space, the universe-volatility knob, and risk-parity per-token weight caps —
the substrate changes that (a) kill the continuous-head boundary-collapse, (b) give the curriculum
its volatility axis (lowvol -> broad -> voltopk), and (c) cap each token's weight inversely to its
volatility so the near-uncorrelated alts can't blow the 30% drawdown DQ gate."""

from __future__ import annotations

import numpy as np
import pandas as pd

from trader.train.event_env import EventRungEnv


def _panel(n=320):
    """Tokens with a deliberate volatility spread (WILD .. CALM) plus RUN, which gets a volume
    spike + runup so a real ignition/entry event fires for the discrete-stepping test."""
    idx = pd.RangeIndex(n) * 3600
    rng = np.random.default_rng(0)
    run = np.ones(n)
    for i in range(50, 92):
        run[i] = run[i - 1] * 1.03                       # runup that ignites
    for i in range(92, n):
        run[i] = run[i - 1] * (1.001 if i % 2 else 0.999)
    run_ret = pd.Series(run, index=idx).pct_change().fillna(0.0)
    run_ret = run_ret + rng.normal(0, 0.003, n)          # small noise -> nonzero trailing vol at selection
    cols = {
        "RUN": run_ret,
        "WILD": pd.Series(rng.normal(0, 0.060, n), index=idx),
        "MID": pd.Series(rng.normal(0, 0.020, n), index=idx),
        "CALM": pd.Series(rng.normal(0, 0.006, n), index=idx),
        "CALMER": pd.Series(rng.normal(0, 0.002, n), index=idx),
    }
    returns = pd.DataFrame(cols)
    btc = pd.Series(np.cumprod(1 + rng.normal(0, 0.004, n)) * 1e4, index=idx)
    vol = pd.DataFrame({c: pd.Series(100.0, index=idx) for c in cols})
    vol.loc[idx[46:58], "RUN"] = 500.0                   # ignition volume spike
    liq = {c: 1e9 for c in cols}
    return returns, btc, vol, liq


def _env(**kw):
    returns, btc, vol, liq = _panel()
    base = dict(volume=vol, k=3, ema_span=10, warmup=30, episode_bars=260, vol_mult=2.5,
                vol_spk=4, vol_base=20, vol_fast=4, stop_k=0.1, cooldown=8, seed=0)
    return EventRungEnv(returns, btc, liq, **{**base, **kw})


# -- discrete action space ----------------------------------------------------

def test_gym_adapter_exposes_discrete_space():
    from trader.train.gym_env import GymEventRungEnv
    returns, btc, vol, liq = _panel()
    g = GymEventRungEnv(returns, btc, liq, volume=vol, k=3, ema_span=10, warmup=30,
                        episode_bars=260, vol_mult=2.5, vol_spk=4, vol_base=20, vol_fast=4,
                        stop_k=0.1, cooldown=8, action_mode="discrete", n_action_levels=4, seed=0)
    import gymnasium.spaces as spaces
    assert isinstance(g.action_space, spaces.Discrete) and g.action_space.n == 4


def test_discrete_level0_skips_and_top_level_funds():
    """idx 0 -> m=0 -> skip (no position); top idx -> m=1 -> funds at the token's cap."""
    skip_env = _env(action_mode="discrete", n_action_levels=4, k=5)  # k=5 -> RUN is in the universe
    skip_env.reset(start=40)
    for _ in range(400):
        if skip_env._done:
            break
        et, _ = skip_env._pending
        skip_env.step([0])                               # always level 0
    assert len(skip_env.pos) == 0                        # never funded anything

    fund_env = _env(action_mode="discrete", n_action_levels=4, k=5)
    fund_env.reset(start=40)
    funded = False
    for _ in range(400):
        if fund_env._done:
            break
        et, _ = fund_env._pending
        _, _, _, info = fund_env.step([3] if et == "entry" else [3])  # max size / hold
        if info.get("trades"):
            funded = True
    assert funded                                        # top level opened at least one position


# -- the universe-volatility knob ---------------------------------------------

def _mean_vol(env, at=30):
    return float(np.mean([env._std[at - 1, env.col_ix[t]] for t in env.universe]))


def test_universe_modes_order_by_volatility():
    top = _env(universe_mode="voltopk"); top.reset(start=40)
    low = _env(universe_mode="lowvol"); low.reset(start=40)
    broad = _env(universe_mode="broad"); broad.reset(start=40)
    assert _mean_vol(top, 40) > _mean_vol(low, 40)        # voltopk is the chaos end
    assert _mean_vol(low, 40) <= _mean_vol(broad, 40) <= _mean_vol(top, 40)
    assert "WILD" in top.universe and "WILD" not in low.universe


# -- risk-parity per-token caps -----------------------------------------------

def test_vol_target_caps_high_vol_tokens_smaller():
    env = _env(universe_mode="broad", vol_target=0.01, cap_floor=0.02, max_entry_frac=0.34)
    env.reset(start=40)
    caps = env._tok_cap
    # every cap within [floor, ceiling]
    assert all(0.02 - 1e-9 <= c <= 0.34 + 1e-9 for c in caps.values())
    # a high-vol token in the universe is capped strictly tighter than a calm one
    vols = {t: env._std[39, env.col_ix[t]] for t in env.universe}
    hi = max(vols, key=vols.get)
    lo = min(vols, key=vols.get)
    assert caps[hi] < caps[lo]


def test_vol_target_zero_is_flat_cap():
    env = _env(vol_target=0.0, max_entry_frac=0.34)
    env.reset(start=40)
    assert all(c == 0.34 for c in env._tok_cap.values())


def test_default_is_continuous_voltopk_flatcap_unchanged():
    """Backward-compat: defaults reproduce the old behavior (continuous, voltopk, flat 0.34 cap)."""
    env = _env()
    env.reset(start=40)
    assert env.action_mode == "continuous" and env.universe_mode == "voltopk"
    assert all(c == env.max_entry_frac for c in env._tok_cap.values())
