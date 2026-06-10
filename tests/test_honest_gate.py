"""The honest gate is STRUCTURAL: every event-rung eval is judged against rung-0 AND Buy&Hold AND
Random, with the regime reported, so 'beats the rule' can never again hide 'lost to the market'.
These tests pin that contract (the orchestration-drift that produced exp1->exp5 went un-caught
because the Buy&Hold bar lived only in prose — now it lives in code, and here)."""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "scripts")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import train_event as te  # noqa: E402


def _panel(n=400):
    """A panel with one clear winner (RUN) so Buy&Hold of the vol-top-k universe is solidly positive."""
    idx = pd.RangeIndex(n) * 3600
    rng = np.random.default_rng(0)
    px = np.ones(n)
    for i in range(60, 220):                       # long, steady runup -> Buy&Hold wins
        px[i] = px[i - 1] * 1.012
    for i in range(220, n):
        px[i] = px[i - 1] * (1.001 if i % 2 else 0.999)
    cols = {"RUN": pd.Series(px, index=idx).pct_change().fillna(0.0),
            "C1": pd.Series(rng.normal(0, 0.02, n), index=idx),
            "C2": pd.Series(rng.normal(0, 0.02, n), index=idx)}
    returns = pd.DataFrame(cols)
    btc = pd.Series(np.cumprod(1 + rng.normal(0, 0.004, n)) * 1e4, index=idx)
    vol = pd.DataFrame({c: pd.Series(100.0, index=idx) for c in cols})
    vol.loc[idx[56:68], "RUN"] = 500.0
    liq = {c: 1e9 for c in cols}
    return returns, btc, vol, liq


# -- the gate decision (pure) -------------------------------------------------

def test_gate_passes_only_when_all_baselines_beaten():
    assert te.honest_gate(pol=0.20, rung0=0.10, buyhold=0.15, random_=-0.05) == (True, None)


def test_gate_binding_is_buyhold_when_market_beats_policy():
    # the exp5 failure mode: policy beats rung-0 + Random but LOSES to Buy&Hold -> FAIL, binding market
    passed, binding = te.honest_gate(pol=-0.047, rung0=-0.094, buyhold=0.068, random_=-0.117)
    assert passed is False and binding == "Buy&Hold"


def test_gate_binding_prefers_market_over_rule():
    # fails both Buy&Hold and rung-0; Buy&Hold (the market) is the binding bar reported first
    _, binding = te.honest_gate(pol=-0.20, rung0=-0.10, buyhold=0.05, random_=-0.30)
    assert binding == "Buy&Hold"


def test_gate_dqs_a_policy_that_breaches_the_drawdown_gate():
    # beats every return bar, but maxDD 35% > 30% -> DQ'd, worthless (return that can't go live)
    passed, binding = te.honest_gate(pol=0.50, rung0=0.10, buyhold=0.15, random_=-0.05,
                                     pol_maxdd=0.35)
    assert passed is False and binding.startswith("DQ")


def test_gate_exempts_a_dqd_rung0_from_being_a_return_bar():
    # rung-0 made +29% but at 40% maxDD (itself DQ'd) -> not a valid live bar; agent need not match it
    passed, _ = te.honest_gate(pol=0.05, rung0=0.29, buyhold=0.02, random_=-0.10,
                               pol_maxdd=0.12, rung0_maxdd=0.40)
    assert passed is True                                   # beats Buy&Hold + Random; DQ'd rung-0 exempt


def test_gate_still_requires_beating_a_surviving_rung0():
    passed, binding = te.honest_gate(pol=0.05, rung0=0.29, buyhold=0.02, random_=-0.10,
                                     pol_maxdd=0.12, rung0_maxdd=0.20)  # rung-0 survives -> real bar
    assert passed is False and binding == "rung-0"


# -- the baselines (computed on the same universe/broker the agent uses) -------

def test_buyhold_over_agent_universe_and_deterministic():
    returns, btc, vol, liq = _panel()
    uni, caps = te.eval_universe_and_caps(returns, btc, liq, vol, dict(k=3, seed=0))
    bh = te.buy_and_hold_return(returns, liq, uni, caps)
    assert bh > 0.0                                  # the runup is in the universe the agent holds
    assert bh == te.buy_and_hold_return(returns, liq, uni, caps)   # no RNG, reproducible


def test_eval_universe_and_caps_mirror_the_env():
    """Buy&Hold/regime must be over the SAME basket the env trades, or the benchmark is rigged."""
    from trader.train.event_env import EventRungEnv
    returns, btc, vol, liq = _panel()
    uni, caps = te.eval_universe_and_caps(returns, btc, liq, vol, dict(k=3, seed=0))
    env = EventRungEnv(returns, btc, liq, volume=vol, k=3, episode_bars=200, seed=0)
    env.reset(start=te.WARMUP)
    assert set(uni) == set(env.universe) and set(caps) == set(env._tok_cap)


def test_buyhold_risk_parity_weights_track_caps():
    """With vol_target>0, Buy&Hold weights are cap-proportional (risk-parity), not equal-weight."""
    returns, btc, vol, liq = _panel()
    uni, caps = te.eval_universe_and_caps(returns, btc, liq, vol,
                                          dict(k=3, vol_target=0.003, cap_floor=0.01))
    assert len(set(round(c, 6) for c in caps.values())) > 1   # caps differ by token volatility


def test_regime_labels_and_fields():
    returns, btc, vol, liq = _panel()
    uni, _ = te.eval_universe_and_caps(returns, btc, liq, vol, dict(k=3, seed=0))
    r = te.eval_regime(returns, btc, uni)
    assert set(r) == {"btc_return", "universe_ew_return", "label"}
    assert r["label"] in {"bull", "bear", "flat"}
    assert r["label"] == ("bull" if r["universe_ew_return"] > 0.10
                          else "bear" if r["universe_ew_return"] < -0.10 else "flat")


def test_random_baseline_runs_through_env_and_is_finite():
    returns, btc, vol, liq = _panel()
    ek = dict(k=3, ema_span=10, warmup=30, episode_bars=300, vol_mult=2.5, vol_spk=4,
              vol_base=20, vol_fast=4, stop_k=0.1, cooldown=8, reward_mode="absolute", seed=0)
    rnd = te.random_baseline_return(returns, btc, liq, vol, ek, n=2, seed=0)
    assert np.isfinite(rnd)


# -- the sweep-level aggregator gate (diagnostics.compare_seeds) ---------------

def _fake_fetch(metrics_by_seed):
    def fetch(url):
        for s, m in metrics_by_seed.items():
            if url.rstrip("/").endswith(f"-s{s}/metrics.json"):
                return m
        raise FileNotFoundError(url)
    return fetch


def test_aggregator_gate_passes_when_mean_beats_all_present_baselines():
    from trader.experiment.diagnostics import compare_seeds
    m = {"total_return_pct": 0.20, "max_drawdown_pct": 0.1, "baseline_return": 0.10,
         "buyhold_return": 0.15, "random_return": -0.05, "gate_pass": True}
    r = compare_seeds("ppo-x", ["0", "1"], fetch=_fake_fetch({"0": m, "1": m}))
    assert r["gate_pass_mean"] is True and r["gate_binding"] is None


def test_aggregator_gate_cannot_pass_without_buyhold():
    """A pre-gate bundle (no buyhold_return) must FAIL — silently checking only rung-0 is the drift."""
    from trader.experiment.diagnostics import compare_seeds
    m = {"total_return_pct": -0.047, "max_drawdown_pct": 0.11, "baseline_return": -0.094}
    r = compare_seeds("ppo-old", ["0"], fetch=_fake_fetch({"0": m}))
    assert r["gate_pass_mean"] is False and r["gate_binding"].startswith("Buy&Hold")


def test_aggregator_gate_binding_is_buyhold_when_market_wins():
    from trader.experiment.diagnostics import compare_seeds
    m = {"total_return_pct": -0.047, "max_drawdown_pct": 0.11, "baseline_return": -0.094,
         "buyhold_return": 0.068, "random_return": -0.117, "gate_pass": False}
    r = compare_seeds("ppo-sel", ["0"], fetch=_fake_fetch({"0": m}))
    assert r["gate_pass_mean"] is False and r["gate_binding"] == "Buy&Hold"
