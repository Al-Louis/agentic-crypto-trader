"""The honest gate is STRUCTURAL (DIRECTION RESET 2026-06-15): every event-rung eval is judged on
SURVIVING the DQ gate AND BEATING the rung-0 RULE, with Buy&Hold and Random COMPUTED and REPORTED
(never binding). Requiring 'beat Buy&Hold' rewarded holding-everything (the rejected basket overlay);
the selective agent sits in cash between ignitions and structurally cannot out-return B&H in a bull.
These tests pin that corrected contract — the rung-0 RULE is the bar, B&H/Random are references."""

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

def test_gate_passes_when_beats_surviving_rung0():
    # beats the (surviving) rung-0 RULE + survives DQ -> PASS, regardless of B&H/Random.
    assert te.honest_gate(pol=0.20, rung0=0.10, buyhold=0.15, random_=-0.05) == (True, None)


def test_gate_beats_rule_loses_buyhold_now_passes():
    # the DIRECTION RESET: a selective policy that BEATS the rung-0 RULE but LOSES to Buy&Hold now
    # PASSES (B&H is a reported reference, never binding) — exactly the bull-market case the
    # selective agent cannot win on B&H but should not be failed for.
    passed, binding = te.honest_gate(pol=0.05, rung0=-0.02, buyhold=0.27, random_=-0.01)
    assert passed is True and binding is None


def test_gate_binding_is_rung0_when_rule_beats_policy():
    # loses to the surviving rung-0 RULE -> FAIL, binding rung-0 (even though it beats B&H here).
    passed, binding = te.honest_gate(pol=-0.094, rung0=-0.047, buyhold=-0.20, random_=-0.30)
    assert passed is False and binding == "rung-0"


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


def test_aggregator_gate_passes_when_mean_beats_rung0_and_survives_dq():
    from trader.experiment.diagnostics import compare_seeds
    m = {"total_return_pct": 0.20, "max_drawdown_pct": 0.1, "baseline_return": 0.10,
         "buyhold_return": 0.15, "random_return": -0.05, "gate_pass": True}
    r = compare_seeds("ppo-x", ["0", "1"], fetch=_fake_fetch({"0": m, "1": m}))
    # beats the rung-0 RULE (0.20 > 0.10) + DD ok -> PASS even though it LOSES B&H (0.20 < 0.15? no
    # — 0.20 > 0.15 here; B&H simply isn't checked). B&H/Random still reported.
    assert r["gate_pass_mean"] is True and r["gate_binding"] is None
    assert r["beats_buyhold"] is True and r["buyhold"] == 0.15   # reported, not binding


def test_aggregator_gate_beats_rule_loses_buyhold_now_passes():
    """DIRECTION RESET: beats the rung-0 RULE but LOSES to Buy&Hold -> PASS (B&H reported, not binding)."""
    from trader.experiment.diagnostics import compare_seeds
    m = {"total_return_pct": 0.05, "max_drawdown_pct": 0.11, "baseline_return": -0.02,
         "buyhold_return": 0.27, "random_return": -0.01, "gate_pass": True}
    r = compare_seeds("ppo-sel", ["0"], fetch=_fake_fetch({"0": m}))
    assert r["gate_pass_mean"] is True and r["gate_binding"] is None
    assert r["beats_buyhold"] is False and r["buyhold"] == 0.27   # reported even when below B&H


def test_aggregator_gate_binding_is_rung0_when_rule_wins():
    from trader.experiment.diagnostics import compare_seeds
    m = {"total_return_pct": -0.047, "max_drawdown_pct": 0.11, "baseline_return": 0.068,
         "buyhold_return": -0.20, "random_return": -0.117, "gate_pass": False}
    r = compare_seeds("ppo-lose", ["0"], fetch=_fake_fetch({"0": m}))
    assert r["gate_pass_mean"] is False and r["gate_binding"] == "rung-0"


def test_aggregator_gate_binding_is_drawdown_when_dq():
    """A config that beats the rung-0 RULE but breaches the DQ gate binds on 'drawdown'."""
    from trader.experiment.diagnostics import compare_seeds
    m = {"total_return_pct": 0.50, "max_drawdown_pct": 0.35, "baseline_return": 0.10,
         "buyhold_return": 0.15, "random_return": -0.05, "gate_pass": False}
    r = compare_seeds("ppo-dq", ["0"], fetch=_fake_fetch({"0": m}))
    assert r["gate_pass_mean"] is False and r["gate_binding"] == "drawdown"


def test_aggregator_gate_binding_is_drawdown_when_both_fail():
    """When a config BOTH loses to the rung-0 RULE AND breaches the DQ gate, the DQ binds first
    (matches weekly_gate's survives_dq-first ordering — the harder, dominating constraint)."""
    from trader.experiment.diagnostics import compare_seeds
    m = {"total_return_pct": -0.05, "max_drawdown_pct": 0.40, "baseline_return": 0.10,
         "buyhold_return": 0.15, "random_return": -0.05, "gate_pass": False}
    r = compare_seeds("ppo-both", ["0"], fetch=_fake_fetch({"0": m}))
    assert r["gate_pass_mean"] is False and r["gate_binding"] == "drawdown"
