"""The long-default basket OVERLAY (`basket_default`, 2026-06-14): the env starts fully long the
risk-parity basket (= Buy&Hold) and events tilt off it; the default action HOLDS. These tests pin
the two load-bearing invariants — starts long, and doing-nothing tracks B&H — plus back-compat."""
from __future__ import annotations

import numpy as np
import pytest

from tests.test_event_env import _panel
from trader.train.event_env import EventRungEnv


def _basket_env(**kw):
    returns, btc, vol, liq = _panel()
    base = dict(volume=vol, k=3, ema_span=10, warmup=30, episode_bars=200, vol_mult=2.5,
                vol_spk=4, vol_base=20, vol_fast=4, stop_k=0.1, cooldown=8, seed=0,
                action_mode="discrete", n_action_levels=4, rule_default=True,
                vol_target=0.005, cap_floor=0.02, reward_mode="relative")
    return EventRungEnv(returns, btc, liq, **{**base, **kw})


def test_basket_default_requires_rule_default():
    with pytest.raises(ValueError):
        _basket_env(basket_default=True, rule_default=False, action_mode="continuous")


def test_basket_default_starts_fully_long():
    env = _basket_env(basket_default=True)
    env.reset(start=40)
    assert len(env.pos) == env.k                                  # the whole basket is held at reset
    exposure = sum(env._pos_value(t) for t in env.pos) / env._equity()
    assert exposure > 0.98                                        # fully invested (cash ~ 0)
    assert env.cash < 0.02 * env.capital
    assert 0.95 * env.capital <= env._equity() <= env.capital     # equity = capital minus entry cost


def test_basket_default_hold_tracks_buyhold_exactly():
    """All-idx0 (exits/profits HOLD, never trim) holds the whole basket -> equity tracks the B&H
    benchmark to float precision, and with no drawdown brake the relative reward of pure holding
    sums to ~0 (doing nothing MATCHES the benchmark — the well-posed gradient: only tilts score)."""
    env = _basket_env(basket_default=True, dd_lambda=0.0)
    env.reset(start=40)
    rule_end = float(env._rule_eq[-1])                            # B&H benchmark equity at episode end
    total_r, info = 0.0, {"equity": env._equity()}
    for _ in range(3000):
        if env._pending[0] == "none":
            break
        _, r, done, info = env.step([0])                         # idx0 everywhere = hold the basket
        total_r += r
        if done:
            break
    assert abs(info["equity"] - rule_end) <= 1e-6 * rule_end      # holding == B&H, exactly
    assert abs(total_r) < 1e-3                                    # holding nets ~0 relative reward


def test_basket_default_cutting_off_the_basket_moves_equity():
    """Cutting names (idx3 on every weakness/profit prompt) tilts OFF the held basket -> a materially
    different outcome from holding. This panel runs up then CRASHES, so the defensive cutter ends
    ABOVE pure holding (it sheds the rolling-over names before the crash) — the learned-exit alpha the
    overlay exists to let the agent express on top of the long-default floor."""
    env = _basket_env(basket_default=True, dd_lambda=0.0)
    env.reset(start=40)
    rule_end = float(env._rule_eq[-1])
    info = {"equity": env._equity()}
    for _ in range(3000):
        etype = env._pending[0]
        if etype == "none":
            break
        a = 3 if etype in ("exit", "profit") else 0              # cut on every weakness/profit prompt
        _, _r, done, info = env.step([a])
        if done:
            break
    assert abs(info["equity"] - rule_end) > 0.01 * rule_end      # the tilt genuinely moves off the B&H floor
    assert info["equity"] > rule_end                             # cutting before the crash beats holding it


def test_basket_default_off_is_byte_identical():
    """Sanity: with basket_default off the env never auto-buys — reset leaves it flat (the prior
    behavior every existing test relies on)."""
    env = _basket_env(basket_default=False)
    env.reset(start=40)
    assert env.pos == {}
    assert env.cash == env.capital
