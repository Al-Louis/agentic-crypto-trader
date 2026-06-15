"""Horizon curriculum: the schedule logic + the anti-cosmetic guards. The #1 TradeSim post-mortem
lesson was a curriculum that only LOGGED phase names and never changed the sampler — so these tests
prove (a) `set_episode_bars` actually moves the sampled episode length, and (b) the schedule drives
the right SEQUENCE of horizon pushes as training progresses."""
from __future__ import annotations

import pytest

from tests.test_event_env import _panel
from trader.train.curriculum import horizon_at, max_horizon, parse_horizon_schedule
from trader.train.event_env import EventRungEnv

SPEC = "672:0.0,336:0.40,168:0.70"


def test_parse_horizon_schedule():
    assert parse_horizon_schedule(SPEC) == [(0.0, 672), (0.40, 336), (0.70, 168)]
    assert parse_horizon_schedule("") == []
    assert parse_horizon_schedule("  ") == []
    with pytest.raises(ValueError):                       # must define progress 0.0 (the start)
        parse_horizon_schedule("336:0.40,168:0.70")


def test_horizon_at_is_a_step_function():
    sched = parse_horizon_schedule(SPEC)
    assert horizon_at(sched, 0.0) == 672
    assert horizon_at(sched, 0.39) == 672
    assert horizon_at(sched, 0.40) == 336                 # threshold is inclusive
    assert horizon_at(sched, 0.69) == 336
    assert horizon_at(sched, 0.70) == 168
    assert horizon_at(sched, 1.0) == 168


def test_max_horizon():
    assert max_horizon(parse_horizon_schedule(SPEC)) == 672


def _env(episode_bars):
    returns, btc, vol, liq = _panel(n=260)
    return EventRungEnv(returns, btc, liq, volume=vol, k=3, ema_span=10, warmup=30,
                        episode_bars=episode_bars, vol_mult=2.5, vol_spk=4, vol_base=20,
                        vol_fast=4, stop_k=0.1, cooldown=8, seed=0)


def test_set_episode_bars_shrinks_safely_and_rejects_growth():
    env = _env(120)                                       # built at the LARGEST horizon
    assert env._max_start == 260 - 120 - 1
    env.set_episode_bars(40)                              # shrink: _max_start WIDENS (always safe)
    assert env.episode_bars == 40 and env._max_start == 260 - 40 - 1
    with pytest.raises(ValueError):                       # grow past the panel -> rejected, not silent
        env.set_episode_bars(300)


def test_curriculum_actually_shifts_the_sampled_horizon():
    """The anti-cosmetic guard (a): mutating episode_bars MOVES the distribution the sampler draws."""
    env = _env(120)
    starts_long, lens_long = [], []
    for s in range(50):
        env.reset(seed=s)
        starts_long.append(env.start)
        lens_long.append(env.end - env.start)
    assert all(L == 120 for L in lens_long)               # sampler honors the live value
    env.set_episode_bars(40)
    starts_short, lens_short = [], []
    for s in range(50):
        env.reset(seed=s)
        starts_short.append(env.start)
        lens_short.append(env.end - env.start)
    assert all(L == 40 for L in lens_short)               # the DISTRIBUTION moved (120 -> 40)
    assert max(starts_short) > max(starts_long)           # and the valid start range widened


def test_curriculum_pushes_each_phase_once_in_order():
    """The anti-cosmetic guard (b): the callback's logic (horizon_at + change-detection) drives the
    sub-envs through each phase exactly once, in DOWN order, at the right progress thresholds."""
    sched = parse_horizon_schedule(SPEC)
    pushed, cur, total = [], None, 1000
    for step in range(total):                             # simulate the callback over training
        target = horizon_at(sched, step / total)
        if target != cur:
            pushed.append((step, target))
            cur = target
    assert [bars for _, bars in pushed] == [672, 336, 168]      # each phase once, ramping DOWN
    assert [step for step, _ in pushed] == [0, 400, 700]        # at progress 0.0 / 0.40 / 0.70
