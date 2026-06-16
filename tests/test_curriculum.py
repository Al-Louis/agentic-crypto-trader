"""Horizon curriculum: the schedule logic + the anti-cosmetic guards. The #1 TradeSim post-mortem
lesson was a curriculum that only LOGGED phase names and never changed the sampler — so these tests
prove (a) `set_episode_bars` actually moves the sampled episode length, and (b) the schedule drives
the right SEQUENCE of horizon pushes as training progresses."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tests.test_event_env import _panel
from trader.train.curriculum import (horizon_at, max_horizon, parse_horizon_schedule,
                                     parse_universe_schedule, universe_at)
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


# -- universe-regime curriculum -----------------------------------------------------------------
USPEC = "lowvol:0.0,broad:0.35,voltopk:0.65"


def test_parse_universe_schedule():
    assert parse_universe_schedule(USPEC) == [(0.0, "lowvol"), (0.35, "broad"), (0.65, "voltopk")]
    assert parse_universe_schedule("") == []
    assert parse_universe_schedule("  ") == []
    with pytest.raises(ValueError):                       # must define progress 0.0 (the start)
        parse_universe_schedule("broad:0.35,voltopk:0.65")
    with pytest.raises(ValueError):                       # an unknown regime is refused, not kept
        parse_universe_schedule("lowvol:0.0,sideways:0.5")


def test_universe_at_is_a_step_function():
    sched = parse_universe_schedule(USPEC)
    assert universe_at(sched, 0.0) == "lowvol"
    assert universe_at(sched, 0.34) == "lowvol"
    assert universe_at(sched, 0.35) == "broad"            # threshold is inclusive
    assert universe_at(sched, 0.64) == "broad"
    assert universe_at(sched, 0.65) == "voltopk"
    assert universe_at(sched, 1.0) == "voltopk"


def _vol_panel(n=260, n_tokens=6):
    """A panel of `n_tokens` tokens with a clean geometric volatility gradient (T0 calmest .. T5 most
    volatile) so `lowvol` and `voltopk` pick DISJOINT universes — the anti-cosmetic guard's fixture."""
    idx = pd.RangeIndex(n) * 3600
    rng = np.random.default_rng(0)
    cols = {f"T{i}": pd.Series(rng.normal(0, 0.001 * (2.0 ** i), n), index=idx) for i in range(n_tokens)}
    returns = pd.DataFrame(cols)
    btc = pd.Series(np.cumprod(1 + rng.normal(0, 0.004, n)) * 1e4, index=idx)
    vol = pd.DataFrame({c: pd.Series(100.0, index=idx) for c in cols})
    liq = {c: 1e9 for c in cols}
    return returns, btc, vol, liq


def _uni_env(mode="voltopk", k=3):
    returns, btc, vol, liq = _vol_panel()
    return EventRungEnv(returns, btc, liq, volume=vol, k=k, ema_span=10, warmup=30,
                        episode_bars=40, vol_mult=2.5, vol_spk=4, vol_base=20, vol_fast=4,
                        stop_k=0.1, cooldown=8, universe_mode=mode, seed=0)


def test_set_universe_mode_rejects_bad_mode():
    env = _uni_env()
    with pytest.raises(ValueError):                       # fails loud, never silently keeps the old mode
        env.set_universe_mode("sideways")


def test_universe_curriculum_actually_shifts_the_sampled_universe():
    """The anti-cosmetic guard: mutating universe_mode MOVES which k tokens reset() samples — lowvol
    picks the calmest, voltopk the most volatile, and on a clean vol gradient the two sets are DISJOINT
    (the TradeSim lesson: a curriculum that only logged phase names never moved the sampler)."""
    env = _uni_env(mode="lowvol", k=3)
    env.set_universe_mode("lowvol")
    env.reset(start=100)
    calm = set(env.universe)
    env.set_universe_mode("voltopk")
    env.reset(start=100)
    chaos = set(env.universe)
    assert calm == {"T0", "T1", "T2"}                     # the 3 calmest (std 0.001..0.004)
    assert chaos == {"T3", "T4", "T5"}                    # the 3 most volatile (std 0.008..0.032)
    assert calm.isdisjoint(chaos)                         # the regime actually moved the universe


def test_universe_curriculum_pushes_each_phase_once_in_order():
    """The anti-cosmetic guard (b): the callback's logic (universe_at + change-detection) drives the
    sub-envs through each regime exactly once, in order, at the right progress thresholds."""
    sched = parse_universe_schedule(USPEC)
    pushed, cur, total = [], None, 1000
    for step in range(total):                             # simulate the callback over training
        target = universe_at(sched, step / total)
        if target != cur:
            pushed.append((step, target))
            cur = target
    assert [m for _, m in pushed] == ["lowvol", "broad", "voltopk"]   # each regime once, easy -> deploy
    assert [step for step, _ in pushed] == [0, 350, 650]             # at progress 0.0 / 0.35 / 0.65


def test_gym_wrapper_exposes_universe_curriculum_hook():
    """Regression (the curu smoke EOFError, 2026-06-16): the VecEnv curriculum callback calls
    env_method('set_universe_mode') on the GYM WRAPPER, not the core env. gym.Env does NOT forward
    unknown attrs to self.core, so GymEventRungEnv MUST expose the passthrough (mirroring
    set_episode_bars). Missing it killed the SubprocVecEnv worker mid-training with EOFError."""
    from trader.train.gym_env import GymEventRungEnv
    returns, btc, vol, liq = _vol_panel()
    env = GymEventRungEnv(returns, btc, liq, volume=vol, k=3, ema_span=10, warmup=30,
                          episode_bars=40, vol_mult=2.5, vol_spk=4, vol_base=20, vol_fast=4,
                          stop_k=0.1, cooldown=8, universe_mode="voltopk", seed=0)
    assert hasattr(env, "set_universe_mode") and hasattr(env, "set_episode_bars")
    env.set_universe_mode("lowvol")
    assert env.core.universe_mode == "lowvol"             # passthrough reached the core env
    env.reset(options={"start": 100})
    assert set(env.core.universe) == {"T0", "T1", "T2"}   # next episode samples the calm regime


def test_vecenv_env_method_drives_universe_curriculum():
    """The EXACT failure path: env_method('set_universe_mode') across a VecEnv must move EVERY
    sub-env's regime — the UniverseCurriculumCallback's mechanism. (Skipped where sb3 is absent.)"""
    pytest.importorskip("stable_baselines3")
    from stable_baselines3.common.vec_env import DummyVecEnv

    from trader.train.gym_env import GymEventRungEnv
    returns, btc, vol, liq = _vol_panel()

    def mk():
        return GymEventRungEnv(returns, btc, liq, volume=vol, k=3, ema_span=10, warmup=30,
                               episode_bars=40, vol_mult=2.5, vol_spk=4, vol_base=20, vol_fast=4,
                               stop_k=0.1, cooldown=8, universe_mode="voltopk", seed=0)
    venv = DummyVecEnv([mk, mk])
    venv.env_method("set_universe_mode", "lowvol")        # what UniverseCurriculumCallback calls
    assert all(c.universe_mode == "lowvol" for c in venv.get_attr("core"))
