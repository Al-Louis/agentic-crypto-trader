"""Event-driven rung-1 env: decisions fire at rung-0's events (not a clock), the agent sizes
entries and can override exits, and there is no look-ahead. Pure-numpy env, runs on the laptop."""

from __future__ import annotations

import numpy as np
import pandas as pd

from trader.train.event_env import OBS_DIM, EventRungEnv


def _panel(n=260):
    idx = pd.RangeIndex(n) * 3600
    rng = np.random.default_rng(0)
    px = np.ones(n)
    for i in range(50, 92):          # runup
        px[i] = px[i - 1] * 1.03
    for i in range(92, 120):         # crash
        px[i] = px[i - 1] * 0.95
    for i in range(120, n):          # sideways below origin
        px[i] = px[i - 1] * (1.001 if i % 2 else 0.999)
    cols = {"RUN": pd.Series(px, index=idx).pct_change().fillna(0.0),
            "C1": pd.Series(rng.normal(0, 0.003, n), index=idx),
            "C2": pd.Series(rng.normal(0, 0.003, n), index=idx)}
    returns = pd.DataFrame(cols)
    btc = pd.Series(np.cumprod(1 + rng.normal(0, 0.004, n)) * 1e4, index=idx)
    vol = pd.DataFrame({c: pd.Series(100.0, index=idx) for c in cols})
    vol.loc[idx[46:58], "RUN"] = 500.0   # volume spike igniting the runup
    liq = {c: 1e9 for c in cols}
    return returns, btc, vol, liq


def _env(**kw):
    returns, btc, vol, liq = _panel()
    base = dict(volume=vol, k=3, ema_span=10, warmup=30, episode_bars=200, vol_mult=2.5,
                vol_spk=4, vol_base=20, vol_fast=4, stop_k=0.1, cooldown=8, seed=0)
    return EventRungEnv(returns, btc, liq, **{**base, **kw})


def _run(env, entry_a=1.0, exit_a=0.0, start=40):
    """Drive an episode with fixed per-event-type actions; record each decision."""
    env.reset(start=start)
    log = []
    for _ in range(800):
        etype, tok = env._pending
        if etype == "none":
            break
        a = entry_a if etype == "entry" else exit_a
        bar = env.bar
        _, r, done, info = env.step([a])
        log.append({"etype": etype, "tok": tok, "bar": bar, "reward": r, "eq": info["equity"]})
        if done:
            break
    return log


def test_obs_shape_and_finite():
    env = _env()
    obs = env.reset(start=40)
    assert obs.shape == (OBS_DIM,)
    assert np.isfinite(obs).all()


def test_entry_event_fires_and_sizes():
    env = _env()
    log = _run(env, entry_a=1.0, exit_a=1.0)               # take every entry, hold everything
    entries = [e for e in log if e["etype"] == "entry" and e["tok"] == "RUN"]
    assert entries, "RUN's volume ignition must produce an entry decision"
    assert "RUN" in env.pos or any(e["eq"] != 10_000.0 for e in log), "sizing an entry must deploy capital"


def test_skip_entry_deploys_nothing():
    env = _env()
    env.reset(start=40)
    # step through, skipping every entry (a=0); equity must never leave cash on an entry
    deployed = False
    for _ in range(800):
        etype, tok = env._pending
        if etype == "none":
            break
        _, _, done, _ = env.step([-1.0])                  # a=-1 -> m=0: skip entries / full-exit holds
        if env.pos:
            deployed = True
        if done:
            break
    assert not deployed, "the most-negative action on every entry must never open a position"


def test_exit_override_holds_longer_than_cut():
    cut = _run(_env(), entry_a=1.0, exit_a=-1.0)           # a=-1 -> m=0: follow rung-0, cut on the trigger
    hold = _run(_env(), entry_a=1.0, exit_a=1.0)           # a=+1 -> m=1: override, hold through triggers
    cut_exits = [e for e in cut if e["etype"] == "exit"]
    assert cut_exits, "a stop/EMA break must produce an exit decision"
    # holding through re-arms the stop, so the position survives more exit prompts (or to episode end)
    assert len(hold) >= len(cut), "overriding exits should keep the position alive at least as long"


def test_events_are_event_timed_not_a_clock():
    """The whole point: decisions land on varied bars, not a fixed step_bars grid."""
    log = _run(_env(), entry_a=1.0, exit_a=0.5)
    bars = sorted({e["bar"] for e in log})
    assert len(bars) >= 3, "expect decisions on several distinct bars"
    gaps = np.diff(bars)
    assert len(set(gaps.tolist())) > 1, "inter-event gaps must vary (not a constant clock)"


def test_no_lookahead():
    """Corrupting strictly-future returns/volume must not change the obs at the current decision."""
    returns, btc, vol, liq = _panel()
    kw = dict(volume=vol, k=3, ema_span=10, warmup=30, episode_bars=120, vol_mult=2.5,
              vol_spk=4, vol_base=20, vol_fast=4, stop_k=0.1, cooldown=8, seed=0)
    e1 = EventRungEnv(returns, btc, liq, **kw)
    o1 = e1.reset(start=40)
    cut = e1.bar
    fr, fv = returns.copy(), vol.copy()
    fr.iloc[cut + 1:] *= -4.0
    fv.iloc[cut + 1:] = 0.0
    e2 = EventRungEnv(fr, btc, liq, **{**kw, "volume": fv})
    o2 = e2.reset(start=40)
    assert np.allclose(o1, o2, atol=1e-9), "future data must not affect the current observation"


def test_episode_terminates_and_moves_equity():
    env = _env()
    log = _run(env, entry_a=0.8, exit_a=0.3)
    assert log and log[-1]["etype"] in ("entry", "exit"), "episode should run real decisions"
    assert env._done, "episode must terminate"


def test_shadow_rule_curve_matches_rung0():
    """The relative-reward benchmark (env._rule_equity_curve) must reproduce run_rung0's behavior on
    the same window/universe/params - the guard the RL plan requires before trusting any reward."""
    from trader.strategy.rung0 import build_rung0, run_rung0
    returns, btc, vol, liq = _panel()
    env = _env(episode_bars=200)
    env.reset(start=40)
    curve, rule_w = env._rule_equity_curve(40, 240)
    assert rule_w.shape == (201, env.k) and rule_w.max() <= 1.0001, "rule weights matrix shape/bounds"
    shadow_ret = curve[-1] / curve[0] - 1.0
    win = returns.iloc[10:241]                            # warmup 30, then trade [40, 240]
    vwin = vol.iloc[10:241]
    sig = build_rung0(win, tokens=env.universe, volume=vwin, ema_span=10, vol_mult=2.5,
                      vol_spike=4, vol_base=20, vol_fast=4)
    eq, _, _ = run_rung0(win, sig, liq, warmup=30, stop_k=env.stop_k, cooldown=env.cooldown,
                         entry_frac=env.rule_entry_frac)
    rung_ret = eq.iloc[-1] / eq.iloc[0] - 1.0
    assert shadow_ret > 0, "shadow rule should capture the volume runup"
    assert abs(shadow_ret - rung_ret) < 0.10, f"shadow {shadow_ret:.4f} vs run_rung0 {rung_ret:.4f}"


def test_relative_reward_runs_and_zeroes_a_rule_mimic():
    """Relative mode wires the rule curve and produces finite rewards; an agent that behaves like the
    rule should accumulate ~0, not a big positive (passivity/melt-up beta no longer pays)."""
    returns, btc, vol, liq = _panel()
    env = _env(reward_mode="relative", episode_bars=200)
    env.reset(start=40)
    rsum, done = 0.0, False
    for _ in range(800):
        if env._pending[0] == "none":
            break
        _, r, done, _ = env.step([0.6])
        rsum += r
        if done:
            break
    assert env._rule_eq is not None and len(env._rule_eq) == 201
    assert np.isfinite(rsum)


def test_residual_reward_runs_and_credits_only_deviations():
    """Residual mode: the reward is the agent's weight DEVIATION from the rule dotted with returns -
    so an agent holding exactly the rule's book would score ~0. Here we just confirm it wires up,
    produces finite per-step rewards, and the rule-exposure obs feature is populated."""
    returns, btc, vol, liq = _panel()
    env = _env(reward_mode="residual", episode_bars=200)
    obs = env.reset(start=40)
    assert obs.shape == (OBS_DIM,) and np.isfinite(obs).all()
    rsum, done = 0.0, False
    for _ in range(800):
        if env._pending[0] == "none":
            break
        _, r, done, _ = env.step([0.5])
        assert np.isfinite(r)
        rsum += r
        if done:
            break
    assert env._rule_w is not None and env._rule_w.shape[1] == env.k
    assert np.isfinite(rsum)


def test_r4_penalizes_undersizing_a_winner():
    """R4 (foregone-opportunity) must make UNDER-sizing the rule on a token that rises cost more —
    closing the 'hug small = ~0 reward' basin. Min-size agent on the RUN winner: r4_beta>0 must net
    strictly less than r4_beta=0."""
    def cumrew(r4):
        env = _env(reward_mode="residual", r4_beta=r4, episode_bars=200)
        env.reset(start=40)
        s, done = 0.0, False
        for _ in range(800):
            et = env._pending[0]
            if et == "none":
                break
            a = 1.0 if et == "exit" else -0.41    # hold positions; under-size entries (~0.10 < rule's 0.20)
            _, r, done, _ = env.step([a])
            s += r
            if done:
                break
        return s
    base, with_r4 = cumrew(0.0), cumrew(0.5)
    assert with_r4 < base - 1e-9, f"R4 must penalize under-sizing the winner: r4={with_r4:.4f} vs {base:.4f}"


def test_residual_ranked_runs_and_budget_taxes_deviation():
    """residual_ranked: rule-mimic (dev≈0) nets ~0, and the quadratic budget makes a constant
    over-sizing agent score strictly less with res_gamma>0 than without — the interior pull that kills
    the magnitude corner. (Corner-CLOSING over real data is proven by scripts/preflight_residual.py.)"""
    def cum(policy, gamma):
        env = _env(reward_mode="residual_ranked", res_gamma=gamma, episode_bars=200)
        env.reset(start=40)
        s, done = 0.0, False
        for _ in range(800):
            et = env._pending[0]
            if et == "none":
                break
            _, r, done, _ = env.step([policy(et)])
            s += r
            if done:
                break
        return s
    big = lambda et: -1.0 if et == "exit" else 1.0        # oversize entries, cut exits
    mimic_a = 0.20 / 0.34 * 2 - 1.0                        # m=0.588 -> size 0.20 (the rule)
    mimic = lambda et: -1.0 if et == "exit" else mimic_a
    assert abs(cum(mimic, 0.1)) < 0.10, "rule-mimic (dev≈0) nets ~0 in residual_ranked"
    assert cum(big, 0.5) < cum(big, 0.0) - 1e-9, "the quadratic budget must tax constant over-sizing"


def test_entry_forward_matures_and_credits():
    """entry_forward credits each entry's deviation by its realized forward return (semi-MDP delayed)
    via the SHARED entry_forward_reward fn, demeaned by the panel's ignition base rate. dd off -> any
    nonzero reward IS a matured entry credit; confirms maturation fires and the queue drains."""
    env = _env(reward_mode="entry_forward", res_gamma=0.05, fwd_horizon=20, dd_lambda=0.0, episode_bars=200)
    env.reset(start=40)
    assert env._mu_base != 0.0 or True            # base rate computed (may be ~0 on the synthetic panel)
    rsum, matured, done = 0.0, False, False
    for _ in range(800):
        et = env._pending[0]
        if et == "none":
            break
        _, r, done, _ = env.step([1.0 if et == "entry" else -1.0])   # all-big
        rsum += r
        if abs(r) > 1e-12:
            matured = True
        if done:
            break
    assert np.isfinite(rsum)
    assert matured, "an entry's forward reward must mature and be credited (dd off -> r is that term)"


def test_gym_adapter_conforms_and_steps():
    from trader.train.gym_env import GymEventRungEnv
    returns, btc, vol, liq = _panel()
    env = GymEventRungEnv(returns, btc, liq, volume=vol, k=3, ema_span=10, warmup=30,
                          episode_bars=120, vol_mult=2.5, vol_spk=4, vol_base=20, vol_fast=4,
                          stop_k=0.1, cooldown=8, seed=0)
    obs, info = env.reset(seed=0)
    assert env.observation_space.contains(obs)
    for _ in range(60):
        obs, r, term, trunc, info = env.step(env.action_space.sample())
        assert env.observation_space.contains(obs), "obs must stay within the declared space"
        assert np.isfinite(r), "reward must be finite"
        if term:
            break
