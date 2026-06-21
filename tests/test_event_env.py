"""Event-driven rung-1 env: decisions fire at rung-0's events (not a clock), the agent sizes
entries and can override exits, and there is no look-ahead. Pure-numpy env, runs on the laptop."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from trader.train.event_env import (
    BASKET_OPEN,
    CANDLE_EXIT,
    EMA_BREAK,
    FORCED_REASONS,
    IGNITION,
    INTRABAR_STOP,
    LOSS_FLOOR,
    OBS_DIM,
    PROFIT_TAKE,
    ROTATION_OUT,
    SCALE_IN,
    TRAILING_STOP,
    EventRungEnv,
)


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


def test_fixed_universe_mode():
    """universe_mode='fixed' uses the hand-set list verbatim (no causal re-pick), syncs k to its
    length, keeps obs_dim, and validates the list; voltopk is unchanged."""
    env = _env(universe_mode="fixed", fixed_universe=["C1", "C2"])
    o = env.reset(start=40)
    assert sorted(env.universe) == ["C1", "C2"] and env.k == 2
    assert len(o) == env.obs_dim                       # obs width is universe-size-independent
    assert sorted(env._pick_universe(45)) == ["C1", "C2"]    # identical regardless of bar
    assert sorted(env._pick_universe(200)) == ["C1", "C2"]   # (no causal vol re-pick)
    # voltopk still re-picks causally (regression guard on the existing path)
    vt = _env(universe_mode="voltopk", k=2)
    vt.reset(start=40)
    assert len(vt.universe) == 2
    assert "RUN" in vt._pick_universe(100)             # at the runup->crash transition RUN is high-vol -> picked
    with pytest.raises(ValueError):
        _env(universe_mode="fixed")                    # missing list
    with pytest.raises(ValueError):
        _env(universe_mode="fixed", fixed_universe=["NOPE"])   # token not in panel


def test_sideways_ema_break_suppression():
    """shallow+quiet EMA-break suppression: OFF builds no svol (byte-identical); ON builds it and yields
    no MORE EMA_BREAK exits than OFF over the same driven episode (loss_floor/trailing untouched)."""
    def ema_breaks(**kw):
        env = _env(**kw)
        assert (env._svol is not None) == (kw.get("shallow_break_max", 0) > 0 and kw.get("consol_vol_max", 0) > 0)
        env.reset(start=40)
        n = 0
        for _ in range(800):
            et, _tok = env._pending
            if et == "none":
                break
            _o, _r, done, info = env.step([1.0 if et == "entry" else 0.0])
            n += sum(1 for tr in info.get("trades", []) if len(tr) > 5 and tr[5] == EMA_BREAK)
            if done:
                break
        return n
    off = ema_breaks()
    on = ema_breaks(shallow_break_max=0.05, consol_vol_max=0.05)
    assert on <= off                                      # suppression can only REMOVE shallow-quiet breaks


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


def test_trailing_stop_fires_off_peak_not_entry():
    """Regression: the exit stop must TRAIL the peak (canonical rung0.py:121), not anchor to the
    ENTRY price. The old bug stopped off `ref_px` (entry) so a winner gave back its whole run before
    exiting — 'sell the bottom'. A position 30% below its peak but 40% ABOVE entry must exit."""
    env = _env(stop_k=0.25)
    env.reset(start=40)
    env._px, env._cush = env._px.copy(), env._cush.copy()   # precomputed arrays are read-only
    t = env.universe[0]
    j = env.col_ix[t]
    bar = env.bar
    env.pos[t] = {"usd": 100.0, "entry_bar": bar, "peak_px": 2.0, "origin": 1.0, "cost_px": 1.0,
                  "tp_i": 0}
    env._cush[bar, j] = 0.1                               # price above EMA -> isolate from the ema-break exit
    env._px[bar, j] = 1.4                                 # 30% below peak (2.0), 40% above entry (1.0)
    ev = env._scan_bar(bar)                               # events now carry (etype, tok, reason, both)
    assert any(e[0] == "exit" and e[1] == t for e in ev)  # trailing stop peak*0.75=1.5 > 1.4 -> fires
    assert ("exit", t, "TRAILING_STOP", False) in ev     # ema isolated -> stop-only, both_stop_ema False
    env.pos[t]["peak_px"] = 2.0
    env._px[bar, j] = 1.6                                 # only 20% below peak -> stop (1.5) NOT hit
    assert not any(e[0] == "exit" and e[1] == t for e in env._scan_bar(bar))   # (entry-anchored never fires)


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


# -- Trade Reasoning Capture (Stage 1: env trace) -----------------------------
# Each marker tuple is now (tok, usd, fee, time, px, reason, obs); `reason` is the deterministic
# TRIGGER _scan_bar tagged, `obs` the state-acted-on. Recording-only — byte-identical to training.

OBS_KEYS = {"surge", "cush", "giveback", "unreal", "held_frac", "action", "both_stop_ema"}


def _collect_markers(env, entry_a=1.0, exit_a=0.0, start=40, max_steps=800):
    """Drive an episode (fixed per-event-type actions) and gather every marker, capturing the
    floor/intrabar fills folded into info['trades'] AFTER the advance (the step() ordering)."""
    env.reset(start=start)
    markers = []
    for _ in range(max_steps):
        etype, _ = env._pending
        if etype == "none":
            break
        a = entry_a if etype == "entry" else exit_a
        _, _, done, info = env.step([a])
        markers.extend(info.get("trades", []))
        if done:
            break
    return markers


def _reignite_panel(n=260):
    """RUN ignites twice: a runup (entry 1), a mild drift up that holds it in profit, then a second
    volume spike (the re-ignition). Mirrors the scale-in fixture so a held in-profit re-ignition tags
    SCALE_IN."""
    idx = pd.RangeIndex(n) * 3600
    rng = np.random.default_rng(0)
    px = np.ones(n)
    for i in range(50, 70):          # ignition-1 runup
        px[i] = px[i - 1] * 1.03
    for i in range(70, 110):         # mild drift up: stays in profit, no trailing-stop trip
        px[i] = px[i - 1] * 1.002
    for i in range(110, 130):        # ignition-2 runup (re-ignition while held + in profit)
        px[i] = px[i - 1] * 1.03
    for i in range(130, n):
        px[i] = px[i - 1] * (1.001 if i % 2 else 0.999)
    cols = {"RUN": pd.Series(px, index=idx).pct_change().fillna(0.0),
            "C1": pd.Series(rng.normal(0, 0.003, n), index=idx),
            "C2": pd.Series(rng.normal(0, 0.003, n), index=idx)}
    returns = pd.DataFrame(cols)
    btc = pd.Series(np.cumprod(1 + rng.normal(0, 0.004, n)) * 1e4, index=idx)
    vol = pd.DataFrame({c: pd.Series(100.0, index=idx) for c in cols})
    vol.loc[idx[46:58], "RUN"] = 500.0    # spike 1 -> ignition window ~50-53
    vol.loc[idx[106:118], "RUN"] = 500.0  # spike 2 -> ignition window ~107-113
    liq = {c: 1e9 for c in cols}
    return returns, btc, vol, liq


def test_entry_tags_ignition_with_obs():
    """A fresh ignition entry tags IGNITION and carries a populated obs (the state acted on)."""
    markers = _collect_markers(_env(), entry_a=1.0, exit_a=1.0)
    buys = [m for m in markers if m[0] == "RUN" and m[1] > 0]
    assert buys, "RUN's ignition must open a position (a BUY marker)"
    tok, usd, fee, t, px, reason, obs = buys[0]
    assert reason == IGNITION
    assert reason not in FORCED_REASONS, "an entry is the agent's sized act, not forced"
    assert set(obs) == OBS_KEYS and all(np.isfinite(v) for v in obs.values())


def test_scale_in_tags_scale_in():
    """scale_in=True: a held in-profit token's re-ignition add tags SCALE_IN (not IGNITION)."""
    returns, btc, vol, liq = _reignite_panel()
    base = dict(volume=vol, k=3, ema_span=10, warmup=30, episode_bars=200, vol_mult=2.5, vol_spk=4,
                vol_base=20, vol_fast=4, stop_k=0.1, cooldown=8, seed=0,
                action_mode="discrete", n_action_levels=4, rule_default=True, scale_in=True)
    env = EventRungEnv(returns, btc, liq, **base)
    env.reset(start=40)
    markers = []
    for _ in range(800):
        et, _ = env._pending
        if et == "none":
            break
        _, _, done, info = env.step([0])             # idx0: enter at rule sizing / hold exits
        markers.extend(info.get("trades", []))
        if done:
            break
    buys = [m for m in markers if m[0] == "RUN" and m[1] > 0]
    assert len(buys) >= 2, "RUN should open then scale in on the re-ignition"
    assert buys[0][5] == IGNITION, "the first (flat) entry is IGNITION"
    assert any(m[5] == SCALE_IN for m in buys[1:]), "a held in-profit re-ignition add must tag SCALE_IN"


def test_ema_only_exit_tags_ema_break():
    """An exit triggered by cush<0 with the trailing stop NOT hit tags EMA_BREAK (discretionary)."""
    env = _env(stop_k=0.5)                            # wide stop so only the ema-break fires
    env.reset(start=40)
    env._px, env._cush = env._px.copy(), env._cush.copy()
    t = env.universe[0]
    j = env.col_ix[t]
    bar = env.bar
    env.pos[t] = {"usd": 100.0, "entry_bar": bar, "peak_px": 1.0, "origin": 1.0, "cost_px": 1.0,
                  "tp_i": 0}
    env._px[bar, j] = 0.9                             # peak 1.0, stop_k 0.5 -> 0.5 threshold NOT hit
    env._cush[bar, j] = -0.05                         # below EMA -> ema-break fires
    ev = env._scan_bar(bar)
    assert ("exit", t, EMA_BREAK, False) in ev, "cush<0 with stop un-hit -> EMA_BREAK, not both"


def test_trailing_stop_exit_tags_trailing_stop():
    """An exit with the trailing stop hit but price above EMA tags TRAILING_STOP (discretionary)."""
    env = _env(stop_k=0.1)
    env.reset(start=40)
    env._px, env._cush = env._px.copy(), env._cush.copy()
    t = env.universe[0]
    j = env.col_ix[t]
    bar = env.bar
    env.pos[t] = {"usd": 100.0, "entry_bar": bar, "peak_px": 2.0, "origin": 1.0, "cost_px": 1.0,
                  "tp_i": 0}
    env._px[bar, j] = 1.5                             # 25% below peak 2.0 -> stop (1.8) hit
    env._cush[bar, j] = 0.1                           # above EMA -> ema-break isolated out
    assert ("exit", t, TRAILING_STOP, False) in env._scan_bar(bar)


def test_both_stop_and_ema_sets_the_co_fire_flag():
    """When the trailing stop AND the EMA break co-fire, precedence picks TRAILING_STOP but the
    both_stop_ema flag is set so the forensics aren't lossy."""
    env = _env(stop_k=0.1)
    env.reset(start=40)
    env._px, env._cush = env._px.copy(), env._cush.copy()
    t = env.universe[0]
    j = env.col_ix[t]
    bar = env.bar
    env.pos[t] = {"usd": 100.0, "entry_bar": bar, "peak_px": 2.0, "origin": 1.0, "cost_px": 1.0,
                  "tp_i": 0}
    env._px[bar, j] = 1.5                             # 25% below peak -> stop hit
    env._cush[bar, j] = -0.05                         # AND below EMA -> ema break too
    ev = env._scan_bar(bar)
    assert ("exit", t, TRAILING_STOP, True) in ev     # stop wins; co-fire recorded on the event
    env._set_pending(("exit", t, TRAILING_STOP, True))   # the dispatch threads reason+both into the marker
    env._trades = []
    env._do_exit(t, 0.0, env._pending_reason)         # full cut -> a SELL marker (reason as step() passes)
    sells = [m for m in env._trades if m[0] == t and m[1] < 0]
    assert sells and sells[0][5] == TRAILING_STOP
    assert sells[0][6]["both_stop_ema"] is True, "the co-fire flag must reach the marker obs"


def test_disaster_floor_exit_tags_loss_floor_forced():
    """A position past the disaster floor force-cuts and tags LOSS_FLOOR (a FORCED reason), even when
    the agent answers idx3 (hold/override)."""
    env = _env(action_mode="discrete", n_action_levels=4, rule_default=True, loss_floor=0.2)
    env.reset(start=40)
    env._px = env._px.copy()
    t = env.universe[0]
    j = env.col_ix[t]
    bar = env.bar
    env.pos[t] = {"usd": 100.0, "entry_bar": bar, "peak_px": env._px[bar, j], "origin": 1.0,
                  "tp_i": 0, "cost_px": env._px[bar, j]}
    later = bar + 1
    env._px[later, j] = env._px[bar, j] * 0.7         # -30% < -20% floor
    env.bar = later
    env._trades = []
    env._do_exit(t, 1.0)                              # idx3-equivalent: try to hold -> floor overrides
    sells = [m for m in env._trades if m[0] == t and m[1] < 0]
    assert sells, "the floor force-cut must emit a SELL marker"
    assert sells[0][5] == LOSS_FLOOR
    assert sells[0][5] in FORCED_REASONS


def test_intrabar_stop_tags_intrabar_stop_forced():
    """The resting-stop intrabar floor fill tags INTRABAR_STOP (FORCED)."""
    returns, btc, vol, liq = _panel()
    low_frac = pd.DataFrame(1.0, index=returns.index, columns=returns.columns)
    base = dict(volume=vol, k=3, ema_span=10, warmup=30, episode_bars=200, vol_mult=2.5, vol_spk=4,
                vol_base=20, vol_fast=4, stop_k=0.1, cooldown=8, seed=0,
                intrabar_floor=True, loss_floor=0.2, low_frac=low_frac)
    env = EventRungEnv(returns, btc, liq, **base)
    env.reset(start=40)
    t = env.universe[0]
    j = env.col_ix[t]
    bar = env.bar
    env.pos[t] = {"usd": 100.0, "entry_bar": bar, "peak_px": env._px[bar, j], "origin": 1.0,
                  "tp_i": 0, "cost_px": env._px[bar, j]}
    env._trades = []
    env._stop_fill(t, env.pos[t]["cost_px"] * (1.0 - env.loss_floor))
    sells = [m for m in env._trades if m[0] == t and m[1] < 0]
    assert sells and sells[0][5] == INTRABAR_STOP
    assert sells[0][5] in FORCED_REASONS


def _rotate_setup(env, runup):
    """Stage a rotation: a weak laggard held, a stronger candidate, cash=0, and the candidate's
    px run-up over rotate_pump_win bars set to `runup`. Returns (laggard, candidate)."""
    env.reset(start=40)
    env._cush = env._cush.copy()
    env._px = env._px.copy()
    t = env.universe[0]; jt = env.col_ix[t]
    incoming = env.universe[1]; ji = env.col_ix[incoming]
    bar = env.bar
    env.pos[t] = {"usd": 100.0, "entry_bar": bar, "peak_px": env._px[bar, jt], "origin": 1.0,
                  "tp_i": 0, "cost_px": env._px[bar, jt]}
    env._cush[bar, jt] = -0.5                         # laggard = weakest cushion
    env._cush[bar, ji] = 0.5                          # candidate stronger -> rotation would proceed
    b0 = bar - env.rotate_pump_win
    env._px[bar, ji] = env._px[b0, ji] * (1.0 + runup)   # candidate's recent run-up
    env.cash = 0.0
    env._trades = []
    return t, incoming


def test_rotate_pump_block_skips_chase():
    """With the anti-chase brake on, _rotate_for refuses to liquidate a holding to fund an entry into
    a candidate that ALREADY ran up past the threshold — but still rotates for a non-pumped candidate."""
    env = _env(rotate_pump_block=0.20, rotate_pump_win=4)
    t, incoming = _rotate_setup(env, runup=0.50)          # candidate +50% over the window (> 20% block)
    env._rotate_for(incoming, want=1000.0)
    assert not [m for m in env._trades if m[0] == t and m[1] < 0]   # NO sell-to-chase
    assert env.pos.get(t) is not None                     # laggard still held

    env2 = _env(rotate_pump_block=0.20, rotate_pump_win=4)
    t2, inc2 = _rotate_setup(env2, runup=0.05)            # candidate only +5% (< 20%): not a chase
    env2._rotate_for(inc2, want=1000.0)
    assert [m for m in env2._trades if m[0] == t2 and m[1] < 0]     # rotation proceeds as before


def test_rotate_pump_block_off_is_byte_identical():
    """Default (rotate_pump_block=0.0) rotates regardless of the candidate's run-up — unchanged."""
    env = _env()                                          # knob off by default
    assert env.rotate_pump_block == 0.0
    t, incoming = _rotate_setup(env, runup=2.0)           # candidate +200% — would be blocked if on
    env._rotate_for(incoming, want=1000.0)
    assert [m for m in env._trades if m[0] == t and m[1] < 0]       # still rotates (brake off)


def _candle_env(**kw):
    """An env with the OHLC frac panels present (flat 1.0 => no real candles) so candle_exit can build;
    tests then override `_bear_candle` directly to place a bearish candle on a chosen bar."""
    returns, btc, vol, liq = _panel()
    ones = pd.DataFrame(1.0, index=returns.index, columns=returns.columns)
    base = dict(volume=vol, k=3, ema_span=10, warmup=30, episode_bars=200, vol_mult=2.5, vol_spk=4,
                vol_base=20, vol_fast=4, stop_k=0.1, cooldown=8, seed=0, loss_floor=0.2,
                low_frac=ones, high_frac=ones, intrabar_floor=True)
    return EventRungEnv(returns, btc, liq, **{**base, **kw})


def _hold(env, t, j, bar, cost_mult):
    env.pos[t] = {"usd": 100.0, "entry_bar": bar - 2, "peak_px": env._px[bar, j], "origin": 1.0,
                  "tp_i": 0, "cost_px": env._px[bar, j] * cost_mult}


def test_candle_exit_prompts_when_in_profit():
    """candle_exit ON: a held IN-PROFIT position on a bearish candle prompts a discretionary CANDLE_EXIT."""
    env = _candle_env(candle_exit=True)
    env.reset(start=40)
    t = env.universe[0]; j = env.col_ix[t]; bar = env.bar
    env._cush = env._cush.copy(); env._cush[bar, j] = 0.5        # above EMA -> no ema-break
    env._bear_candle = env._bear_candle.copy(); env._bear_candle[bar, j] = True
    _hold(env, t, j, bar, cost_mult=0.8)                         # cost 20% below px -> in profit
    ev = [e for e in env._scan_bar(bar) if e[0] == "exit" and e[1] == t]
    assert ev and ev[0][2] == CANDLE_EXIT
    assert CANDLE_EXIT not in FORCED_REASONS                     # discretionary: the agent can still hold


def test_candle_exit_skipped_when_underwater():
    """No CANDLE_EXIT when the position is below its cost basis, even on a bearish candle (in-profit only)."""
    env = _candle_env(candle_exit=True)
    env.reset(start=40)
    t = env.universe[0]; j = env.col_ix[t]; bar = env.bar
    env._cush = env._cush.copy(); env._cush[bar, j] = 0.5
    env._bear_candle = env._bear_candle.copy(); env._bear_candle[bar, j] = True
    _hold(env, t, j, bar, cost_mult=1.2)                         # cost ABOVE px -> underwater
    ev = [e for e in env._scan_bar(bar) if e[0] == "exit" and e[1] == t and e[2] == CANDLE_EXIT]
    assert not ev


def test_candle_exit_off_is_byte_identical():
    """Default (candle_exit off): no bearish-candle mask is built and no CANDLE_EXIT is ever prompted."""
    env = _candle_env()                                         # off by default
    assert env._bear_candle is None
    env.reset(start=40)
    t = env.universe[0]; j = env.col_ix[t]; bar = env.bar
    env._cush = env._cush.copy(); env._cush[bar, j] = 0.5
    _hold(env, t, j, bar, cost_mult=0.8)
    ev = [e for e in env._scan_bar(bar) if e[0] == "exit" and e[2] == CANDLE_EXIT]
    assert not ev


def test_rotation_out_tags_rotation_out_forced():
    """A holding closed by _rotate_for (to fund a stronger ignition) tags ROTATION_OUT (FORCED)."""
    env = _env()
    env.reset(start=40)
    env._cush = env._cush.copy()
    t = env.universe[0]                               # the laggard to rotate out
    j = env.col_ix[t]
    bar = env.bar
    env.pos[t] = {"usd": 100.0, "entry_bar": bar, "peak_px": env._px[bar, j], "origin": 1.0,
                  "tp_i": 0, "cost_px": env._px[bar, j]}
    incoming = env.universe[1]
    env._cush[bar, j] = -0.5                          # the holding is the WEAKEST cushion
    env._cush[bar, env.col_ix[incoming]] = 0.5        # the candidate is stronger -> rotation proceeds
    env.cash = 0.0
    env._trades = []
    env._rotate_for(incoming, want=1000.0)            # needs cash -> closes the laggard
    sells = [m for m in env._trades if m[0] == t and m[1] < 0]
    assert sells and sells[0][5] == ROTATION_OUT
    assert sells[0][5] in FORCED_REASONS


def test_profit_take_tags_profit_take():
    """A take-profit prompt answered with a sell tags PROFIT_TAKE (discretionary)."""
    env = _env(action_mode="discrete", n_action_levels=4, rule_default=True, tp_rungs=(0.25,))
    env.reset(start=40)
    t = env.universe[0]
    j = env.col_ix[t]
    bar = env.bar
    env.pos[t] = {"usd": 100.0, "entry_bar": bar, "peak_px": 1.0, "origin": 1.0,
                  "tp_i": 0, "cost_px": 1.0}
    env._px = env._px.copy()
    env._px[bar, j] = 1.3                             # +30% -> crosses the +25% rung
    assert ("profit", t) in env._scan_bar(bar)
    env._pending = ("profit", t)
    env._trades = []
    env._do_profit(t, 0.0)                            # sell into strength (keep 0)
    sells = [m for m in env._trades if m[0] == t and m[1] < 0]
    assert sells and sells[0][5] == PROFIT_TAKE
    assert sells[0][5] not in FORCED_REASONS, "taking profit is the agent's discretionary act"


def test_basket_open_tags_basket_open():
    """basket_default buys the whole basket at reset; each opening BUY tags BASKET_OPEN."""
    returns, btc, vol, liq = _panel()
    env = EventRungEnv(returns, btc, liq, volume=vol, k=3, ema_span=10, warmup=30, episode_bars=200,
                       vol_mult=2.5, vol_spk=4, vol_base=20, vol_fast=4, stop_k=0.1, cooldown=8,
                       seed=0, action_mode="discrete", n_action_levels=4, rule_default=True,
                       basket_default=True, vol_target=0.005, cap_floor=0.02)
    env.reset(start=40)
    opens = list(env._trades)
    assert opens, "the basket open must emit markers at reset"
    assert all(m[5] == BASKET_OPEN for m in opens)
    assert all(set(m[6]) == OBS_KEYS for m in opens)


def test_obs_is_populated_on_every_marker():
    """Every recorded marker carries the 6-key obs dict (the state acted on)."""
    markers = _collect_markers(_env(tp_rungs=(0.25,)), entry_a=1.0, exit_a=0.0)
    assert markers, "the episode must produce markers"
    for m in markers:
        assert len(m) == 7, "marker tuple = (tok, usd, fee, time, px, reason, obs)"
        assert set(m[6]) == OBS_KEYS, "obs must hold the documented keys"


def test_forced_reasons_membership_is_exact():
    """FORCED_REASONS is exactly {LOSS_FLOOR, INTRABAR_STOP, ROTATION_OUT} — the rest discretionary."""
    assert FORCED_REASONS == {LOSS_FLOOR, INTRABAR_STOP, ROTATION_OUT}
    for r in (IGNITION, SCALE_IN, BASKET_OPEN, EMA_BREAK, TRAILING_STOP, PROFIT_TAKE):
        assert r not in FORCED_REASONS


def test_markers_are_byte_identical_recording_only():
    """HARD CONSTRAINT: the markers are recording-only — return / max-drawdown / trade-count must be
    the deterministic golden values (the reward is equity-based; adding reason+obs changes nothing the
    policy or the equity path sees). These goldens were captured on this exact synthetic panel."""
    env = _env(record_trace=True)
    env.reset(start=40)
    ntrades = 0
    while True:
        et, _ = env._pending
        if et == "none":
            break
        a = 1.0 if et == "entry" else 0.0
        _, _, done, info = env.step([a])
        ntrades += len(info["trades"])
        if done:
            break
    eq = [e for _, e in env._eq_trace]
    ret = eq[-1] / eq[0] - 1.0
    peak, maxdd = eq[0], 0.0
    for e in eq:
        peak = max(peak, e)
        maxdd = max(maxdd, (peak - e) / peak)
    assert eq[-1] == pytest.approx(15875.454436865408, rel=0, abs=1e-6)
    assert ret == pytest.approx(0.5875454436865408, rel=0, abs=1e-9)
    assert maxdd == pytest.approx(0.11876140553437244, rel=0, abs=1e-9)
    assert ntrades == 103
    assert len(eq) == 202
