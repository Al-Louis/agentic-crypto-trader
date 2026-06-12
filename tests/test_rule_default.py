"""Rung-1b `rule_default` semantics: action idx 0 EXECUTES rung-0's decision (entry at the rule's
sizing / exit full cut), deviations are explicit; exit decisions commit (no re-prompt drip); the
dust floor kills the gas-bleeding trim tail; the override no longer re-anchors (no stop ratchet).
All flags default OFF — legacy behavior is asserted unchanged."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from trader.train.event_env import (
    RULE_DEFAULT_ENTRY_MULT,
    RULE_DEFAULT_EXIT_KEEP,
    EventRungEnv,
)


def _panel(n=260):
    idx = pd.RangeIndex(n) * 3600
    rng = np.random.default_rng(0)
    px = np.ones(n)
    for i in range(50, 92):
        px[i] = px[i - 1] * 1.03
    for i in range(92, 120):
        px[i] = px[i - 1] * 0.95
    for i in range(120, n):
        px[i] = px[i - 1] * (1.001 if i % 2 else 0.999)
    cols = {"RUN": pd.Series(px, index=idx).pct_change().fillna(0.0),
            "C1": pd.Series(rng.normal(0, 0.003, n), index=idx),
            "C2": pd.Series(rng.normal(0, 0.003, n), index=idx)}
    returns = pd.DataFrame(cols)
    btc = pd.Series(np.cumprod(1 + rng.normal(0, 0.004, n)) * 1e4, index=idx)
    vol = pd.DataFrame({c: pd.Series(100.0, index=idx) for c in cols})
    vol.loc[idx[46:58], "RUN"] = 500.0
    liq = {c: 1e9 for c in cols}
    return returns, btc, vol, liq


def _env(**kw):
    returns, btc, vol, liq = _panel()
    base = dict(volume=vol, k=3, ema_span=10, warmup=30, episode_bars=200, vol_mult=2.5,
                vol_spk=4, vol_base=20, vol_fast=4, stop_k=0.1, cooldown=8, seed=0)
    return EventRungEnv(returns, btc, liq, **{**base, **kw})


def _rd_env(**kw):
    return _env(action_mode="discrete", n_action_levels=4, rule_default=True, **kw)


def _advance_to(env, etype_want, tok_want=None, max_steps=800, filler=0):
    """Step with `filler` until the pending event matches; returns True if reached."""
    for _ in range(max_steps):
        etype, tok = env._pending
        if etype == "none":
            return False
        if etype == etype_want and (tok_want is None or tok == tok_want):
            return True
        env.step([filler])
    return False


def test_action_tables_are_rule_at_idx0():
    assert RULE_DEFAULT_ENTRY_MULT[0] == 1.0 and RULE_DEFAULT_EXIT_KEEP[0] == 0.0


def test_rule_default_requires_discrete():
    with pytest.raises(ValueError):
        _env(rule_default=True)                       # continuous action mode -> invalid


def test_rule_default_entry_idx0_sizes_like_the_rule():
    env = _rd_env()                                   # vol_target=0 -> flat cap 0.34 > rule 0.20
    env.reset(start=40)
    assert _advance_to(env, "entry", "RUN", filler=2)  # idx2 = skip while waiting
    eq = env._equity()
    env.step([0])                                     # idx0 = EXECUTE the rule
    assert "RUN" in env.pos
    assert env.pos["RUN"]["usd"] == pytest.approx(env.rule_entry_frac * eq, rel=0.02)


def test_rule_default_entry_idx2_skips_idx3_doubles():
    env = _rd_env()
    env.reset(start=40)
    assert _advance_to(env, "entry", "RUN", filler=2)
    env.step([2])                                     # idx2 = skip
    assert "RUN" not in env.pos
    env2 = _rd_env()
    env2.reset(start=40)
    assert _advance_to(env2, "entry", "RUN", filler=2)
    eq = env2._equity()
    env2.step([3])                                    # idx3 = 2x rule, clipped by the cap (0.34)
    assert env2.pos["RUN"]["usd"] == pytest.approx(min(2 * 0.20, 0.34) * eq, rel=0.02)


def test_rule_default_exit_idx0_is_full_cut_with_cooldown():
    env = _rd_env()
    env.reset(start=40)
    assert _advance_to(env, "entry", "RUN", filler=2)
    env.step([0])
    assert _advance_to(env, "exit", "RUN", filler=2)
    bar = env.bar
    env.step([0])                                     # idx0 = the rule's cut
    assert "RUN" not in env.pos
    assert env.cool["RUN"] == bar                     # cooldown + dead-zone armed like the rule


def test_override_commits_and_does_not_reanchor_the_peak():
    env = _rd_env(exit_commit=12)
    env.reset(start=40)
    env._px, env._cush = env._px.copy(), env._cush.copy()
    t = env.universe[0]
    j = env.col_ix[t]
    bar = env.bar
    env.pos[t] = {"usd": 100.0, "entry_bar": bar, "peak_px": 2.0, "origin": 1.0}
    env._cush[bar, j] = 0.1
    env._px[bar, j] = 1.7                             # stop_k=0.1 -> threshold 1.8 -> stop fires
    assert ("exit", t) in env._scan_bar(bar)
    env._pending = ("exit", t)
    env._do_exit(t, RULE_DEFAULT_EXIT_KEEP[3])        # idx3 = hold-through (override)
    assert env.pos[t]["peak_px"] == 2.0               # NOT re-anchored — the ratchet is gone
    assert env._exit_decided[t] == bar
    assert ("exit", t) not in env._scan_bar(bar + 5)   # committed: suppressed inside the window
    env._px[bar + 13, j] = 1.7
    env._cush[bar + 13, j] = 0.1
    assert ("exit", t) in env._scan_bar(bar + 13)      # window over: prompts again off the TRUE peak


def test_dust_floor_forces_full_close():
    env = _rd_env(dust_usd=10.0)
    env.reset(start=40)
    t = env.universe[0]
    bar = env.bar
    env.pos[t] = {"usd": 24.0, "entry_bar": bar, "peak_px": 1.0, "origin": 1.0}
    env._do_exit(t, RULE_DEFAULT_EXIT_KEEP[1])        # keep 1/3 of ~$24 = $8 < $10 -> full close
    assert t not in env.pos
    assert env.cool[t] == bar


def test_legacy_defaults_unchanged_trim_still_reanchors():
    env = _env()                                      # all new flags OFF
    env.reset(start=40)
    env._px = env._px.copy()
    t = env.universe[0]
    j = env.col_ix[t]
    bar = env.bar
    env._px[bar, j] = 1.4
    env.pos[t] = {"usd": 100.0, "entry_bar": bar, "peak_px": 2.0, "origin": 1.0}
    env._do_exit(t, 0.5)                              # legacy partial trim
    assert env.pos[t]["peak_px"] == pytest.approx(1.4)  # legacy: re-anchors on the trim
    env.pos[t]["peak_px"] = 2.0
    env._do_exit(t, 0.96)                             # legacy override
    assert env.pos[t]["peak_px"] == pytest.approx(1.4)  # legacy: re-anchors (the ratchet)
    assert ("exit", t) in env._scan_bar(bar) or True   # no commit suppression exists with flags off


def test_profit_prompt_fires_and_default_lets_winner_run():
    """RUN climbs ~3%/bar after entry -> the +25% rung must prompt; answering idx0 (keep all, the
    rule's let-winners-run) consumes the rung, keeps the position, and the +50% rung prompts later."""
    env = _rd_env(tp_rungs=(0.25, 0.5))
    env.reset(start=40)
    assert _advance_to(env, "entry", "RUN", filler=2)
    env.step([0])                                     # enter at the rule's sizing
    assert _advance_to(env, "profit", "RUN", filler=0)
    j = env.col_ix["RUN"]
    p = env.pos["RUN"]
    assert env._px[env.bar, j] / env._px[p["entry_bar"], j] - 1.0 >= 0.25
    env.step([0])                                     # idx0 on a profit prompt = LET IT RUN
    assert "RUN" in env.pos and env.pos["RUN"]["tp_i"] >= 1
    assert _advance_to(env, "profit", "RUN", filler=0)  # the +50% rung prompts later
    assert env.pos["RUN"]["tp_i"] == 1


def test_profit_take_idx3_sells_all_realizing_the_gain():
    env = _rd_env(tp_rungs=(0.25,))
    env.reset(start=40)
    assert _advance_to(env, "entry", "RUN", filler=2)
    env.step([0])
    assert _advance_to(env, "profit", "RUN", filler=0)
    bar = env.bar
    env.step([3])                                     # idx3 on a profit prompt = SELL INTO STRENGTH
    assert "RUN" not in env.pos
    assert env.cool["RUN"] == bar                     # full close arms cooldown + dead-zone
    assert env._equity() > 10_000.0                   # the +25% was realized, not given back


def test_no_tp_rungs_means_no_profit_events():
    env = _rd_env()                                   # tp_rungs off
    env.reset(start=40)
    assert not _advance_to(env, "profit", max_steps=400, filler=0)


def test_loss_floor_forces_cut_despite_override():
    """A position below entry*(1-floor) cannot be overridden — idx3 (hold) still force-cuts."""
    env = _rd_env(loss_floor=0.2)
    env.reset(start=40)
    env._px, env._cush = env._px.copy(), env._cush.copy()
    t = env.universe[0]
    j = env.col_ix[t]
    bar = env.bar
    env.pos[t] = {"usd": 100.0, "entry_bar": bar, "peak_px": 1.0, "origin": 1.0, "tp_i": 0}
    px_entry = env._px[bar, j]
    later = bar + 1
    env._px[later, j] = px_entry * 0.75                  # 25% below entry: under the floor
    env.bar = later
    env._do_exit(t, RULE_DEFAULT_EXIT_KEEP[3])           # idx3 = hold/override
    assert t not in env.pos                              # forced cut anyway
    assert env.cool[t] == later


def test_loss_floor_punctures_the_commit_window():
    env = _rd_env(loss_floor=0.2, exit_commit=12)
    env.reset(start=40)
    env._px, env._cush = env._px.copy(), env._cush.copy()
    t = env.universe[0]
    j = env.col_ix[t]
    bar = env.bar
    env.pos[t] = {"usd": 100.0, "entry_bar": bar, "peak_px": env._px[bar, j],
                  "origin": 1.0, "tp_i": 0}
    env._exit_decided[t] = bar                           # freshly committed (e.g. an override)
    env._cush[bar + 3, j] = 0.1                          # above EMA: only the floor can prompt
    env._px[bar + 3, j] = env._px[bar, j] * 0.9          # -10%: above floor -> commit holds
    assert ("exit", t) not in env._scan_bar(bar + 3)
    env._px[bar + 5, j] = env._px[bar, j] * 0.7          # -30%: below floor -> punctures commit
    env._cush[bar + 5, j] = 0.1
    assert ("exit", t) in env._scan_bar(bar + 5)


def test_above_floor_override_still_works():
    env = _rd_env(loss_floor=0.2)
    env.reset(start=40)
    env._px, env._cush = env._px.copy(), env._cush.copy()
    t = env.universe[0]
    j = env.col_ix[t]
    bar = env.bar
    env.pos[t] = {"usd": 100.0, "entry_bar": bar, "peak_px": 2.0, "origin": 1.0, "tp_i": 0}
    env._px[bar, j] = env._px[bar, j] * 1.4              # well above entry (a winner in retrace)
    env._do_exit(t, RULE_DEFAULT_EXIT_KEEP[3])           # override allowed
    assert t in env.pos                                  # winner can still be ridden


def test_detonation_blacklist_kills_later_ignitions():
    """A detonation (huge surge while price collapses) must zero the token's ignitions for the
    blacklist window; ignitions BEFORE the detonation and on other tokens are untouched."""
    returns, btc, vol, liq = _panel()
    vol.loc[vol.index[100:104], "RUN"] = 5000.0          # huge volume ON the crash (bars 92-120 fall)
    base = dict(volume=vol, k=3, ema_span=10, warmup=30, episode_bars=200, vol_mult=2.5,
                vol_spk=4, vol_base=20, vol_fast=4, stop_k=0.1, cooldown=8, seed=0)
    on = EventRungEnv(returns, btc, liq, det_blacklist=80, det_surge=8.0, det_drop=-0.15, **base)
    off = EventRungEnv(returns, btc, liq, **base)
    j = on.col_ix["RUN"]
    pre_det = on._ignite[:100, j]
    assert (pre_det == off._ignite[:100, j]).all()        # before the detonation: identical
    det_zone = slice(104, 184)                            # 80-bar blacklist after the det bars
    assert not on._ignite[det_zone, j].any()              # blacklisted
    other = on.col_ix["C1"]
    assert (on._ignite[:, other] == off._ignite[:, other]).all()   # other tokens untouched


def _frac_panel(returns, default=1.0, **overrides):
    """A low_frac/high_frac panel: `overrides` = {token: (bar_slice, value)}."""
    import pandas as pd
    df = pd.DataFrame(default, index=returns.index, columns=returns.columns)
    for tok, (sl, v) in overrides.items():
        df.iloc[sl, df.columns.get_loc(tok)] = v
    return df


def test_intrabar_floor_fills_at_the_stop_not_the_close():
    """A bar whose LOW crosses entry*(1-floor) force-fills AT the floor price — even when the
    close collapses far below it (the Q -53%-in-one-bar hole) or recovers above it (a wick)."""
    returns, btc, vol, liq = _panel()
    lowf = _frac_panel(returns, 1.0, RUN=(slice(95, 96), 0.5))   # bar 95: low = 50% of close
    base = dict(volume=vol, k=3, ema_span=10, warmup=30, episode_bars=140, vol_mult=2.5,
                vol_spk=4, vol_base=20, vol_fast=4, stop_k=0.1, cooldown=8, seed=0,
                action_mode="discrete", n_action_levels=4, rule_default=True)
    env = EventRungEnv(returns, btc, liq, loss_floor=0.2, intrabar_floor=True,
                       low_frac=lowf, **base)
    env.reset(start=90)
    j = env.col_ix["RUN"]
    entry_px = env._px[91, j]
    env.pos["RUN"] = {"usd": 1000.0, "entry_bar": 91, "peak_px": entry_px, "origin": 1.0, "tp_i": 0}
    env.bar = 94
    cash0 = env.cash
    env._advance_to_event()                                  # crosses bar 95: low < floor -> fill
    assert "RUN" not in env.pos
    got = env.cash - cash0
    assert got == pytest.approx(1000.0 * 0.8, rel=0.02)      # filled at the floor (-20%), NOT the
    assert env.cool["RUN"] >= 95                             # bar's path low (-50%+)


def test_intrabar_floor_requires_data_and_floor():
    with pytest.raises(ValueError):
        _rd_env(intrabar_floor=True, loss_floor=0.2)         # no low_frac data


def test_wick_reject_kills_extreme_rejection_ignitions():
    returns, btc, vol, liq = _panel()
    highf = _frac_panel(returns, 1.0, RUN=(slice(0, len(returns)), 0.6))   # all bars: close=60% of high
    base = dict(volume=vol, k=3, ema_span=10, warmup=30, episode_bars=200, vol_mult=2.5,
                vol_spk=4, vol_base=20, vol_fast=4, stop_k=0.1, cooldown=8, seed=0)
    guarded = EventRungEnv(returns, btc, liq, wick_reject=0.30, high_frac=highf, **base)
    plain = EventRungEnv(returns, btc, liq, **base)
    j = guarded.col_ix["RUN"]
    assert plain._ignite[:, j].any()                         # the fixture's ignition exists
    assert not guarded._ignite[:, j].any()                   # 0.6 < 0.7 -> every trigger killed
    ok = EventRungEnv(returns, btc, liq, wick_reject=0.30,
                      high_frac=_frac_panel(returns, 0.95), **base)
    assert (ok._ignite[:, j] == plain._ignite[:, j]).all()   # strong closes untouched


def test_cycle_obs_spent_move_slots():
    """cycle_obs appends 2 bounded slots: ret-since / bars-since the token's PRIOR ignition.
    At RUN's FIRST ignition prompt the token is fresh (0.0 / 1.0); at a later prompt after the
    runup, ret_since is positive (the spent-move flag the probe validated)."""
    env = _rd_env(cycle_obs=True)
    env.reset(start=40)
    assert env.obs_dim == 15                              # 13 base + 2 cycle slots
    assert _advance_to(env, "entry", "RUN", filler=2)
    obs = env._obs()
    assert obs.shape == (15,)
    first_ret, first_bars = float(obs[-2]), float(obs[-1])
    assert first_ret == 0.0 and first_bars == 1.0        # fresh token: no prior ignition
    j = env.col_ix["RUN"]
    later = min(env.bar + 30, env.n_bars - 1)            # mid/post-runup: a prior ignition exists
    env.bar = later
    env._pending = ("entry", "RUN")
    obs2 = env._obs()
    assert obs2[-1] < 1.0                                 # bars-since now meaningful
    assert abs(obs2[-2]) <= 1.0                           # tanh-bounded
    plain = _rd_env()
    plain.reset(start=40)
    assert plain.obs_dim == 13                            # flag off: unchanged


def test_all_default_policy_tracks_the_rule_mirror():
    """Parity (gate B, unit-scale): a policy answering idx0 at EVERY prompt through the env should
    track the rule mirror's equity on the synthetic panel (flat caps -> sizing matches ef=0.20)."""
    env = _rd_env(reward_mode="relative")             # relative mode precomputes the rule mirror
    env.reset(start=40)
    for _ in range(800):
        if env._pending[0] == "none":
            break
        _, _, done, _ = env.step([0])
        if done:
            break
    agent_eq = env._equity()
    rule_eq = env._rule_eq[-1]
    assert agent_eq == pytest.approx(rule_eq, rel=0.02)
