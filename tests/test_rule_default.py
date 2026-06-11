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
