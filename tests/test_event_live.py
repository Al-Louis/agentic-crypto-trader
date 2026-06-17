"""The live event-driven driver's orchestration: cold-week window math + hourly fill-diff.

Exercises `trader.agent.event_live` against the RECORDED panel with a deterministic fake
predictor (idx 0 = execute the rung-0 rule), so the window selection, the reuse of the
validated `evaluate_event_policy`, and the determinism the fill-diff relies on are proven
without torch / the checkpoint. The model-loading path (`_predict_fn`) is not exercised here —
it needs the ef-s2 artifact + sb3-contrib (covered by the on-box dry-run gate).
"""

import numpy as np
import pandas as pd
import pytest

from trader.agent.event_live import (MONDAY_PHASE, WARMUP, WEEK_SECS, LiveEventTrader,
                                      cold_week_window, new_fills, week_start_for)

# importing event_live put scripts/ + src/ on the path
from train_rl import build_volume_panel, load_data  # noqa: E402


# --- pure window math (synthetic, no data) ----------------------------------

def test_week_start_for_is_monday_midnight_utc():
    # MONDAY_PHASE itself is a Monday 00:00; anything within the next 7d maps back to it
    assert week_start_for(MONDAY_PHASE) == MONDAY_PHASE
    assert week_start_for(MONDAY_PHASE + 3600) == MONDAY_PHASE
    assert week_start_for(MONDAY_PHASE + WEEK_SECS - 1) == MONDAY_PHASE
    assert week_start_for(MONDAY_PHASE + WEEK_SECS) == MONDAY_PHASE + WEEK_SECS
    # every result is an exact Monday phase
    for t in (MONDAY_PHASE + 99_999, MONDAY_PHASE + 5 * WEEK_SECS + 777):
        assert week_start_for(t) % WEEK_SECS == MONDAY_PHASE


def _synthetic_returns(n_bars: int, start: int, n_tok: int = 8) -> pd.DataFrame:
    idx = [start + i * 3600 for i in range(n_bars)]
    return pd.DataFrame(np.zeros((n_bars, n_tok)), index=idx,
                        columns=[f"T{i}" for i in range(n_tok)])


def test_cold_week_window_prepads_warmup_and_truncates_at_now():
    # build a panel that starts WARMUP+50 bars before a Monday open and runs 168 into the week
    ws = MONDAY_PHASE + 40 * WEEK_SECS
    panel_start = ws - (WARMUP + 50) * 3600
    r = _synthetic_returns(WARMUP + 50 + 168, panel_start)
    now = ws + 100 * 3600                       # 100 bars into the week
    win, got_ws, i0 = cold_week_window(r, now)
    assert got_ws == ws
    assert int(win.index[0]) == ws - WARMUP * 3600        # exactly WARMUP prepad before the open
    assert int(win.index[-1]) == now                       # truncated at the just-closed bar
    assert len(win) == WARMUP + 100 + 1


def test_cold_week_window_needs_warmup_and_present_open():
    ws = MONDAY_PHASE + 40 * WEEK_SECS
    r = _synthetic_returns(50 + 168, ws - 50 * 3600)       # only 50 bars before the open
    with pytest.raises(ValueError, match="warmup"):
        cold_week_window(r, ws + 10 * 3600)
    r2 = _synthetic_returns(200, ws + WEEK_SECS)           # panel begins AFTER this week
    with pytest.raises(ValueError, match="not present"):
        cold_week_window(r2, ws + 10 * 3600)


# --- end-to-end over a real recorded week (fake predictor, no torch) --------

def _ef_like_prov() -> dict:
    """A complete, valid EventRungEnv provenance for the orchestration test — rule_default
    discrete (ef-s2's head) but WITHOUT the intrabar/wick frac panels, so no OHLC-frac build is
    needed. The orchestration is config-agnostic; the real ef-s2 config is exercised on-box."""
    return {"k": 8, "max_entry_frac": 0.34, "stop_k": 0.25, "cooldown": 48,
            "dd_lambda": 2.0, "dd_soft": 0.15, "reward_mode": "absolute", "r4_beta": 0.0,
            "res_gamma": 0.0, "fwd_horizon": 24, "ungate": False, "action_mode": "discrete",
            "n_action_levels": 4, "universe_mode": "voltopk", "vol_target": 0.005,
            "cap_floor": 0.02, "harvest_obs": False, "rule_default": True, "basket_default": False,
            "exit_commit": 12, "dust_usd": 0.0, "tp_rungs": "0.25,0.5,1,2", "loss_floor": 0.2,
            "det_blacklist": 0, "scale_in": False, "cycle_obs": False, "no_btc_obs": False,
            "universe_lookback": 0, "intrabar_floor": False, "wick_reject": 0.0,
            "recurrent": False, "seed": 0}


@pytest.fixture(scope="module")
def recorded():
    returns, btc, _anchor, liq = load_data()
    vol = build_volume_panel(list(returns.columns), returns.index)
    return returns, btc, liq, vol


def _first_full_week(returns):
    """The first Monday-open in the panel with a full WARMUP behind it and a full week ahead."""
    idx = [int(t) for t in returns.index]
    have = set(idx)
    pos = {t: i for i, t in enumerate(idx)}
    for t in idx:
        if t % WEEK_SECS == MONDAY_PHASE and pos[t] >= WARMUP and (t + 167 * 3600) in have:
            return t
    raise AssertionError("no full Monday week in recorded data")


def test_evaluate_week_runs_and_is_deterministic_for_the_fill_diff(recorded):
    returns, btc, liq, vol = recorded
    ws = _first_full_week(returns)
    trader = LiveEventTrader(_ef_like_prov())              # no model needed — predict_fn injected
    env_kwargs = trader.env_kwargs(returns)
    rule = lambda obs: np.array([0])                       # idx 0 = execute the rung-0 rule  # noqa: E731

    now1 = ws + 90 * 3600
    res1 = trader.evaluate_week(returns, btc, liq, vol, now1, env_kwargs, predict_fn=rule)
    assert res1["week_start"] == ws
    assert isinstance(res1["equity"], pd.Series) and len(res1["equity"]) >= 90
    assert res1["win_index"][-1] == now1

    now2 = ws + 140 * 3600                                 # +50 bars later in the SAME week
    res2 = trader.evaluate_week(returns, btc, liq, vol, now2, env_kwargs, predict_fn=rule)

    # determinism the diff relies on: every fill on a bar present in run 1 is IDENTICAL in run 2
    f1 = res1["fills"]
    f2_upto = [f for f in res2["fills"] if f.time <= now1]
    assert len(f1) == len(f2_upto)
    for a, b in zip(f1, f2_upto):
        assert (a.token, a.time, a.side) == (b.token, b.time, b.side)
        assert a.usd == pytest.approx(b.usd) and a.price == pytest.approx(b.price)

    # the hourly diff: fills strictly after now1 are exactly run 2's tail (this hour's new trades)
    fresh = new_fills(res2["records"], after_ts=now1)
    assert all(f.time > now1 for f in fresh)
    assert all(f.time <= now2 for f in fresh)
    assert len(fresh) == len(res2["fills"]) - len(f2_upto)


def test_new_fills_on_fresh_week_surfaces_whole_stream(recorded):
    returns, btc, liq, vol = recorded
    ws = _first_full_week(returns)
    trader = LiveEventTrader(_ef_like_prov())
    env_kwargs = trader.env_kwargs(returns)
    rule = lambda obs: np.array([0])  # noqa: E731
    res = trader.evaluate_week(returns, btc, liq, vol, ws + 120 * 3600, env_kwargs, predict_fn=rule)
    # after_ts before the week open -> every fill this week is "new" (the cold-week reset case)
    assert len(new_fills(res["records"], after_ts=ws - 1)) == len(res["fills"])
