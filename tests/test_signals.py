"""The decision-tape tally (trading/signals.json): the pure day-bucketing + an instrumented
replay over the recorded panel with a fake predictor (no torch)."""

import numpy as np
import pytest

from trader.agent.event_live import MONDAY_PHASE, WARMUP, WEEK_SECS, LiveEventTrader
from trader.agent.signals import replay_decisions, tally

from train_rl import build_volume_panel, load_data  # noqa: E402 (event_live set the path)

RULE = lambda obs: np.array([0])  # noqa: E731 — execute the rung-0 rule


def _prov() -> dict:
    return {"k": 8, "max_entry_frac": 0.34, "stop_k": 0.25, "cooldown": 48, "dd_lambda": 2.0,
            "dd_soft": 0.15, "reward_mode": "absolute", "r4_beta": 0.0, "res_gamma": 0.0,
            "fwd_horizon": 24, "ungate": False, "action_mode": "discrete", "n_action_levels": 4,
            "universe_mode": "voltopk", "vol_target": 0.005, "cap_floor": 0.02,
            "harvest_obs": False, "rule_default": True, "basket_default": False, "exit_commit": 12,
            "dust_usd": 0.0, "tp_rungs": "0.25,0.5,1,2", "loss_floor": 0.2, "det_blacklist": 0,
            "scale_in": False, "cycle_obs": False, "no_btc_obs": False, "universe_lookback": 0,
            "intrabar_floor": False, "wick_reject": 0.0, "recurrent": False, "seed": 0}


@pytest.fixture(scope="module")
def panels():
    returns, btc, _a, liq = load_data()
    return returns, btc, liq, build_volume_panel(list(returns.columns), returns.index)


def test_tally_buckets_by_day_with_participation():
    events = [
        {"type": "entry", "token": "HUMA", "time": 1, "time_utc": "2026-06-17T20:00:00Z",
         "action_idx": 0, "executed": True},
        {"type": "entry", "token": "SKYAI", "time": 2, "time_utc": "2026-06-17T10:00:00Z",
         "action_idx": 2, "executed": False},
        {"type": "exit", "token": "HUMA", "time": 3, "time_utc": "2026-06-18T14:00:00Z",
         "action_idx": 0, "executed": True},
    ]
    t = tally(1781481600, events, generated="G")
    assert t["totals"] == {"signals_seen": 2, "executed": 1, "ignored": 1, "exits": 1,
                           "participation": 0.5}
    by = {d["date"]: d for d in t["days"]}
    assert by["2026-06-17"]["signals_seen"] == 2 and by["2026-06-17"]["ignored"] == 1
    assert by["2026-06-18"]["exits"] == 1
    assert t["week_start"] == 1781481600


def test_tally_empty():
    t = tally(0, [], generated="G")
    assert t["totals"]["signals_seen"] == 0 and t["totals"]["participation"] is None


def _week_with_fills(returns, btc, liq, vol):
    idx = [int(x) for x in returns.index]
    have, pos = set(idx), {x: i for i, x in enumerate(idx)}
    tr = LiveEventTrader(_prov())
    ek = tr.env_kwargs(returns)
    for t in idx:
        if t % WEEK_SECS == MONDAY_PHASE and pos[t] >= WARMUP and (t + 160 * 3600) in have:
            _ws, events = replay_decisions(tr, returns, btc, liq, vol, t + 167 * 3600, ek, predict_fn=RULE)
            if any(e["type"] == "entry" for e in events):
                return t, ek, events
    raise AssertionError("no week with entry decisions in recorded data")


def test_replay_decisions_captures_entries_and_tally_is_consistent(panels):
    returns, btc, liq, vol = panels
    ws, _ek, events = _week_with_fills(returns, btc, liq, vol)
    entries = [e for e in events if e["type"] == "entry"]
    assert len(entries) >= 1
    for e in entries:
        assert {"type", "token", "time", "time_utc", "action_idx", "executed"} <= set(e)
        assert e["time_utc"].endswith("Z")
    t = tally(ws, events)
    assert t["totals"]["signals_seen"] == len(entries)
    assert t["totals"]["executed"] + t["totals"]["ignored"] == len(entries)
