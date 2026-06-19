"""The paper-trading runner: fill recording, the guardrail audit, week reset, and no
double-recording across hourly ticks — exercised against the recorded panel with a fake
"execute-the-rule" predictor (no torch / no checkpoint, no network: panels injected)."""

import numpy as np
import pandas as pd
import pytest

from trader.agent import store
from trader.agent.event_live import WARMUP, WEEK_SECS, MONDAY_PHASE, LiveEventTrader
from trader.agent.event_runner import EventRunner, forward_run_policy
from trader.risk import Policy

from train_rl import build_volume_panel, load_data  # noqa: E402 (event_live set the path)

RULE = lambda obs: np.array([0])  # noqa: E731 — idx 0 = execute the rung-0 rule


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
    returns, btc, _anchor, liq = load_data()
    vol = build_volume_panel(list(returns.columns), returns.index)
    return returns, btc, liq, vol


def _first_full_week(returns):
    idx = [int(t) for t in returns.index]
    have, pos = set(idx), {t: i for i, t in enumerate(idx)}
    for t in idx:
        if t % WEEK_SECS == MONDAY_PHASE and pos[t] >= WARMUP and (t + 167 * 3600) in have:
            return t
    raise AssertionError("no full recorded week")


def _runner(tmp_path, **kw):
    trader = LiveEventTrader(_prov())
    return EventRunner(trader, selection=[], agent_ledger_path=tmp_path / "agent.jsonl", **kw)


def test_tick_records_fills_equity_and_heartbeat(tmp_path, panels):
    returns, *_ = panels
    ws = _first_full_week(returns)
    runner = _runner(tmp_path)
    res = runner.tick(ws + 120 * 3600, panels=panels, predict_fn=RULE, refresh_data=False)

    assert res.week_start == ws and res.new_week is True
    assert res.equity_usd > 0 and 0.0 <= res.drawdown_pct <= 100.0
    rows = store.read_rows(tmp_path / "agent.jsonl")
    kinds = [r["kind"] for r in rows]
    assert "equity" in kinds and "heartbeat" in kinds
    fills = [r for r in rows if r["kind"] == "fill"]
    assert len(fills) == res.fills_recorded + res.fills_blocked
    if fills:                                            # fills carry the trigger reason + the bar
        assert {"token", "trigger", "bar_ts", "guardrail_ok"} <= set(fills[0])


def test_no_double_recording_across_ticks_same_week(tmp_path, panels):
    returns, *_ = panels
    ws = _first_full_week(returns)
    runner = _runner(tmp_path)
    runner.tick(ws + 90 * 3600, panels=panels, predict_fn=RULE, refresh_data=False)
    n1 = sum(1 for r in store.read_rows(tmp_path / "agent.jsonl") if r["kind"] == "fill")
    runner.tick(ws + 150 * 3600, panels=panels, predict_fn=RULE, refresh_data=False)
    rows = store.read_rows(tmp_path / "agent.jsonl")
    n2 = sum(1 for r in rows if r["kind"] == "fill")
    assert n2 >= n1                                      # only NEW fills added, never re-recorded
    bar_ts = [r["bar_ts"] for r in rows if r["kind"] == "fill"]
    assert bar_ts == sorted(bar_ts)                      # chronological, no rewind/duplication
    assert all(t <= ws + 150 * 3600 for t in bar_ts)


def test_week_rollover_resets_and_starts_new_session(tmp_path, panels):
    returns, *_ = panels
    idx = [int(t) for t in returns.index]
    have, pos = set(idx), {t: i for i, t in enumerate(idx)}
    weeks = [t for t in idx if t % WEEK_SECS == MONDAY_PHASE and pos[t] >= WARMUP
             and (t + 167 * 3600) in have]
    assert len(weeks) >= 2
    w1, w2 = weeks[0], weeks[1]
    runner = _runner(tmp_path)
    r1 = runner.tick(w1 + 100 * 3600, panels=panels, predict_fn=RULE, refresh_data=False)
    r2 = runner.tick(w2 + 100 * 3600, panels=panels, predict_fn=RULE, refresh_data=False)
    assert r1.new_week and r2.new_week                   # each Monday opens a fresh cold session
    assert r2.week_start == w2 and runner._week_start == w2


def test_guardrail_blocks_off_allowlist_fills(tmp_path, panels):
    returns, *_ = panels
    ws = _first_full_week(returns)
    # a policy whose allowlist excludes every universe token -> every buy/sell is NOT_ALLOWLISTED
    locked = Policy(allowlist=frozenset({"NOTHING"}), per_trade_usd=1e9, daily_usd=1e12,
                    max_slippage_pct=1.0, drawdown_stop_pct=30.0, lifetime_usd_ceiling=1e15,
                    chain="bsc")
    runner = _runner(tmp_path, policy=locked)
    res = runner.tick(ws + 120 * 3600, panels=panels, predict_fn=RULE, refresh_data=False)
    rows = store.read_rows(tmp_path / "agent.jsonl")
    refusals = [r for r in rows if r["kind"] == "refusal"]
    if res.fills_recorded + res.fills_blocked > 0:       # the rule traded this week
        assert res.fills_blocked > 0 and res.fills_recorded == 0
        assert len(refusals) == res.fills_blocked
        assert all("NOT_ALLOWLISTED" in r["refusals"] for r in refusals)


def _week_with_fills(returns, btc, liq, vol):
    """A Monday week where the rule predictor produces >=1 fill, so the diff tests aren't vacuous."""
    idx = [int(t) for t in returns.index]
    have, pos = set(idx), {t: i for i, t in enumerate(idx)}
    trader = LiveEventTrader(_prov())
    ek = trader.env_kwargs(returns)
    for t in idx:
        if t % WEEK_SECS == MONDAY_PHASE and pos[t] >= WARMUP and (t + 160 * 3600) in have:
            res = trader.evaluate_week(returns, btc, liq, vol, t + 167 * 3600, ek, predict_fn=RULE)
            if res["fills"]:
                return t, ek
    raise AssertionError("no week with rule fills in recorded data")


def test_offset_now_records_every_fill_once(tmp_path, panels):
    """Regression for the wall-clock-vs-bar-time cursor bug: ticks fire a few min PAST the bar
    (real HH:03 schedule), so a wall-clock cursor drops fills whose bar-time < the prior tick's
    wall-time (the dropped HUMA EMA_BREAK exit). Every env fill must be recorded EXACTLY once."""
    returns, btc, liq, vol = panels
    ws, ek = _week_with_fills(returns, btc, liq, vol)
    runner = _runner(tmp_path)
    OFF = 183                                              # minutes past the bar, like the live timer
    for h in (40, 60, 80, 100, 120, 140, 160):
        runner.tick(ws + h * 3600 + OFF, panels=panels, predict_fn=RULE, refresh_data=False)
    rows = store.read_rows(tmp_path / "agent.jsonl")
    got = sorted((r["bar_ts"], r["token"], "buy" if r["from"] == "USDT" else "sell")
                 for r in rows if r["kind"] == "fill")
    last_now = ws + 160 * 3600 + OFF
    res = LiveEventTrader(_prov())  # ground truth: one replay to the last bar
    truth = sorted((f.time, f.token, f.side)
                   for f in res.evaluate_week(returns, btc, liq, vol, last_now, ek, predict_fn=RULE)["fills"])
    assert len(truth) >= 1                                  # the week genuinely traded
    assert got == truth                                    # no drop (the bug), no duplicate


def test_restart_does_not_duplicate_fills(tmp_path, panels):
    """A process restart (fresh runner on the SAME ledger) must NOT re-record the week's fills —
    the cursor resumes from the ledger, not ws-1 (the duplicate-on-restart bug)."""
    returns, btc, liq, vol = panels
    ws, _ek = _week_with_fills(returns, btc, liq, vol)
    led = tmp_path / "agent.jsonl"
    now = ws + 150 * 3600
    EventRunner(LiveEventTrader(_prov()), selection=[], agent_ledger_path=led).tick(
        now, panels=panels, predict_fn=RULE, refresh_data=False)
    n1 = sum(1 for r in store.read_rows(led) if r["kind"] == "fill")
    assert n1 >= 1
    # brand-new runner (a systemd restart) on the same ledger + same now -> no new bars
    EventRunner(LiveEventTrader(_prov()), selection=[], agent_ledger_path=led).tick(
        now, panels=panels, predict_fn=RULE, refresh_data=False)
    n2 = sum(1 for r in store.read_rows(led) if r["kind"] == "fill")
    assert n2 == n1                                        # idempotent — restart re-recorded nothing


def test_fill_price_is_real_usd_not_env_index(tmp_path, panels):
    """The fill `price` is the real USD close (≈ market price), with the env's internal
    return-index kept as `price_index`."""
    import json
    import os
    returns, btc, liq, vol = panels
    ws, _ek = _week_with_fills(returns, btc, liq, vol)
    sel_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            "data", "selection.json")
    selection = [{"symbol": s["symbol"], "pair_address": s["pair_address"]}
                 for s in json.load(open(sel_path, encoding="utf-8"))]
    runner = EventRunner(LiveEventTrader(_prov()), selection=selection,
                         agent_ledger_path=tmp_path / "agent.jsonl")
    runner.tick(ws + 167 * 3600, panels=panels, predict_fn=RULE, refresh_data=False)
    fills = [r for r in store.read_rows(tmp_path / "agent.jsonl") if r["kind"] == "fill"]
    assert fills
    from trader.agent.live_data import build_close_panel
    cp = build_close_panel(selection, returns.index)
    f = fills[0]
    assert "price_index" in f and 0.5 < f["price_index"] < 2.0      # the index sits near 1.0
    assert f["price"] == pytest.approx(float(cp.at[f["bar_ts"], f["token"]]), rel=1e-6)  # real USD


def test_forward_run_policy_allows_universe_and_cash_leg():
    pol = forward_run_policy(["ADA", "zec"], capital=10_000.0)
    assert "ADA" in pol.allowlist and "ZEC" in pol.allowlist and "USDT" in pol.allowlist
    assert pol.drawdown_stop_pct == 30.0 and pol.chain == "bsc"
    assert pol.per_trade_usd == 10_000.0
