"""The paper-trading runner: fill recording, the guardrail audit, week reset, and no
double-recording across hourly ticks — exercised against the recorded panel with a fake
"execute-the-rule" predictor (no torch / no checkpoint, no network: panels injected)."""

import numpy as np
import pandas as pd
import pytest

from trader.agent import store
from trader.agent.event_live import WARMUP, WEEK_SECS, MONDAY_PHASE, LiveEventTrader
from trader.agent.event_runner import EventRunner, forward_run_policy, live_forward_policy
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


class _FakeExec:
    """Stand-in for `execute_trade` — records each call (intent + policy + dry_run) and returns a
    scripted result, so the live-signing WIRING is tested without any network/keychain/real funds."""

    def __init__(self, result=None):
        self.calls = []
        self.result = result if result is not None else {
            "tx_hash": "0x" + "cd" * 32, "status": "confirmed", "usd": 0.40}

    def __call__(self, intent, policy, *, dry_run=False):
        self.calls.append({"from": intent.from_asset, "to": intent.to_asset, "usd": intent.usd,
                           "slippage_pct": intent.slippage_pct, "dry_run": dry_run, "policy": policy})
        return self.result


_TIGHT = Policy(allowlist=frozenset({"BNB", "USDT"}), per_trade_usd=0.5, daily_usd=1.5,
                max_slippage_pct=1.0, drawdown_stop_pct=30.0, lifetime_usd_ceiling=3.0, chain="bsc")


class _FakeExecBnbOut:
    """USD executor whose result reports the REALIZED BNB it bought (out_amount/out_symbol), so the
    compliance BUY can capture the qty for the M4 amount-in unwind. BNB at a flat $600."""

    def __init__(self):
        self.calls = []

    def __call__(self, intent, policy, *, dry_run=False):
        self.calls.append({"from": intent.from_asset, "to": intent.to_asset, "usd": intent.usd})
        return {"tx_hash": "0x" + "bb" * 32, "status": "confirmed", "usd": intent.usd,
                "out_amount": intent.usd / 600.0, "out_symbol": "BNB"}


class _FakeAmountExec:
    """Amount-in executor (stands in for execute_swap_amount) — records the exact qty it was asked
    to swap so the test can assert the SELL unwinds the BUY's BNB quantity, not a USD notional."""

    def __init__(self):
        self.calls = []

    def __call__(self, frm, to, amount, policy, *, dry_run=False):
        self.calls.append({"from": frm, "to": to, "amount": amount, "policy": policy})
        return {"tx_hash": "0x" + "ee" * 32, "status": "confirmed", "amount": amount,
                "out_amount": amount * 600.0, "out_symbol": "USDT"}


class _FakeExecRefuse:
    """USD executor that REFUSES every call (e.g. a real drawdown/cap block) — for asserting the
    compliance SELL never signs when the BUY didn't land."""

    def __init__(self):
        self.calls = []

    def __call__(self, intent, policy, *, dry_run=False):
        self.calls.append({"from": intent.from_asset, "to": intent.to_asset})
        return {"refused": ["DRAWDOWN_STOP"], "phase": "intent"}


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


# -- daily >=1-trade/day compliance overlay (Rule-1) ------------------------

def test_compliance_action_schedule():
    import datetime as dt

    from trader.agent.compliance import compliance_action

    def at(h):
        return int(dt.datetime(2026, 6, 24, h, 3, tzinfo=dt.timezone.utc).timestamp())
    assert compliance_action(at(1)) == "buy"
    assert compliance_action(at(23)) == "sell"
    assert compliance_action(at(0)) is None and compliance_action(at(12)) is None


def test_compliance_positions_pure():
    """The simulate_weekly overlay builder: one BNB round-trip per UTC day, hours [1,23], rising
    intraday -> profit, flat -> loss (cost only), frac=0 -> nothing."""
    import datetime as dt

    from trader.agent.compliance import DEFAULT_FRAC, compliance_positions
    WEEK = 7 * 24 * 3600
    ws = (1_700_000_000 // WEEK) * WEEK + 345600           # a Monday 00:00 UTC
    rising = lambda ts: 600.0 + (int(ts) % 86400) / 86400 * 60.0   # higher later in the day
    pos, pnl = compliance_positions(ws, ws + WEEK, rising, frac=DEFAULT_FRAC, capital=10_000.0)
    assert len(pos) == 7 and all(p["kind"] == "compliance" for p in pos)
    assert all(dt.datetime.fromtimestamp(p["entry_t"], dt.timezone.utc).hour == 1 for p in pos)
    assert all(dt.datetime.fromtimestamp(p["exit_t"], dt.timezone.utc).hour == 23 for p in pos)
    assert pnl > 0                                          # bought low (01:00), sold high (23:00)
    _, pnl_flat = compliance_positions(ws, ws + WEEK, lambda ts: 600.0, capital=10_000.0)
    assert pnl_flat < 0                                     # flat price -> only the AMM cost
    assert compliance_positions(ws, ws + WEEK, lambda ts: 600.0, frac=0.0)[0] == []


def test_compliance_round_trip_records_floor_fills(tmp_path, panels):
    """BUY 3% BNB at 01:00, SELL it back at 23:00 — two recorded fills (the >=1-trade/day floor),
    allowlisted, off the env book, sized at 3% of equity."""
    returns, *_ = panels
    ws = _first_full_week(returns)
    runner = _runner(tmp_path, bnb_price_fn=lambda ts: 600.0)
    rb = runner.tick(ws + (2 * 24 + 1) * 3600, panels=panels, predict_fn=RULE, refresh_data=False)   # Wed 01:00
    rs = runner.tick(ws + (2 * 24 + 23) * 3600, panels=panels, predict_fn=RULE, refresh_data=False)  # Wed 23:00
    comp = [r for r in store.read_rows(tmp_path / "agent.jsonl")
            if r["kind"] == "fill" and r.get("compliance")]
    assert sorted(r["reason"] for r in comp) == ["COMPLIANCE_BUY", "COMPLIANCE_SELL"]
    assert rb.compliance_trades == 1 and rs.compliance_trades == 1
    assert all(r["guardrail_ok"] and r["token"] == "BNB" for r in comp)   # BNB allowlisted -> not blocked
    buy = next(r for r in comp if r["reason"] == "COMPLIANCE_BUY")
    assert buy["from"] == "USDT" and buy["to"] == "BNB"
    assert buy["usd_in"] == pytest.approx(0.03 * rb.equity_usd, rel=1e-6)   # 3% of equity


def test_compliance_idempotent_same_day(tmp_path, panels):
    returns, *_ = panels
    ws = _first_full_week(returns)
    runner = _runner(tmp_path, bnb_price_fn=lambda ts: 600.0)
    buy_ts = ws + (2 * 24 + 1) * 3600
    runner.tick(buy_ts, panels=panels, predict_fn=RULE, refresh_data=False)
    r2 = runner.tick(buy_ts + 90, panels=panels, predict_fn=RULE, refresh_data=False)   # re-tick same hour
    buys = [r for r in store.read_rows(tmp_path / "agent.jsonl") if r.get("reason") == "COMPLIANCE_BUY"]
    assert len(buys) == 1 and r2.compliance_trades == 0     # no double-buy


def test_compliance_pnl_tracked(tmp_path, panels):
    """BNB +10% intraday -> the 3% sleeve realizes ~+10% of its notional (minus the small AMM cost)."""
    import datetime as dt
    returns, *_ = panels
    ws = _first_full_week(returns)

    def px(ts):
        return 660.0 if dt.datetime.fromtimestamp(int(ts), dt.timezone.utc).hour == 23 else 600.0
    runner = _runner(tmp_path, bnb_price_fn=px)
    runner.tick(ws + (2 * 24 + 1) * 3600, panels=panels, predict_fn=RULE, refresh_data=False)
    runner.tick(ws + (2 * 24 + 23) * 3600, panels=panels, predict_fn=RULE, refresh_data=False)
    buy_usd = next(r["usd_in"] for r in store.read_rows(tmp_path / "agent.jsonl")
                   if r.get("reason") == "COMPLIANCE_BUY")
    assert runner._compliance_pnl > 0
    assert runner._compliance_pnl == pytest.approx(0.10 * buy_usd, rel=0.1)   # ~10% gain less cost


def test_compliance_disabled_when_frac_zero(tmp_path, panels):
    returns, *_ = panels
    ws = _first_full_week(returns)
    runner = _runner(tmp_path, compliance_frac=0.0, bnb_price_fn=lambda ts: 600.0)
    r = runner.tick(ws + (2 * 24 + 1) * 3600, panels=panels, predict_fn=RULE, refresh_data=False)
    assert r.compliance_trades == 0
    assert not any(x.get("compliance") for x in store.read_rows(tmp_path / "agent.jsonl"))


# -- live signing sleeve (the TWAK execution path; gated on mode=="live" + an executor) -----

def test_compliance_live_signs_at_tiny_notional_and_records_tx(tmp_path, panels):
    """In live mode the compliance BUY routes through the executor at `live_compliance_usd` (NOT
    the $10k-book frac*equity), under the tight live policy, and the tx hash lands on the fill."""
    returns, *_ = panels
    ws = _first_full_week(returns)
    ex = _FakeExec()
    runner = _runner(tmp_path, mode="live", bnb_price_fn=lambda ts: 600.0,
                     execute_fn=ex, live_policy=_TIGHT, live_compliance_usd=0.40)
    rb = runner.tick(ws + (2 * 24 + 1) * 3600, panels=panels, predict_fn=RULE, refresh_data=False)
    assert rb.compliance_trades == 1
    buy = next(c for c in ex.calls if c["from"] == "USDT" and c["to"] == "BNB")
    assert buy["usd"] == pytest.approx(0.40) and buy["policy"] is _TIGHT and buy["dry_run"] is False
    comp = [r for r in store.read_rows(tmp_path / "agent.jsonl")
            if r["kind"] == "fill" and r.get("compliance")]
    assert comp and comp[0]["tx_hash"] == ex.result["tx_hash"] and comp[0]["exec_status"] == "confirmed"


def test_paper_mode_never_calls_the_executor(tmp_path, panels):
    """Default (paper) keeps the fill row byte-identical: the executor is never called and no
    exec fields are added — the EC2 paper service is unaffected by the sleeve."""
    returns, *_ = panels
    ws = _first_full_week(returns)
    ex = _FakeExec()
    runner = _runner(tmp_path, mode="paper", bnb_price_fn=lambda ts: 600.0, execute_fn=ex)
    runner.tick(ws + (2 * 24 + 1) * 3600, panels=panels, predict_fn=RULE, refresh_data=False)
    assert ex.calls == []
    comp = [r for r in store.read_rows(tmp_path / "agent.jsonl")
            if r["kind"] == "fill" and r.get("compliance")]
    assert comp and "tx_hash" not in comp[0] and "exec_status" not in comp[0]


def test_strategy_fills_route_through_executor_in_live(tmp_path, panels):
    """Every guardrail-passed strategy fill is signed via the executor in live mode (the env-fill
    half of the sleeve), and the tx hash is recorded on each allowed fill."""
    returns, btc, liq, vol = panels
    ws, _ek = _week_with_fills(returns, btc, liq, vol)
    ex = _FakeExec()
    runner = EventRunner(LiveEventTrader(_prov()), selection=[],
                         agent_ledger_path=tmp_path / "agent.jsonl",
                         mode="live", execute_fn=ex, compliance_frac=0.0,
                         live_bankroll_usd=10_000.0)          # scale 1.0 (== env book): executor gets full usd
    runner.tick(ws + 167 * 3600, panels=panels, predict_fn=RULE, refresh_data=False)
    fills = [r for r in store.read_rows(tmp_path / "agent.jsonl")
             if r["kind"] == "fill" and not r.get("compliance")]
    allowed = [f for f in fills if f["guardrail_ok"]]
    assert allowed and all(f.get("tx_hash") == ex.result["tx_hash"] for f in allowed)
    assert len(ex.calls) == len(allowed)                   # one executor call per allowed fill, none for blocked


def test_live_signing_requires_bankroll_or_policy(tmp_path):
    """M1 guard: arming live signing with NEITHER a bankroll NOR an explicit live_policy refuses at
    construction — otherwise real swaps would be capped by the $10k-scale env-parity policy."""
    with pytest.raises(ValueError, match="live signing requires"):
        EventRunner(LiveEventTrader(_prov()), selection=[], agent_ledger_path=tmp_path / "a.jsonl",
                    mode="live", execute_fn=_FakeExec())   # no live_bankroll_usd, no live_policy


def test_live_policy_auto_derived_from_bankroll_not_env_book(tmp_path, panels):
    """M1 fix: with a bankroll and no explicit live_policy, the executor is handed a policy whose caps
    are sized to the REAL bankroll (live_forward_policy), NOT the $10k-scale self.policy."""
    returns, btc, liq, vol = panels
    ws, _ek = _week_with_fills(returns, btc, liq, vol)
    ex = _FakeExec()
    runner = EventRunner(LiveEventTrader(_prov()), selection=[], agent_ledger_path=tmp_path / "agent.jsonl",
                         mode="live", execute_fn=ex, compliance_frac=0.0, live_bankroll_usd=100.0)
    runner.tick(ws + 167 * 3600, panels=panels, predict_fn=RULE, refresh_data=False)
    assert ex.calls                                         # the rule traded
    pol = ex.calls[0]["policy"]
    assert pol.per_trade_usd == pytest.approx(100.0 * 0.34 * 1.25)   # bankroll-scaled
    assert pol.per_trade_usd < 10_000.0                    # NOT the $10k env-parity cap


def test_zero_bankroll_scales_to_zero_not_full_size(tmp_path):
    """M2: an explicit 0.0 bankroll (depleted wallet) => scale 0.0 (=> $0 intents the executor
    refuses), NOT a silent fall-through to scale 1.0 / full $10k size."""
    r = EventRunner(LiveEventTrader(_prov()), selection=[], agent_ledger_path=tmp_path / "a.jsonl",
                    mode="live", execute_fn=_FakeExec(), live_bankroll_usd=0.0)
    assert r._live_scale == 0.0


def test_nonpositive_capital_rejected(tmp_path):
    with pytest.raises(ValueError, match="capital must be"):
        EventRunner(LiveEventTrader(_prov()), selection=[], agent_ledger_path=tmp_path / "a.jsonl",
                    capital=0.0)


def test_compliance_sell_unwinds_exact_bnb_qty_amount_in(tmp_path, panels):
    """M4: the live compliance SELL routes through the AMOUNT-in executor with the EXACT BNB the BUY
    acquired (captured as bnb_qty on the BUY fill), not a USD notional that could over/under-shoot."""
    returns, *_ = panels
    ws = _first_full_week(returns)
    usd_ex, amt_ex = _FakeExecBnbOut(), _FakeAmountExec()
    runner = _runner(tmp_path, mode="live", bnb_price_fn=lambda ts: 600.0,
                     execute_fn=usd_ex, execute_amount_fn=amt_ex, live_bankroll_usd=100.0)
    runner.tick(ws + (2 * 24 + 1) * 3600, panels=panels, predict_fn=RULE, refresh_data=False)   # BUY 01:00
    runner.tick(ws + (2 * 24 + 23) * 3600, panels=panels, predict_fn=RULE, refresh_data=False)  # SELL 23:00
    comp = [r for r in store.read_rows(tmp_path / "agent.jsonl")
            if r["kind"] == "fill" and r.get("compliance")]
    buy = next(r for r in comp if r["reason"] == "COMPLIANCE_BUY")
    assert buy.get("bnb_qty") and buy["bnb_qty"] > 0       # BUY captured the realized BNB qty
    assert amt_ex.calls and amt_ex.calls[0]["from"] == "BNB" and amt_ex.calls[0]["to"] == "USDT"
    assert amt_ex.calls[0]["amount"] == pytest.approx(buy["bnb_qty"])   # SELL unwinds the EXACT qty
    # the compliance BUY went via the USD executor (USDT->BNB); the SELL did NOT (it's the only amount-in)
    assert any(c["from"] == "USDT" and c["to"] == "BNB" for c in usd_ex.calls)
    assert not any(c["from"] == "BNB" and c["to"] == "USDT" for c in usd_ex.calls)  # SELL avoided USD path


def test_compliance_sell_skips_when_buy_did_not_land(tmp_path, panels):
    """Safety (review M4): if the live BUY is refused (no real BNB), the SELL must NOT sign a
    USD-sized swap (it would sell the wallet's gas buffer). No captured qty => SELL skips signing."""
    returns, *_ = panels
    ws = _first_full_week(returns)
    usd_ex, amt_ex = _FakeExecRefuse(), _FakeAmountExec()
    runner = _runner(tmp_path, mode="live", bnb_price_fn=lambda ts: 600.0,
                     execute_fn=usd_ex, execute_amount_fn=amt_ex, live_bankroll_usd=100.0)
    runner.tick(ws + (2 * 24 + 1) * 3600, panels=panels, predict_fn=RULE, refresh_data=False)   # BUY (refused)
    runner.tick(ws + (2 * 24 + 23) * 3600, panels=panels, predict_fn=RULE, refresh_data=False)  # SELL
    comp = [r for r in store.read_rows(tmp_path / "agent.jsonl")
            if r["kind"] == "fill" and r.get("compliance")]
    buy = next(r for r in comp if r["reason"] == "COMPLIANCE_BUY")
    sell = next(r for r in comp if r["reason"] == "COMPLIANCE_SELL")
    assert "bnb_qty" not in buy                             # refused BUY captured no qty
    assert amt_ex.calls == []                               # SELL never attempted amount-in
    assert sell.get("exec_status") == "skipped" and sell.get("exec_skipped") == "no_bought_qty"
    assert not any(c["from"] == "BNB" and c["to"] == "USDT" for c in usd_ex.calls)  # never sold the buffer


def test_strategy_fills_scale_to_real_bankroll(tmp_path, panels):
    """Live fills are the env's $10k-denominated usd re-based to the bankroll by a fixed scale
    (bankroll/$10k). $100 bankroll => scale 0.01 => a 10% weight ($1k env) signs ~$10."""
    returns, btc, liq, vol = panels
    ws, _ek = _week_with_fills(returns, btc, liq, vol)
    ex = _FakeExec()
    runner = EventRunner(LiveEventTrader(_prov()), selection=[],
                         agent_ledger_path=tmp_path / "agent.jsonl",
                         mode="live", execute_fn=ex, compliance_frac=0.0, live_bankroll_usd=100.0)
    runner.tick(ws + 167 * 3600, panels=panels, predict_fn=RULE, refresh_data=False)
    allowed = [r for r in store.read_rows(tmp_path / "agent.jsonl")
               if r["kind"] == "fill" and not r.get("compliance") and r["guardrail_ok"]]
    assert allowed and len(ex.calls) == len(allowed)
    for call, f in zip(ex.calls, allowed):                  # one signed call per allowed fill, in order
        assert call["usd"] == pytest.approx(f["usd_in"] * 0.01, rel=1e-9)   # re-based to 1% of the $10k book
        assert call["usd"] < f["usd_in"]                    # scaled DOWN (env usd is $10k-scale)


def test_min_notional_skips_dust_fills(tmp_path, panels):
    """A scaled fill below `min_notional_usd` is skipped on-chain (the env still records the paper
    fill; only the live mirror is skipped) — no executor call, exec_status='skipped'."""
    returns, btc, liq, vol = panels
    ws, _ek = _week_with_fills(returns, btc, liq, vol)
    ex = _FakeExec()
    runner = EventRunner(LiveEventTrader(_prov()), selection=[],
                         agent_ledger_path=tmp_path / "agent.jsonl", mode="live", execute_fn=ex,
                         compliance_frac=0.0, live_bankroll_usd=100.0, min_notional_usd=1e6)
    runner.tick(ws + 167 * 3600, panels=panels, predict_fn=RULE, refresh_data=False)
    assert ex.calls == []                                   # everything below $1e6 -> nothing signed
    allowed = [r for r in store.read_rows(tmp_path / "agent.jsonl")
               if r["kind"] == "fill" and not r.get("compliance") and r["guardrail_ok"]]
    assert allowed and all(r.get("exec_status") == "skipped" for r in allowed)
    # buys are dust-skipped (below_min_notional); their SELLs skip earlier as no_onchain_position
    # (the dust buy never confirmed an on-chain position) — both are valid "nothing signed" skips
    assert all(r.get("exec_skipped") in ("below_min_notional", "no_onchain_position") for r in allowed)
    assert any(r.get("exec_skipped") == "below_min_notional" for r in allowed)   # dust guard still fires


def test_live_forward_policy_scales_caps_to_bankroll():
    pol = live_forward_policy(["UB", "zec"], 100.0)
    assert {"UB", "ZEC", "USDT", "BNB"} <= pol.allowlist    # universe + cash leg + compliance leg
    assert pol.per_trade_usd == pytest.approx(100.0 * 0.34 * 1.25)   # covers the max scaled entry
    assert pol.daily_usd == pytest.approx(400.0) and pol.drawdown_stop_pct == 30.0
    assert pol.chain == "bsc"


def test_live_forward_policy_allowlists_token_contracts():
    pol = live_forward_policy(["UB"], 100.0, asset_ids=["c20000714_t0xABCdef"])
    assert "UB" in pol.allowlist                            # the realized SYMBOL (quote-phase check)
    assert "C20000714_T0XABCDEF" in pol.allowlist           # the CONTRACT assetId (intent-phase check)


def test_sign_live_resolves_token_symbol_to_contract(tmp_path):
    """TWAK can't resolve microcap tickers, so a live token leg is swapped for its BEP-20 contract
    (assetId); BNB/USDT stay as symbols."""
    ex = _FakeExec()
    sel = [{"symbol": "UB", "pair_address": "0xpair", "token_address": "0x40b8129B"}]
    runner = EventRunner(LiveEventTrader(_prov()), selection=sel,
                         agent_ledger_path=tmp_path / "a.jsonl", mode="live", execute_fn=ex,
                         live_policy=_TIGHT)
    runner._sign_live("USDT", "UB", 1.0, 1.0, prescaled=True)
    assert ex.calls[-1]["from"] == "USDT"                   # cash leg: bare symbol (TWAK resolves it)
    assert ex.calls[-1]["to"] == "c20000714_t0x40b8129B"    # token leg: the contract assetId
    runner._sign_live("UB", "USDT", 1.0, 1.0, prescaled=True)   # and the sell direction
    assert ex.calls[-1]["from"] == "c20000714_t0x40b8129B" and ex.calls[-1]["to"] == "USDT"


def test_forward_run_policy_allows_universe_and_cash_leg():
    pol = forward_run_policy(["ADA", "zec"], capital=10_000.0)
    assert "ADA" in pol.allowlist and "ZEC" in pol.allowlist and "USDT" in pol.allowlist
    assert "BNB" in pol.allowlist                          # the daily compliance round-trip leg
    assert pol.drawdown_stop_pct == 30.0 and pol.chain == "bsc"
    assert pol.per_trade_usd == 10_000.0


# --- IDENTITY-DEDUP fill recording (the fix for sbq-s1's dropped week-open ignition) -------------
# The old forward-only cursor dropped a fill the env back-dates to a bar already behind the cursor
# (a lagged ignition, surfaced late and attributed to its origin bar). Dedup records by
# (bar_ts, token, side) against the ledger: a late fill is caught, and a recorded/seeded one is
# never re-signed.

from trader.agent.event_live import fills_from_records  # noqa: E402


class _FakeTrader:
    """evaluate_week with SCRIPTED records per now_ts — exercises the dedup/drop logic deterministically
    (no real env / torch). `records_by_now[now_ts]` is a list of evaluate_event_policy-shaped records."""

    recurrent = False

    def __init__(self, ws, records_by_now):
        self._ws = int(ws)
        self._by_now = records_by_now

    def env_kwargs(self, returns):
        return {}

    def evaluate_week(self, returns, btc, liq, vol, now_ts, env_kwargs, *, predict_fn=None):
        recs = self._by_now[int(now_ts)]
        bars = sorted({int(r["time"]) for r in recs} | {self._ws})
        eq = pd.Series([10_000.0] * len(bars), index=bars)
        toks = sorted({f["token"] for r in recs for f in r["fills"]}) or ["UB"]
        return {"week_start": self._ws, "equity": eq, "records": recs, "universe": toks,
                "win_index": bars, "fills": fills_from_records(recs)}


def _rec(bar, token, usd, reason="IGNITION"):
    f = {"token": token, "usd": float(usd), "fee": 0.0, "time": int(bar), "px": 1.0,
         "reason": reason, "obs": {}}
    return {"time": int(bar), "weights": {}, "trades_usd": {token: float(usd)},
            "trade_fees": {token: 0.0}, "fills": [f]}


_DUMMY_PANELS = (pd.DataFrame(index=[0]), None, None, None)   # FakeTrader ignores panel content


def test_dedup_catches_a_backdated_fill_the_cursor_would_drop(tmp_path):
    """THE regression for the live miss: a fill surfaces LATE on an EARLIER bar (the env's lagged
    ignition). The old forward cursor — advanced past the later bar — dropped it; dedup records it."""
    ws = 1782086400
    Y, X = ws + 5 * 3600, ws + 1 * 3600           # later bar Y seen first; earlier bar X surfaces later
    by_now = {ws + 5 * 3600 + 200: [_rec(Y, "AAA", 100.0)],
              ws + 6 * 3600 + 200: [_rec(Y, "AAA", 100.0), _rec(X, "BBB", 120.0)]}
    runner = EventRunner(_FakeTrader(ws, by_now), selection=[],
                         agent_ledger_path=tmp_path / "a.jsonl", compliance_frac=0.0)
    runner.tick(ws + 5 * 3600 + 200, panels=_DUMMY_PANELS, refresh_data=False)
    runner.tick(ws + 6 * 3600 + 200, panels=_DUMMY_PANELS, refresh_data=False)
    fills = {(r["bar_ts"], r["token"]) for r in store.read_rows(tmp_path / "a.jsonl")
             if r.get("kind") == "fill"}
    assert (Y, "AAA") in fills and (X, "BBB") in fills          # the BACK-DATED fill is caught, not dropped


def test_dedup_idempotent_on_the_backdated_fill(tmp_path):
    """A later tick (with X now 'old') must not re-record either fill — identity dedup is idempotent."""
    ws = 1782086400
    Y, X = ws + 5 * 3600, ws + 1 * 3600
    recs = [_rec(Y, "AAA", 100.0), _rec(X, "BBB", 120.0)]
    by_now = {ws + 6 * 3600 + 200: recs, ws + 7 * 3600 + 200: recs}
    runner = EventRunner(_FakeTrader(ws, by_now), selection=[],
                         agent_ledger_path=tmp_path / "a.jsonl", compliance_frac=0.0)
    runner.tick(ws + 6 * 3600 + 200, panels=_DUMMY_PANELS, refresh_data=False)
    runner.tick(ws + 7 * 3600 + 200, panels=_DUMMY_PANELS, refresh_data=False)
    n = sum(1 for r in store.read_rows(tmp_path / "a.jsonl") if r.get("kind") == "fill")
    assert n == 2                                              # each recorded exactly once


def test_seeded_missed_fill_is_not_resigned_live(tmp_path):
    """The 'don't chase' guard: a fill SEEDED as `missed` in the ledger (the UB the live run dropped)
    is deduped — the executor is NEVER called for it, so the wallet doesn't chase a stale entry."""
    ws = 1782086400
    bar = ws                                                  # the week-open ignition bar (UB)
    store.append({"kind": "fill", "mode": "live", "compliance": False, "from": "USDT", "to": "UB",
                  "token": "UB", "usd_in": 17.8, "usd_out": 17.8, "cost_usd": 0.0, "bar_ts": bar,
                  "reason": "IGNITION", "trigger": "IGNITION", "guardrail_ok": True,
                  "guardrail_codes": [], "exec_status": "missed", "tx_hash": None},
                 tmp_path / "a.jsonl", now=None)
    ex = _FakeExec()
    by_now = {ws + 3 * 3600 + 200: [_rec(bar, "UB", 1820.0)]}   # env still holds the UB entry at the open
    runner = EventRunner(_FakeTrader(ws, by_now), selection=[], agent_ledger_path=tmp_path / "a.jsonl",
                         mode="live", execute_fn=ex, live_policy=_TIGHT, live_bankroll_usd=100.0,
                         min_notional_usd=0.0, compliance_frac=0.0)
    runner.tick(ws + 3 * 3600 + 200, panels=_DUMMY_PANELS, refresh_data=False)
    assert ex.calls == []                                     # UB deduped -> NEVER chased on-chain


_WIDE = Policy(allowlist=frozenset({"USDT", "UB", "AAA", "BNB"}), per_trade_usd=1e6, daily_usd=1e6,
               max_slippage_pct=1.0, drawdown_stop_pct=30.0, lifetime_usd_ceiling=1e9, chain="bsc")


def test_unbacked_sell_is_skipped_not_signed_live(tmp_path):
    """C1/C2 guard: when the env EXITS a token the wallet never bought (a seeded-missed entry — same
    as a guardrail-blocked one), the SELL must NOT be signed — no unbacked on-chain swap of phantom
    funds. It is recorded skipped:no_onchain_position instead."""
    ws = 1782086400
    store.append({"kind": "fill", "mode": "live", "compliance": False, "from": "USDT", "to": "UB",
                  "token": "UB", "usd_in": 17.8, "usd_out": 17.8, "cost_usd": 0.0, "bar_ts": ws,
                  "reason": "IGNITION", "trigger": "IGNITION", "guardrail_ok": True,
                  "guardrail_codes": [], "exec_status": "missed", "tx_hash": None},
                 tmp_path / "a.jsonl", now=None)
    ex = _FakeExec()
    sell_bar = ws + 4 * 3600
    by_now = {ws + 4 * 3600 + 200: [_rec(ws, "UB", 1820.0),
                                    _rec(sell_bar, "UB", -900.0, "TRAILING_STOP")]}
    runner = EventRunner(_FakeTrader(ws, by_now), selection=[], agent_ledger_path=tmp_path / "a.jsonl",
                         mode="live", execute_fn=ex, live_policy=_WIDE, live_bankroll_usd=100.0,
                         min_notional_usd=0.0, compliance_frac=0.0)
    runner.tick(ws + 4 * 3600 + 200, panels=_DUMMY_PANELS, refresh_data=False)
    assert ex.calls == []                                     # BUY deduped (missed), SELL skipped (unbacked)
    sells = [r for r in store.read_rows(tmp_path / "a.jsonl")
             if r.get("kind") == "fill" and r.get("from") == "UB"]
    assert sells and all(r.get("exec_status") == "skipped"
                         and r.get("exec_skipped") == "no_onchain_position" for r in sells)


def test_sell_signs_when_the_buy_actually_landed_live(tmp_path):
    """The guard is not over-broad: a SELL of a token whose BUY confirmed on-chain DOES sign."""
    ws = 1782086400
    ex = _FakeExec()                                          # default result is status=confirmed
    buy_bar, sell_bar = ws + 1 * 3600, ws + 4 * 3600
    by_now = {ws + 1 * 3600 + 200: [_rec(buy_bar, "AAA", 500.0)],
              ws + 4 * 3600 + 200: [_rec(buy_bar, "AAA", 500.0),
                                    _rec(sell_bar, "AAA", -500.0, "TRAILING_STOP")]}
    runner = EventRunner(_FakeTrader(ws, by_now), selection=[], agent_ledger_path=tmp_path / "a.jsonl",
                         mode="live", execute_fn=ex, live_policy=_WIDE, live_bankroll_usd=100.0,
                         min_notional_usd=0.0, compliance_frac=0.0)
    runner.tick(ws + 1 * 3600 + 200, panels=_DUMMY_PANELS, refresh_data=False)   # BUY lands (confirmed)
    runner.tick(ws + 4 * 3600 + 200, panels=_DUMMY_PANELS, refresh_data=False)   # env exits -> SELL signs
    sides = [(c["from"], c["to"]) for c in ex.calls]
    assert ("USDT", "AAA") in sides and ("AAA", "USDT") in sides   # the buy AND the BACKED sell both signed


def test_partial_exits_all_sign_after_a_confirmed_buy_live(tmp_path):
    """A position bought on-chain can be trimmed in MULTIPLE partial sells — EVERY trim signs. The
    guard checks 'has a confirmed buy', not a net count that would wrongly skip later trims."""
    ws = 1782086400
    ex = _FakeExec()                                          # confirmed
    buy = ws + 1 * 3600
    by_now = {ws + 1 * 3600 + 200: [_rec(buy, "AAA", 600.0)],
              ws + 5 * 3600 + 200: [_rec(buy, "AAA", 600.0),
                                    _rec(ws + 3 * 3600, "AAA", -200.0, "TP1"),
                                    _rec(ws + 5 * 3600, "AAA", -200.0, "TP2")]}
    runner = EventRunner(_FakeTrader(ws, by_now), selection=[], agent_ledger_path=tmp_path / "a.jsonl",
                         mode="live", execute_fn=ex, live_policy=_WIDE, live_bankroll_usd=100.0,
                         min_notional_usd=0.0, compliance_frac=0.0)
    runner.tick(ws + 1 * 3600 + 200, panels=_DUMMY_PANELS, refresh_data=False)
    runner.tick(ws + 5 * 3600 + 200, panels=_DUMMY_PANELS, refresh_data=False)
    assert len([c for c in ex.calls if c["from"] == "AAA"]) == 2   # BOTH trims signed, not just the first


def test_latest_token_prices_for_wallet_recon(tmp_path):
    """The wallet-recon price source: USDT=1.0, each token's LATEST close, NaN excluded, BNB from anchor."""
    runner = EventRunner(_FakeTrader(0, {}), selection=[], agent_ledger_path=tmp_path / "a.jsonl",
                         compliance_frac=0.0, bnb_price_fn=lambda ts: 612.0)
    runner._close_panel = pd.DataFrame({"UB": [1.0, 2.5], "ZEC": [10.0, float("nan")]},
                                       index=[1000, 2000])
    px = runner.latest_token_prices(2000)
    assert px["USDT"] == 1.0 and px["UB"] == 2.5 and px["BNB"] == 612.0
    assert "ZEC" not in px            # NaN latest close -> excluded (no fabricated price)
