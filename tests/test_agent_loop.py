"""The autonomous loop in paper mode — deterministic tick, paper-fill accounting, crash
recovery, and the live-mode-requires-flag refusal.

No network: a `FakeFeed` scripts prices and a stub `DecisionCore` scripts intents, so every
assertion is exact. Covers the four required cases (task spec):
  * loop tick with fake feed + stub core (deterministic equity/heartbeat/PnL rows)
  * paper-fill accounting (units in/out, AMM cost charged, caps debited)
  * restart-state recovery (a fresh Loop re-derives positions + tick pointer from disk)
  * live-mode-requires-flag refusal (live unreachable without the explicit opt-in)
"""

from pathlib import Path

import pytest

from trader.agent import store
from trader.agent.decide import HoldCore, Intent, Observation
from trader.agent.feed import FakeFeed
from trader.agent.loop import DUST_EQUITY_USD, Loop, LoopConfig
from trader.agent.paper import DEFAULT_LIQUIDITY_USD, fill
from trader.sim.broker import amm_cost_usd


def _cfg(tmp_path: Path, **kw) -> LoopConfig:
    return LoopConfig(universe=["BNB", "USDT"], mode=kw.pop("mode", "paper"),
                      tick_seconds=0, max_ticks=kw.pop("max_ticks", 1),
                      agent_ledger_path=tmp_path / "agent.jsonl",
                      risk_ledger_path=tmp_path / "risk.jsonl", **kw)


class ScriptedCore:
    """Fires a fixed intent list on the first tick, then holds."""

    name = "scripted"

    def __init__(self, intents):
        self._intents = intents
        self._fired = False

    def decide(self, obs: Observation):
        if self._fired:
            return []
        self._fired = True
        return list(self._intents)


def _seed_position(path: Path, symbol: str, units: float, price: float):
    store.append({"kind": "fill", "mode": "paper", "from": "CASH", "to": symbol,
                  "usd_in": 0, "usd_out": 0, "units_from": 0.0, "units_to": units,
                  "price_from": 0, "price_to": price}, path)


# --- deterministic tick ----------------------------------------------------

def test_hold_tick_marks_equity_and_heartbeat(tmp_path):
    cfg = _cfg(tmp_path, max_ticks=1)
    _seed_position(cfg.agent_ledger_path, "BNB", 0.01, 600.0)
    loop = Loop(cfg, FakeFeed([{"BNB": 600.0, "USDT": 1.0}]), HoldCore(), sleep=lambda s: None)
    n = loop.run()
    assert n == 1
    rows = store.read_rows(cfg.agent_ledger_path)
    kinds = [r["kind"] for r in rows]
    assert "equity" in kinds and "heartbeat" in kinds
    eq = next(r for r in rows if r["kind"] == "equity")
    assert eq["equity_usd"] == pytest.approx(6.0)  # 0.01 BNB * $600
    assert eq["drawdown_pct"] == pytest.approx(0.0)
    assert any(r["kind"] == "fill" for r in rows[:1])  # the seed fill is first; no new fills


def test_hold_core_never_trades(tmp_path):
    cfg = _cfg(tmp_path, max_ticks=3)
    _seed_position(cfg.agent_ledger_path, "BNB", 0.01, 600.0)
    loop = Loop(cfg, FakeFeed([{"BNB": 600.0, "USDT": 1.0}]), HoldCore(), sleep=lambda s: None)
    loop.run()
    rows = store.read_rows(cfg.agent_ledger_path)
    # one seed fill only; HoldCore adds no fills, ever
    assert sum(1 for r in rows if r["kind"] == "fill") == 1


# --- paper-fill accounting --------------------------------------------------

def test_paper_fill_accounting_matches_cost_model():
    prices = {"BNB": 600.0, "USDT": 1.0}
    f = fill("BNB", "USDT", 1.0, prices)
    expect_cost = amm_cost_usd(1.0, DEFAULT_LIQUIDITY_USD)
    assert f.cost_usd == pytest.approx(expect_cost)
    assert f.usd_out == pytest.approx(1.0 - expect_cost)
    assert f.units_from == pytest.approx(1.0 / 600.0)        # $1 of BNB removed
    assert f.units_to == pytest.approx((1.0 - expect_cost))  # USDT @ $1


def test_paper_fill_needs_live_prices_both_legs():
    with pytest.raises(ValueError):
        fill("BNB", "USDT", 1.0, {"USDT": 1.0})  # BNB price missing -> refuse, no fabrication


def test_loop_paper_trade_updates_book_and_debits_caps(tmp_path):
    cfg = _cfg(tmp_path, max_ticks=1)
    _seed_position(cfg.agent_ledger_path, "BNB", 0.01, 600.0)
    core = ScriptedCore([Intent("BNB", "USDT", 1.0, reason="rebal")])
    loop = Loop(cfg, FakeFeed([{"BNB": 600.0, "USDT": 1.0}]), core, sleep=lambda s: None)
    loop.run()
    st = store.derive_state(cfg.agent_ledger_path)
    assert st.units("BNB") == pytest.approx(0.01 - 1.0 / 600.0)
    assert st.units("USDT") > 0
    # the paper spend counts against the SAME risk-ledger caps the live run uses
    from trader.risk import ledger as risk_ledger
    rstate = risk_ledger.state_from_ledger(cfg.risk_ledger_path)
    assert rstate.spent_today_usd == pytest.approx(1.0)


# --- guardrail refusal ------------------------------------------------------

def test_out_of_policy_intent_refused_not_obeyed(tmp_path):
    cfg = _cfg(tmp_path, max_ticks=1)
    _seed_position(cfg.agent_ledger_path, "BNB", 1.0, 600.0)
    # $5 exceeds the $2 per-trade cap -> must be refused, never filled
    core = ScriptedCore([Intent("BNB", "USDT", 5.0, reason="too-big")])
    loop = Loop(cfg, FakeFeed([{"BNB": 600.0, "USDT": 1.0}]), core, sleep=lambda s: None)
    loop.run()
    rows = store.read_rows(cfg.agent_ledger_path)
    refusal = next(r for r in rows if r["kind"] == "refusal")
    assert "PER_TRADE_CAP" in refusal["refusals"]
    assert not any(r["kind"] == "fill" and r.get("from") == "BNB" for r in rows
                   if r["mode"] == "paper")


# --- crash recovery ---------------------------------------------------------

def test_restart_rederives_state_from_disk(tmp_path):
    cfg = _cfg(tmp_path, max_ticks=2)
    _seed_position(cfg.agent_ledger_path, "BNB", 0.02, 600.0)
    core = ScriptedCore([Intent("BNB", "USDT", 1.0)])
    loop = Loop(cfg, FakeFeed([{"BNB": 600.0, "USDT": 1.0}]), core, sleep=lambda s: None)
    loop.run()
    snap = store.derive_state(cfg.agent_ledger_path)

    # a brand-new Loop (simulating a process restart) must see identical state
    loop2 = Loop(cfg, FakeFeed([{"BNB": 600.0, "USDT": 1.0}]), HoldCore(), sleep=lambda s: None)
    assert loop2.state.positions == pytest.approx(snap.positions)
    assert loop2.state.tick == snap.tick
    assert loop2.state.tick >= 2  # the tick pointer advanced and persisted


def test_malformed_ledger_refuses_to_start(tmp_path):
    p = tmp_path / "agent.jsonl"
    p.write_text("{not json}\n", encoding="utf-8")
    cfg = _cfg(tmp_path, max_ticks=1)
    with pytest.raises(store.StoreError):
        Loop(cfg, FakeFeed([{"BNB": 600.0}]), HoldCore())


# --- clean shutdown ----------------------------------------------------------

def test_request_stop_wakes_the_inter_tick_wait(tmp_path):
    """A stop signal during the (hour-long) inter-tick wait must return promptly — the
    systemd `stop` path. time.sleep resumes after a signal handler (PEP 475); the loop's
    default Event-based wait must not."""
    import threading
    import time as _time

    cfg = _cfg(tmp_path, max_ticks=None)
    cfg = LoopConfig(universe=cfg.universe, mode="paper", tick_seconds=3600.0,
                     max_ticks=None, agent_ledger_path=cfg.agent_ledger_path,
                     risk_ledger_path=cfg.risk_ledger_path)
    loop = Loop(cfg, FakeFeed([{"BNB": 600.0, "USDT": 1.0}]), HoldCore())  # default wait
    t = threading.Thread(target=loop.run)
    t.start()
    _time.sleep(0.2)          # let the first tick finish and the loop enter its wait
    loop.request_stop()
    t.join(timeout=5.0)       # must wake immediately, not after 3600s
    assert not t.is_alive()


# --- mode gating ------------------------------------------------------------

def test_live_requires_explicit_mode_string():
    paper = LoopConfig(universe=["BNB"], mode="paper")
    typo = LoopConfig(universe=["BNB"], mode="LIVE")   # wrong case -> NOT live
    live = LoopConfig(universe=["BNB"], mode="live")
    assert not paper.is_live and not typo.is_live and live.is_live


def test_live_intent_routes_only_through_execute_trade(tmp_path):
    cfg = _cfg(tmp_path, mode="live", max_ticks=1)
    _seed_position(cfg.agent_ledger_path, "BNB", 1.0, 600.0)
    calls = []

    def fake_execute(intent, policy, *, ledger_path):
        calls.append(intent)
        return {"tx_hash": "0x" + "ab" * 32, "status": "confirmed", "usd": 1.0}

    core = ScriptedCore([Intent("BNB", "USDT", 1.0)])
    loop = Loop(cfg, FakeFeed([{"BNB": 600.0, "USDT": 1.0}]), core,
                execute_fn=fake_execute, sleep=lambda s: None)
    loop.run()
    assert len(calls) == 1  # the live intent went through execute_trade, nothing else
    rows = store.read_rows(cfg.agent_ledger_path)
    live_fill = next(r for r in rows if r["kind"] == "fill" and r["mode"] == "live")
    assert live_fill["tx_hash"].startswith("0x")


def test_main_refuses_live_without_env(monkeypatch):
    import trader.agent.__main__ as m
    monkeypatch.setattr(m.config, "get", lambda name, default=None: None)  # env unset
    rc = m.main(["--mode", "live", "--ticks", "1", "--interval", "0"])
    assert rc == 2  # refused: live needs AGENT_ALLOW_LIVE=1


def test_main_mode_from_env_still_needs_live_opt_in(monkeypatch):
    # the systemd contract: no --mode argv, TRADER_MODE comes from the env-file —
    # and live via env hits the same AGENT_ALLOW_LIVE gate as live via flag
    import trader.agent.__main__ as m
    env = {m.MODE_ENV: "live"}
    monkeypatch.setattr(m.config, "get", lambda name, default=None: env.get(name, default))
    rc = m.main(["--ticks", "1", "--interval", "0"])
    assert rc == 2


def test_main_refuses_garbage_mode_env(monkeypatch):
    # a typo'd TRADER_MODE must refuse loudly, never silently fall back to paper
    import trader.agent.__main__ as m
    env = {m.MODE_ENV: "lvie"}
    monkeypatch.setattr(m.config, "get", lambda name, default=None: env.get(name, default))
    rc = m.main(["--ticks", "1", "--interval", "0"])
    assert rc == 2


def test_main_flag_overrides_env_mode(monkeypatch):
    # explicit --mode live overrides TRADER_MODE=paper and is gated as live
    import trader.agent.__main__ as m
    env = {m.MODE_ENV: "paper"}
    monkeypatch.setattr(m.config, "get", lambda name, default=None: env.get(name, default))
    rc = m.main(["--mode", "live", "--ticks", "1", "--interval", "0"])
    assert rc == 2  # gated as live, proving the flag won over the env's paper


# --- scoring-mirror dust mark ----------------------------------------------

def test_below_dust_equity_flagged(tmp_path):
    cfg = _cfg(tmp_path, max_ticks=1)
    _seed_position(cfg.agent_ledger_path, "BNB", 0.001, 600.0)  # $0.60 < $1 dust line
    loop = Loop(cfg, FakeFeed([{"BNB": 600.0, "USDT": 1.0}]), HoldCore(), sleep=lambda s: None)
    summary = loop.tick()
    assert summary["below_dust"] is True
    assert summary["equity_usd"] < DUST_EQUITY_USD
