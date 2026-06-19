"""The trading/ telemetry publisher — projection shapes, the percent->fraction boundary,
the local put path, and the loop's fail-safe hook (a broken publisher never stops a tick).

No network: `publish_trading` is exercised against a local directory target (the same
`put_bytes` code path s3:// takes, minus boto3), and the loop integration uses FakeFeed.
"""

import datetime as _dt
import json
from pathlib import Path

import pytest

from trader.agent import store
from trader.agent.decide import HoldCore
from trader.agent.feed import FakeFeed
from trader.agent.loop import Loop, LoopConfig
from trader.agent.publish import build_publisher, project, publish_trading

# the fill's TRADE bar (exact UTC hour) — distinct from its write `ts`; on the same UTC day as
# the marks below so it counts toward trades_today
_BAR_TS = int(_dt.datetime(2026, 6, 12, 1, 0, 0, tzinfo=_dt.timezone.utc).timestamp())

ROWS = [
    {"kind": "fill", "mode": "paper", "from": "CASH", "to": "BNB", "usd_in": 1.0,
     "usd_out": 0.99, "cost_usd": 0.01, "units_from": 1.0, "units_to": 0.00165, "bar_ts": _BAR_TS,
     "price_from": 1.0, "price_to": 600.0, "reason": "rebal", "ts": "2026-06-12T01:00:00+00:00"},
    {"kind": "refusal", "mode": "paper", "intent": {"from": "BNB", "to": "USDT", "usd": 5.0},
     "refusals": ["PER_TRADE_CAP"], "ts": "2026-06-12T01:00:01+00:00"},
    {"kind": "equity", "mode": "paper", "tick": 0, "equity_usd": 10.0, "peak_usd": 10.0,
     "drawdown_pct": 0.0, "below_dust": False, "ts": "2026-06-12T01:00:02+00:00"},
    {"kind": "heartbeat", "mode": "paper", "tick": 0, "equity_usd": 10.0,
     "ts": "2026-06-12T01:00:02+00:00"},
    {"kind": "equity", "mode": "paper", "tick": 1, "equity_usd": 9.5, "peak_usd": 10.0,
     "drawdown_pct": 5.0, "below_dust": False, "ts": "2026-06-12T02:00:02+00:00"},
    {"kind": "heartbeat", "mode": "paper", "tick": 1, "equity_usd": 9.5,
     "ts": "2026-06-12T02:00:02+00:00"},
]


# --- projection ---------------------------------------------------------------

def test_project_shapes_and_generated():
    files = project(ROWS)
    assert set(files) == {"heartbeat.json", "status.json", "equity.json", "trades.json"}
    hb = files["heartbeat.json"]
    # generated = newest mark ts from the ROWS, never the wall clock (stale publishes as stale)
    assert hb["generated"] == "2026-06-12T02:00:02+00:00"
    assert hb["mode"] == "paper" and hb["tick"] == 1 and hb["equity_usd"] == 9.5


def test_project_normalizes_drawdown_percent_to_fraction():
    files = project(ROWS)
    assert files["status.json"]["drawdown"] == pytest.approx(0.05)  # ledger 5.0% -> 0.05
    series = files["equity.json"]["series"]
    assert [p["drawdown"] for p in series] == [pytest.approx(0.0), pytest.approx(0.05)]


def test_project_counts_trades_against_daily_floor():
    files = project(ROWS)
    st = files["status.json"]
    assert st["trades_today"] == 1 and st["daily_floor_ok"] is True
    assert st["n_fills"] == 1 and st["n_refusals"] == 1
    # count by TRADE day (bar_ts), NOT the write `ts`: a fill whose bar is an earlier UTC day must
    # not count toward today, even if it was written (ts) today — the post-restart re-record case
    older_bar = int(_dt.datetime(2026, 6, 11, 23, 0, 0, tzinfo=_dt.timezone.utc).timestamp())
    old = [dict(ROWS[0], bar_ts=older_bar), *ROWS[1:]]   # same write ts (today), older trade bar
    assert project(old)["status.json"]["trades_today"] == 0
    assert project(old)["status.json"]["daily_floor_ok"] is False


def test_fill_time_is_the_trade_bar_in_utc():
    """Published fills carry the TRADE time (their bar, exact-hour UTC) — `ts` overwritten to it,
    plus `time` (unix sec) + `time_utc`; the write time is kept as `recorded_ts`."""
    f = project(ROWS)["trades.json"]["fills"][0]
    assert f["time"] == _BAR_TS
    assert f["time_utc"] == "2026-06-12T01:00:00Z"
    assert f["ts"] == "2026-06-12T01:00:00Z"            # consumers reading `ts` see the trade time
    assert f["recorded_ts"] == "2026-06-12T01:00:00+00:00"  # original write time preserved
    assert f["time_utc"].endswith(":00:00Z")            # exact hour, UTC


def test_project_empty_ledger_publishes_nothing():
    assert project([]) == {}
    assert project([ROWS[0]]) == {}  # a fill alone has no mark to stamp `generated` from


# --- the put path (local target = same code path as s3, minus boto3) -----------

def test_publish_trading_writes_valid_json(tmp_path):
    written = publish_trading(ROWS, str(tmp_path))
    assert len(written) == 4
    hb = json.loads((tmp_path / "heartbeat.json").read_text(encoding="utf-8"))
    assert hb["generated"] == "2026-06-12T02:00:02+00:00"
    trades = json.loads((tmp_path / "trades.json").read_text(encoding="utf-8"))
    assert trades["fills"][0]["to"] == "BNB"
    assert trades["refusals"][0]["refusals"] == ["PER_TRADE_CAP"]


# --- loop integration -----------------------------------------------------------

def _cfg(tmp_path: Path, **kw) -> LoopConfig:
    return LoopConfig(universe=["BNB", "USDT"], mode="paper", tick_seconds=0,
                      max_ticks=kw.pop("max_ticks", 1),
                      agent_ledger_path=tmp_path / "agent.jsonl",
                      risk_ledger_path=tmp_path / "risk.jsonl", **kw)


def test_loop_publishes_after_each_tick(tmp_path):
    cfg = _cfg(tmp_path, max_ticks=2)
    target = tmp_path / "trading"
    publisher = build_publisher(cfg.agent_ledger_path, str(target))
    loop = Loop(cfg, FakeFeed([{"BNB": 600.0, "USDT": 1.0}]), HoldCore(),
                sleep=lambda s: None, publisher=publisher)
    assert loop.run() == 2
    hb = json.loads((target / "heartbeat.json").read_text(encoding="utf-8"))
    assert hb["tick"] == 1  # the publish after the second tick wins
    rows = store.read_rows(cfg.agent_ledger_path)
    assert hb["generated"] == max(r["ts"] for r in rows
                                  if r["kind"] in ("heartbeat", "equity"))


def test_broken_publisher_never_stops_the_loop(tmp_path, capsys):
    cfg = _cfg(tmp_path, max_ticks=2)

    def explode():
        raise RuntimeError("s3 is down")

    loop = Loop(cfg, FakeFeed([{"BNB": 600.0, "USDT": 1.0}]), HoldCore(),
                sleep=lambda s: None, publisher=explode)
    assert loop.run() == 2  # both ticks complete despite the publisher failing
    rows = store.read_rows(cfg.agent_ledger_path)
    assert sum(1 for r in rows if r["kind"] == "heartbeat") == 2
    assert "publish warning" in capsys.readouterr().err
