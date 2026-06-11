"""Risk-ledger persistence: spend derivation, the high-water mark, and fail-closed reads."""

from datetime import datetime, timezone

from trader.risk import ledger

NOW = datetime(2026, 6, 11, 12, 0, tzinfo=timezone.utc)
YESTERDAY = datetime(2026, 6, 10, 12, 0, tzinfo=timezone.utc)


def path(tmp_path):
    return tmp_path / "risk_ledger.jsonl"


def test_append_creates_dirs_and_round_trips(tmp_path):
    p = tmp_path / "nested" / "risk_ledger.jsonl"
    ledger.append({"kind": "attempt", "usd": 1.5}, p, now=NOW)
    rows = ledger.read_rows(p)
    assert rows == [{"kind": "attempt", "usd": 1.5, "ts": NOW.isoformat(timespec="seconds")}]


def test_missing_file_is_a_fresh_ledger(tmp_path):
    state = ledger.state_from_ledger(path(tmp_path), now=NOW)
    assert state.available and state.spent_lifetime_usd == 0.0
    assert state.equity_usd is None and state.high_water_usd is None


def test_daily_resets_lifetime_does_not(tmp_path):
    p = path(tmp_path)
    ledger.append({"kind": "attempt", "usd": 2.0}, p, now=YESTERDAY)
    ledger.append({"kind": "attempt", "usd": 1.0}, p, now=NOW)
    state = ledger.state_from_ledger(p, now=NOW)
    assert state.spent_today_usd == 1.0
    assert state.spent_lifetime_usd == 3.0


def test_results_add_gas_refusals_count_nothing(tmp_path):
    p = path(tmp_path)
    ledger.append({"kind": "attempt", "usd": 1.0}, p, now=NOW)
    ledger.append({"kind": "result", "tx_hash": "0xabc", "status": "confirmed",
                   "usd": 1.0, "gas_usd": 0.25}, p, now=NOW)         # gas adds; notional doesn't double
    ledger.append({"kind": "refusal", "phase": "intent", "usd": 99.0,
                   "refusals": [{"code": "PER_TRADE_CAP"}]}, p, now=NOW)
    state = ledger.state_from_ledger(p, now=NOW)
    assert state.spent_today_usd == 1.0
    assert state.spent_lifetime_usd == 1.25


def test_high_water_and_latest_equity(tmp_path):
    p = path(tmp_path)
    for eq in (8.0, 10.0, 7.0):
        ledger.append_equity(eq, p, now=NOW)
    state = ledger.state_from_ledger(p, now=NOW)
    assert state.high_water_usd == 10.0 and state.equity_usd == 7.0


def test_missing_ts_counts_today_conservative(tmp_path):
    p = path(tmp_path)
    with open(p, "w", encoding="utf-8") as f:
        f.write('{"kind": "attempt", "usd": 2.0}\n')                 # no ts at all
    assert ledger.state_from_ledger(p, now=NOW).spent_today_usd == 2.0


def test_malformed_ledger_fails_closed(tmp_path):
    p = path(tmp_path)
    ledger.append({"kind": "attempt", "usd": 1.0}, p, now=NOW)
    with open(p, "a", encoding="utf-8") as f:
        f.write("{not json\n")
    state = ledger.state_from_ledger(p, now=NOW)
    assert not state.available and "ledger unreadable" in state.detail


def test_non_object_row_fails_closed(tmp_path):
    p = path(tmp_path)
    with open(p, "w", encoding="utf-8") as f:
        f.write("[1, 2]\n")
    assert not ledger.state_from_ledger(p, now=NOW).available
