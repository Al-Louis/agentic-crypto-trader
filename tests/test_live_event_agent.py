"""The LIVE launcher's safety gates + parser — the part that must fail closed. The full live run
needs the model + torch + network, but the triple-gate (which arms real-money signing) is pure and
unit-tested here."""
import pytest

from trader.agent import live_event_agent as lea


def _env(monkeypatch, **kv):
    monkeypatch.setattr(lea.config, "get", lambda k, *a: kv.get(k))


def test_gates_refuse_until_all_three_set(monkeypatch):
    _env(monkeypatch)                                       # nothing set
    with pytest.raises(SystemExit):
        lea._require_live_gates(dry_run=False)
    _env(monkeypatch, TRADER_MODE="live")                  # missing AGENT_ALLOW_LIVE
    with pytest.raises(SystemExit):
        lea._require_live_gates(dry_run=False)
    _env(monkeypatch, TRADER_MODE="live", AGENT_ALLOW_LIVE="1")   # missing the CONFIRM gate
    with pytest.raises(SystemExit):
        lea._require_live_gates(dry_run=False)
    _env(monkeypatch, TRADER_MODE="live", AGENT_ALLOW_LIVE="1", AGENT_LIVE_CONFIRM="1")
    lea._require_live_gates(dry_run=False)                  # all three -> arms (no raise)


def test_dry_run_skips_the_confirm_gate(monkeypatch):
    # --dry-run never signs, so it needs only the first two gates, NOT AGENT_LIVE_CONFIRM.
    _env(monkeypatch, TRADER_MODE="live", AGENT_ALLOW_LIVE="1")
    lea._require_live_gates(dry_run=True)                   # ok
    _env(monkeypatch, AGENT_ALLOW_LIVE="1")                 # but still needs TRADER_MODE=live
    with pytest.raises(SystemExit):
        lea._require_live_gates(dry_run=True)


def test_parser_defaults():
    args = lea.build_parser().parse_args(["--run-dir", "runs-rl/x"])
    assert args.bankroll_usd is None                        # default: read from the wallet at startup
    assert args.min_notional_usd == 0.50 and args.capital == 10_000.0 and not args.dry_run
