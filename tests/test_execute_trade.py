"""`execute_trade` end-to-end with a FAKE twak CLI — the Step-4 refusal matrix, live-shaped.

Asserts the two-phase design: an out-of-policy intent never reaches the network; an
in-policy intent is re-judged on the QUOTE's numbers (realized USD, route symbols, implied
slippage) before the (mocked) swap signs; the attempt row hits the ledger before signing;
and every state failure refuses STATE_UNAVAILABLE. The quote texts reuse the captured
fixture shape (human prefix + JSON, no USD field)."""

import json
from pathlib import Path

from trader.execution.execute import execute_trade, extract_tx_hash, parse_tx_status
from trader.risk import TradeIntent, ledger

FIX = Path(__file__).parent / "fixtures"
QUOTE_OK = (FIX / "twak_quote_bnb_usdt.txt").read_text(encoding="utf-8")
TX = "0x" + "ab" * 32


def quote_text(usd=1.0, in_sym="BNB", out_sym="USDT", slip_pct=1.0, impact="0",
               price=605.0) -> str:
    """Synthesize stdout in the captured shape: `$… (@ $price)` line + the JSON object."""
    in_amt, out_amt = usd / price, usd
    body = {"input": f"{in_amt:.18f} {in_sym}", "output": f"{out_amt:.18f} {out_sym}",
            "minReceived": f"{out_amt * (1 - slip_pct / 100):.18f} {out_sym}",
            "provider": "Native", "priceImpact": impact}
    return f"${usd:g} USD ≈ {in_amt:.18f} {in_sym} (@ ${price})\n" + json.dumps(body, indent=2)


class FakeCli:
    """Scriptable twak surface; records calls so tests can assert what never ran."""

    def __init__(self, quote_out=QUOTE_OK, swap_out=None, tx_out=None):
        self.quote_out, self.swap_out = quote_out, swap_out or {"txHash": TX}
        self.tx_out = tx_out or {"status": "success"}
        self.calls = []

    def quote(self, *a, **kw):
        self.calls.append("quote")
        if isinstance(self.quote_out, Exception):
            raise self.quote_out
        return self.quote_out

    def swap(self, *a, **kw):
        self.calls.append("swap")
        if isinstance(self.swap_out, Exception):
            raise self.swap_out
        return self.swap_out

    def tx_status(self, *a, **kw):
        self.calls.append("tx_status")
        return self.tx_out if not isinstance(self.tx_out, list) else self.tx_out.pop(0)


def run(intent, tmp_path, cli=None, **kw):
    cli = cli or FakeCli()
    res = execute_trade(intent, ledger_path=tmp_path / "ledger.jsonl", cli=cli,
                        poll_interval_s=0, sleep=lambda s: None, **kw)
    return res, cli


def intent(**kw) -> TradeIntent:
    base = dict(from_asset="BNB", to_asset="USDT", usd=1.0, chain="bsc", slippage_pct=1.0)
    base.update(kw)
    return TradeIntent(**base)


def test_in_policy_trade_lands_with_tx_hash(tmp_path):
    res, cli = run(intent(), tmp_path)
    assert res["tx_hash"] == TX and res["status"] == "confirmed"
    assert cli.calls == ["quote", "swap", "tx_status"]
    rows = ledger.read_rows(tmp_path / "ledger.jsonl")
    assert [r["kind"] for r in rows] == ["attempt", "result"]
    assert rows[1]["tx_hash"] == TX
    # spend persisted: the realized quote USD now counts against the caps
    state = ledger.state_from_ledger(tmp_path / "ledger.jsonl")
    assert state.spent_lifetime_usd == res["usd"] > 0.99


def test_out_of_policy_intent_refused_before_any_network(tmp_path):
    res, cli = run(intent(usd=5.0), tmp_path)            # the Step-6 negative proof, in vitro
    assert res["refused"] == ["PER_TRADE_CAP"] and res["phase"] == "intent"
    assert cli.calls == []                               # refusal cost zero network calls
    rows = ledger.read_rows(tmp_path / "ledger.jsonl")
    assert [r["kind"] for r in rows] == ["refusal"]      # audit trail, no spend
    assert ledger.state_from_ledger(tmp_path / "ledger.jsonl").spent_lifetime_usd == 0.0


def test_quote_recheck_catches_slippage_the_intent_hid(tmp_path):
    # The intent asks 1% but the quote's minReceived implies 2% — refuse, never adjust.
    res, cli = run(intent(), tmp_path, cli=FakeCli(quote_out=quote_text(slip_pct=2.0)))
    assert res["refused"] == ["SLIPPAGE_BOUND"] and res["phase"] == "quote"
    assert "swap" not in cli.calls


def test_quote_recheck_uses_realized_usd_not_the_wish(tmp_path):
    # Intent says $1.90 (under the cap) but the quote VALUES at $2.50 — the truth wins.
    res, cli = run(intent(usd=1.9), tmp_path, cli=FakeCli(quote_out=quote_text(usd=2.5)))
    assert res["refused"] == ["PER_TRADE_CAP"] and res["phase"] == "quote"
    assert "swap" not in cli.calls


def test_quote_recheck_route_allowlist(tmp_path):
    res, cli = run(intent(), tmp_path, cli=FakeCli(quote_out=quote_text(out_sym="CAKE")))
    assert res["refused"] == ["NOT_ALLOWLISTED"] and res["phase"] == "quote"
    assert "swap" not in cli.calls


def test_quote_failure_fails_closed(tmp_path):
    res, cli = run(intent(), tmp_path, cli=FakeCli(quote_out=RuntimeError("twak swap timed out")))
    assert res["refused"] == ["STATE_UNAVAILABLE"] and res["phase"] == "quote"
    assert "swap" not in cli.calls


def test_unvaluable_quote_fails_closed(tmp_path):
    # No (@ $price) prefix and no stable leg: USD unknowable -> refuse, never guess.
    bare = json.dumps({"input": "0.0016 BNB", "output": "2.4 CAKE",
                       "minReceived": "2.37 CAKE", "priceImpact": "0"})
    res, _ = run(intent(), tmp_path, cli=FakeCli(quote_out=bare))
    assert res["refused"] == ["STATE_UNAVAILABLE"] and res["phase"] == "quote"


def test_unreadable_ledger_refuses_before_quote(tmp_path):
    (tmp_path / "ledger.jsonl").write_text("{corrupt\n", encoding="utf-8")
    res, cli = run(intent(), tmp_path)
    assert res["refused"] == ["STATE_UNAVAILABLE"] and res["phase"] == "intent"
    assert cli.calls == []


def test_daily_cap_binds_from_persisted_attempts(tmp_path):
    p = tmp_path / "ledger.jsonl"
    for _ in range(3):
        ledger.append({"kind": "attempt", "usd": 2.0}, p)            # $6 attempted today
    res, cli = run(intent(), tmp_path)
    assert res["refused"] == ["DAILY_CAP"]
    assert cli.calls == []


def test_swap_error_records_result_and_keeps_spend(tmp_path):
    res, _ = run(intent(), tmp_path, cli=FakeCli(swap_out=RuntimeError("rc=1")))
    assert res["tx_hash"] is None and "swap failed" in res["error"]
    rows = ledger.read_rows(tmp_path / "ledger.jsonl")
    assert [r["kind"] for r in rows] == ["attempt", "result"]
    assert rows[1]["status"] == "swap_error"
    # conservative: the attempt still counts — an unknown outcome may have moved money
    assert ledger.state_from_ledger(tmp_path / "ledger.jsonl").spent_lifetime_usd > 0.99


def test_missing_tx_hash_is_an_error_not_a_pass(tmp_path):
    res, _ = run(intent(), tmp_path, cli=FakeCli(swap_out={"ok": True}))
    assert res["tx_hash"] is None and "no tx hash" in res["error"]
    assert ledger.read_rows(tmp_path / "ledger.jsonl")[1]["status"] == "unknown"


def test_poll_until_confirmed(tmp_path):
    cli = FakeCli(tx_out=[{"status": "pending"}, {"status": "pending"}, {"status": "success"}])
    res, cli = run(intent(), tmp_path, cli=cli)
    assert res["status"] == "confirmed"
    assert cli.calls.count("tx_status") == 3


def test_extract_tx_hash_shapes():
    assert extract_tx_hash({"txHash": TX}) == TX
    assert extract_tx_hash({"result": {"transactionHash": TX}}) == TX
    assert extract_tx_hash({"hash": "not-hex"}) is None
    assert extract_tx_hash({}) is None


def test_parse_tx_status_shapes():
    assert parse_tx_status({"status": "success"}) == "confirmed"
    assert parse_tx_status({"status": "0x1"}) == "confirmed"
    assert parse_tx_status({"status": 1}) == "confirmed"
    assert parse_tx_status({"status": "reverted"}) == "failed"
    assert parse_tx_status({"status": "0x0"}) == "failed"
    assert parse_tx_status({}) == "pending"


def test_parse_tx_status_boolean_shape():
    # the shape twak v0.19.0 actually returned on the live dust trade (2026-06-11)
    assert parse_tx_status({"confirmed": True, "pending": False, "failed": False}) == "confirmed"
    assert parse_tx_status({"confirmed": False, "pending": True, "failed": False}) == "pending"
    assert parse_tx_status({"confirmed": False, "pending": False, "failed": True}) == "failed"
    # failed wins over a contradictory confirmed flag (conservative)
    assert parse_tx_status({"confirmed": True, "failed": True}) == "failed"


# -- dry-run: the safe pre-flight (full check, quote, re-check — but sign/write NOTHING) -----

def test_dry_run_checks_quote_but_signs_and_writes_nothing(tmp_path):
    res, cli = run(intent(), tmp_path, dry_run=True)
    assert res["dry_run"] is True and res["would_execute"] is True and res["status"] == "dry_run"
    assert res["usd"] > 0.99 and res["quote"]["out_symbol"] == "USDT"
    assert cli.calls == ["quote"]                          # quoted, never signed, never polled
    assert ledger.read_rows(tmp_path / "ledger.jsonl") == []   # no attempt/result row written


def test_dry_run_still_refuses_out_of_policy_intent(tmp_path):
    res, cli = run(intent(usd=5.0), tmp_path, dry_run=True)
    assert res["refused"] == ["PER_TRADE_CAP"] and res["phase"] == "intent"
    assert cli.calls == []                                 # refusal short-circuits before the quote


def test_dry_run_refuses_on_quote_implied_slippage(tmp_path):
    # the quote re-check binds even in dry-run: minReceived implies 2% vs the 1% cap -> refuse.
    res, cli = run(intent(), tmp_path, dry_run=True, cli=FakeCli(quote_out=quote_text(slip_pct=2.0)))
    assert res["refused"] == ["SLIPPAGE_BOUND"] and res["phase"] == "quote"
    assert cli.calls == ["quote"] and "swap" not in cli.calls
