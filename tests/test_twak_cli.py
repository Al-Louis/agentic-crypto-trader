"""twak CLI wrapper — quote parsing (against CAPTURED fixtures) + the custody guards.

The fixtures are real `twak swap --quote-only --json` stdout captured 2026-06-11 (unfunded
wallet; quotes are read-only). Key quirks they pin: a human line precedes the JSON even
with --json, the JSON carries NO USD field, and `1 - minReceived/output` equals the
requested slippage tolerance exactly (1% default fixture, 0.5% fixture). No network here.
"""

from pathlib import Path

import pytest

from trader.execution import twak_cli as tc

FIX = Path(__file__).parent / "fixtures"
QUOTE_BNB_USDT = (FIX / "twak_quote_bnb_usdt.txt").read_text(encoding="utf-8")
QUOTE_USDT_BNB = (FIX / "twak_quote_usdt_bnb.txt").read_text(encoding="utf-8")


def test_parse_quote_bnb_to_usdt():
    q = tc.parse_quote(QUOTE_BNB_USDT)
    assert (q["in_symbol"], q["out_symbol"]) == ("BNB", "USDT")
    assert q["implied_slippage_pct"] == 1.0          # the default --slippage tolerance
    assert q["price_impact_pct"] == 0.0
    assert q["provider"] == "Native"
    # USD valuation: max(input @ $605.13 ≈ $1.00, stable out leg $0.993) — conservative.
    assert q["usd_value"] == pytest.approx(1.0, abs=0.01)


def test_parse_quote_usdt_to_bnb_half_pct():
    q = tc.parse_quote(QUOTE_USDT_BNB)
    assert (q["in_symbol"], q["out_symbol"]) == ("USDT", "BNB")
    assert q["implied_slippage_pct"] == 0.5          # matches the requested --slippage 0.5
    assert q["usd_value"] == pytest.approx(1.0, abs=0.01)


def test_parse_quote_missing_fields_raises():
    for drop in ("input", "output", "minReceived", "priceImpact"):
        broken = QUOTE_BNB_USDT.replace(f'"{drop}"', f'"{drop}_gone"')
        with pytest.raises(tc.QuoteParseError):
            tc.parse_quote(broken)


def test_parse_quote_unparseable_leg_raises():
    with pytest.raises(tc.QuoteParseError):
        tc.parse_quote('{"input": "???", "output": "1 USDT", '
                       '"minReceived": "0.99 USDT", "priceImpact": "0"}')


def test_no_stable_leg_and_no_price_line_means_unvaluable():
    # e.g. a BNB->CAKE quote with the human prefix stripped: cannot be valued -> None
    # (execute_trade then refuses STATE_UNAVAILABLE — fail closed, never guess).
    q = tc.parse_quote('{"input": "0.0016 BNB", "output": "2.4 CAKE", '
                       '"minReceived": "2.37 CAKE", "priceImpact": "0.1"}')
    assert q["usd_value"] is None
    assert q["price_impact_pct"] == 0.1


def test_extract_json_needs_an_object():
    with pytest.raises(tc.TwakError):
        tc.extract_json("Error: no credentials\n")
    assert tc.extract_json('noise {"a": 1} trailing')["a"] == 1


def test_arg_builders_never_carry_password():
    qa = tc.quote_args("BNB", "USDT", 1.0, chain="bsc", slippage_pct=1.0)
    sa = tc.swap_args("BNB", "USDT", 1.0, chain="bsc", slippage_pct=1.0)
    assert "--quote-only" in qa and "--quote-only" not in sa
    for args in (qa, sa):
        assert not any(a.startswith("--password") for a in args)
        assert "--json" in args and args[:2] == ["swap", "--usd"]


def test_check_args_refuses_password_and_hostile_args():
    with pytest.raises(tc.TwakError, match="password"):
        tc._check_args(["swap", "--password", "x"])
    with pytest.raises(tc.TwakError, match="password"):
        tc._check_args(["swap", "--password=x"])
    for evil in ("USDT && del /q *", "USDT;rm", "U$DT", 'a"b'):
        with pytest.raises(tc.TwakError, match="unsafe"):
            tc._check_args(["swap", evil])
    assert tc._check_args(["tx", "0xabc123", "--chain", "bsc", "--json"])
