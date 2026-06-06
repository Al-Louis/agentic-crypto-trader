"""Tests for the GoPlus forensic scoring (pure; no network)."""

from trader.data import goplus as gp

# A clean, established token (open source, no dangerous powers, locked LP).
CLEAN = {
    "is_open_source": "1", "is_honeypot": "0", "is_mintable": "0",
    "transfer_pausable": "0", "is_blacklisted": "0", "is_proxy": "0",
    "buy_tax": "0", "sell_tax": "0", "holder_count": "235937",
    "owner_percent": "0", "lp_holder_count": "3",
    "lp_holders": [{"is_locked": "1", "percent": "0.9"}],
}


def test_parse_coerces_strings():
    sec = gp.parse_security(CLEAN)
    assert sec["available"] and sec["is_open_source"] is True
    assert sec["is_honeypot"] is False
    assert sec["sell_tax"] == 0.0
    assert sec["holder_count"] == 235937
    assert sec["lp_locked_pct"] == 0.9


def test_parse_missing_is_unavailable():
    assert gp.parse_security({}) == {"available": False}
    assert gp.verdict(gp.parse_security({}))["verdict"] == "unknown"


def test_clean_token_is_ok():
    v = gp.verdict(gp.parse_security(CLEAN))
    assert v["verdict"] == "ok"
    assert v["score"] == 0


def test_honeypot_is_blocked():
    raw = dict(CLEAN, is_honeypot="1")
    v = gp.verdict(gp.parse_security(raw))
    assert v["verdict"] == "block"
    assert "honeypot (cannot sell)" in v["flags"]


def test_extreme_tax_is_blocked():
    v = gp.verdict(gp.parse_security(dict(CLEAN, sell_tax="0.9")))
    assert v["verdict"] == "block"


def test_can_take_back_ownership_is_blocked():
    v = gp.verdict(gp.parse_security(dict(CLEAN, can_take_back_ownership="1")))
    assert v["verdict"] == "block"


def test_dangerous_powers_accumulate_to_block():
    # mintable + pausable + blacklist = 9 >= BLOCK_SCORE
    raw = dict(CLEAN, is_mintable="1", transfer_pausable="1", is_blacklisted="1")
    v = gp.verdict(gp.parse_security(raw))
    assert v["verdict"] == "block"
    assert v["score"] >= gp.BLOCK_SCORE


def test_single_soft_flag_is_warn():
    v = gp.verdict(gp.parse_security(dict(CLEAN, is_proxy="1")))
    assert v["verdict"] == "warn"
    assert "upgradeable proxy" in v["flags"]


def test_unlocked_lp_and_owner_concentration_flag():
    # low holder count: unlocked LP is a real rug signal here
    raw = dict(CLEAN, holder_count="500", owner_percent="0.6",
               lp_holders=[{"is_locked": "0", "percent": "0.8"}])
    flags = gp.risk_flags(gp.parse_security(raw))
    assert any("owner holds" in f for f in flags)
    assert any("LP only" in f for f in flags)


def test_unlocked_lp_ignored_for_established_token():
    # high holder count (Binance-Peg style): unlocked LP is normal, not scored
    raw = dict(CLEAN, holder_count="500000",
               lp_holders=[{"is_locked": "0", "percent": "0.8"}])
    sec = gp.parse_security(raw)
    assert not any("LP only" in f for f in gp.risk_flags(sec))


def test_moderate_sell_tax_warns():
    v = gp.verdict(gp.parse_security(dict(CLEAN, sell_tax="0.06")))
    assert v["verdict"] == "warn"
    assert any("sell tax" in f for f in v["flags"])
