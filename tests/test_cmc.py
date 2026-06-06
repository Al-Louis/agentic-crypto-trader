"""Tests for CMC contract-resolution helpers (pure; no API key / network)."""

from trader.data import cmc
from trader.data import dexscreener as ds


# --- pick_canonical -------------------------------------------------------

def test_pick_canonical_prefers_active_then_rank():
    cands = [
        {"id": 9, "symbol": "ETH", "rank": 5000, "is_active": 1},
        {"id": 1027, "symbol": "ETH", "rank": 2, "is_active": 1},
        {"id": 3, "symbol": "ETH", "rank": 1, "is_active": 0},   # better rank but inactive
    ]
    assert cmc.pick_canonical(cands)["id"] == 1027


def test_pick_canonical_falls_back_to_inactive_if_none_active():
    cands = [{"id": 3, "symbol": "X", "rank": 10, "is_active": 0}]
    assert cmc.pick_canonical(cands)["id"] == 3


def test_pick_canonical_empty():
    assert cmc.pick_canonical([]) is None


# --- bsc_contract_from_info ----------------------------------------------

def _info(contract_list, platform=None):
    return {"contract_address": contract_list, "platform": platform}


def test_bsc_contract_from_contract_list_by_name():
    info = _info([
        {"contract_address": "0xETHMAINNET", "platform": {"name": "Ethereum"}},
        {"contract_address": "0xBSCADDR", "platform": {"name": "BNB Smart Chain",
                                                        "coin": {"symbol": "BNB"}}},
    ])
    assert cmc.bsc_contract_from_info(info) == "0xBSCADDR"


def test_bsc_contract_matches_by_coin_symbol():
    info = _info([{"contract_address": "0xABC",
                   "platform": {"name": "Some Chain", "coin": {"symbol": "BNB"}}}])
    assert cmc.bsc_contract_from_info(info) == "0xABC"


def test_bsc_contract_none_when_no_bsc_deployment():
    info = _info([{"contract_address": "0xETH", "platform": {"name": "Ethereum"}}])
    assert cmc.bsc_contract_from_info(info) is None


def test_bsc_contract_fallback_to_primary_platform():
    info = _info([], platform={"name": "BNB Smart Chain", "token_address": "0xPRIMARY"})
    assert cmc.bsc_contract_from_info(info) == "0xPRIMARY"


# --- resolve_bsc_contracts (fetch mocked) --------------------------------

def test_resolve_bsc_contracts_end_to_end(monkeypatch):
    fake_map = {
        "ETH": [{"id": 1027, "symbol": "ETH", "name": "Ethereum", "rank": 2, "is_active": 1}],
        "FOO": [
            {"id": 5, "symbol": "FOO", "name": "Foo", "rank": 800, "is_active": 1},
            {"id": 6, "symbol": "FOO", "name": "Foo Scam", "rank": 9000, "is_active": 1},
        ],
    }
    fake_info = {
        "1027": _info([{"contract_address": "0xETHonBSC",
                        "platform": {"name": "BNB Smart Chain"}}]),
        "5": _info([{"contract_address": "0xFOO", "platform": {"name": "BNB Smart Chain"}}]),
    }
    monkeypatch.setattr(cmc, "fetch_map", lambda syms, key: fake_map)
    monkeypatch.setattr(cmc, "fetch_info", lambda ids, key: fake_info)

    res = cmc.resolve_bsc_contracts(["ETH", "FOO", "MISSING"], "KEY")
    assert res["ETH"]["cmc_id"] == 1027
    assert res["ETH"]["bsc_contract"] == "0xETHonBSC"
    assert res["FOO"]["cmc_id"] == 5            # canonical (best rank) of the 2
    assert res["FOO"]["n_candidates"] == 2
    assert res["MISSING"]["bsc_contract"] is None
    assert res["MISSING"]["cmc_id"] is None


# --- dexscreener contract-path -------------------------------------------

def test_summarize_token_pairs_picks_deepest_bsc_pool():
    pairs = [
        {"chainId": "bsc", "pairAddress": "0xLOW", "liquidity": {"usd": 1000},
         "baseToken": {"symbol": "X", "name": "X", "address": "0xT"},
         "quoteToken": {"symbol": "WBNB"}, "priceUsd": "1.0"},
        {"chainId": "bsc", "pairAddress": "0xHIGH", "liquidity": {"usd": 9000},
         "baseToken": {"symbol": "X", "name": "X", "address": "0xT"},
         "quoteToken": {"symbol": "USDT"}, "priceUsd": "1.0"},
        {"chainId": "ethereum", "pairAddress": "0xETH", "liquidity": {"usd": 99999},
         "baseToken": {"symbol": "X"}, "quoteToken": {"symbol": "USDC"}},
    ]
    row = ds.summarize_token_pairs("X", pairs, now_ms=1_700_000_000_000)
    assert row["status"] == "resolved"
    assert row["pair_address"] == "0xHIGH"     # deepest BSC pool, ignores ETH pool
    assert row["liq_usd"] == 9000
