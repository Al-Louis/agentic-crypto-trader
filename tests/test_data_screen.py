"""Unit tests for the data-spike pure helpers (no network)."""

from trader.data import dexscreener as ds
from trader.data import geckoterminal as gt
from trader.data.eligible import ELIGIBLE_SYMBOLS, STABLES, eligible_symbols


def _pair(symbol, liq, chain="bsc", **kw):
    p = {
        "chainId": chain,
        "pairAddress": kw.get("pair", "0xPAIR"),
        "dexId": kw.get("dex", "pancakeswap"),
        "baseToken": {"symbol": symbol, "name": kw.get("name", symbol), "address": kw.get("addr", "0xTOK")},
        "quoteToken": {"symbol": kw.get("quote", "WBNB")},
        "priceUsd": kw.get("price", "1.0"),
        "liquidity": {"usd": liq},
        "volume": {"h24": kw.get("vol", 1000)},
        "priceChange": {"h1": kw.get("h1", 1.0), "h6": kw.get("h6", -2.0), "h24": kw.get("h24", 5.0)},
        "txns": {"h24": {"buys": 3, "sells": 2}},
        "pairCreatedAt": kw.get("created", 1_000_000_000_000),
    }
    return p


# --- eligible universe ----------------------------------------------------

def test_eligible_is_deduped_and_nonempty():
    assert len(ELIGIBLE_SYMBOLS) == len(set(ELIGIBLE_SYMBOLS))
    assert "ETH" in ELIGIBLE_SYMBOLS and "CAKE" in ELIGIBLE_SYMBOLS
    # source prose lists SLX twice; dedupe must collapse it
    assert ELIGIBLE_SYMBOLS.count("SLX") == 1


def test_eligible_symbols_can_drop_stables():
    with_stables = eligible_symbols(include_stables=True)
    without = eligible_symbols(include_stables=False)
    assert len(without) < len(with_stables)
    assert not (set(without) & STABLES)


# --- dexscreener parsing --------------------------------------------------

def test_bsc_matches_filters_chain_and_symbol():
    raw = {"pairs": [
        _pair("CAKE", 100, chain="bsc"),
        _pair("CAKE", 999, chain="ethereum"),   # wrong chain
        _pair("NOTCAKE", 500, chain="bsc"),       # wrong symbol
    ]}
    m = ds.bsc_matches(raw, "CAKE")
    assert len(m) == 1 and m[0]["chainId"] == "bsc"


def test_summarize_picks_deepest_liquidity():
    raw = {"pairs": [
        _pair("X", 1_000, pair="0xLOW"),
        _pair("X", 9_000, pair="0xHIGH"),
    ]}
    s = ds.summarize("X", raw, now_ms=1_000_000_000_000)
    assert s["status"] == "resolved"
    assert s["pair_address"] == "0xHIGH"
    assert s["liq_usd"] == 9_000
    assert s["n_bsc"] == 2


def test_summarize_flags_ambiguity_when_runnerup_close():
    raw = {"pairs": [_pair("X", 1_000), _pair("X", 900)]}  # 900 > 0.25*1000
    assert ds.summarize("X", raw)["ambiguous"] is True
    raw2 = {"pairs": [_pair("X", 1_000), _pair("X", 100)]}  # 100 < 0.25*1000
    assert ds.summarize("X", raw2)["ambiguous"] is False


def test_summarize_unresolved_when_no_bsc_pair():
    raw = {"pairs": [_pair("X", 1_000, chain="ethereum")]}
    s = ds.summarize("X", raw)
    assert s["status"] == "unresolved" and s["n_bsc"] == 0


def test_vol_proxy_averages_abs_changes():
    s = {"chg_h1": 1.0, "chg_h6": -2.0, "chg_h24": 6.0}
    assert ds.vol_proxy(s) == 3.0


# --- geckoterminal analysis ----------------------------------------------

def test_candle_span_days():
    ohlcv = [[86_400, 1, 1, 1, 1, 1], [0, 1, 1, 1, 1, 1]]
    assert gt.candle_span_days(ohlcv) == 1.0
    assert gt.candle_span_days([]) == 0.0


def test_realized_vol_constant_series_is_zero():
    ohlcv = [[i, 1, 1, 1, 10.0, 1] for i in range(5)]
    assert gt.realized_vol(ohlcv) == 0.0


def test_realized_vol_none_when_sparse():
    assert gt.realized_vol([[0, 1, 1, 1, 10, 1]]) is None


def test_realized_vol_positive_for_moving_series():
    closes = [10, 11, 9, 12, 8]
    ohlcv = [[i, 0, 0, 0, c, 0] for i, c in enumerate(closes)]
    rv = gt.realized_vol(ohlcv)
    assert rv is not None and rv > 0


def test_endpoint_keyless_by_default(monkeypatch):
    monkeypatch.delenv("COINGECKO_API_KEY", raising=False)
    base, headers = gt._endpoint()
    assert base == gt.GT and "x-cg-demo-api-key" not in headers and "x-cg-pro-api-key" not in headers


def test_endpoint_demo_key_routes_to_coingecko(monkeypatch):
    monkeypatch.setenv("COINGECKO_API_KEY", "CG-test123")
    monkeypatch.delenv("COINGECKO_API_TIER", raising=False)        # default tier = demo
    base, headers = gt._endpoint()
    assert base == gt.CG_DEMO and headers["x-cg-demo-api-key"] == "CG-test123"


def test_endpoint_pro_tier_uses_pro_host_and_header(monkeypatch):
    monkeypatch.setenv("COINGECKO_API_KEY", "CG-test123")
    monkeypatch.setenv("COINGECKO_API_TIER", "pro")
    base, headers = gt._endpoint()
    assert base == gt.CG_PRO and headers["x-cg-pro-api-key"] == "CG-test123"
    # the on-chain path shape is identical, so only the base + auth differ from keyless
    assert base.endswith("/onchain")
