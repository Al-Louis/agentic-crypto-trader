"""CMC live-quote parsing — the canonical-pick fix + the missing-price omission.

No network: feeds `parse_quotes` the exact response *shape* observed live on 2026-06-11
(the "BNB" collision: 4 records — real BNB rank-4 priced, BNB AI, BNBTiger, and an inactive
null-price BeanBox). Asserts the canonical major wins over the collisions and that a symbol
with no priced record is omitted (never a false zero) — the bug that silently dropped BNB
when the parser took `recs[0]`."""

from trader.data.cmc_market import Quote, fetch_quotes, parse_quotes


def _rec(id_, name, sym, rank, price, active=1):
    quote = {"USD": {"price": price, "percent_change_24h": 1.0, "volume_24h": 100.0,
                     "last_updated": "2026-06-11T00:00:00Z"}} if price is not None \
        else {"USD": {"price": None}}
    return {"id": id_, "name": name, "symbol": sym, "cmc_rank": rank,
            "is_active": active, "quote": quote}


def test_canonical_pick_beats_ticker_collisions():
    # the live BNB collision shape, deliberately ordered so recs[0] is the WRONG (null) record
    payload = {"data": {"BNB": [
        _rec(39373, "BeanBox", "BNB", None, None, active=0),
        _rec(38157, "BNB AI", "BNB", 7592, 2.7e-05),
        _rec(1839, "BNB", "BNB", 4, 603.92),
        _rec(38210, "BNBTiger Inu", "BNB", 7604, 3.3e-05),
    ]}}
    out = parse_quotes(payload)
    assert out["BNB"].price_usd == 603.92  # rank-4 active priced major wins, not recs[0]


def test_missing_price_is_omitted_not_zero():
    payload = {"data": {"DEAD": [_rec(1, "Dead", "DEAD", None, None, active=0)]}}
    out = parse_quotes(payload)
    assert "DEAD" not in out  # no false zero — the loop reads absence as "no observation"


def test_dict_and_list_data_shapes_both_parse():
    as_list = {"data": {"USDT": [_rec(825, "Tether", "USDT", 3, 1.0)]}}
    as_dict = {"data": {"USDT": _rec(825, "Tether", "USDT", 3, 1.0)}}
    assert parse_quotes(as_list)["USDT"].price_usd == 1.0
    assert parse_quotes(as_dict)["USDT"].price_usd == 1.0


def test_non_positive_and_nan_prices_dropped():
    payload = {"data": {
        "Z": [_rec(1, "Z", "Z", 1, 0.0)],
        "N": [_rec(2, "N", "N", 2, float("nan"))],
        "G": [_rec(3, "G", "G", 3, 5.0)],
    }}
    out = parse_quotes(payload)
    assert set(out) == {"G"} and out["G"].price_usd == 5.0


def test_fetch_quotes_drops_non_ascii_symbols(monkeypatch):
    seen = {}

    def fake_get(endpoint, params, api_key, timeout=40):
        seen["symbols"] = params["symbol"]
        return {"data": {"USDT": [_rec(825, "Tether", "USDT", 3, 1.0)]}}

    monkeypatch.setattr("trader.data.cmc_market._get", fake_get)
    out = fetch_quotes(["USDT", "币安人生", "NIGHT"], "k")
    # the CJK ticker is dropped before the request (CMC 400s a batch on one bad symbol)
    assert "币安人生" not in seen["symbols"]
    assert isinstance(out["USDT"], Quote)
