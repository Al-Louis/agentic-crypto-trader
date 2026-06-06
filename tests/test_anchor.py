"""Tests for the ccxt anchor downloader (pure pagination; no network)."""

from trader.data import anchor


class FakeExchange:
    """Serves an ascending candle list like ccxt.fetch_ohlcv(since=, limit=)."""

    def __init__(self, candles):
        self.candles = candles
        self.calls = 0

    def fetch_ohlcv(self, symbol, timeframe, since=None, limit=1000):
        self.calls += 1
        return [c for c in self.candles if c[0] >= (since or 0)][:limit]


def _candles(n, tf=60_000):
    return [[i * tf, 1.0, 1.1, 0.9, 1.0 + i, 100.0] for i in range(n)]


def test_slug_and_path():
    assert anchor._slug("BTC/USDT") == "BTC_USDT"
    assert anchor._path("data/anchor", "BNB/USDT", "1h").endswith("BNB_USDT/1h.parquet".replace("/", anchor.os.sep))


def test_paginate_returns_full_range_across_pages():
    candles = _candles(2500)               # 3 pages at limit=1000
    ex = FakeExchange(candles)
    rows = anchor.paginate_ohlcv(ex, "BTC/USDT", "1m", since_ms=0, until_ms=2500 * 60_000, limit=1000)
    assert len(rows) == 2500
    assert ex.calls == 3
    assert [r[0] for r in rows] == [c[0] for c in candles]


def test_paginate_respects_until():
    ex = FakeExchange(_candles(2500))
    rows = anchor.paginate_ohlcv(ex, "BTC/USDT", "1m", since_ms=0, until_ms=1500 * 60_000, limit=1000)
    assert all(r[0] < 1500 * 60_000 for r in rows)
    assert len(rows) == 1500


def test_paginate_stops_without_progress():
    # exchange that keeps returning the same first candle => must not loop forever
    class Stuck:
        def fetch_ohlcv(self, *a, **k):
            return [[0, 1, 1, 1, 1, 1]]
    rows = anchor.paginate_ohlcv(Stuck(), "BTC/USDT", "1m", since_ms=0, until_ms=10 * 60_000)
    assert len(rows) <= 1


def test_load_anchor_missing_is_empty():
    df = anchor.load_anchor("BTC/USDT", "1m", root="data/_nonexistent_anchor_")
    assert df.empty
