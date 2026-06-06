"""Tests for the resumable OHLCV downloader (offline; GeckoTerminal mocked)."""

import urllib.error

import pytest

from trader.data import downloader as dlmod
from trader.data import geckoterminal as gt
from trader.data.downloader import OHLCVDownloader, load_ohlcv, slug, tf_key

QUIET = lambda *a, **k: None  # noqa: E731


def make_history(n, step=3600, start_ts=1_700_000_000):
    """`n` candles as GeckoTerminal returns them: newest-first [ts,o,h,l,c,v]."""
    rows = [[start_ts + i * step, 1.0, 1.1, 0.9, 1.0 + i * 0.001, 100.0] for i in range(n)]
    rows.sort(key=lambda r: r[0], reverse=True)
    return rows


def fake_fetch(history, fail_on_call=None, http_429_on_call=None):
    state = {"calls": 0}

    def _fetch(pool, timeframe="hour", aggregate=1, limit=1000, currency="usd",
               network="bsc", before_timestamp=None, timeout=30):
        state["calls"] += 1
        if fail_on_call and state["calls"] == fail_on_call:
            raise RuntimeError("simulated crash")
        if http_429_on_call and state["calls"] == http_429_on_call:
            raise urllib.error.HTTPError("http://x", 429, "Too Many Requests", {}, None)
        rows = history if before_timestamp is None else [r for r in history if r[0] < before_timestamp]
        return rows[:limit]

    return _fetch, state


def _dl(tmp_path, **kw):
    return OHLCVDownloader(root=str(tmp_path), min_interval=0, max_days=10 ** 9,
                           page_limit=1000, logger=QUIET, **kw)


# --- helpers --------------------------------------------------------------

def test_slug_and_tf_key():
    assert slug("CAKE") == "CAKE"
    assert slug("币安人生") == "____"           # non-ASCII -> safe
    assert slug("a/b:c") == "a_b_c"
    assert tf_key("minute", 5) == "minute_5"


# --- full paginated download ---------------------------------------------

def test_full_download_paginates_and_dedupes(tmp_path, monkeypatch):
    history = make_history(2500)              # 3 pages at limit=1000
    fetch, state = fake_fetch(history)
    monkeypatch.setattr(gt, "fetch_ohlcv", fetch)

    ent = _dl(tmp_path).download("CAKE", "0xPOOL", "hour", 1)
    assert ent["complete"] is True
    assert ent["pages"] == 3
    assert state["calls"] == 3

    df = load_ohlcv("CAKE", "0xPOOL", "hour", 1, root=str(tmp_path))
    assert len(df) == 2500
    assert df["timestamp"].is_monotonic_increasing      # sorted oldest->newest
    assert df["timestamp"].is_unique


# --- resume after a mid-download crash ------------------------------------

def test_resume_after_crash(tmp_path, monkeypatch):
    history = make_history(2500)
    fetch, _ = fake_fetch(history, fail_on_call=3)       # crash entering page 3
    monkeypatch.setattr(gt, "fetch_ohlcv", fetch)

    with pytest.raises(RuntimeError):
        _dl(tmp_path).download("X", "0xP", "hour", 1)

    # a fresh instance reloads the manifest: 2 pages checkpointed, not complete
    dl2 = _dl(tmp_path)
    assert dl2.manifest["0xP|hour_1"]["pages"] == 2
    assert dl2.manifest["0xP|hour_1"]["complete"] is False

    fetch2, _ = fake_fetch(history)                      # fixed feed
    monkeypatch.setattr(gt, "fetch_ohlcv", fetch2)
    ent = dl2.download("X", "0xP", "hour", 1)            # resumes from page 3
    assert ent["complete"] is True and ent["pages"] == 3

    df = load_ohlcv("X", "0xP", "hour", 1, root=str(tmp_path))
    assert len(df) == 2500 and df["timestamp"].is_unique


def test_completed_series_is_skipped(tmp_path, monkeypatch):
    history = make_history(500)                          # single short page -> complete
    fetch, state = fake_fetch(history)
    monkeypatch.setattr(gt, "fetch_ohlcv", fetch)
    dl = _dl(tmp_path)
    dl.download("Z", "0xZ", "day", 1)
    calls_after_first = state["calls"]
    dl.download("Z", "0xZ", "day", 1)                    # second run: no new fetches
    assert state["calls"] == calls_after_first


# --- 429 backoff ----------------------------------------------------------

def test_429_is_retried(tmp_path, monkeypatch):
    history = make_history(50)                           # one short page
    fetch, state = fake_fetch(history, http_429_on_call=1)  # first call rate-limited
    monkeypatch.setattr(gt, "fetch_ohlcv", fetch)
    monkeypatch.setattr(dlmod.time, "sleep", QUIET)      # don't actually wait

    ent = _dl(tmp_path).download("R", "0xR", "minute", 1)
    assert ent["complete"] is True
    assert ent["pages"] == 1
    assert state["calls"] == 2                           # 429 then success
