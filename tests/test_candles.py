"""The per-token candlestick publisher (trading/candles/) — payload shape + the local put path."""

import json
import os

import pytest

from trader.agent.candles import build_candle_payload, publish_candles

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SEL = os.path.join(REPO, "data", "selection.json")

pytestmark = pytest.mark.skipif(not os.path.isfile(SEL),
                                reason="recorded data/selection.json absent")


def _selection():
    return [{"symbol": s["symbol"], "pair_address": s["pair_address"]}
            for s in json.load(open(SEL, encoding="utf-8"))]


def test_build_candle_payload_shape():
    s = _selection()[0]
    p = build_candle_payload(s["symbol"], s["pair_address"], window_bars=100, generated="G")
    assert p is not None
    assert p["interval_seconds"] == 3600 and p["token"] == s["symbol"]
    assert 0 < len(p["candles"]) <= 100
    assert set(p["candles"][0]) == {"t", "o", "h", "l", "c", "v"}
    ts = [c["t"] for c in p["candles"]]
    assert ts == sorted(ts)                                 # chronological
    assert all(c["t"] < 1e12 for c in p["candles"])         # unix SECONDS, not ms


def test_publish_candles_writes_files_and_index(tmp_path):
    sel = _selection()[:3]
    target = str(tmp_path / "trading")                      # local path = same put_bytes code path
    n = publish_candles(sel, target, window_bars=50)
    assert n >= 1
    idx = json.loads((tmp_path / "trading" / "candles" / "index.json").read_text(encoding="utf-8"))
    assert idx["interval_seconds"] == 3600 and len(idx["tokens"]) == n
    for t in idx["tokens"]:
        f = json.loads((tmp_path / "trading" / "candles" / f"{t['slug']}.json").read_text(encoding="utf-8"))
        assert f["token"] == t["symbol"] and len(f["candles"]) == t["n"] <= 50
        assert f["candles"][-1]["t"] == t["last"]
