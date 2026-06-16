"""build_portfolio_artifacts marker placement: event-env fills must be drawn at the REAL candle
close of their bar (a LOCATION on the chart), not at the env's returns-index `_px` value scaled by
the token's global first close — the bug that floated Q's buy to 0.0219 above a 0.0185 high. PnL is
read from token_pnl.json (the env ledger), so a marker price is purely for placement."""
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
import train_rl  # noqa: E402


def _ohlcv(ts, closes):
    return pd.DataFrame({"timestamp": ts, "open": closes, "high": closes,
                         "low": closes, "close": [float(c) for c in closes],
                         "volume": [1.0] * len(ts)})


def test_event_marker_lands_on_candle_close_not_px_index(monkeypatch):
    ts = [1_700_000_000, 1_700_003_600, 1_700_007_200]      # 3 hourly bars
    oh = _ohlcv(ts, [100, 110, 121])                        # close rises 100 -> 110 -> 121
    monkeypatch.setattr(train_rl, "_load_token_ohlcv", lambda t: oh if t == "ZEC" else None)
    # f["px"] is the env's returns-index basis (~1.0), deliberately NOT a real price. The OLD code did
    # f["px"] * first_close = 0.95 * 100 = 95 (off the candle); the fix must use the candle close 110.
    records = [{"time": ts[1], "weights": {"ZEC": 0.5},
                "fills": [{"token": "ZEC", "time": ts[1], "px": 0.95, "usd": 5000.0, "fee": 2.0}]}]
    _w, candles, trades = train_rl.build_portfolio_artifacts(records, ["ZEC"], ts[0], ts[2])

    m = trades["ZEC"][0]
    assert m["price"] == 110.0                              # candle close at the fill bar (not 95)
    assert m["side"] == "buy" and m["usd"] == 5000.0 and m["fee"] == 2.0
    assert len(candles["ZEC"]) == 3                         # candles still emitted from the real OHLCV
