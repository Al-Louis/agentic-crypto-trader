"""Publish per-token live candlesticks under the `trading/candles/` CDN prefix (hourly).

Projects the box's `data/ohlcv/hour_1/<token>` parquet store into static JSON the frontend can
chart — `trading/candles/<slug>.json` per token + a `trading/candles/index.json` directory.
Stays within the EC2 instance role's put-only `trading/*` grant (no new IAM). Candle shape
`{t,o,h,l,c,v}` (t = unix seconds) matches the simulated-trades page's candle convention; overlay
`trading/trades.json` markers and cross-reference `market_metrics.json`'s `selected` for the
currently-traded vol-top-8. See [[Apentic Data Contract]] §trading/.
"""

from __future__ import annotations

import json
import math
from datetime import datetime, timezone

from trader.agent.live_data import OHLCV_ROOT
from trader.data.downloader import load_ohlcv
from trader.report.apentic import _slug

INTERVAL_SECONDS = 3600
DEFAULT_WINDOW_BARS = 168          # trailing 7 days of hourly candles (quick-glance charts)
SIG_FIGS = 6                       # round OHLCV to N significant figures (chart precision; trims size)


def _round(x: float) -> float:
    """Round to `SIG_FIGS` significant figures — handles both microcap ($0.000012) and large
    ($446) prices without per-token scale assumptions; full 15-digit floats are wasteful for a chart."""
    if x == 0 or not math.isfinite(x):
        return 0.0
    return round(x, -int(math.floor(math.log10(abs(x)))) + (SIG_FIGS - 1))


def build_candle_payload(symbol: str, pool: str, *, window_bars: int = DEFAULT_WINDOW_BARS,
                         generated: str, root: str = OHLCV_ROOT) -> dict | None:
    """The published candle doc for one token (trailing `window_bars` 1h candles), or None if no
    OHLCV is cached. Timestamps normalized to unix seconds."""
    df = load_ohlcv(symbol, pool, "hour", 1, root=root)
    if df.empty:
        return None
    df = df.tail(window_bars)
    ts = df["timestamp"].to_numpy()
    to_sec = (lambda t: int(t) // 1000) if len(ts) and ts.max() > 1e12 else (lambda t: int(t))
    candles = [{"t": to_sec(t), "o": _round(o), "h": _round(h), "l": _round(low), "c": _round(c),
                "v": _round(v)}
               for t, o, h, low, c, v in zip(df["timestamp"], df["open"], df["high"],
                                             df["low"], df["close"], df["volume"])]
    return {"token": symbol, "slug": _slug(symbol), "generated": generated,
            "interval_seconds": INTERVAL_SECONDS, "candles": candles}


def publish_candles(selection: list[dict], target: str, *, window_bars: int = DEFAULT_WINDOW_BARS,
                    generated: str | None = None) -> int:
    """Publish a candle file per token + an index under `<target>/candles/` (target already ends
    in the `trading` prefix). `no-cache` so the dashboard sees each hour's refresh. Returns the
    number of token files written."""
    from remote_train.publish import join, put_bytes  # noqa: PLC0415 — boto3 stays optional

    generated = generated or datetime.now(timezone.utc).isoformat()
    index: list[dict] = []
    written = 0
    for s in selection:
        payload = build_candle_payload(s["symbol"], s["pair_address"],
                                       window_bars=window_bars, generated=generated)
        if payload is None:
            continue
        data = json.dumps(payload, separators=(",", ":")).encode("utf-8")
        put_bytes(join(target, f"candles/{payload['slug']}.json"), data,
                  content_type="application/json", cache_control="no-cache")
        written += 1
        index.append({"symbol": payload["token"], "slug": payload["slug"],
                      "n": len(payload["candles"]),
                      "last": payload["candles"][-1]["t"] if payload["candles"] else None})
    idx = {"generated": generated, "interval_seconds": INTERVAL_SECONDS, "tokens": index}
    put_bytes(join(target, "candles/index.json"),
              json.dumps(idx, separators=(",", ":")).encode("utf-8"),
              content_type="application/json", cache_control="no-cache")
    return written
