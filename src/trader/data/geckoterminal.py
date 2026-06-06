"""GeckoTerminal (CoinGecko on-chain) client — historical DEX OHLCV for BSC.

No API key (keyless public API, ~30 req/min). OHLCV is keyed by **pool address**
(supplied by the DexScreener screen), timeframes ``day``/``hour``/``minute`` with
aggregations (minute: 1/5/15), up to 1000 candles per request, history ~6 months
back. This is the training-history source the DexScreener snapshots can't provide.

Fetch is stdlib-only (urllib); analysis helpers are pure and testable.
"""

from __future__ import annotations

import json
import math
import urllib.error
import urllib.request

GT = "https://api.geckoterminal.com/api/v2"
_HEADERS = {"User-Agent": "act-data-spike/0.1", "Accept": "application/json"}


def _get_json(url: str, timeout: int = 30) -> dict:
    req = urllib.request.Request(url, headers=_HEADERS)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def fetch_ohlcv(
    pool: str,
    timeframe: str = "hour",
    aggregate: int = 1,
    limit: int = 1000,
    currency: str = "usd",
    network: str = "bsc",
    before_timestamp: int | None = None,
    timeout: int = 30,
) -> list[list[float]]:
    """OHLCV rows ``[ts, open, high, low, close, volume]`` (newest first).

    `before_timestamp` (unix seconds) paginates further back for full history.
    """
    url = (
        f"{GT}/networks/{network}/pools/{pool}/ohlcv/{timeframe}"
        f"?aggregate={aggregate}&limit={limit}&currency={currency}"
    )
    if before_timestamp is not None:
        url += f"&before_timestamp={before_timestamp}"
    raw = _get_json(url, timeout=timeout)
    attrs = ((raw.get("data") or {}).get("attributes") or {})
    return attrs.get("ohlcv_list") or []


# --- pure analysis helpers ------------------------------------------------

def candle_span_days(ohlcv: list[list[float]]) -> float:
    """Calendar span covered by the candles, in days."""
    if not ohlcv:
        return 0.0
    ts = [row[0] for row in ohlcv]
    return round((max(ts) - min(ts)) / 86_400.0, 2)


def realized_vol(ohlcv: list[list[float]]) -> float | None:
    """Sample std-dev of per-candle log returns of close. None if too sparse."""
    closes = [row[4] for row in ohlcv if row[4]]
    if len(closes) < 3:
        return None
    rets = [
        math.log(closes[i] / closes[i - 1])
        for i in range(1, len(closes))
        if closes[i - 1] > 0
    ]
    if len(rets) < 2:
        return None
    mean = sum(rets) / len(rets)
    var = sum((r - mean) ** 2 for r in rets) / (len(rets) - 1)
    return round(math.sqrt(var), 5)
