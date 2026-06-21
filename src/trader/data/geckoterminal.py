"""GeckoTerminal (CoinGecko on-chain) client — historical DEX OHLCV for BSC.

Keyless public API by default (~30 req/min, shared bucket). Set `COINGECKO_API_KEY` to route the
SAME on-chain pool data through the keyed CoinGecko endpoint (dedicated, higher rate limit) — this
is **not** a feed change: identical pools, identical candles (`data.attributes.ohlcv_list`), just
authenticated, so **no train/serve skew** (the champion trained on this exact GeckoTerminal source).
`COINGECKO_API_TIER=demo|pro` (default demo) picks the host + auth header.

OHLCV is keyed by **pool address** (supplied by the DexScreener screen), timeframes
``day``/``hour``/``minute`` with aggregations (minute: 1/5/15), up to 1000 candles per request,
history ~6 months back. Fetch is stdlib-only (urllib); analysis helpers are pure and testable.
"""

from __future__ import annotations

import json
import math
import os
import urllib.error
import urllib.request

# Keyless public GeckoTerminal (~30 req/min, shared) — the default.
GT = "https://api.geckoterminal.com/api/v2"
# Keyed CoinGecko on-chain endpoints: SAME GeckoTerminal data + response shape, dedicated quota.
CG_DEMO = "https://api.coingecko.com/api/v3/onchain"
CG_PRO = "https://pro-api.coingecko.com/api/v3/onchain"
_HEADERS = {"User-Agent": "act-data-spike/0.1", "Accept": "application/json"}


def _endpoint() -> tuple[str, dict]:
    """(base_url, headers) chosen from the env at call time: `COINGECKO_API_KEY` set => the keyed
    CoinGecko on-chain endpoint (`COINGECKO_API_TIER=demo|pro`, default demo, picks host + header);
    unset => keyless public GeckoTerminal. The path after the base is identical, so only base + auth
    differ — same data, higher limit, zero skew."""
    key = os.environ.get("COINGECKO_API_KEY")
    if not key:
        return GT, dict(_HEADERS)
    if (os.environ.get("COINGECKO_API_TIER") or "demo").strip().lower() == "pro":
        return CG_PRO, {**_HEADERS, "x-cg-pro-api-key": key}
    return CG_DEMO, {**_HEADERS, "x-cg-demo-api-key": key}


def _get_json(url: str, headers: dict | None = None, timeout: int = 30) -> dict:
    req = urllib.request.Request(url, headers=headers or _HEADERS)
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
    base, headers = _endpoint()
    url = (
        f"{base}/networks/{network}/pools/{pool}/ohlcv/{timeframe}"
        f"?aggregate={aggregate}&limit={limit}&currency={currency}"
    )
    if before_timestamp is not None:
        url += f"&before_timestamp={before_timestamp}"
    raw = _get_json(url, headers=headers, timeout=timeout)
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
