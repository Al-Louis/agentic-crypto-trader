"""BTC/BNB factor-anchor OHLCV via ccxt — the "Bitcoin-is-King" factor data.

binance.com is geo-restricted here, so we use **Binance.US** (same data family as the
TradeSim seed, deep 1-minute history, 1000 candles/request). Public klines need no API
key. Columns match the GeckoTerminal store — `[timestamp(ms), open, high, low, close,
volume]` — so the ported indicator pipeline (`trader.features.indicators`) runs uniformly
on anchor and alt data.

The anchor is the BTC (and BNB) series the factor model regresses alts against:
`r_alt = α + β·r_btc + ε` (vault "Trading Strategies"). Re-running is **incremental** —
it appends only candles newer than the cached Parquet.
"""

from __future__ import annotations

import os

import pandas as pd

OHLCV_COLS = ["timestamp", "open", "high", "low", "close", "volume"]
TF_MS = {"1m": 60_000, "5m": 300_000, "1h": 3_600_000, "1d": 86_400_000}
DEFAULT_ROOT = os.path.join("data", "anchor")


def _slug(symbol: str) -> str:
    return symbol.replace("/", "_")


def _path(root: str, symbol: str, timeframe: str) -> str:
    return os.path.join(root, _slug(symbol), f"{timeframe}.parquet")


def paginate_ohlcv(exchange, symbol: str, timeframe: str, since_ms: int, until_ms: int,
                   limit: int = 1000) -> list[list]:
    """Page ccxt `fetch_ohlcv` forward from `since` to `until` (ascending klines)."""
    tf = TF_MS[timeframe]
    rows: list[list] = []
    cursor = since_ms
    while cursor < until_ms:
        batch = exchange.fetch_ohlcv(symbol, timeframe, since=cursor, limit=limit)
        if not batch:
            break
        rows.extend(batch)
        last = batch[-1][0]
        if last <= cursor:        # no forward progress — stop
            break
        cursor = last + tf
        if len(batch) < limit:    # reached the present
            break
    return [r for r in rows if r[0] < until_ms]


def load_anchor(symbol: str, timeframe: str, root: str = DEFAULT_ROOT) -> pd.DataFrame:
    """Read a cached anchor series (with a `datetime` column), oldest->newest."""
    path = _path(root, symbol, timeframe)
    if not os.path.exists(path):
        return pd.DataFrame(columns=OHLCV_COLS)
    df = pd.read_parquet(path)
    df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    return df


def download_anchor(symbols, timeframes, days: int, root: str = DEFAULT_ROOT,
                    exchange_id: str = "binanceus", logger=print) -> dict:
    """Download/refresh anchor OHLCV for `symbols` × `timeframes` over the last `days`.

    Incremental: if a Parquet exists, only candles newer than its last timestamp are
    fetched, then merged + deduped.
    """
    import ccxt

    ex = getattr(ccxt, exchange_id)({"enableRateLimit": True, "timeout": 30_000})
    now = ex.milliseconds()
    out: dict = {}

    for symbol in symbols:
        for tf in timeframes:
            path = _path(root, symbol, tf)
            os.makedirs(os.path.dirname(path), exist_ok=True)
            existing = pd.read_parquet(path) if os.path.exists(path) else None

            since = int(now - days * 86_400_000)
            if existing is not None and len(existing):
                since = int(existing["timestamp"].max()) + TF_MS[tf]  # incremental

            new_rows = paginate_ohlcv(ex, symbol, tf, since, now) if since < now else []
            df_new = pd.DataFrame(new_rows, columns=OHLCV_COLS)
            df = pd.concat([existing, df_new], ignore_index=True) if existing is not None else df_new
            df = (df.drop_duplicates("timestamp").sort_values("timestamp").reset_index(drop=True))
            df["timestamp"] = df["timestamp"].astype("int64")
            df.to_parquet(path, index=False)

            out[(symbol, tf)] = len(df)
            logger(f"  {symbol:9} {tf:3}: +{len(df_new):6} new, {len(df):7} total -> {path}")
    return out
