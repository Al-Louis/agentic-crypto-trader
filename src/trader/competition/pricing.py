"""USD prices for participant holdings — current (live) and baseline (at the window start).

**Source: CMC k-line** (`/v1/k-line/candles`, by token CONTRACT ADDRESS, USD OHLCV). One call returns
the token's full recent hourly series, so the SAME fetch serves both the current price (latest bar) and
any historical hour (the bar covering that ts) — and being an aggregated by-address USD price it has no
"deepest pool" confusion (the DexScreener base-token bug that mispriced TRX as HTX and ZEC as AAVE).
Stables = 1.0; BNB from the local anchor parquet. DexScreener (base-matched) remains a last-resort
fallback only when CMC has no series for a token.

All price reads are read-only and best-effort: a miss yields no entry (the pure payload builder then
flags that holding's value null + the wallet stale), never a crash.
"""

from __future__ import annotations

import json
import os
import time
import urllib.parse
import urllib.request

from trader import config

WBNB = "0xBB4CdB9CBd36B01bD1cBaEBF2De08d9173bc095c"
OHLCV_ROOT = os.path.join("data", "ohlcv")   # local hourly store (matches the live feed's path)
CMC_KLINE = "https://pro-api.coinmarketcap.com/v1/k-line/candles"
_CMC_SERIES: dict[str, dict[int, float]] = {}   # addr_lower -> {bar_open_ts(secs): close_usd}, per-process cache


def _cmc_series(addr: str, *, interval: str = "1h", limit: int = 300, sleep: float = 0.12
                ) -> dict[int, float]:
    """`{bar_open_ts: close_usd}` for a token's CMC k-line series (cached per process). Empty on any
    miss/error. One fetch covers ~limit hours, reused for both the current price and every historical
    hour, so a full board prices each token with a single CMC call."""
    a = addr.lower()
    if a in _CMC_SERIES:
        return _CMC_SERIES[a]
    key = config.get("CMC_API_KEY")
    series: dict[int, float] = {}
    if key:
        q = {"platform": "bsc", "address": addr, "interval": interval, "unit": "usd", "limit": str(limit)}
        req = urllib.request.Request(f"{CMC_KLINE}?{urllib.parse.urlencode(q)}",
                                     headers={"X-CMC_PRO_API_KEY": key, "Accept": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=40) as r:
                data = json.loads(r.read()).get("data") or []
            series = {int(c[5]) // 1000: float(c[3]) for c in data if c[3]}   # candle=[o,h,l,c,v,time_ms,count]
        except Exception:  # noqa: BLE001 — best-effort; a miss falls back to DexScreener
            series = {}
        if sleep:
            time.sleep(sleep)
    _CMC_SERIES[a] = series
    return series


def _cmc_now(addr: str) -> float | None:
    s = _cmc_series(addr)
    return s[max(s)] if s else None


def _cmc_at(addr: str, ts: int) -> float | None:
    """CMC close of the bar covering `ts` (open <= ts), else the earliest bar if `ts` predates the series."""
    s = _cmc_series(addr)
    if not s:
        return None
    earlier = [t for t in s if t <= ts]
    return s[max(earlier)] if earlier else s[min(s)]


def _deepest_price_usd(pairs: list[dict], token_addr: str | None = None) -> float | None:
    """`priceUsd` of the deepest-liquidity BSC pair where our token is the BASE token.

    DexScreener's `priceUsd` is the price of the pair's BASE token, so a pool where our token is the
    QUOTE side reports the OTHER token's price. Without the base-token guard, the deepest pool wins
    blindly: e.g. TRX's contract's deepest BSC pool ($3.4M) is HTX-based, whose `priceUsd` is ~$1.7e-6,
    valuing TRX at ~$0 (the 2026-06-26 16Z phantom -45% dip). When `token_addr` is given we only trust
    base-matching pairs; a token with no such pool stays unpriced (flagged) rather than mis-valued."""
    addr = (token_addr or "").lower()
    best, best_liq = None, -1.0
    for p in pairs or []:
        if p.get("chainId") != "bsc":
            continue
        if addr and ((p.get("baseToken") or {}).get("address") or "").lower() != addr:
            continue
        liq = float((p.get("liquidity") or {}).get("usd") or 0.0)
        px = p.get("priceUsd")
        if px and liq > best_liq:
            best, best_liq = float(px), liq
    return best


STABLE_BAND = 0.10   # honor the is_stable $1 peg only when CMC agrees the token is within 10% of $1


def _peg_to_dollar(is_stable: bool, cmc_px: float | None) -> bool:
    """Treat a flagged stablecoin as exactly $1.0 ONLY when CMC confirms it's ~$1 (or has no data at all,
    e.g. DAI). A token flagged is_stable but priced by CMC far from $1 is a MISCLASSIFICATION — e.g.
    `STABLE` (~$0.04) or `EURI` (EUR-pegged, ~$1.14) — and must get its real CMC price, not $1."""
    return is_stable and (cmc_px is None or abs(cmc_px - 1.0) <= STABLE_BAND)


def current_prices(held_symbols, universe: list[dict], *, root: str | None = None,
                   sleep: float = 0.2) -> dict[str, float]:
    """`{symbol: usd}` for the held symbols. BNB -> anchor; genuine USD stablecoins -> 1.0; everything
    else (incl. misclassified "stables") -> CMC k-line latest bar (DexScreener base-matched last resort)."""
    by_sym = {u["symbol"]: u for u in universe}
    px: dict[str, float] = {}
    for sym in set(held_symbols):
        if sym == "BNB":
            v = _bnb_anchor_close(root=root) or _cmc_now(WBNB)
            if v:
                px[sym] = v
            continue
        u = by_sym.get(sym) or {}
        addr = u.get("contract")
        cmc = _cmc_now(addr) if addr else None
        if _peg_to_dollar(sym == "USDT" or u.get("is_stable"), cmc):
            px[sym] = 1.0
        elif cmc is not None:
            px[sym] = cmc
        elif addr:
            v = _dex_fallback(addr)
            if v:
                px[sym] = v
    return px


def _dex_fallback(address: str) -> float | None:
    """Last-resort price when CMC has no series for a token: DexScreener's base-matched deepest pool."""
    from trader.data import dexscreener  # noqa: PLC0415
    try:
        return _deepest_price_usd(dexscreener.token_pairs(address), address)
    except Exception:  # noqa: BLE001 — best-effort; a miss => unpriced (flagged), not a crash
        return None


def start_prices(held_symbols, universe: list[dict], start_ts: int, *,
                 current: dict[str, float] | None = None,
                 root: str | None = None) -> tuple[dict[str, float], list[str]]:
    """`({symbol: usd}, approx_symbols)` at `start_ts`. Stables=1; BNB from the anchor bar; everything
    else from the CMC k-line bar covering `start_ts` (true per-hour history by address). A token CMC
    doesn't cover falls back to the local OHLCV bar, then the current price (flagged in `approx`)."""
    root = root or OHLCV_ROOT
    by_sym = {u["symbol"]: u for u in universe}
    current = current or {}
    px: dict[str, float] = {}
    approx: list[str] = []
    for sym in set(held_symbols):
        if sym == "BNB":
            v = _bnb_anchor_close(at_ts=start_ts)
            if v is not None:
                px[sym] = v
            elif "BNB" in current:
                px[sym] = current["BNB"]
                approx.append("BNB")
            continue
        u = by_sym.get(sym) or {}
        addr = u.get("contract")
        cmc = _cmc_at(addr, start_ts) if addr else None
        if _peg_to_dollar(sym == "USDT" or u.get("is_stable"), cmc):
            px[sym] = 1.0
            continue
        v = cmc
        if v is None:                          # CMC gap -> local OHLCV bar
            v = _ohlcv_close_at(sym, u.get("pair_address"), start_ts, root) if u.get("pair_address") else None
        if v is not None:
            px[sym] = v
        elif sym in current:                   # last resort: current price, flagged approximate
            px[sym] = current[sym]
            approx.append(sym)
    return px, sorted(set(approx))


def _ohlcv_close_at(symbol: str, pair: str, ts: int, root: str) -> float | None:
    """Close of the local hourly bar with `timestamp <= ts` (the bar covering the window start)."""
    try:
        from trader.data.downloader import load_ohlcv  # noqa: PLC0415

        df = load_ohlcv(symbol, pair, "hour", 1, root=root)
        if df.empty:
            return None
        sub = df[df["timestamp"] <= ts]
        row = (sub if not sub.empty else df).iloc[-1]
        v = float(row["close"])
        return v if (v == v and v > 0) else None
    except Exception:  # noqa: BLE001
        return None


def _bnb_anchor_close(*, at_ts: int | None = None, root: str | None = None) -> float | None:
    """BNB USD close from the anchor parquet — latest, or the bar `<= at_ts`. Same source the live
    runner prices BNB with (parity)."""
    try:
        import pandas as pd  # noqa: PLC0415

        path = os.path.join("data", "anchor", "BNB_USDT", "1h.parquet")
        a = pd.read_parquet(path).sort_values("timestamp")
        if at_ts is not None:
            sub = a[a["timestamp"] <= at_ts]
            a = sub if not sub.empty else a
        v = float(a["close"].iloc[-1])
        return v if (v == v and v > 0) else None
    except Exception:  # noqa: BLE001
        return None
