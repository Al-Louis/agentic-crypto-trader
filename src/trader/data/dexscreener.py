"""DexScreener client — token *screening* for BSC (no API key, ~60 req/min).

DexScreener provides live snapshots only (liquidity, volume, price-change,
pool age) — **no historical OHLCV**. So this module is the screening surface
that turns the eligible universe into a ranked, risk-characterized shortlist;
historical candles come from `trader.data.geckoterminal`.

Fetch is stdlib-only (urllib) so the spike runs with zero extra deps. Parsing
helpers are pure and unit-testable.
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.parse
import urllib.request

BASE = "https://api.dexscreener.com"
_HEADERS = {"User-Agent": "act-data-spike/0.1", "Accept": "application/json"}


def _get_json(url: str, timeout: int = 20, retries: int = 1) -> dict:
    """GET + parse JSON, with one polite backoff on HTTP 429."""
    for attempt in range(retries + 1):
        try:
            req = urllib.request.Request(url, headers=_HEADERS)
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return json.loads(r.read())
        except urllib.error.HTTPError as e:
            if e.code == 429 and attempt < retries:
                time.sleep(3.0)
                continue
            raise


def search(symbol: str, timeout: int = 20) -> dict:
    """Raw DexScreener search response for a symbol/name/address query."""
    q = urllib.parse.quote(symbol)
    return _get_json(f"{BASE}/latest/dex/search?q={q}", timeout=timeout)


# --- pure parsing helpers -------------------------------------------------

def _liq(pair: dict) -> float:
    return float((pair.get("liquidity") or {}).get("usd") or 0.0)


def bsc_matches(raw: dict, symbol: str) -> list[dict]:
    """BSC pairs whose *base* token symbol matches (case-insensitive)."""
    out = []
    for p in raw.get("pairs") or []:
        if p.get("chainId") != "bsc":
            continue
        base_sym = ((p.get("baseToken") or {}).get("symbol") or "")
        if base_sym.lower() == symbol.lower():
            out.append(p)
    return out


def summarize(symbol: str, raw: dict, now_ms: float | None = None) -> dict:
    """Reduce a search response to one row for `symbol` (symbol-search path).

    Picks the deepest-liquidity BSC pair as canonical; flags ambiguity when a
    runner-up pair has comparable liquidity (likely a same-ticker different
    contract — a real shitcoin hazard).
    """
    matches = sorted(bsc_matches(raw, symbol), key=_liq, reverse=True)
    return _build_row(symbol, matches, now_ms)


def token_pairs(address: str, chain: str = "bsc", timeout: int = 20) -> list[dict]:
    """Pairs for a specific token *contract* (the v1 token-pairs API).

    Used after CMC resolves the canonical BSC contract — keys off the right
    address instead of guessing by ticker. Returns a list of pair dicts.
    """
    q = urllib.parse.quote(address)
    data = _get_json(f"{BASE}/token-pairs/v1/{chain}/{q}", timeout=timeout)
    return data if isinstance(data, list) else (data.get("pairs") or [])


def summarize_token_pairs(symbol: str, pairs: list[dict],
                          now_ms: float | None = None) -> dict:
    """Reduce contract-resolved pairs to one row (deepest-liquidity BSC pool).

    Ambiguity is already resolved by the contract address, so this just screens
    the right token's pools — `ambiguous` here means multiple *pools* for the
    same token, not a ticker collision.
    """
    bsc = sorted((p for p in pairs if p.get("chainId") == "bsc"), key=_liq, reverse=True)
    return _build_row(symbol, bsc, now_ms)


def _build_row(symbol: str, matches: list[dict], now_ms: float | None = None) -> dict:
    """Reduce a ranked list of BSC pairs (deepest first) to one screen row."""
    if not matches:
        return {"symbol": symbol, "status": "unresolved", "n_bsc": 0}

    best = matches[0]
    liq = _liq(best)
    second_liq = _liq(matches[1]) if len(matches) > 1 else 0.0
    pc = best.get("priceChange") or {}
    vol = best.get("volume") or {}
    txns = (best.get("txns") or {}).get("h24") or {}
    created = best.get("pairCreatedAt")
    now_ms = now_ms if now_ms is not None else time.time() * 1000
    age_days = round((now_ms - created) / 86_400_000, 1) if created else None

    return {
        "symbol": symbol,
        "status": "resolved",
        "name": (best.get("baseToken") or {}).get("name"),
        "token_address": (best.get("baseToken") or {}).get("address"),
        "pair_address": best.get("pairAddress"),
        "dex": best.get("dexId"),
        "quote": (best.get("quoteToken") or {}).get("symbol"),
        "price_usd": float(best.get("priceUsd") or 0) or None,
        "liq_usd": round(liq, 2),
        "vol_h24": round(float(vol.get("h24") or 0), 2),
        "chg_h1": pc.get("h1"),
        "chg_h6": pc.get("h6"),
        "chg_h24": pc.get("h24"),
        "txns_h24": int(txns.get("buys") or 0) + int(txns.get("sells") or 0),
        "age_days": age_days,
        "n_bsc": len(matches),
        "second_liq": round(second_liq, 2),
        # runner-up within 25% of best liquidity => low-confidence resolution
        "ambiguous": len(matches) > 1 and second_liq > 0.25 * liq,
    }


def vol_proxy(summary: dict) -> float:
    """Cheap intraday volatility proxy from |price-change| across windows.

    A screening-stage stand-in only; realized vol on real candles comes from
    `geckoterminal.realized_vol`.
    """
    vals = [abs(float(summary.get(k) or 0)) for k in ("chg_h1", "chg_h6", "chg_h24")]
    return round(sum(vals) / len(vals), 3) if vals else 0.0
