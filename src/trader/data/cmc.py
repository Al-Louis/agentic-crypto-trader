"""CoinMarketCap Pro API client — canonical BSC contract resolution.

Fixes the 35% symbol-resolution ambiguity from the screen (vault "Simulated
Market"): map each eligible symbol -> canonical CMC id -> the token's **BNB Smart
Chain** contract address, so DexScreener pool selection keys off the right
contract instead of guessing by ticker.

Flow:
  /v1/cryptocurrency/map?symbol=...   -> candidate ids per symbol (+ rank, active)
  pick_canonical()                    -> the real token (active, best rank)
  /v2/cryptocurrency/info?id=...      -> contract_address[] across chains
  bsc_contract_from_info()            -> the BNB Smart Chain deployment

Needs CMC_API_KEY (free Basic/Hobbyist tier). Pure parsing helpers are testable
without a key; fetch is stdlib urllib.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.parse
import urllib.request

BASE = "https://pro-api.coinmarketcap.com"
BSC_PLATFORM_NAMES = ("bnb smart chain", "binance smart chain", "bnb chain")


def _get(endpoint: str, params: dict, api_key: str, timeout: int = 30) -> dict:
    url = f"{BASE}{endpoint}?{urllib.parse.urlencode(params)}"
    req = urllib.request.Request(url, headers={
        "X-CMC_PRO_API_KEY": api_key, "Accept": "application/json",
    })
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def _chunks(seq, n):
    seq = list(seq)
    for i in range(0, len(seq), n):
        yield seq[i:i + n]


# --- fetch wrappers -------------------------------------------------------

def _map_into(out: dict, chunk: list[str], api_key: str) -> None:
    d = _get("/v1/cryptocurrency/map", {"symbol": ",".join(chunk)}, api_key)
    for row in d.get("data") or []:
        out.setdefault((row.get("symbol") or "").upper(), []).append(row)


def fetch_map(symbols: list[str], api_key: str) -> dict[str, list[dict]]:
    """symbol(upper) -> list of candidate map rows (id, rank, is_active, ...).

    A single unsupported symbol (e.g. non-ASCII tickers) makes CMC 400 the whole
    comma-batch, so a failing batch is bisected to isolate and skip the offender
    rather than losing every symbol in it.
    """
    out: dict[str, list[dict]] = {}

    def rec(chunk: list[str]) -> None:
        if not chunk:
            return
        try:
            _map_into(out, chunk, api_key)
        except urllib.error.HTTPError as e:
            if e.code == 400 and len(chunk) > 1:
                mid = len(chunk) // 2
                rec(chunk[:mid])
                rec(chunk[mid:])
            elif e.code == 400:
                pass  # single unresolvable symbol — skip it
            else:
                raise

    for chunk in _chunks(symbols, 100):
        rec(chunk)
    return out


def fetch_info(ids: list, api_key: str) -> dict[str, dict]:
    """CMC id(str) -> info record (with contract_address[])."""
    out: dict[str, dict] = {}
    for chunk in _chunks([str(i) for i in ids], 100):
        d = _get("/v2/cryptocurrency/info", {"id": ",".join(chunk)}, api_key)
        out.update(d.get("data") or {})
    return out


# --- pure parsing helpers -------------------------------------------------

def pick_canonical(candidates: list[dict]) -> dict | None:
    """The real token among ticker collisions: active first, then best rank."""
    if not candidates:
        return None
    active = [c for c in candidates if c.get("is_active", 1)]
    pool = active or candidates
    return min(pool, key=lambda c: (c.get("rank") is None, c.get("rank") or 1e9))


def bsc_contract_from_info(info_entry: dict) -> str | None:
    """The BNB Smart Chain contract address from an info record, or None."""
    for ca in info_entry.get("contract_address") or []:
        plat = ca.get("platform") or {}
        name = (plat.get("name") or "").lower()
        coin_sym = ((plat.get("coin") or {}).get("symbol") or "").upper()
        if any(b in name for b in BSC_PLATFORM_NAMES) or coin_sym == "BNB":
            return ca.get("contract_address")
    # fallback: token's primary platform if it is itself BSC
    plat = info_entry.get("platform") or {}
    if plat and any(b in (plat.get("name") or "").lower() for b in BSC_PLATFORM_NAMES):
        return plat.get("token_address")
    return None


# --- orchestration --------------------------------------------------------

def resolve_bsc_contracts(symbols: list[str], api_key: str) -> dict[str, dict]:
    """symbol -> {cmc_id, name, rank, n_candidates, bsc_contract}."""
    mp = fetch_map(symbols, api_key)
    picks: dict[str, dict] = {}
    for sym in symbols:
        c = pick_canonical(mp.get(sym.upper(), []))
        if c:
            picks[sym] = c

    info = fetch_info([p["id"] for p in picks.values()], api_key) if picks else {}

    out: dict[str, dict] = {}
    for sym in symbols:
        pick = picks.get(sym)
        if not pick:
            out[sym] = {"cmc_id": None, "name": None, "rank": None,
                        "n_candidates": len(mp.get(sym.upper(), [])), "bsc_contract": None}
            continue
        entry = info.get(str(pick["id"])) or {}
        out[sym] = {
            "cmc_id": pick["id"],
            "name": pick.get("name"),
            "rank": pick.get("rank"),
            "n_candidates": len(mp.get(sym.upper(), [])),
            "bsc_contract": bsc_contract_from_info(entry),
        }
    return out
