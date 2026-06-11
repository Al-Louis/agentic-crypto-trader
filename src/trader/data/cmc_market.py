"""CoinMarketCap live-quote client — the loop's market-data READ surface.

This is the *read step* of the autonomous loop: one batched call to CMC's
`/v2/cryptocurrency/quotes/latest` returns a USD price (plus 24h change / volume)
for every symbol in the eligible universe. Proven 2026-06-11 on the project key:
full 147-symbol ASCII coverage, **1 credit per call** (the symbol list is batched,
not per-symbol), 150k monthly credits — orders of magnitude of headroom for an
hourly loop (≈720 calls/month). See [[Tech Stack]] §"Data sources" and
[[Real-time Monitoring]].

CMC Agent Hub vs plain REST (recon — vault [[Tech Stack]]):
  * The same quote data is reachable three ways: this **Pro REST** endpoint
    (`X-CMC_PRO_API_KEY`), the **CMC CLI** (a Go wrapper over the same REST), and
    the **CMC Agent Hub MCP** (which adds **x402** pay-per-request billing in the
    path). The loop uses **plain Pro REST** — it is the lowest-dependency surface,
    needs no MCP runtime, and the project key already covers the whole universe on
    the included credit budget. **x402 is recon-only this engagement** (no payments):
    it matters only if a future need exceeds the free credit tier, at which point
    the Agent Hub MCP's pay-per-call path becomes the fallback. Not needed now.

Fail-closed: a missing key, HTTP error, or a symbol with no parseable USD price
raises / is omitted — the caller (the loop's read step) must treat a missing price
as "no observation for that symbol this tick", never as a zero.

Fetch is stdlib urllib (no new deps); parsing helpers are pure and unit-testable.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass

BASE = "https://pro-api.coinmarketcap.com"
QUOTES_LATEST = "/v2/cryptocurrency/quotes/latest"


class CmcError(RuntimeError):
    """A CMC request failed (HTTP error, transport error, or unparseable body)."""


@dataclass(frozen=True)
class Quote:
    """One symbol's live USD quote — the loop's per-tick observation for an asset."""

    symbol: str
    price_usd: float
    pct_change_24h: float | None = None
    volume_24h_usd: float | None = None
    last_updated: str | None = None


def _get(endpoint: str, params: dict, api_key: str, timeout: int = 40) -> dict:
    url = f"{BASE}{endpoint}?{urllib.parse.urlencode(params)}"
    req = urllib.request.Request(
        url, headers={"X-CMC_PRO_API_KEY": api_key, "Accept": "application/json"}
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read())
    except urllib.error.HTTPError as e:  # surface CMC's structured error message
        try:
            body = json.loads(e.read())
            msg = (body.get("status") or {}).get("error_message") or str(e)
        except Exception:  # noqa: BLE001
            msg = str(e)
        raise CmcError(f"CMC HTTP {e.code}: {msg}") from e
    except (urllib.error.URLError, TimeoutError, OSError) as e:
        raise CmcError(f"CMC transport error: {type(e).__name__}: {e}") from e


def _chunks(seq: list[str], n: int):
    for i in range(0, len(seq), n):
        yield seq[i:i + n]


# --- pure parsing ----------------------------------------------------------

def _valid_price(rec: dict, convert: str) -> float | None:
    q = ((rec.get("quote") or {}).get(convert)) or {}
    price = q.get("price")
    if not isinstance(price, (int, float)) or price != price or price <= 0:
        return None  # NaN/None/non-positive
    return float(price)


def _pick_canonical(recs: list[dict], convert: str) -> dict | None:
    """The real token among ticker collisions, restricted to records with a valid USD price.

    CMC v2 maps a symbol to a *list* of records (e.g. "BNB" -> BNB, BNB AI, BNBTiger,
    BeanBox); blindly taking `recs[0]` once resolved BNB to an inactive null-price coin.
    Mirror `cmc.pick_canonical`: prefer active records, then best `cmc_rank`. Only records
    that actually carry a price are eligible (a rank-4 coin with a missing quote loses to a
    priced one — but in practice the canonical major is both top-ranked and priced).
    """
    priced = [r for r in recs if isinstance(r, dict) and _valid_price(r, convert) is not None]
    if not priced:
        return None
    active = [r for r in priced if r.get("is_active", 1)]
    pool = active or priced
    return min(pool, key=lambda r: (r.get("cmc_rank") is None, r.get("cmc_rank") or 1e12))


def parse_quotes(payload: dict, convert: str = "USD") -> dict[str, Quote]:
    """Reduce a `quotes/latest` response to `{SYMBOL_UPPER: Quote}`.

    On ticker collisions the canonical (active, best-rank, priced) record wins via
    `_pick_canonical` — never a blind `[0]`. A symbol with no priced record is omitted
    (no false zero — the loop reads a missing key as "no observation"). Tolerant of both
    list and dict `data` shapes.
    """
    out: dict[str, Quote] = {}
    data = payload.get("data") or {}
    for sym, recs in data.items():
        recs = recs if isinstance(recs, list) else [recs]
        rec = _pick_canonical(recs, convert)
        if rec is None:
            continue
        q = (rec.get("quote") or {}).get(convert) or {}
        out[str(sym).upper()] = Quote(
            symbol=str(rec.get("symbol") or sym).upper(),
            price_usd=float(q["price"]),
            pct_change_24h=q.get("percent_change_24h"),
            volume_24h_usd=q.get("volume_24h"),
            last_updated=q.get("last_updated"),
        )
    return out


# --- fetch -----------------------------------------------------------------

def fetch_quotes(symbols: list[str], api_key: str, *, convert: str = "USD",
                 batch: int = 100, timeout: int = 40) -> dict[str, Quote]:
    """Live USD quotes for `symbols` -> `{SYMBOL_UPPER: Quote}`.

    Non-ASCII tickers (a handful of the eligible universe are CJK/emoji) are dropped
    before the request — CMC 400s the whole comma-batch on one bad symbol. Batched in
    100s; each batch is 1 credit. Symbols with no parseable USD price are simply absent
    from the result. Raises CmcError only on a transport/HTTP failure of the request
    itself (fail closed at the call site).
    """
    clean = [s for s in symbols if s and s.isascii()]
    out: dict[str, Quote] = {}
    for chunk in _chunks(clean, batch):
        payload = _get(QUOTES_LATEST, {"symbol": ",".join(chunk), "convert": convert},
                       api_key, timeout=timeout)
        out.update(parse_quotes(payload, convert=convert))
    return out
