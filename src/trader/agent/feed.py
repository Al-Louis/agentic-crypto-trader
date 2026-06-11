"""Price feeds — the loop's READ step behind one interface.

`PriceFeed.prices(symbols) -> {SYMBOL: usd_price}` is all the loop needs to read the
market each tick. Two implementations:

  * `CmcPriceFeed` — live CMC `quotes/latest` (the proven feed, [[Tech Stack]] §data).
    One batched call per tick, 1 credit. A missing symbol is simply absent (no zero).
  * `FakeFeed` — deterministic scripted prices for tests (no network), so the whole
    loop is testable offline ([[Simulated Market]] discipline: validate offline first).

The contract is intentionally tiny — swapping CMC for GeckoTerminal/DexScreener (the
proven keyless fallback if CMC ever tier-gates the loop) is a new `PriceFeed`, nothing
in `loop.py` changes. The loop needs A feed, not a specific vendor.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from trader.data import cmc_market


@runtime_checkable
class PriceFeed(Protocol):
    """A read-only source of live USD prices keyed by symbol."""

    name: str

    def prices(self, symbols: list[str]) -> dict[str, float]:
        """`{SYMBOL_UPPER: usd_price}` for the symbols that resolved (others omitted)."""
        ...


class CmcPriceFeed:
    """Live CMC quotes feed. Holds the API key; raises `cmc_market.CmcError` on transport
    failure so the loop can treat a failed read as a *skipped tick* (fail closed — no
    decision on stale/empty data), not as zero prices."""

    name = "cmc"

    def __init__(self, api_key: str, *, convert: str = "USD"):
        self._key = api_key
        self._convert = convert

    def prices(self, symbols: list[str]) -> dict[str, float]:
        quotes = cmc_market.fetch_quotes(symbols, self._key, convert=self._convert)
        return {sym: q.price_usd for sym, q in quotes.items()}


class FakeFeed:
    """Deterministic in-memory feed for tests. `script` is a list of price dicts, one per
    tick; the last is repeated if the loop runs past the script. Symbols absent from a
    tick's dict are omitted (exercises the missing-observation path)."""

    name = "fake"

    def __init__(self, script: list[dict[str, float]]):
        if not script:
            raise ValueError("FakeFeed needs at least one price frame")
        self._script = [dict(frame) for frame in script]
        self._i = 0

    def prices(self, symbols: list[str]) -> dict[str, float]:
        frame = self._script[min(self._i, len(self._script) - 1)]
        self._i += 1
        wanted = {s.upper() for s in symbols}
        return {k.upper(): float(v) for k, v in frame.items() if k.upper() in wanted}
