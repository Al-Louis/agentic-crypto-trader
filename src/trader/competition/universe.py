"""The eligible-token set participants can hold — the `balanceOf` universe.

Source: `data/resolved.json` (the screening step's symbol -> BSC-contract resolution, 148 eligible
tokens). MUST be read as UTF-8 (it contains a CJK ticker; the cp1252 default crashes). Native BNB is
handled separately (Multicall3 `getEthBalance`), not as an ERC-20 here.
"""

from __future__ import annotations

import json

from trader.data.eligible import STABLES

USDT_BSC = "0x55d398326f99059fF775485246999027B3197955"  # canonical BSC-USDT, 18 decimals

# Canonical BSC contracts for majors the screening step mis-resolved onto OTHER tokens' contracts
# (ticker collisions verified on-chain 2026-06-22): resolved.json mapped USDC->UB's contract,
# TRX->HTX's, USDF->USDf's, so balanceOf double-counted the real holder. Force the real address so
# each token reads its own balance, and the dedupe-by-contract net below kills any residual collision.
CANONICAL = {
    "USDT": USDT_BSC,
    "USDC": "0x8AC76a51cc950d9822D68b83fE1Ad97B32Cd580d",
    "TRX": "0xCE7de646e7208a4Ef112cb6ed5038FA6cC6b12e3",
}


def load_universe(path: str = "data/resolved.json") -> list[dict]:
    """`[{symbol, contract, pair_address, is_stable, cmc_id}, ...]` for every resolved eligible token
    with a BSC contract. Majors are forced to their canonical contract (`CANONICAL`), then the list is
    **deduped by contract address** so a ticker collision can never make `balanceOf` count one holding
    twice (the USDC==UB bug). One symbol per contract; first occurrence wins."""
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)
    out: dict[str, dict] = {}
    for r in raw:
        sym = r.get("symbol")
        addr = CANONICAL.get(sym) or r.get("token_address") or r.get("bsc_contract")
        if not sym or not addr or r.get("status") not in (None, "resolved"):
            continue
        if sym in out:
            continue
        out[sym] = {
            "symbol": sym,
            "contract": addr,
            "pair_address": r.get("pair_address"),
            "is_stable": bool(r.get("is_stable")) or sym in STABLES,
            "cmc_id": r.get("cmc_id"),
        }
    for sym, addr in CANONICAL.items():                 # ensure majors present + canonical
        e = out.setdefault(sym, {"symbol": sym, "contract": addr, "pair_address": None,
                                 "is_stable": sym in STABLES or sym in ("USDT", "USDC"), "cmc_id": None})
        e["contract"] = addr

    # dedupe by contract — guards against any residual ticker collision (e.g. USDF sharing USDf's addr)
    seen: set[str] = set()
    deduped: list[dict] = []
    for u in out.values():
        c = (u["contract"] or "").lower()
        if not c or c in seen:
            continue
        seen.add(c)
        deduped.append(u)
    return deduped
