"""Pool registry — per-pool metadata the decoders and panels need.

Built once from ``data/selection.json`` plus on-chain probes (``slot0()``
succeeds -> V3, else ``getReserves()`` -> V2; ``token0/token1`` orientation;
ERC-20 ``decimals`` for both sides) and cached to ``data/chain/_pools.json``.

``token_side`` is which side (0/1) of the pool is *our* universe token; the
other side is the quote (WBNB/BTCB/USDT/USDC/USD1). ``quote_anchor`` names
the USD-conversion series: "USD" for stables, else the anchor pair symbol.
"""

from __future__ import annotations

import json
import os

from trader.chain.rpc import BscRpc, RpcError

DEFAULT_PATH = os.path.join("data", "chain", "_pools.json")

_SLOT0 = "0x3850c7bd"
_GET_RESERVES = "0x0902f1ac"
_TOKEN0 = "0x0dfe1681"
_TOKEN1 = "0xd21220a7"
_FEE = "0xddca3f43"
_DECIMALS = "0x313ce567"

# quote symbol -> how to get USD: 1.0 for stables, else anchor close series
QUOTE_ANCHOR = {"USDT": "USD", "USDC": "USD", "USD1": "USD",
                "WBNB": "BNB_USDT", "BTCB": "BTC_USDT"}


def _probe(rpc: BscRpc, to: str, data: str) -> str | None:
    """eth_call that treats a revert as 'this contract lacks that method'
    (some endpoints return '0x' for reverts, others raise code 3)."""
    try:
        res = rpc.eth_call(to, data)
    except RpcError as e:
        if "revert" in (e.message or "").lower():
            return None
        raise
    return res if res and res != "0x" else None


def build_registry(selection_path: str = os.path.join("data", "selection.json"),
                   out_path: str = DEFAULT_PATH, rpc: BscRpc | None = None,
                   logger=print) -> list[dict]:
    rpc = rpc or BscRpc()
    sel = json.load(open(selection_path, encoding="utf-8"))
    dec_cache: dict[str, int] = {}

    def decimals(token: str) -> int:
        t = token.lower()
        if t not in dec_cache:
            dec_cache[t] = int(rpc.eth_call(token, _DECIMALS), 16)
        return dec_cache[t]

    pools = []
    for s in sel:
        pool, tok = s["pair_address"], s["token_address"].lower()
        version = "v3" if _probe(rpc, pool, _SLOT0) else (
            "v2" if _probe(rpc, pool, _GET_RESERVES) else None)
        if version is None:
            raise RuntimeError(f"{s['symbol']}: pool {pool} answers neither slot0 nor getReserves")
        token0 = "0x" + rpc.eth_call(pool, _TOKEN0)[-40:]
        token1 = "0x" + rpc.eth_call(pool, _TOKEN1)[-40:]
        side = 0 if token0.lower() == tok else (1 if token1.lower() == tok else None)
        if side is None:
            raise RuntimeError(f"{s['symbol']}: token {tok} is neither side of pool {pool}")
        fee = int(rpc.eth_call(pool, _FEE), 16) if version == "v3" else 2500
        ent = {
            "symbol": s["symbol"], "pool": pool, "version": version,
            "token_side": side, "token0": token0, "token1": token1,
            "dec0": decimals(token0), "dec1": decimals(token1),
            "fee_ppm": fee, "quote": s["quote"],
            "quote_anchor": QUOTE_ANCHOR.get(s["quote"], "USD"),
            "tier": s.get("tier"),
        }
        pools.append(ent)
        logger(f"  {ent['symbol']:10} {version} side={side} "
               f"dec=({ent['dec0']},{ent['dec1']}) quote={ent['quote']}")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    tmp = out_path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(pools, f, indent=1)
    os.replace(tmp, out_path)
    return pools


def load_registry(path: str = DEFAULT_PATH) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)
