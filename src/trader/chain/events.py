"""PancakeSwap pair/pool event signatures and decoders -> unified flat rows.

The universe mixes PancakeSwap **V2** pairs (Sync/Swap/Mint/Burn) and **V3**
pools (Swap/Mint/Burn/Collect — Pancake's V3 Swap adds two protocol-fee
fields, so its topic0 differs from Uniswap V3's; confirmed empirically on the
Q pool). All amounts are decoded to a single **pool-perspective** convention:

    positive = tokens INTO the pool, negative = tokens OUT of the pool.

So a trader *buying* the token shows the token side negative (token left the
pool). On V3, ``Burn`` is the liquidity-removal *decision* (no tokens move)
and ``Collect`` is the actual withdrawal transfer (principal + fees) — both
are kept, with distinct event labels, so probes can choose the signal.
Amounts are decimal-normalized floats (research precision, not accounting).

Unified row columns (absent fields are None):
    block, log_index, tx_hash, event, a0, a1, liquidity, price1per0, tick,
    r0, r1, amount_l, sender, recipient
"""

from __future__ import annotations

# --- topic0 ---------------------------------------------------------------

V2_SYNC = "0x1c411e9a96e071241c2f21f7726b17ae89e3cab4c78be50e062b03a9fffbbad1"
V2_SWAP = "0xd78ad95fa46c994b6551d0da85fc275fe613ce37657fb8d5e3d130840159d822"
V2_MINT = "0x4c209b5fc8ad50758f13e2e1088ba56a560dff690a1c6fef26394f4c03821c4f"
V2_BURN = "0xdccd412f0b1252819cb1fd330b93224ca42612892bb3f4f789976e6d81936496"

# PancakeSwap V3: Swap(...,uint128 protocolFeesToken0, uint128 protocolFeesToken1)
V3_SWAP = "0x19b47279256b2a23a1665c810c8d55a1758940ee09377d4f8d26497a3577dc83"
V3_MINT = "0x7a53080ba414158be7ec69b987b5fb7d07dee101fe85488f0853ae16239d0bde"
V3_BURN = "0x0c396cd989a39f4459b5fa1aed6a9a8dcdbc45908acfd67e028cd568da98982c"
V3_COLLECT = "0x70935338e69775456a85ddef226c395fb668b63fa0115f5f20610b388e6ca9c0"

ALL_TOPICS = [V2_SYNC, V2_SWAP, V2_MINT, V2_BURN,
              V3_SWAP, V3_MINT, V3_BURN, V3_COLLECT]

ROW_COLUMNS = ["block", "log_index", "tx_hash", "event", "a0", "a1",
               "liquidity", "price1per0", "tick", "r0", "r1", "amount_l",
               "sender", "recipient"]

_TWO_96 = 2 ** 96


def _u(data: str, i: int) -> int:
    """i-th 32-byte word of the data field as unsigned int."""
    return int(data[2 + 64 * i: 2 + 64 * (i + 1)], 16)


def _signed(v: int, bits: int) -> int:
    return v - (1 << bits) if v >= (1 << (bits - 1)) else v


def _addr_topic(topic: str) -> str:
    return "0x" + topic[-40:]


def _addr_word(data: str, i: int) -> str:
    return "0x" + data[2 + 64 * i + 24: 2 + 64 * (i + 1)]


def sqrt_price_to_price(sqrt_price_x96: int, dec0: int, dec1: int) -> float:
    """token1-per-token0 price, decimal-adjusted."""
    p = (sqrt_price_x96 / _TWO_96) ** 2
    return p * (10 ** (dec0 - dec1))


def decode_log(log: dict, dec0: int, dec1: int) -> dict | None:
    """Decode one eth_getLogs entry into a unified row (None = not ours)."""
    t0 = log["topics"][0].lower()
    data = log["data"]
    s0, s1 = 10.0 ** dec0, 10.0 ** dec1
    row = {
        "block": int(log["blockNumber"], 16),
        "log_index": int(log["logIndex"], 16),
        "tx_hash": log["transactionHash"],
        "event": None, "a0": None, "a1": None, "liquidity": None,
        "price1per0": None, "tick": None, "r0": None, "r1": None,
        "amount_l": None, "sender": None, "recipient": None,
    }
    tp = log["topics"]

    if t0 == V2_SYNC:
        # Sync(uint112 reserve0, uint112 reserve1)
        row.update(event="sync", r0=_u(data, 0) / s0, r1=_u(data, 1) / s1)
    elif t0 == V2_SWAP:
        # Swap(idx sender, a0In, a1In, a0Out, a1Out, idx to)
        a0in, a1in, a0out, a1out = (_u(data, i) for i in range(4))
        row.update(event="swap", a0=(a0in - a0out) / s0, a1=(a1in - a1out) / s1,
                   sender=_addr_topic(tp[1]), recipient=_addr_topic(tp[2]))
    elif t0 == V2_MINT:
        # Mint(idx sender, amount0, amount1)
        row.update(event="mint", a0=_u(data, 0) / s0, a1=_u(data, 1) / s1,
                   sender=_addr_topic(tp[1]))
    elif t0 == V2_BURN:
        # Burn(idx sender, amount0, amount1, idx to)
        row.update(event="burn", a0=-_u(data, 0) / s0, a1=-_u(data, 1) / s1,
                   sender=_addr_topic(tp[1]), recipient=_addr_topic(tp[2]))
    elif t0 == V3_SWAP:
        # Swap(idx sender, idx recipient, int256 a0, int256 a1, uint160 sqrtP,
        #      uint128 L, int24 tick, uint128 pFee0, uint128 pFee1)
        amount0 = _signed(_u(data, 0), 256)
        amount1 = _signed(_u(data, 1), 256)
        row.update(event="swap", a0=amount0 / s0, a1=amount1 / s1,
                   liquidity=_u(data, 3) / (10 ** ((dec0 + dec1) / 2)),
                   price1per0=sqrt_price_to_price(_u(data, 2), dec0, dec1),
                   tick=_signed(_u(data, 4) & 0xFFFFFF, 24),
                   sender=_addr_topic(tp[1]), recipient=_addr_topic(tp[2]))
    elif t0 == V3_MINT:
        # Mint(address sender, idx owner, idx tickLower, idx tickUpper,
        #      uint128 amount, amount0, amount1) — data: [sender, L, a0, a1]
        row.update(event="mint", a0=_u(data, 2) / s0, a1=_u(data, 3) / s1,
                   amount_l=float(_u(data, 1)),
                   sender=_addr_topic(tp[1]))  # owner — the LP wallet/manager
    elif t0 == V3_BURN:
        # Burn(idx owner, idx tickLower, idx tickUpper, uint128 amount, a0, a1)
        row.update(event="burn", a0=-_u(data, 1) / s0, a1=-_u(data, 2) / s1,
                   amount_l=-float(_u(data, 0)), sender=_addr_topic(tp[1]))
    elif t0 == V3_COLLECT:
        # Collect(idx owner, address recipient, idx tickLower, idx tickUpper,
        #         uint128 amount0, uint128 amount1) — data: [recipient, a0, a1]
        row.update(event="collect", a0=-_u(data, 1) / s0, a1=-_u(data, 2) / s1,
                   sender=_addr_topic(tp[1]), recipient=_addr_word(data, 0))
    else:
        return None
    return row
