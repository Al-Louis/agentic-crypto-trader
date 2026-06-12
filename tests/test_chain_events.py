"""Decoder tests for trader.chain.events — synthetic logs with known payloads."""

import math

from trader.chain import events


def _w(v: int, bits: int = 256) -> str:
    """One 32-byte ABI word, two's complement for negatives."""
    return format(v & ((1 << 256) - 1), "064x")


def _log(topic0, topics_rest, data_words, block=100, idx=3):
    return {
        "blockNumber": hex(block),
        "logIndex": hex(idx),
        "transactionHash": "0x" + "ab" * 32,
        "topics": [topic0] + topics_rest,
        "data": "0x" + "".join(data_words),
    }


ADDR_A = "0x" + "11" * 20
ADDR_B = "0x" + "22" * 20
TOPIC_A = "0x" + "00" * 12 + "11" * 20
TOPIC_B = "0x" + "00" * 12 + "22" * 20


def test_v2_sync():
    lg = _log(events.V2_SYNC, [], [_w(5 * 10 ** 18), _w(2 * 10 ** 18)])
    r = events.decode_log(lg, 18, 18)
    assert r["event"] == "sync"
    assert r["r0"] == 5.0 and r["r1"] == 2.0
    assert r["block"] == 100 and r["log_index"] == 3


def test_v2_swap_pool_perspective_sign():
    # trader sells 10 token0 in, receives 4 token1 out
    lg = _log(events.V2_SWAP, [TOPIC_A, TOPIC_B],
              [_w(10 * 10 ** 18), _w(0), _w(0), _w(4 * 10 ** 18)])
    r = events.decode_log(lg, 18, 18)
    assert r["event"] == "swap"
    assert r["a0"] == 10.0          # into pool
    assert r["a1"] == -4.0          # out of pool
    assert r["sender"] == ADDR_A and r["recipient"] == ADDR_B


def test_v2_burn_negative():
    lg = _log(events.V2_BURN, [TOPIC_A, TOPIC_B],
              [_w(3 * 10 ** 18), _w(7 * 10 ** 18)])
    r = events.decode_log(lg, 18, 18)
    assert r["event"] == "burn"
    assert r["a0"] == -3.0 and r["a1"] == -7.0


def test_v3_swap_signed_amounts_and_price():
    # buy: 100 token0 (USDT) in, 50 token1 out; sqrtPriceX96 for price 2.0
    sqrt_px = int(math.sqrt(2.0) * 2 ** 96)
    lg = _log(events.V3_SWAP, [TOPIC_A, TOPIC_B],
              [_w(100 * 10 ** 18), _w(-50 * 10 ** 18), _w(sqrt_px),
               _w(9 * 10 ** 18), _w(-887272), _w(0), _w(0)])
    r = events.decode_log(lg, 18, 18)
    assert r["a0"] == 100.0 and r["a1"] == -50.0
    assert abs(r["price1per0"] - 2.0) < 1e-9
    assert r["tick"] == -887272
    assert r["liquidity"] == 9.0


def test_v3_swap_decimal_adjustment():
    # dec0=6 (e.g. XAUt), dec1=18: price1per0 must scale by 10**(6-18)
    sqrt_px = int(math.sqrt(4.0e12) * 2 ** 96)   # raw price 4e12 -> adjusted 4.0
    lg = _log(events.V3_SWAP, [TOPIC_A, TOPIC_B],
              [_w(10 ** 6), _w(-4 * 10 ** 18), _w(sqrt_px), _w(0), _w(0), _w(0), _w(0)])
    r = events.decode_log(lg, 6, 18)
    assert r["a0"] == 1.0 and r["a1"] == -4.0
    assert abs(r["price1per0"] - 4.0) < 1e-6


def test_v3_mint_layout():
    # data: [sender, L, amount0, amount1]; owner is topics[1]
    lg = _log(events.V3_MINT, [TOPIC_A, _w(-100), _w(200)],
              [_w(int(ADDR_B, 16)), _w(7 * 10 ** 18), _w(2 * 10 ** 18), _w(3 * 10 ** 18)])
    r = events.decode_log(lg, 18, 18)
    assert r["event"] == "mint"
    assert r["a0"] == 2.0 and r["a1"] == 3.0
    assert r["amount_l"] == float(7 * 10 ** 18)
    assert r["sender"] == ADDR_A                 # the position OWNER


def test_v3_burn_negative():
    lg = _log(events.V3_BURN, [TOPIC_A, _w(-100), _w(200)],
              [_w(5 * 10 ** 18), _w(1 * 10 ** 18), _w(6 * 10 ** 18)])
    r = events.decode_log(lg, 18, 18)
    assert r["a0"] == -1.0 and r["a1"] == -6.0
    assert r["amount_l"] == -float(5 * 10 ** 18)


def test_v3_collect_recipient_from_data():
    lg = _log(events.V3_COLLECT, [TOPIC_A, _w(-100), _w(200)],
              [_w(int(ADDR_B, 16)), _w(2 * 10 ** 18), _w(10 ** 18)])
    r = events.decode_log(lg, 18, 18)
    assert r["event"] == "collect"
    assert r["a0"] == -2.0 and r["a1"] == -1.0
    assert r["recipient"] == ADDR_B


def test_unknown_topic_returns_none():
    lg = _log("0x" + "ff" * 32, [], [_w(1)])
    assert events.decode_log(lg, 18, 18) is None
