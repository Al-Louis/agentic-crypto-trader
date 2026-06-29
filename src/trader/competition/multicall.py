"""Multicall3 `aggregate3` ABI encode/decode + a batched balance reader (stdlib-only).

Collapses N token `balanceOf` reads (+ native BNB via Multicall3's own `getEthBalance`) into ~1
`eth_call` per wallet per block. Works at an arbitrary block tag, so it serves both the live snapshot
(`latest`) and the baseline read at the June-22-00:00-UTC start block (against an archive RPC). The
`eth_call` transport is injected (`call(method, params)`) so encode/decode stay pure + unit-testable.

ABI background — `aggregate3(Call3[])` where `Call3 = (address target, bool allowFailure, bytes
callData)` returns `Result[]` where `Result = (bool success, bytes returnData)`. Selector lookups are
the standard keccak[:4] values, hard-coded (no eth-abi dependency).
"""

from __future__ import annotations

MULTICALL3 = "0xcA11bde05977b3631167028862bE2a173976CA11"  # canonical, same on every EVM chain

AGGREGATE3 = "82ad56cb"      # aggregate3((address,bool,bytes)[])
BALANCE_OF = "70a08231"      # balanceOf(address)
GET_ETH_BALANCE = "4d2301cc"  # Multicall3.getEthBalance(address)
DECIMALS = "313ce567"        # decimals()


def _u(n: int) -> str:
    return f"{int(n):064x}"


def _addr(a: str) -> str:
    return a.lower().replace("0x", "").rjust(64, "0")


def _bytes(data_hex: str) -> str:
    """ABI-encode a `bytes` value: length word + right-padded data."""
    b = data_hex.lower().replace("0x", "")
    nbytes = len(b) // 2
    pad = (64 - len(b) % 64) % 64
    return _u(nbytes) + b + "0" * pad


def calldata_balance_of(wallet: str) -> str:
    return BALANCE_OF + _addr(wallet)


def calldata_get_eth_balance(wallet: str) -> str:
    return GET_ETH_BALANCE + _addr(wallet)


def calldata_decimals() -> str:
    return DECIMALS


def encode_aggregate3(calls: list[tuple[str, str]]) -> str:
    """`calls` = `[(target_address, calldata_hex), ...]` (allowFailure is always true so one bad token
    can't sink the batch). Returns the full `eth_call` data hex (`0x…`)."""
    n = len(calls)
    heads, tails, running = [], [], n * 32  # tails start after the n head words
    for target, data in calls:
        tup = _addr(target) + _u(1) + _u(0x60) + _bytes(data)  # (addr, allowFailure=1, off=0x60, bytes)
        heads.append(_u(running))
        tails.append(tup)
        running += len(tup) // 2
    array_body = _u(n) + "".join(heads) + "".join(tails)
    return "0x" + AGGREGATE3 + _u(0x20) + array_body


def decode_aggregate3(result_hex: str) -> list[tuple[bool, str]]:
    """Decode the `Result[]` return into `[(success, returnData_hex), ...]`."""
    h = result_hex.lower().replace("0x", "")

    def word(i: int) -> str:
        return h[i * 64:(i + 1) * 64]

    arr_off = int(word(0), 16) // 32
    n = int(word(arr_off), 16)
    base = arr_off + 1                         # first head word of the array's element region
    out: list[tuple[bool, str]] = []
    for i in range(n):
        tup = base + int(word(base + i), 16) // 32
        success = bool(int(word(tup), 16))
        ret_off = int(word(tup + 1), 16) // 32
        ret_len = int(word(tup + ret_off), 16)
        start = (tup + ret_off + 1) * 64
        out.append((success, "0x" + h[start:start + ret_len * 2]))
    return out


def _to_int(ret_hex: str) -> int | None:
    b = ret_hex.replace("0x", "")
    return int(b, 16) if b else None


def multicall_values(call, calls: list[tuple[str, str]], *, block: str = "latest",
                     chunk: int = 120, mc3: str = MULTICALL3) -> list[tuple[bool, str]]:
    """Run `calls` through Multicall3 in chunks, returning `(success, returnData)` per call in order.
    `call(method, params)` is the injected JSON-RPC transport (e.g. `BscRpc.call`)."""
    results: list[tuple[bool, str]] = []
    for i in range(0, len(calls), chunk):
        batch = calls[i:i + chunk]
        data = encode_aggregate3(batch)
        raw = call("eth_call", [{"to": mc3, "data": data}, block])
        results.extend(decode_aggregate3(raw))
    return results


def read_decimals(call, tokens: list[dict], *, block: str = "latest") -> dict[str, int]:
    """`{address_lower: decimals}` for tokens with a `contract`/`token_address`/`address` field.
    Defaults to 18 when `decimals()` reverts or returns empty (common for proxy/odd tokens)."""
    addrs = [_token_addr(t) for t in tokens]
    addrs = [a for a in addrs if a]
    res = multicall_values(call, [(a, calldata_decimals()) for a in addrs], block=block)
    out: dict[str, int] = {}
    for a, (ok, ret) in zip(addrs, res):
        v = _to_int(ret) if ok else None
        out[a.lower()] = int(v) if (v is not None and 0 <= v <= 36) else 18
    return out


def _token_addr(t: dict) -> str | None:
    return t.get("contract") or t.get("token_address") or t.get("address")


def read_holdings(call, wallet: str, tokens: list[dict], decimals: dict[str, int], *,
                  block: str = "latest", include_bnb: bool = True) -> dict[str, float]:
    """Read `{symbol: qty}` for `wallet` over `tokens` (+ native BNB) at `block`, scaled by decimals.
    `tokens` = `[{symbol, contract|token_address|address}, ...]`. A failed/zero call -> 0.0 (kept, so
    the symbol still appears). Pure given the injected `call`."""
    calls: list[tuple[str, str]] = []
    syms: list[str] = []
    for t in tokens:
        a = _token_addr(t)
        if not a or not t.get("symbol"):
            continue
        calls.append((a, calldata_balance_of(wallet)))
        syms.append(str(t["symbol"]))
    if include_bnb:
        calls.append((MULTICALL3, calldata_get_eth_balance(wallet)))
        syms.append("BNB")

    res = multicall_values(call, calls, block=block)
    out: dict[str, float] = {}
    for t, sym, (ok, ret) in zip(tokens + ([{}] if include_bnb else []), syms, res):
        raw = _to_int(ret) if ok else None
        if sym == "BNB":
            dec = 18
        else:
            a = (_token_addr(t) or "").lower()
            dec = decimals.get(a, 18)
        out[sym] = (raw / (10 ** dec)) if raw else 0.0
    return out
