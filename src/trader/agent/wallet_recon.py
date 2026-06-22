"""On-chain wallet reconciliation -> `trading/wallet.json`: the ACTUAL competition-wallet equity +
PnL, read from chain — NOT the $10k env book the rest of `trading/*` reports.

Why this exists: `equity.json`/`status.json` are the env's $10k cold-weekly BOOK (the strategy's
notional performance, what the leaderboard ranks). The real wallet is ~$100 and is what the
competition actually scores. This layer reads the real holdings on-chain, prices them with the SAME
OHLCV source the model + candle feed use (parity), and publishes the real equity/PnL.

Design: **read-only, fail-safe, additive, and FLAG-GATED** — nothing here runs unless the live
launcher is started with `--publish-wallet`, so the proven loop + every existing `trading/*` file stay
byte-identical until this is validated on the box. Holdings come from ERC-20 `balanceOf` over the
KNOWN universe (+ BNB via `eth_getBalance`, + USDT), so a token the strategy bought is captured even if
a wallet tool wouldn't list it. `build_wallet_payload` is pure (prices injected) for offline testing.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone

from trader.agent.candles import _round           # sig-fig rounding (chart/JSON precision)

BALANCEOF_SELECTOR = "0x70a08231"                  # keccak("balanceOf(address)")[:4]
DECIMALS_SELECTOR = "0x313ce567"                   # keccak("decimals()")[:4]
WEI = 10 ** 18                                     # BNB native decimals
USDT_BSC = "0x55d398326f99059fF775485246999027B3197955"   # BSC-USDT (BSC-USD), 18 decimals


def _erc20_balance(rpc, contract: str, address: str, decimals: int) -> float:
    """ERC-20 `balanceOf(address)` via `eth_call`, scaled by `decimals`. 0.0 on an empty/`0x` result."""
    data = BALANCEOF_SELECTOR + address.lower().replace("0x", "").rjust(64, "0")
    raw = rpc.eth_call(contract, data)
    return (int(raw, 16) / (10 ** decimals)) if raw and raw not in ("0x", "0x0") else 0.0


def _erc20_decimals(rpc, contract: str) -> int:
    """ERC-20 `decimals()` via `eth_call` (default 18 if the call returns nothing)."""
    raw = rpc.eth_call(contract, DECIMALS_SELECTOR)
    return int(raw, 16) if raw and raw not in ("0x", "0x0") else 18


def read_holdings_onchain(address: str, assets: list[dict], *, rpc=None) -> dict:
    """Real wallet holdings `{symbol: qty}` read on-chain: native BNB (`eth_getBalance`) + each asset's
    ERC-20 `balanceOf`. `assets` = `[{symbol, contract[, decimals]}]` (USDT is just one asset). Decimals
    are read once per token if not provided. Network — injected `rpc` in tests."""
    if rpc is None:
        from trader.chain.rpc import BscRpc          # noqa: PLC0415 (network)
        rpc = BscRpc()
    out: dict[str, float] = {}
    bnb = rpc.call("eth_getBalance", [address, "latest"])
    out["BNB"] = (int(bnb, 16) / WEI) if bnb and bnb not in ("0x", "0x0") else 0.0
    for a in assets:
        c, sym = a.get("contract"), a.get("symbol")
        if not c or not sym:
            continue
        dec = a.get("decimals")
        if dec is None:
            dec = _erc20_decimals(rpc, c)
        out[str(sym)] = _erc20_balance(rpc, c, address, int(dec))
    return out


def build_wallet_payload(holdings: dict, prices: dict, *, baseline_usd: float | None,
                         address: str, generated: str | None = None, stale: bool = False) -> dict:
    """PURE: given `{symbol: qty}` holdings + `{symbol: usd_price}`, compute real equity, PnL vs
    `baseline_usd` (the funded cost basis; None -> PnL omitted), and a per-token breakdown. USDT is
    priced at 1.0; a missing price -> that holding's value is null and excluded from equity (and the
    payload is flagged so the frontend can show it's incomplete)."""
    generated = generated or datetime.now(timezone.utc).isoformat()
    rows, equity, missing = [], 0.0, False
    for sym, qty in holdings.items():
        px = 1.0 if str(sym).upper() == "USDT" else prices.get(sym)
        val = (float(qty) * float(px)) if (px is not None) else None
        if val is None:
            missing = True
        else:
            equity += val
        rows.append({"token": sym, "qty": _round(float(qty)),
                     "price_usd": (_round(float(px)) if px is not None else None),
                     "value_usd": (_round(val) if val is not None else None)})
    pnl = (equity - baseline_usd) if baseline_usd else None
    pnl_pct = (pnl / baseline_usd * 100.0) if (baseline_usd and pnl is not None) else None
    return {"generated": generated, "address": address, "source": "onchain",
            "stale": bool(stale or missing),
            "equity_usd": _round(equity), "baseline_usd": baseline_usd,
            "pnl_usd": (_round(pnl) if pnl is not None else None),
            "pnl_pct": (round(pnl_pct, 2) if pnl_pct is not None else None),
            "holdings": sorted(rows, key=lambda r: -(r["value_usd"] or 0.0))}


def publish_wallet(target: str, *, address: str, assets: list[dict], prices: dict,
                   baseline_usd: float | None, holdings_fn=read_holdings_onchain,
                   generated: str | None = None) -> dict:
    """Read on-chain holdings, build the payload, and PUT `<target>/wallet.json` (no-cache, same
    put-only path as the other feeds). Returns the payload. Raises on a read/put failure — the caller
    wraps it fail-safe (a wallet-recon error must never stop a trading tick)."""
    from remote_train.publish import join, put_bytes   # noqa: PLC0415 — boto3 stays optional
    holdings = holdings_fn(address, assets)
    payload = build_wallet_payload(holdings, prices, baseline_usd=baseline_usd, address=address,
                                   generated=generated)
    put_bytes(join(target, "wallet.json"),
              json.dumps(payload, separators=(",", ":")).encode("utf-8"),
              content_type="application/json", cache_control="no-cache")
    return payload
