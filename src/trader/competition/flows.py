"""Deposit-proof cost basis from a wallet's transfer history (NodeReal `nr_getAssetTransfers`).

The leaderboard's honest metric is `PnL = current_equity − net_deposited_capital`, where deposits and
withdrawals are EXTERNAL capital flows, not DEX trades. Classification is **per-transaction**: a tx
where the wallet both sends and receives is a swap/trade (internal — excluded); a tx with only
inbound is a deposit, only outbound a withdrawal. Because NodeReal surfaces native-BNB legs
(external + internal), even BNB↔token swaps show both sides and classify correctly.

Cost basis counts only **fundable assets** — stablecoins (USD 1.0) and BNB (priced at deposit time) —
because agents are funded in USDT/BNB. Random token inflows (airdrops) are NOT treated as invested
capital (they surface as gains if later sold), so they can't distort the denominator.
"""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone

from trader.data.eligible import STABLES


def _utc_day(ts: int) -> str:
    return datetime.fromtimestamp(int(ts), timezone.utc).strftime("%Y-%m-%d")


def _fundable_usd(leg: dict, *, bnb_price_now: float | None,
                  bnb_price_at=None) -> float | None:
    """USD value of a transfer leg IF it is fundable capital (stablecoin or BNB), else None."""
    sym = (leg.get("asset") or "").upper()
    qty = float(leg.get("qty") or 0.0)
    if qty <= 0:
        return None
    if sym == "USDT" or sym in STABLES:
        return qty
    if sym == "BNB":
        px = None
        if bnb_price_at and leg.get("ts"):
            px = bnb_price_at(leg["ts"])
        px = px or bnb_price_now
        return qty * px if px else None
    return None


def _counted_usd(leg: dict, counted: set[str], prices: dict, *, bnb_price_now, bnb_price_at) -> float | None:
    """USD value of a leg IF it is a COUNTED asset (one our equity values: the eligible universe incl
    USDT/stables, + native/wrapped BNB), else None. `prices` is {symbol: usd} for eligible alts; an
    unpriced counted leg returns 0.0 (still counted, just no value)."""
    sym = (leg.get("asset") or "").upper()
    qty = float(leg.get("qty") or 0.0)
    if qty <= 0:
        return None
    if sym in ("BNB", "WBNB"):
        px = (bnb_price_at(leg["ts"]) if (bnb_price_at and leg.get("ts")) else None) or bnb_price_now
        return qty * px if px else 0.0
    if (leg.get("contract") or "").lower() in counted:
        if sym == "USDT" or sym in STABLES:
            return qty
        px = (prices or {}).get(leg.get("asset"))
        return qty * px if px else 0.0
    return None


def wallet_cost_basis(nr, wallet: str, *, from_block: int, to_block: int,
                      bnb_price_now: float | None, bnb_price_at=None,
                      eligible_contracts: set[str] | None = None,
                      counted_contracts: set[str] | None = None, prices: dict | None = None) -> dict:
    """Net deposited capital + trading activity for `wallet` over the block range.

    Capital flows (deposits/withdrawals) come from single-direction txs; swap txs (both directions)
    are trades. When `eligible_contracts` (lowercased non-stable eligible token addresses) is given,
    a swap that ACQUIRES one of those tokens counts as a real competition trade — the honest
    "is this wallet actually trading?" signal (vs hold-flat depositors and spam-airdrop farmers,
    whose swaps only ever produce BNB/stables).

    `trade_days` (UTC days with >=1 swap of ANY kind) drives the >=1-trade/day DQ rule; the
    `eligible`/`n_eligible_buys` signals drive ranking only — keep them separate.

    Returns `{net_deposited, gross_deposits, gross_withdrawals, n_deposits, n_withdrawals,
      first_funding_ts, nonfundable_deposit_assets, n_swaps, n_eligible_buys, traded_eligible,
      eligible_tokens_traded, trade_days, last_trade_ts}`."""
    inbound = nr.asset_transfers(to_address=wallet, from_block=from_block, to_block=to_block)
    outbound = nr.asset_transfers(from_address=wallet, from_block=from_block, to_block=to_block)
    for t in inbound:
        t["dir"] = "in"
    for t in outbound:
        t["dir"] = "out"
    eligible = eligible_contracts or set()
    counted = counted_contracts or eligible      # assets our equity values (eligible universe + BNB)

    by_hash: dict[str, list] = defaultdict(list)
    for t in inbound + outbound:
        if (t.get("qty") or 0) > 0 and t.get("hash"):
            by_hash[t["hash"]].append(t)

    deposits = withdrawals = boundary_flow = 0.0
    n_dep = n_wd = n_swaps = 0
    first_ts = None
    last_trade_ts = None
    nonfundable: list[str] = []
    eligible_traded: set[str] = set()
    # `trade_days` = UTC days with >=1 swap of ANY kind. The competition's daily-trade rule is just
    # ">=1 trade/day", and a swap is a trade — so a BNB<->USDT keepalive or any stable-pair swap MUST
    # count (excluding them falsely DQs compliant wallets). The non-stable-alt set (`eligible`) is used
    # ONLY for the ranking signals below (n_eligible_buys / traded_eligible), never for the DQ gate.
    trade_days: set[str] = set()
    for legs in by_hash.values():
        dirs = {leg["dir"] for leg in legs}
        if "in" in dirs and "out" in dirs:
            n_swaps += 1                           # swap/trade — not a capital flow
            ts = max((leg.get("ts") or 0) for leg in legs)
            if ts:                                 # ANY swap satisfies the daily-trade rule
                trade_days.add(_utc_day(ts))
                last_trade_ts = max(last_trade_ts or 0, ts)
            # BOUNDARY flow: a swap that converts an UNCOUNTED asset (BTCB, spam, anything outside the
            # eligible universe) <-> a COUNTED one injects/removes value our equity tracks but didn't
            # earn by eligible trading. Treat the net counted-value change as a CAPITAL FLOW (so e.g.
            # selling Bitcoin-held-at-open for USDT is capital in, not profit). Counted<->counted swaps
            # are pure trades (skipped).
            counted_in = counted_out = 0.0
            has_uncounted = False
            for leg in legs:                       # non-stable eligible ACQUISITION -> ranking only
                if (leg.get("contract") or "") in eligible and leg["dir"] == "in":
                    eligible_traded.add(leg.get("asset") or leg["contract"])
                cu = _counted_usd(leg, counted, prices or {},
                                  bnb_price_now=bnb_price_now, bnb_price_at=bnb_price_at)
                if cu is None:
                    has_uncounted = True
                elif leg["dir"] == "in":
                    counted_in += cu
                else:
                    counted_out += cu
            if has_uncounted:
                boundary_flow += counted_in - counted_out
            continue
        cap_dir = "in" if "in" in dirs else "out"
        for leg in legs:
            usd = _fundable_usd(leg, bnb_price_now=bnb_price_now, bnb_price_at=bnb_price_at)
            if usd is None:
                if cap_dir == "in":
                    nonfundable.append((leg.get("asset") or "?"))
                continue
            if cap_dir == "in":
                deposits += usd
                n_dep += 1
                if leg.get("ts") and (first_ts is None or leg["ts"] < first_ts):
                    first_ts = leg["ts"]
            else:
                withdrawals += usd
                n_wd += 1

    return {
        "net_deposited": round(deposits - withdrawals, 2),
        "boundary_flow": round(boundary_flow, 2),     # value crossing the eligible<->uncounted boundary
        "net_capital_in": round(deposits - withdrawals + boundary_flow, 2),   # total non-trading inflow
        "gross_deposits": round(deposits, 2),
        "gross_withdrawals": round(withdrawals, 2),
        "n_deposits": n_dep,
        "n_withdrawals": n_wd,
        "first_funding_ts": first_ts,
        "nonfundable_deposit_assets": sorted(set(nonfundable)),
        "n_swaps": n_swaps,
        "n_eligible_buys": len(eligible_traded),
        "traded_eligible": bool(eligible_traded),
        "eligible_tokens_traded": sorted(eligible_traded),
        "trade_days": sorted(trade_days),
        "last_trade_ts": last_trade_ts,
    }
