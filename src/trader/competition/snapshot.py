"""Orchestrate one leaderboard snapshot: read every participant wallet's holdings now and at the
window start, price both, and assemble a ranked board. Reuses the live agent's PURE valuation
function (`trader.agent.wallet_recon.build_wallet_payload`) so the equity/PnL math is identical to the
one we already trust — without importing or touching the live loop.

Read-only: balances via Multicall3 against an archive RPC, prices via DexScreener + local stores. No
key, no signer.
"""

from __future__ import annotations

import json
import os
import time
from datetime import datetime, timedelta, timezone

WINDOW_DAYS = 7   # competition live window length (Jun 22-28 UTC)


def completed_window_days(start_ts: int, now_ts: float, window_days: int = WINDOW_DAYS) -> list[str]:
    """UTC day strings (YYYY-MM-DD) that are FULLY elapsed within the window as of `now_ts`. A day
    only counts once its 00:00-next-day boundary has passed (and is within the window) — so the
    daily-trade DQ never fires on the in-progress day."""
    end_ts = start_ts + window_days * 86400
    d0 = datetime.fromtimestamp(start_ts, timezone.utc).date()
    out = []
    for i in range(window_days):
        day = d0 + timedelta(days=i)
        day_end = datetime(day.year, day.month, day.day, tzinfo=timezone.utc).timestamp() + 86400
        if day_end <= min(now_ts, end_ts):
            out.append(day.strftime("%Y-%m-%d"))
    return out

from trader.agent.wallet_recon import build_wallet_payload
from trader.competition import flows
from trader.competition import multicall as mc
from trader.competition import pricing

DECIMALS_CACHE = os.path.join("data", "competition", "decimals.json")


def _load_decimals_cache(path: str = DECIMALS_CACHE) -> dict[str, int]:
    try:
        with open(path, encoding="utf-8") as f:
            return {k.lower(): int(v) for k, v in json.load(f).items()}
    except Exception:  # noqa: BLE001
        return {}


def _save_decimals_cache(dec: dict[str, int], path: str = DECIMALS_CACHE) -> None:
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(dec, f)
    except Exception:  # noqa: BLE001
        pass


def get_decimals(rpc_call, universe: list[dict], *, refresh: bool = False) -> dict[str, int]:
    """Token decimals `{address_lower: decimals}`, cached on disk (constant per token)."""
    cache = {} if refresh else _load_decimals_cache()
    missing = [u for u in universe if (u.get("contract") or "").lower() not in cache]
    if missing:
        cache.update(mc.read_decimals(rpc_call, missing))
        _save_decimals_cache(cache)
    return cache


def _nonzero(holdings: dict[str, float]) -> set[str]:
    return {s for s, q in holdings.items() if q and q > 0}


def build_leaderboard(participants: list[dict], universe: list[dict], *, rpc_call,
                      nr=None, window_start_block: int, window_start_ts: int,
                      equity_history: dict[str, list] | None = None,
                      ohlcv_root: str | None = None, dd_gate: float = 0.30, log=print
                      ) -> tuple[dict, dict[str, dict]]:
    """Return `(leaderboard, wallet_payloads)` for the COMPETITION-WINDOW board.

    Everything is scoped to the live window `[window_start_block, latest]`:

      E0           = equity at the window-open block (archive Multicall), valued at window-start prices
      E1           = equity now (Multicall `latest`), current prices
      window_flows = net fundable deposits − withdrawals DURING the window (NodeReal from open block)
      capital_basis = E0 + window_flows           # capital deployed for the window
      window PnL$   = E1 − capital_basis           # deposit/withdrawal-proof
      window PnL%   = PnL$ / capital_basis

    The three contest DQ rules are enforced per wallet (any failure DQs): (1) held an eligible token at
    the window open, (2) >=1 swap every completed UTC day, (3) wallet value never below $1 (checked at
    open/now and across `equity_history` — past hourly equity per wallet from the series). ALL non-DQ
    wallets are ranked by PnL% (no minimum-capital rule). Equities use the shared pure
    `build_wallet_payload`; in-window flows/trades come from `trader.competition.flows`."""
    equity_history = equity_history or {}
    decimals = get_decimals(rpc_call, universe)
    bnb_now = pricing._bnb_anchor_close()
    bnb_at = (lambda ts: pricing._bnb_anchor_close(at_ts=ts)) if bnb_now else None
    latest = (nr.block_number() - 100) if nr is not None else rpc_call("eth_blockNumber", [])
    latest = int(latest, 16) if isinstance(latest, str) else latest   # margin: NodeReal's transfer index lags the head
    start_blk_hex = hex(window_start_block)
    eligible_contracts = {(u.get("contract") or "").lower() for u in universe
                          if u.get("contract") and not u.get("is_stable")}
    # `counted` = everything our equity values (the whole eligible universe incl USDT/stables + wrapped
    # BNB; native BNB handled by symbol). Used to neutralize uncounted<->counted conversions in PnL.
    counted_contracts = {(u.get("contract") or "").lower() for u in universe if u.get("contract")}
    counted_contracts.add("0xbb4cdb9cbd36b01bd1cbaebf2de08d9173bc095c")   # WBNB
    now_ts = datetime.now(timezone.utc).timestamp()
    comp_days = completed_window_days(window_start_ts, now_ts)
    window_end_ts = window_start_ts + WINDOW_DAYS * 86400
    cur_day = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    cur_day_in_window = window_start_ts <= now_ts < window_end_ts

    # 1) holdings now + at the window open (2 Multicalls/wallet; archive read for the open block)
    now_h: dict[str, dict] = {}
    start_h: dict[str, dict] = {}
    held: set[str] = set()
    held_start: set[str] = set()
    for i, p in enumerate(participants, 1):
        w = p["wallet"]
        now_h[w] = mc.read_holdings(rpc_call, w, universe, decimals, block="latest")
        start_h[w] = mc.read_holdings(rpc_call, w, universe, decimals, block=start_blk_hex)
        held |= _nonzero(now_h[w])
        held_start |= _nonzero(start_h[w])
        if i % 20 == 0 or i == len(participants):
            log(f"  read holdings {i}/{len(participants)}")

    cur_px = pricing.current_prices(held | held_start, universe, root=ohlcv_root)
    start_px, approx = pricing.start_prices(held_start, universe, window_start_ts,
                                            current=cur_px, root=ohlcv_root)
    log(f"  priced {len(cur_px)} current / {len(start_px)} window-open symbols")

    # 2) in-window flows + trading activity per wallet (NodeReal, scoped to the window)
    cost: dict[str, dict] = {}
    if nr is not None:
        n_err = 0
        def _fetch(wallet):
            return flows.wallet_cost_basis(nr, wallet, from_block=window_start_block, to_block=latest,
                                           bnb_price_now=bnb_now, bnb_price_at=bnb_at,
                                           eligible_contracts=eligible_contracts,
                                           counted_contracts=counted_contracts, prices=cur_px)

        for i, p in enumerate(participants, 1):
            w = p["wallet"]
            try:
                cost[w] = _fetch(w)
            except Exception as e:  # noqa: BLE001 — a flaky/rate-limited API must not sink the snapshot
                n_err += 1
                if n_err <= 3:
                    log(f"  flows FAILED for {w[:10]}: {type(e).__name__} {str(e)[:60]}")
            if i % 20 == 0 or i == len(participants):
                log(f"  in-window flows {i}/{len(participants)} ({n_err} errors)")

        # Confirm-before-DQ: the bulk sweep sometimes returns INCOMPLETE transfers under NodeReal load
        # (a wallet that traded looks like it missed a day -> false DQ + wrong PnL). A back-to-back
        # re-fetch is throttled the same way, so re-check each suspect in a SEPARATE PACED pass, UNIONing
        # trade_days across paced reads (monotonic — more data only adds days) and keeping the fuller cost
        # basis, until the wallet covers every completed day or the retries are exhausted.
        # Skip when the source is the incremental cache: it accumulates every transfer across runs and
        # re-serves them from memory, so a re-fetch yields nothing new (only wasted sleeps). The paced
        # pass only helps the live API's transient partial reads.
        suspects = ([] if getattr(nr, "cached", False)
                    else [w for w, cb in cost.items()
                          if cb and (set(comp_days) - set(cb.get("trade_days", [])))])
        log(f"  confirming {len(suspects)} DQ-suspect wallets (paced re-fetch)")
        for w in suspects:
            best = cost[w]
            for _ in range(3):
                time.sleep(0.4)                       # space the read out of the bulk-load throttle window
                try:
                    cb2 = _fetch(w)
                except Exception:  # noqa: BLE001
                    continue
                merged = sorted(set(best.get("trade_days", [])) | set(cb2.get("trade_days", [])))
                best = cb2 if cb2["n_swaps"] > best["n_swaps"] else best
                best["trade_days"] = merged
                if not (set(comp_days) - set(merged)):  # now covers all completed days -> confirmed active
                    break
            cost[w] = best

    # 3) value + rank (window-scoped) + disqualification checks
    rows: list[dict] = []
    payloads: dict[str, dict] = {}
    for p in participants:
        w = p["wallet"]
        cb = cost.get(w)
        e1_payload = build_wallet_payload({s: q for s, q in now_h[w].items() if q > 0},
                                          cur_px, baseline_usd=None, address=w)
        e0_payload = build_wallet_payload({s: q for s, q in start_h[w].items() if q > 0},
                                          start_px, baseline_usd=None, address=w)
        e1, e0 = e1_payload["equity_usd"], e0_payload["equity_usd"]
        # window_flows = ALL non-trading capital in: external deposits/withdrawals + value crossing the
        # eligible<->uncounted boundary (e.g. selling Bitcoin/spam held at open). So PnL reflects only
        # eligible-asset trading, not converting uncounted assets into counted ones.
        window_flows = cb["net_capital_in"] if cb else None
        capital_basis = round(e0 + (window_flows or 0.0), 2)
        pnl_usd = round(e1 - capital_basis, 2) if window_flows is not None else None
        pnl_pct = (round(pnl_usd / capital_basis * 100, 2)
                   if (pnl_usd is not None and capital_basis > 0) else None)
        traded = bool(cb and cb.get("traded_eligible"))
        # === competition DQ rules (any single failure disqualifies) ===
        # (1) ENTRY: the wallet must HOLD AN ELIGIBLE TOKEN at the window open (Jun 22 00:00). The
        #     eligible list excludes BNB/WBNB, so a wallet holding only gas-BNB (or nothing — a late
        #     entrant) at open is DQ'd. USDT and every listed token count.
        e0_eligible_usd = sum((h["value_usd"] or 0.0) for h in e0_payload["holdings"]
                              if h["token"] != "BNB")
        entered = e0_eligible_usd > 0.01
        # (2) DAILY TRADE: >=1 swap on EVERY completed UTC day of the window (incl. Jun 22).
        trade_days = set((cb or {}).get("trade_days", []))
        missing_days = [d for d in comp_days if d not in trade_days]
        # (3) EQUITY FLOOR: total wallet value must never fall below $1, at any point. Checked at the
        #     open, now, and across every captured hourly snapshot (series) — a single breach is a DQ.
        eq_points = [v for v in (e0, e1, *equity_history.get(w, [])) if v is not None]
        floor_min = min(eq_points) if eq_points else e1
        dq_reasons = []
        if not entered:
            dq_reasons.append("No eligible token held at window open (Jun 22 00:00)")
        if missing_days:
            dq_reasons.append(f"No trade on {missing_days[0]}")
        if floor_min < 1.0:
            dq_reasons.append(f"Wallet value fell below $1 (min ${floor_min:.2f})")
        disqualified = bool(dq_reasons)
        dq_reason = "; ".join(dq_reasons) if dq_reasons else None
        # at-risk = today (in-progress) has no trade yet — a warning, not yet a DQ.
        dq_risk = (not disqualified) and cur_day_in_window and (cur_day not in trade_days)
        # rank ALL active (non-DQ) wallets — there is no minimum-capital rule; a $5 wallet can compete.
        ranked = (not disqualified) and pnl_pct is not None
        row = {
            "wallet": w,
            "equity_usd": e1,
            "equity_open_usd": e0,
            "window_flows_usd": window_flows,
            "boundary_flow_usd": (cb or {}).get("boundary_flow"),
            "capital_basis_usd": capital_basis if window_flows is not None else None,
            "pnl_usd": pnl_usd,
            "pnl_pct": pnl_pct,
            "traded_in_window": traded,
            "n_eligible_buys": (cb or {}).get("n_eligible_buys", 0),
            "n_swaps": (cb or {}).get("n_swaps", 0),
            "trade_days": sorted(trade_days),
            "entered": entered,
            "e0_eligible_usd": round(e0_eligible_usd, 2),
            "floor_min_usd": round(floor_min, 2),
            "ranked": ranked,
            "disqualified": disqualified,
            "dq_reason": dq_reason,
            "dq_risk": dq_risk,
            "n_holdings": sum(1 for h in e1_payload["holdings"] if (h["value_usd"] or 0) > 0),
            "stale": bool(e1_payload["stale"]),
            "registered_ts": p.get("registered_ts"),
        }
        rows.append(row)
        payloads[w] = {**e1_payload, "equity_open_usd": e0, "window_flows_usd": window_flows,
                       "capital_basis_usd": row["capital_basis_usd"], "pnl_usd": pnl_usd,
                       "pnl_pct": pnl_pct, "cost_basis": cb, "ranked": ranked,
                       "traded_in_window": traded, "disqualified": disqualified,
                       "dq_reason": dq_reason, "dq_risk": dq_risk,
                       "registered_ts": p.get("registered_ts")}

    # ranked active traders first (by PnL%); other active wallets next (by equity); DQ'd last.
    rows.sort(key=lambda r: (not r["disqualified"], r["ranked"],
                             r["pnl_pct"] if (r["ranked"] and r["pnl_pct"] is not None) else -1e18,
                             r["equity_usd"]), reverse=True)
    for rank, r in enumerate(rows, 1):
        r["rank"] = rank

    leaderboard = {
        "generated": datetime.now(timezone.utc).isoformat(),
        "metric": "window_pnl_vs_capital_basis",
        "window": {"start_block": window_start_block, "start_ts": window_start_ts,
                   "start_utc": datetime.fromtimestamp(window_start_ts, timezone.utc).isoformat(),
                   "end_utc": datetime.fromtimestamp(window_end_ts, timezone.utc).isoformat(),
                   "completed_days": comp_days},
        "dd_gate": dd_gate,
        "floor_usd": 1.0,
        "n_participants": len(rows),
        "n_traded_in_window": sum(1 for r in rows if r["traded_in_window"]),
        "n_entered": sum(1 for r in rows if r["entered"]),
        "n_ranked": sum(1 for r in rows if r["ranked"]),
        "n_disqualified": sum(1 for r in rows if r["disqualified"]),
        "n_dq_risk": sum(1 for r in rows if r["dq_risk"]),
        "total_equity_usd": round(sum(r["equity_usd"] for r in rows), 2),
        "rows": rows,
    }
    return leaderboard, payloads
