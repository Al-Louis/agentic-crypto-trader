"""The agent loop's append-only store + crash-safe state derivation.

A sibling to `risk/ledger.py` (the *guardrail* ledger): this is the *portfolio* ledger
(`data/agent_ledger.jsonl`) the loop owns. The risk ledger remains the source of truth for
spend caps and is still written by `execute_trade`; this store holds the loop's portfolio
positions, equity marks, heartbeats and per-tick decisions so the **loop can re-derive its
full state from disk after a crash** ([[Remote Capabilities]] idempotency rule). Nothing is
held only in memory.

Row kinds (one JSON object per line, UTC `ts` added on write):
  {"kind":"fill",      ts, mode, from, to, usd_in, usd_out, cost_usd,
                       units_from, units_to, price_from, price_to, reason, tx_hash?}
  {"kind":"equity",    ts, mode, equity_usd, drawdown_pct, peak_usd}   # hourly PnL mark
  {"kind":"heartbeat", ts, mode, tick, equity_usd}                     # dead-man input
  {"kind":"refusal",   ts, mode, intent, refusals}                     # guardrail audit

Positions re-derive by replaying `fill` rows (units in/out per symbol). Cash is held as a
position under the `CASH` pseudo-symbol (USD-pegged). Fail-closed read: a present-but-
malformed ledger raises `StoreError` so the loop refuses to start on untrusted state rather
than trading on a guessed portfolio.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

AGENT_LEDGER_PATH = Path("data/agent_ledger.jsonl")
CASH = "CASH"  # the USD-pegged cash leg, tracked as a position for uniform accounting


class StoreError(RuntimeError):
    """The agent ledger exists but cannot be trusted (malformed row / unreadable)."""


def _now() -> datetime:
    return datetime.now(timezone.utc)


def append(row: dict, path: Path = AGENT_LEDGER_PATH, *, now: datetime | None = None) -> dict:
    """Append one event row (adds a UTC `ts` if absent). Returns the written row."""
    row = dict(row)
    row.setdefault("ts", (now or _now()).isoformat(timespec="seconds"))
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, sort_keys=True) + "\n")
    return row


def read_rows(path: Path = AGENT_LEDGER_PATH) -> list[dict]:
    """All rows oldest-first. Missing file -> [] (fresh); malformed -> StoreError."""
    if not path.exists():
        return []
    rows: list[dict] = []
    try:
        with open(path, encoding="utf-8") as f:
            for i, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise StoreError(f"row {i} is not an object")
                rows.append(row)
    except StoreError:
        raise
    except Exception as e:  # noqa: BLE001 — malformed/unreadable => state can't be trusted
        raise StoreError(f"agent ledger unreadable: {e}") from e
    return rows


@dataclass(frozen=True)
class PortfolioState:
    """The loop's portfolio, re-derived from the ledger. `positions` is SYMBOL -> token units;
    `CASH` is USD. `peak_usd` is the running equity high-water (drawdown anchor); `tick` is the
    next tick index (so a restart continues the count). All derived from disk, never memory."""

    positions: dict[str, float] = field(default_factory=dict)
    peak_usd: float | None = None
    last_equity_usd: float | None = None
    tick: int = 0
    n_fills: int = 0

    def units(self, symbol: str) -> float:
        return self.positions.get(symbol.upper(), 0.0)

    def cash(self) -> float:
        return self.positions.get(CASH, 0.0)


def apply_fill(positions: dict[str, float], row: dict) -> None:
    """Mutate `positions` by one fill row's token deltas (used by replay and live updates)."""
    frm = str(row.get("from", "")).upper()
    to = str(row.get("to", "")).upper()
    positions[frm] = positions.get(frm, 0.0) - float(row.get("units_from") or 0.0)
    positions[to] = positions.get(to, 0.0) + float(row.get("units_to") or 0.0)


def derive_state(path: Path = AGENT_LEDGER_PATH) -> PortfolioState:
    """Replay the ledger into a `PortfolioState`. Raises `StoreError` on an untrusted ledger."""
    rows = read_rows(path)
    positions: dict[str, float] = {}
    peak: float | None = None
    last_equity: float | None = None
    n_fills = 0
    max_tick = -1
    for r in rows:
        kind = r.get("kind")
        if kind == "fill":
            apply_fill(positions, r)
            n_fills += 1
        elif kind == "equity":
            eq = r.get("equity_usd")
            if isinstance(eq, (int, float)):
                last_equity = float(eq)
                peak = float(eq) if peak is None else max(peak, float(eq))
            pk = r.get("peak_usd")
            if isinstance(pk, (int, float)):
                peak = float(pk) if peak is None else max(peak, float(pk))
        if kind in ("heartbeat", "equity"):
            t = r.get("tick")
            if isinstance(t, int):
                max_tick = max(max_tick, t)
    # drop dust-negative/zero positions that are float noise, keep CASH always
    positions = {k: v for k, v in positions.items() if k == CASH or abs(v) > 1e-15}
    return PortfolioState(positions=positions, peak_usd=peak, last_equity_usd=last_equity,
                          tick=max_tick + 1, n_fills=n_fills)
