"""Append-only risk ledger — the caps' persistence (JSONL at `data/risk_ledger.jsonl`).

A crashed-and-restarted loop must stay capped, so daily / lifetime spend and the equity
high-water mark are **derived from disk on every check**, never held in memory. One JSON
row per event:

  {"kind": "attempt", ts, intent, quote, usd}        # written BEFORE the swap is signed
  {"kind": "result",  ts, tx_hash, status, gas_usd}  # confirmation outcome
  {"kind": "refusal", ts, phase, intent, refusals}   # audit trail (counts no spend)
  {"kind": "equity",  ts, equity_usd}                # wallet equity marks (drawdown anchor)

Spend accounting is **conservative**: every *attempt* counts its full notional the moment
it is written (an attempt whose outcome is unknown may still have moved money), and result
rows add realized `gas_usd` on top. Both swap directions count notional — a $1 round trip
consumes $2 of the daily budget. Days are UTC.

Fail closed: a present-but-unparseable ledger yields `RiskState(available=False)` (every
check then refuses with STATE_UNAVAILABLE). A *missing* file is a legitimate fresh ledger.
The file is git-ignored — it can reference tx hashes and sizes, never secrets.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from trader.risk.checks import RiskState

LEDGER_PATH = Path("data/risk_ledger.jsonl")


class LedgerError(RuntimeError):
    """The ledger exists but cannot be trusted (malformed row / unreadable)."""


def _now() -> datetime:
    return datetime.now(timezone.utc)


def append(row: dict, path: Path = LEDGER_PATH, *, now: datetime | None = None) -> dict:
    """Append one event row (adds a UTC `ts` if absent). Returns the written row."""
    row = dict(row)
    row.setdefault("ts", (now or _now()).isoformat(timespec="seconds"))
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, sort_keys=True) + "\n")
    return row


def append_equity(equity_usd: float, path: Path = LEDGER_PATH,
                  *, now: datetime | None = None) -> dict:
    """Record a wallet-equity mark (feeds the high-water / drawdown stop)."""
    return append({"kind": "equity", "equity_usd": float(equity_usd)}, path, now=now)


def read_rows(path: Path = LEDGER_PATH) -> list[dict]:
    """All rows, oldest first. Missing file -> [] (fresh); malformed -> LedgerError."""
    if not path.exists():
        return []
    rows = []
    try:
        with open(path, encoding="utf-8") as f:
            for i, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise LedgerError(f"row {i} is not an object")
                rows.append(row)
    except LedgerError:
        raise
    except Exception as e:  # malformed JSON / unreadable file => the caps can't be trusted
        raise LedgerError(f"ledger unreadable: {e}") from e
    return rows


def _usd(row: dict) -> float:
    v = row.get("usd")
    return float(v) if isinstance(v, (int, float)) else 0.0


def _is_today(row: dict, now: datetime) -> bool:
    """UTC same-day test. A row with a missing/odd ts counts as today (conservative)."""
    ts = row.get("ts")
    if not isinstance(ts, str) or len(ts) < 10:
        return True
    return ts[:10] == now.strftime("%Y-%m-%d")


def spent_today_usd(rows: list[dict], now: datetime) -> float:
    return sum(_usd(r) for r in rows if r.get("kind") == "attempt" and _is_today(r, now))


def spent_lifetime_usd(rows: list[dict]) -> float:
    notional = sum(_usd(r) for r in rows if r.get("kind") == "attempt")
    gas = sum(float(r.get("gas_usd") or 0.0) for r in rows if r.get("kind") == "result")
    return notional + gas


def state_from_ledger(path: Path = LEDGER_PATH, *, now: datetime | None = None) -> RiskState:
    """Derive the `RiskState` the checks consume. NEVER raises — failure => unavailable."""
    now = now or _now()
    try:
        rows = read_rows(path)
        equity = [float(r["equity_usd"]) for r in rows
                  if r.get("kind") == "equity" and isinstance(r.get("equity_usd"), (int, float))]
        return RiskState(
            spent_today_usd=spent_today_usd(rows, now),
            spent_lifetime_usd=spent_lifetime_usd(rows),
            equity_usd=equity[-1] if equity else None,
            high_water_usd=max(equity) if equity else None,
            available=True,
        )
    except Exception as e:  # noqa: BLE001 — fail closed: unreadable state refuses every trade
        return RiskState(available=False, detail=f"{type(e).__name__}: {e}"[:200])
