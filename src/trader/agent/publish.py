"""The loop's telemetry publisher — project the agent ledger into the `trading/` surface.

[[Apentic Data Contract]] §trading/: the producer is the trading host itself, writing
through a put-only role scoped to `trading/*` (no delete, no list — the no-delete posture).
Freshness comes from the CloudFront CachingDisabled behavior on `trading/*`, never from
invalidations, so this module only ever PUTs. Files written each tick:

  heartbeat.json  {generated, mode, tick, equity_usd}              # the dead-man input
  status.json     one-object summary: equity/peak/drawdown/below_dust, trades_today
                  vs the >=1/day floor, fill + refusal counts
  equity.json     the full hourly series [{ts, equity_usd, drawdown}]
  trades.json     the trade log: fills (tx_hash when live) + guardrail refusals

Convention boundary: ledger rows carry `drawdown_pct` as a PERCENT (4.2 = 4.2%); the
published contract uses FRACTIONS — normalized here and nowhere else.

`project` is pure (rows in, file-shaped dicts out) so the surface is testable offline;
`publish_trading` does the I/O via `remote_train.publish.put_bytes` (s3:// or a local
path — local lets tests and dev runs exercise the real write path). Publishing is
telemetry, not trading: the loop wraps its publisher call fail-safe, and a broken S3
path must never stop a tick.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from trader.agent import store

# The >=1 trade/day competition floor the status surface reports against.
DAILY_TRADE_FLOOR = 1


def _fraction(pct: float | None) -> float | None:
    """Ledger percent -> contract fraction (the one normalization boundary)."""
    return None if pct is None else pct / 100.0


def _utc_iso(ts_secs) -> str | None:
    """Unix seconds -> exact UTC ISO (e.g. 2026-06-17T16:00:00Z). None if not a usable ts."""
    try:
        return datetime.fromtimestamp(int(ts_secs), timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    except (TypeError, ValueError, OverflowError, OSError):
        return None


def _trade_time(fill: dict) -> str | None:
    """A fill's TRADE time = its bar timestamp in UTC (exact hour). NOT `ts`, which is the
    wall-clock time the row was written during the replay (≈ now on a restart, not the trade)."""
    return _utc_iso(fill.get("bar_ts"))


def project(rows: list[dict]) -> dict[str, dict]:
    """Project ledger rows into {filename: json-object}. Empty ledger -> {} (nothing to say).

    `generated` is the newest heartbeat/equity `ts` — derived from the rows, not the clock,
    so the projection is deterministic and a stale ledger publishes *as* stale (the dead-man
    indicator must age when the loop stops, not be refreshed by the publisher).
    """
    fills = [r for r in rows if r.get("kind") == "fill"]
    refusals = [r for r in rows if r.get("kind") == "refusal"]
    equity_rows = [r for r in rows if r.get("kind") == "equity"]

    marks = [r for r in rows if r.get("kind") in ("heartbeat", "equity")]  # ledger order
    if not marks:
        return {}
    generated = max(str(r.get("ts") or "") for r in marks)
    # Same-second ticks tie on `ts`; the ledger is append-only chronological, so row order
    # (the enumerate index) is the honest tie-break.
    newest = max(enumerate(marks), key=lambda iv: (str(iv[1].get("ts") or ""), iv[0]))[1]
    mode = str(newest.get("mode") or "paper")

    last_eq = equity_rows[-1] if equity_rows else {}
    today = generated[:10]  # current UTC date (newest mark's write time ≈ now)
    # count by the TRADE day (bar time), not the write time: after a restart the whole week is
    # re-recorded "now", so a write-time count would mark every fill as today (and the >=1/day
    # floor would read wrong). A trade counts on the UTC day it actually executed.
    trades_today = sum(1 for f in fills if (_trade_time(f) or "")[:10] == today)

    heartbeat = {
        "generated": generated,
        "mode": mode,
        "tick": newest.get("tick"),
        "equity_usd": newest.get("equity_usd"),
    }
    status = {
        "generated": generated,
        "mode": mode,
        "tick": last_eq.get("tick"),
        "equity_usd": last_eq.get("equity_usd"),
        "peak_usd": last_eq.get("peak_usd"),
        "drawdown": _fraction(last_eq.get("drawdown_pct")),
        "below_dust": bool(last_eq.get("below_dust")),
        "trades_today": trades_today,
        "daily_floor_ok": trades_today >= DAILY_TRADE_FLOOR,
        "n_fills": len(fills),
        "n_refusals": len(refusals),
    }
    if last_eq.get("live_scale") is not None:               # live-only; absent in paper (byte-identical).
        status["live_scale"] = last_eq["live_scale"]        # fill REAL usd = book usd_in * live_scale.
    equity = {
        "generated": generated,
        "mode": mode,
        "series": [
            {"ts": r.get("ts"), "equity_usd": r.get("equity_usd"),
             "drawdown": _fraction(r.get("drawdown_pct"))}
            for r in equity_rows
        ],
    }
    # Present each fill's time as the TRADE time (its bar, exact-hour UTC): `time` = unix seconds,
    # `time_utc` = ISO Z, and `ts` is overwritten to the trade time so any consumer reading `ts`
    # shows when the trade happened, not when the replay wrote the row. The write time is kept as
    # `recorded_ts` for debugging.
    pub_fills = []
    for f in fills:
        tt = _trade_time(f)
        pf = dict(f)
        pf["recorded_ts"] = f.get("ts")
        if tt is not None:
            pf["time"] = int(f["bar_ts"])
            pf["time_utc"] = tt
            pf["ts"] = tt
        pub_fills.append(pf)
    trades = {
        "generated": generated,
        "mode": mode,
        "fills": pub_fills,
        "refusals": refusals,
    }
    return {"heartbeat.json": heartbeat, "status.json": status,
            "equity.json": equity, "trades.json": trades}


def publish_trading(rows: list[dict], target: str) -> list[str]:
    """Project + PUT every file under `target` (s3://bucket/trading or a local dir).

    Returns the URIs written. `Cache-Control: no-cache` belt-and-braces the CachingDisabled
    behavior for any client that caches on its own.
    """
    from remote_train.publish import join, put_bytes  # noqa: PLC0415 — boto3 stays optional

    written: list[str] = []
    for name, obj in project(rows).items():
        data = json.dumps(obj, sort_keys=True).encode("utf-8")
        written.append(put_bytes(join(target, name), data,
                                 content_type="application/json", cache_control="no-cache"))
    return written


def build_publisher(ledger_path: Path, target: str):
    """The loop's zero-arg publish hook: re-read the ledger from disk (the crash-safe source
    of truth — never in-memory state) and push the projection. Raises on failure; the loop's
    call site is the fail-safe boundary."""
    def _publish() -> list[str]:
        return publish_trading(store.read_rows(ledger_path), target)
    return _publish
