"""Retained hourly snapshots + a compact per-wallet time series for timeline views.

Each capture is archived under `snapshots/<id>/leaderboard.json` (id = the capture hour, e.g.
`2026-06-22T23Z`) and indexed in `snapshots/index.json`, so any past board is recoverable. A single
compact `series.json` accumulates one light point per wallet per hour (rank/equity/PnL/flags) — the
frontend reads it once to chart how every wallet moves over the window. Re-running within the same
hour REPLACES that hour's entry (idempotent), so a manual re-run never double-counts.

Operates on the LOCAL canonical competition dir (history accumulates there); the publisher mirrors the
changed files to the CDN. Pure file I/O, no network.
"""

from __future__ import annotations

import json
import os
from datetime import datetime


def snapshot_id(generated_iso: str) -> str:
    """Capture hour id from an ISO timestamp, e.g. '2026-06-22T23:05:46+00:00' -> '2026-06-22T23Z'."""
    dt = datetime.fromisoformat(generated_iso.replace("Z", "+00:00"))
    return dt.strftime("%Y-%m-%dT%HZ")


def _read_json(path: str, default):
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return default


def _write_json(path: str, obj) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, separators=(",", ":"))


def _upsert(lst: list[dict], entry: dict, key: str = "id") -> list[dict]:
    """Replace the element whose `key` matches `entry[key]`, else append; keep sorted by `key`."""
    out = [e for e in lst if e.get(key) != entry[key]]
    out.append(entry)
    return sorted(out, key=lambda e: e.get(key) or "")


def update_history(leaderboard: dict, comp_dir: str) -> list[str]:
    """Archive this capture + update the index and series under `comp_dir` (the local `competition/`
    dir). Returns the list of relative paths written (for the publisher to mirror)."""
    sid = snapshot_id(leaderboard["generated"])
    gen = leaderboard["generated"]

    # 1) full-board archive (immutable once written for a given hour)
    archive_rel = f"snapshots/{sid}/leaderboard.json"
    _write_json(os.path.join(comp_dir, archive_rel), leaderboard)

    # 2) snapshot index
    idx_path = os.path.join(comp_dir, "snapshots", "index.json")
    idx = _read_json(idx_path, {"snapshots": []})
    idx["generated"] = gen
    idx["snapshots"] = _upsert(idx.get("snapshots", []), {
        "id": sid, "generated": gen,
        "n_participants": leaderboard.get("n_participants"),
        "n_ranked": leaderboard.get("n_ranked"),
        "n_disqualified": leaderboard.get("n_disqualified"),
        "n_dq_risk": leaderboard.get("n_dq_risk"),
        "total_equity_usd": leaderboard.get("total_equity_usd"),
    })
    _write_json(idx_path, idx)

    # 3) compact per-wallet series (one light point per wallet per hour)
    ser_path = os.path.join(comp_dir, "series.json")
    ser = _read_json(ser_path, {"snapshots": [], "wallets": {}})
    ser["generated"] = gen
    ser["snapshots"] = _upsert(ser.get("snapshots", []), {"id": sid, "generated": gen})
    wallets = ser.setdefault("wallets", {})
    for r in leaderboard["rows"]:
        pt = {"id": sid, "rank": r["rank"], "equity_usd": r["equity_usd"],
              "pnl_pct": r["pnl_pct"], "capital_basis_usd": r.get("capital_basis_usd"),
              "ranked": r["ranked"], "disqualified": r["disqualified"],
              "traded_in_window": r["traded_in_window"]}
        arr = [p for p in wallets.get(r["wallet"], []) if p.get("id") != sid]
        arr.append(pt)
        wallets[r["wallet"]] = sorted(arr, key=lambda p: p.get("id") or "")
    _write_json(ser_path, ser)

    return ["series.json", "snapshots/index.json", archive_rel]
