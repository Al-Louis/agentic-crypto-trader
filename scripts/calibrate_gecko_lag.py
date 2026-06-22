"""Calibrate DEFAULT_TICK_OFFSET: measure GeckoTerminal's candle finalization lag at an hour boundary.

Waits until just after the next HH:00 UTC, then polls each selection pool (PACED, so the measurement
never 429-storms) and records when Gecko first serves the just-closed (HH-1):00 bar. Reports the
per-pool lag + distribution so we can set the tick offset to cover the real lag (e.g. P90 + margin).

Run in the background a little before the top of an hour:
    PYTHONPATH=src python scripts/calibrate_gecko_lag.py
Writes data/gecko_lag_calib.json and prints a summary. Read-only (no box, no trading).
"""
from __future__ import annotations

import json
import os
import time

from trader.agent.event_agent import load_selection
from trader.data import geckoterminal as gt

PACE_S = 3.0            # per-pool pacing (mirror the live loop's GeckoTerminal courtesy)
SWEEP_GAP_S = 75.0      # rest between sweeps of the still-missing pools
MAX_WAIT_S = 1500.0     # give up after 25 min (perma-stale pools never settle)
OUT = os.path.join("data", "gecko_lag_calib.json")


def has_bar(pool: str, target: int) -> bool:
    """Does Gecko currently serve a finalized bar opening at `target` for this pool?"""
    try:
        rows = gt.fetch_ohlcv(pool, timeframe="hour", aggregate=1, limit=6, network="bsc")
    except Exception:  # noqa: BLE001 — a transient error this sweep; try again next sweep
        return False
    return any(int(r[0]) == target for r in rows)


def main() -> None:
    sel = [s for s in load_selection() if s.get("pair_address")]
    now = int(time.time())
    boundary = ((now // 3600) + 1) * 3600                 # next HH:00
    if boundary - now < 20:                               # too close to race cleanly -> next hour
        boundary += 3600
    target = boundary - 3600                              # the (HH-1):00 bar that closes AT boundary
    print(f"[calib] now={now} next_boundary={boundary} target_bar={target} "
          f"(measuring {len(sel)} pools); sleeping {boundary - now}s until the close...", flush=True)
    time.sleep(max(0, boundary - now))

    lag: dict[str, float | None] = {s["symbol"]: None for s in sel}
    missing = list(sel)
    while missing and (time.time() - boundary) < MAX_WAIT_S:
        for i, s in enumerate(missing):
            if i:
                time.sleep(PACE_S)
            if has_bar(s["pair_address"], target):
                lag[s["symbol"]] = round(time.time() - boundary, 1)
        missing = [s for s in missing if lag[s["symbol"]] is None]
        settled = [v for v in lag.values() if v is not None]
        print(f"[calib] +{int(time.time() - boundary)}s  settled={len(settled)}/{len(sel)}  "
              f"still_missing={[s['symbol'] for s in missing]}", flush=True)
        if missing:
            time.sleep(SWEEP_GAP_S)

    settled = sorted(v for v in lag.values() if v is not None)
    never = [k for k, v in lag.items() if v is None]
    pct = (lambda p: settled[min(len(settled) - 1, int(p * (len(settled) - 1)))] if settled else None)
    summary = {
        "boundary": boundary, "target_bar": target, "n_pools": len(sel),
        "n_settled": len(settled), "n_never": len(never),
        "lag_seconds_sorted": settled,
        "median_s": pct(0.5), "p90_s": pct(0.9), "max_s": (settled[-1] if settled else None),
        "never_settled": never, "per_pool": lag,
    }
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print("\n=== GECKO FINALIZATION LAG (seconds after the hour close) ===", flush=True)
    print(f"settled {len(settled)}/{len(sel)}  median={summary['median_s']}s  "
          f"P90={summary['p90_s']}s  max={summary['max_s']}s  never={never}", flush=True)
    print(f"suggested offset ~ P90 + 2min = {(summary['p90_s'] or 0) + 120:.0f}s "
          f"(current default 900s).  wrote {OUT}", flush=True)


if __name__ == "__main__":
    main()
