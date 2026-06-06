"""Data spike — screen the eligible universe on BSC and propose a risk-tiered set.

Pipeline (see vault "Tech Stack" / "Trading Strategies"):
  1. Resolve each eligible symbol to its deepest-liquidity BSC pair (DexScreener).
  2. Rank by real on-chain liquidity; characterize volatility; flag ambiguity,
     stables, and an un-tradeable dust floor.
  3. Bucket the tradeable, non-stable set into major/mid/degen risk tiers and
     propose ~20 spread across them.
  4. Validate GeckoTerminal OHLCV (history depth + granularity) for the proposal.

Run:  .venv/Scripts/python.exe scripts/screen_universe.py
Out:  data/universe_screen.{json,csv}, data/proposed20.json
"""

from __future__ import annotations

import csv
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from trader.data import dexscreener as ds  # noqa: E402
from trader.data import geckoterminal as gt  # noqa: E402
from trader.data.eligible import ELIGIBLE_SYMBOLS, STABLES  # noqa: E402

LIQ_FLOOR = 50_000.0  # USD pool liquidity to count as "tradeable"
TARGET = 20
OUT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data"))
os.makedirs(OUT, exist_ok=True)

CSV_COLS = [
    "symbol", "status", "tier", "is_stable", "tradeable", "liq_usd", "vol_h24",
    "vol_proxy", "chg_h24", "age_days", "n_bsc", "ambiguous", "dex", "quote",
    "token_address", "pair_address", "name",
]


def screen() -> list[dict]:
    rows: list[dict] = []
    n = len(ELIGIBLE_SYMBOLS)
    for i, sym in enumerate(ELIGIBLE_SYMBOLS, 1):
        try:
            s = ds.summarize(sym, ds.search(sym))
            if s.get("status") == "resolved":
                s["vol_proxy"] = ds.vol_proxy(s)
                s["is_stable"] = sym in STABLES
                s["tradeable"] = s["liq_usd"] >= LIQ_FLOOR
        except Exception as e:  # noqa: BLE001 — spike: never let one token kill the run
            s = {"symbol": sym, "status": "error", "error": repr(e)}
        rows.append(s)
        print(f"[{i:3}/{n}] {sym:12} {s.get('status'):10} "
              f"liq={s.get('liq_usd')} n_bsc={s.get('n_bsc')}", flush=True)
        time.sleep(0.3)
    return rows


def assign_tiers(rows: list[dict]) -> list[dict]:
    tradeable = [r for r in rows if r.get("tradeable") and not r.get("is_stable")]
    tradeable.sort(key=lambda r: r["liq_usd"], reverse=True)
    nt = len(tradeable)
    third = max(1, nt // 3)
    for idx, r in enumerate(tradeable):
        r["tier"] = "major" if idx < third else ("mid" if idx < 2 * third else "degen")
    return tradeable


def propose(tradeable: list[dict]) -> list[dict]:
    majors = [r for r in tradeable if r["tier"] == "major"][:7]
    mids = sorted((r for r in tradeable if r["tier"] == "mid"),
                  key=lambda r: r["liq_usd"], reverse=True)[:7]
    degens = sorted((r for r in tradeable if r["tier"] == "degen"),
                    key=lambda r: r.get("vol_proxy", 0), reverse=True)[:6]
    return majors + mids + degens


def validate_ohlcv(proposed: list[dict]) -> list[dict]:
    out = []
    for r in proposed:
        rec = {"symbol": r["symbol"], "pair": r.get("pair_address"), "tier": r.get("tier")}
        try:
            day = gt.fetch_ohlcv(r["pair_address"], timeframe="day", limit=1000)
            rec.update(ok=bool(day), day_candles=len(day),
                       history_days=gt.candle_span_days(day),
                       daily_vol=gt.realized_vol(day))
        except Exception as e:  # noqa: BLE001
            rec.update(ok=False, error=repr(e))
        out.append(rec)
        print(f"  OHLCV {rec['symbol']:12} candles={rec.get('day_candles')} "
              f"hist_days={rec.get('history_days')} ok={rec.get('ok')}", flush=True)
        time.sleep(2.1)
    return out


def minute_probe(pair: str) -> dict:
    """Confirm 1-minute granularity is actually available (front-run/sweep edges)."""
    try:
        m = gt.fetch_ohlcv(pair, timeframe="minute", aggregate=1, limit=1000)
        return {"ok": bool(m), "candles": len(m), "span_days": gt.candle_span_days(m)}
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "error": repr(e)}


def main() -> None:
    # Windows consoles default to cp1252; the universe contains non-ASCII tickers.
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:  # noqa: BLE001
        pass
    rows = screen()

    with open(os.path.join(OUT, "universe_screen.json"), "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2, ensure_ascii=False)
    with open(os.path.join(OUT, "universe_screen.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=CSV_COLS, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)

    tradeable = assign_tiers(rows)
    proposed = propose(tradeable)
    validated = validate_ohlcv(proposed)
    mprobe = minute_probe(proposed[0]["pair_address"]) if proposed else {}

    with open(os.path.join(OUT, "proposed20.json"), "w", encoding="utf-8") as f:
        json.dump({
            "proposed": [{k: r.get(k) for k in CSV_COLS} for r in proposed],
            "ohlcv_validation": validated,
            "minute_probe": mprobe,
        }, f, indent=2, ensure_ascii=False)

    resolved = [r for r in rows if r.get("status") == "resolved"]
    print("\n=== SUMMARY ===")
    print(f"total symbols       : {len(rows)}")
    print(f"resolved on BSC     : {len(resolved)}")
    print(f"unresolved/error    : {len(rows) - len(resolved)}")
    print(f"tradeable (liq>={int(LIQ_FLOOR):,}): "
          f"{len([r for r in resolved if r.get('tradeable')])}")
    print(f"stables             : {len([r for r in resolved if r.get('is_stable')])}")
    print(f"ambiguous resolves  : {len([r for r in resolved if r.get('ambiguous')])}")
    print(f"proposed set        : {len(proposed)}")
    ok = [v for v in validated if v.get("ok")]
    print(f"OHLCV ok            : {len(ok)}/{len(validated)}")
    if ok:
        spans = sorted(v["history_days"] for v in ok)
        print(f"  median history_days: {spans[len(spans)//2]}")
    print(f"1-min probe         : {mprobe}")
    print(f"\nartifacts -> {OUT}")


if __name__ == "__main__":
    main()
