"""Apply the corrected selection criteria to the screen -> a risk-tiered proposal.

Reads data/universe_screen.json (from scripts/screen_universe.py), writes
data/selection.json (consumable by scripts/download_ohlcv.py), and prints the
tiered table plus the tokens needing CMC contract verification.

Run: .venv/Scripts/python.exe scripts/select_universe.py [--major 7 --mid 7 --degen 6]
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from trader.data import dexscreener as ds  # noqa: E402
from trader.data import select as sel  # noqa: E402

KEEP = ["symbol", "tier", "pair_address", "token_address", "bsc_contract", "cmc_id",
        "liq_usd", "vol_h24", "turnover", "vol_proxy", "needs_verification",
        "dex", "quote", "name"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--screen", default="data/resolved.json")
    ap.add_argument("--out", default="data/selection.json")
    ap.add_argument("--major", type=int, default=7)
    ap.add_argument("--mid", type=int, default=7)
    ap.add_argument("--degen", type=int, default=6)
    ap.add_argument("--liq-floor", type=float, default=sel.LIQ_FLOOR)
    ap.add_argument("--turnover-floor", type=float, default=sel.TURNOVER_FLOOR)
    args = ap.parse_args()

    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:  # noqa: BLE001
        pass

    rows = json.load(open(args.screen, encoding="utf-8"))
    for r in rows:  # resolved.json rows lack the screen-stage vol_proxy
        if r.get("status") == "resolved" and "vol_proxy" not in r:
            r["vol_proxy"] = ds.vol_proxy(r)
    cands = sel.candidates(rows, args.liq_floor, args.turnover_floor)
    chosen = sel.tier(cands, args.major, args.mid, args.degen)

    out = [{k: r.get(k) for k in KEEP} for r in chosen]
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)

    print(f"candidates passing gates (liq>={args.liq_floor:,.0f}, "
          f"turnover>={args.turnover_floor}): {len(cands)}")
    print(f"selected (tiered): {len(chosen)}  ->  {args.out}\n")
    print(f"  {'tier':6} {'symbol':12} {'liq_usd':>13} {'vol_h24':>13} {'turnover':>8} {'vp':>7} verify?")
    for r in chosen:
        flag = "  ⚠ CMC" if r.get("needs_verification") else ""
        print(f"  {r['tier']:6} {r['symbol']:12} {r['liq_usd']:>13,.0f} "
              f"{r.get('vol_h24',0):>13,.0f} {r.get('turnover',0):>8.2f} "
              f"{r.get('vol_proxy',0):>7}{flag}")

    nv = [r["symbol"] for r in chosen if r.get("needs_verification")]
    print(f"\n  needs CMC contract verification ({len(nv)}): {', '.join(nv) or 'none'}")


if __name__ == "__main__":
    main()
