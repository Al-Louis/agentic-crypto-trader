"""Forensic gate — rug/honeypot-screen tokens via GoPlus (vault "Security and Encryption").

Reads a selection (default the locked 20) or the full resolved universe, screens each
token's BSC contract through GoPlus, and grades it block / warn / ok. Tokens graded
`block` should not enter the tradeable set.

Run:
  .venv/Scripts/python.exe scripts/forensics.py                  # screen data/selection.json
  .venv/Scripts/python.exe scripts/forensics.py --all            # screen all resolved tokens
Out: data/forensics.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from trader.data import goplus  # noqa: E402

ORDER = {"block": 0, "unknown": 1, "warn": 2, "ok": 3}


def _load(path: str) -> list[dict]:
    data = json.load(open(path, encoding="utf-8"))
    rows = data.get("proposed", data) if isinstance(data, dict) else data
    return [r for r in rows if r.get("bsc_contract")]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--selection", default="data/selection.json")
    ap.add_argument("--all", action="store_true", help="screen data/resolved.json instead")
    ap.add_argument("--out", default="data/forensics.json")
    args = ap.parse_args()

    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:  # noqa: BLE001
        pass

    rows = _load("data/resolved.json" if args.all else args.selection)

    # cache: GoPlus keyless is rate-limit-flaky (empties non-deterministically), so
    # keep good verdicts across runs and only re-screen missing/unknown tokens. Re-run
    # until nothing is left unknown.
    prev = {}
    if os.path.exists(args.out):
        for rec in json.load(open(args.out, encoding="utf-8")):
            prev[(rec.get("contract") or "").lower()] = rec
    need = [r for r in rows
            if prev.get(r["bsc_contract"].lower(), {}).get("verdict") in (None, "unknown")]
    print(f"{len(rows)} tokens; {len(need)} need screening, {len(rows) - len(need)} cached")
    raw = (goplus.fetch_token_security([r["bsc_contract"] for r in need],
                                       retries=2, sleep=2.0, logger=print) if need else {})

    def record(r, sec, v):
        return {"symbol": r["symbol"], "tier": r.get("tier"), "contract": r["bsc_contract"],
                "verdict": v["verdict"], "score": v["score"], "flags": v["flags"],
                "holder_count": sec.get("holder_count"), "sell_tax": sec.get("sell_tax"),
                "owner_percent": sec.get("owner_percent"), "lp_locked_pct": sec.get("lp_locked_pct")}

    results = []
    for r in rows:
        c = r["bsc_contract"].lower()
        fresh = raw.get(c)
        if fresh:
            sec = goplus.parse_security(fresh)
            results.append(record(r, sec, goplus.verdict(sec)))
        elif c in prev:
            results.append(prev[c])                     # keep cached verdict
        else:
            results.append(record(r, {}, goplus.verdict(goplus.parse_security({}))))

    results.sort(key=lambda x: (ORDER.get(x["verdict"], 9), -(x["score"] or 0)))
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\n  {'verdict':7} {'symbol':12} {'tier':7} {'score':>5} {'holders':>9}  flags")
    for r in results:
        print(f"  {r['verdict']:7} {r['symbol']:12} {str(r.get('tier')):7} "
              f"{str(r['score']):>5} {str(r.get('holder_count')):>9}  {', '.join(r['flags'])}")

    from collections import Counter
    c = Counter(r["verdict"] for r in results)
    print(f"\n  summary: {dict(c)}")
    blocked = [r["symbol"] for r in results if r["verdict"] == "block"]
    if blocked:
        print(f"  ⛔ BLOCK (remove from tradeable set): {', '.join(blocked)}")
    print(f"  -> {args.out}")


if __name__ == "__main__":
    main()
