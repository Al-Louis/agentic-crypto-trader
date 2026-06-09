"""Simulation-integrity audit across every published bundle — catch silent accounting bugs that are
nearly impossible to spot by eye (the BANANAS31 re-rank marker bug was found only by reading trades).

Check #1 — PnL reconciliation: the sum of per-token PnL (from the published markers + candles) must
equal the headline portfolio profit (from the equity curve). A large gap => the markers don't account
for the equity — i.e. trades are missing or mispriced. Run this as a gate before trusting any
per-token analysis. More invariants (price-series consistency, weight conservation, fee totals) to
follow.

    python scripts/audit_bundles.py [--threshold 0.05]   # flag gaps > 5% of capital
"""
from __future__ import annotations

import argparse
import json
import urllib.request

HOST = "https://data.alexlouis.dev"
CAP = 10_000.0


def g(u):
    try:
        with urllib.request.urlopen(u, timeout=20) as r:
            return json.load(r)
    except Exception:  # noqa: BLE001
        return None


def token_pnl_sum(rid, universe):
    total = 0.0
    for u in universe:
        slug = u["slug"]
        candles = g(f"{HOST}/{rid}/tk_{slug}_candles.json")
        trades = g(f"{HOST}/{rid}/tk_{slug}_trades.json")
        if not candles:
            continue
        last = candles[-1]["close"]
        qty = cash = fees = 0.0
        for m in (trades or []):
            px = m.get("price") or 0.0
            if not px:
                continue
            q = m["usd"] / px
            fees += m.get("fee", 0.0)
            if m["side"] == "buy":
                qty += q; cash -= m["usd"]
            else:
                qty -= q; cash += m["usd"]
        total += cash + qty * last - fees
    return total


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--threshold", type=float, default=0.05, help="flag |gap| > this fraction of capital")
    args = p.parse_args()
    man = g(f"{HOST}/manifest.json") or []
    bundles = [e for e in man if e.get("kind") == "portfolio"]
    print(f"auditing {len(bundles)} portfolio bundles — PnL reconciliation\n")
    print(f"  {'bundle':22}{'rerank':>7}{'headline$':>11}{'token-sum$':>11}{'gap$':>9}{'':>3}")
    flagged = []
    for e in sorted(bundles, key=lambda x: x["id"]):
        rid = e["id"]
        m = g(f"{HOST}/{rid}/metrics.json")
        info = g(f"{HOST}/{rid}/run_info.json")
        if not m or not info:
            continue
        headline = (m.get("total_return_pct") or 0) * CAP
        tsum = token_pnl_sum(rid, info.get("universe", []))
        gap = tsum - headline
        rr = (m.get("provenance") or {}).get("rerank_every", "?")
        ok = abs(gap) <= args.threshold * CAP
        if not ok:
            flagged.append((rid, gap, rr))
        print(f"  {rid:22}{str(rr):>7}{headline:>+11,.0f}{tsum:>+11,.0f}{gap:>+9,.0f}{'  OK' if ok else '  ! FLAG'}")
    print()
    if flagged:
        print(f"! {len(flagged)} bundle(s) FAIL reconciliation (gap > {args.threshold:.0%} of capital):")
        for rid, gap, rr in flagged:
            print(f"    {rid}  gap ${gap:+,.0f}  (rerank_every={rr})")
        print("  -> for rerank_every>=1 bundles this is the known marker bug (fixed 1d26881); "
              "re-publish them. A FLAG on a rerank_every=0 bundle would be a NEW bug.")
    else:
        print("OK: all bundles reconcile — no silent accounting mismatch detected")


if __name__ == "__main__":
    main()
