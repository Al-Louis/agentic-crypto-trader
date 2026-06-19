"""CALIBRATE the anti-chase rotation brake from the PUBLISHED simulated_trades.json (the realized,
already-reconciled trades the user is reviewing) — no policy re-run, no reconstruction. For every
position: realized net return = exit_price/entry_price - 1 - cost; run-up over the prior `--win` hours
from that asset's published candles (entries with < win hours of prior candles are EXCLUDED and counted);
rotation-likely = some OTHER asset in the same week has an exit at this entry's hour (the FF->ZEC #2 swap
co-occurs at one timestamp). Bin by run-up, split rotation-likely vs cash, report mean net + win + $size.

  python scripts/calib_published_runup.py path/to/fxsbq_s1.json [--win 24] [--cost 0.01]
"""
from __future__ import annotations
import argparse, json, sys
from collections import defaultdict

BINS = [(-9, 0), (0, .05), (.05, .10), (.10, .15), (.15, .20), (.20, .30), (.30, .50), (.50, 9)]


def runup_at(candles, t, win_h):
    """px at t / px at t-win_h - 1, from the asset's hourly candles. None if < win_h of prior history."""
    by_t = {int(c["t"]): c["c"] for c in candles}
    t0 = t - win_h * 3600
    if t not in by_t or t0 not in by_t or by_t[t0] <= 0:
        return None
    return by_t[t] / by_t[t0] - 1.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path"); ap.add_argument("--win", type=int, default=24); ap.add_argument("--cost", type=float, default=0.01)
    a = ap.parse_args()
    d = json.load(open(a.path))
    rows, excluded, splits = [], 0, defaultdict(int)
    for w in d["weeks"]:
        # all exit timestamps in this week, per token, for the rotation (co-exit) proxy
        exits = defaultdict(set)
        for asset in w["assets"]:
            for p in asset["positions"]:
                exits[asset["symbol"]].add(int(p["exit_t"]))
        for asset in w["assets"]:
            sym = asset["symbol"]
            for p in asset["positions"]:
                et = int(p["entry_t"])
                ru = runup_at(asset.get("candles", []), et, a.win)
                if ru is None:
                    excluded += 1
                    continue
                net = p["exit_price"] / p["entry_price"] - 1.0 - a.cost
                size = p["qty"] * p["entry_price"]
                rot = any(et in ts for s, ts in exits.items() if s != sym)   # another token sold THIS hour
                rows.append({"ru": ru, "net": net, "size": size, "rot": rot, "split": w["split"]})
                splits[w["split"]] += 1
    print(f"{d['meta']['source_run']} | run-up win {a.win}h | cost {a.cost:.0%} | "
          f"positions used={len(rows)} excluded(<{a.win}h hist)={excluded} | splits={dict(splits)}")

    def table(name, subset):
        print(f"\n=== {name}  (n={len(subset)}) ===")
        print(f"  {'runup bin':>12} | {'n':>4} {'mean_net':>9} {'win':>5} {'$size-wtd':>10} {'sum$PnL':>9}")
        for lo, hi in BINS:
            b = [r for r in subset if lo <= r["ru"] < hi]
            if not b:
                continue
            n = len(b)
            mn = sum(r["net"] for r in b) / n
            win = sum(r["net"] > 0 for r in b) / n
            wtd = sum(r["net"] * r["size"] for r in b) / sum(r["size"] for r in b)
            spnl = sum(r["net"] * r["size"] for r in b)
            lab = f"{lo:+.0%}..{hi:+.0%}" if hi < 9 else f"{lo:+.0%}+    "
            print(f"  {lab:>12} | {n:4d} {mn:+8.2%} {win:4.0%} {wtd:+9.2%} {spnl:+8.0f}")

    table("ROTATION-LIKELY entries (co-exit same hour)", [r for r in rows if r["rot"]])
    table("CASH entries (no co-exit)", [r for r in rows if not r["rot"]])
    table("ALL entries", rows)
    # per-split persistence for the rotation subset (does a high-run-up penalty hold OOS?)
    print("\n-- ROTATION-LIKELY, run-up >= 15%, per split (OOS persistence) --")
    for sp in ("train", "val", "test"):
        hi = [r for r in rows if r["rot"] and r["ru"] >= 0.15 and r["split"] == sp]
        if hi:
            print(f"   {sp:6} n={len(hi):3d}  mean_net={sum(r['net'] for r in hi)/len(hi):+.2%}  "
                  f"win={sum(r['net']>0 for r in hi)/len(hi):.0%}")


if __name__ == "__main__":
    main()
