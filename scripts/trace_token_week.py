"""Forensic: what did a published sim ACTUALLY do on one token's big-rip week? Reads the bundle's
per-asset candles + folded positions (the exact fills, cost baked in) and shows entry/exit vs the
intra-week rip — so "captured / missed / lost" gets a mechanical answer, not a guess.

    python scripts/trace_token_week.py --run-id <id> --token FF [--week-start <unix or YYYY-MM-DD>]

No torch — pure JSON over the published simulated_trades.json. DESKTOP (bundles live in runs-rl/).
"""
from __future__ import annotations
import argparse, json, os
from datetime import datetime, timezone


def dts(t): return datetime.fromtimestamp(int(t), timezone.utc).strftime("%m-%d %H:%M")
def dday(t): return datetime.fromtimestamp(int(t), timezone.utc).strftime("%Y-%m-%d")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run-id", required=True)
    p.add_argument("--token", required=True)
    p.add_argument("--week-start", default=None, help="unix ts or YYYY-MM-DD to pick the week; "
                   "default = the token's biggest-rip week")
    args = p.parse_args()
    d = json.load(open(os.path.join("runs-rl", args.run_id, "simulated_trades.json")))

    want = None
    if args.week_start:
        want = (int(datetime.strptime(args.week_start, "%Y-%m-%d").replace(tzinfo=timezone.utc).timestamp())
                if "-" in args.week_start else int(args.week_start))

    best = None
    for w in d["weeks"]:
        a = next((x for x in w["assets"] if x["symbol"] == args.token), None)
        if not a or not a.get("candles"):
            continue
        if want is not None and not (w["start"] <= want < w["end"]):
            continue
        cs = a["candles"]
        o = cs[0]["o"]
        hi = max(c["h"] for c in cs)
        rip = hi / o - 1.0 if o else 0.0
        if want is not None or best is None or rip > best[0]:
            best = (rip, w, a)
            if want is not None:
                break
    if not best:
        print(f"{args.token} not present (with candles) in any matching week of {args.run_id}")
        return

    rip, w, a = best
    cs = a["candles"]
    o, c_close = cs[0]["o"], cs[-1]["c"]
    hi = max(cs, key=lambda c: c["h"]); lo = min(cs, key=lambda c: c["l"])
    print(f"=== {args.run_id} | {args.token} | {w['label']} {dday(w['start'])}..{dday(w['end'])} ===")
    print(f"  price: open {o:.4g}  HIGH {hi['h']:.4g}@{dts(hi['t'])}  low {lo['l']:.4g}@{dts(lo['t'])}  close {c_close:.4g}")
    print(f"  intra-week RIP from open: {rip*100:+.0f}%   open->close: {(c_close/o-1)*100:+.0f}%")
    alloc = a.get("alloc_usd")
    print(f"  vol_rank {a.get('vol_rank')}  alloc_usd {alloc}")
    pos = a.get("positions", [])
    tot = 0.0
    print(f"  positions: {len(pos)}")
    for q in pos:
        pnl = q["qty"] * (q["exit_price"] - q["entry_price"]); tot += pnl
        ent_vs_hi = q["entry_price"] / hi["h"] - 1.0
        held_h = (q["exit_t"] - q["entry_t"]) / 3600.0
        print(f"    BUY {dts(q['entry_t'])} @{q['entry_price']:.4g} -> SELL {dts(q['exit_t'])} @{q['exit_price']:.4g}"
              f"  held {held_h:.0f}h  pnl ${pnl:+.0f}  (entry was {ent_vs_hi*100:+.0f}% vs the high)")
    print(f"  TOKEN WEEK PnL: ${tot:+.0f}   |   oracle buy-open/sell-high on alloc: "
          f"${(alloc or 0)*rip:+.0f}")


if __name__ == "__main__":
    main()
