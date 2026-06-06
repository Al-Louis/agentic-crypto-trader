"""Upper-tail sweep — the tournament objective: maximize P(big week) under a DQ cap.

It's a leaderboard (top-5 win), so median return loses; we want the strategy with the best
*upper tail* that still rarely breaches the 30% gate. Since entry-timing alpha is dead here
(it churns thin pools), the lever is **which fixed subset to buy-and-hold**: concentration
(k) and a volatility / beta tilt. Each candidate is resampled over many 7-day windows.

Caveat: subsets are defined from full-sample token stats (vol/beta) — a mild in-sample
selection bias for the analysis; a live entry would fix the subset from pre-window stats.

Run:  .venv/Scripts/python.exe scripts/tail_sweep.py [--samples 500 --bar 0.15]
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from trader.sim.backtest import NEVER  # noqa: E402
from trader.sim.resample import evaluate_windows  # noqa: E402
from trader.sim.strategies import static_subset  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", type=int, default=500)
    ap.add_argument("--capital", type=float, default=10_000.0)
    ap.add_argument("--bar", type=float, default=0.15, help="'contender' weekly return threshold")
    args = ap.parse_args()
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:  # noqa: BLE001
        pass

    log_ret, beta = {}, {}
    for f in sorted(glob.glob("data/features/*_factor.parquet")):
        sym = os.path.basename(f)[:-len("_factor.parquet")]
        df = pd.read_parquet(f).set_index("timestamp")
        log_ret[sym] = df["r_alt"]
        beta[sym] = df["beta_btc"].dropna().mean()
    if not log_ret:
        print("no factor features — run scripts/build_factor_features.py first")
        return
    returns = np.expm1(pd.DataFrame(log_ret).sort_index())
    sel = json.load(open("data/selection.json", encoding="utf-8"))
    liq = {s["symbol"]: (s.get("liq_usd") or 0.0) for s in sel}
    tier = {s["symbol"]: s.get("tier") for s in sel}

    vol = returns.std().sort_values(ascending=False)          # realized hourly vol per token
    beta_s = pd.Series(beta).sort_values(ascending=False)
    syms = list(returns.columns)

    candidates = {
        "all-20": syms,
        "vol-top3": list(vol.head(3).index),
        "vol-top5": list(vol.head(5).index),
        "vol-top8": list(vol.head(8).index),
        "vol-bot8 (defensive)": list(vol.tail(8).index),
        "beta-top5": list(beta_s.head(5).index),
        "meme tier": [s for s in syms if tier.get(s) == "meme"],
        "anchor tier": [s for s in syms if tier.get(s) == "anchor"],
    }

    print(f"upper-tail sweep: {len(syms)} alts, {args.samples} windows, capital "
          f"${args.capital:,.0f}, contender bar = +{args.bar:.0%}\n")
    print(f"  {'portfolio':22} {'k':>3} {'med':>7} {'p5':>7} {'p90':>7} {'p95':>7} "
          f"{f'P(>{args.bar:.0%})':>8} {'P(DQ)':>6} {'TOURNEY':>8}")
    rows = []
    for name, subset in candidates.items():
        if not subset:
            continue
        df = evaluate_windows(returns, static_subset(subset), liq, rebalance_every=NEVER,
                              n_samples=args.samples, capital=args.capital, seed=42)
        big = df["ret"] > args.bar
        tourney = float((big & ~df["dq"]).mean())             # P(contender week AND not DQ'd)
        rows.append({
            "name": name, "k": len(subset),
            "med": df["ret"].median(), "p5": df["ret"].quantile(0.05),
            "p90": df["ret"].quantile(0.90), "p95": df["ret"].quantile(0.95),
            "p_big": float(big.mean()), "p_dq": float(df["dq"].mean()), "tourney": tourney,
        })
    for r in sorted(rows, key=lambda r: r["tourney"], reverse=True):
        print(f"  {r['name']:22} {r['k']:>3} {r['med']:>+6.1%} {r['p5']:>+6.1%} "
              f"{r['p90']:>+6.1%} {r['p95']:>+6.1%} {r['p_big']:>7.0%} {r['p_dq']:>5.0%} "
              f"{r['tourney']:>7.0%}")

    print(f"\n  TOURNEY = P(weekly return > +{args.bar:.0%} AND not DQ'd) — the contender rate.")
    print("  Higher upper tail (p90/p95/P(big)) wins a leaderboard; watch the P(DQ) it costs.")


if __name__ == "__main__":
    main()
