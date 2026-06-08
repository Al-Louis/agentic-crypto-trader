"""Refined regime overlay — find a gate that keeps the upside while capping the bear tail.

The blunt all-or-nothing 72h gate was overpriced (cut the tournament rate in half). This
sweeps refined exposure variants — partial de-risk (half cash vs full), a whipsaw dead-band,
and extreme-stress-only gating — vs the ungated tilt, reporting the tournament objective AND
the bear-week conditioning. The winner keeps `TOURNEY` near ungated while cutting bear-week
drawdown/DQ.

Run:  .venv/Scripts/python.exe scripts/regime_refine.py [--k 8 --samples 500]
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

from trader.data.anchor import load_anchor  # noqa: E402
from trader.features.regime import stress_exposure, trend_exposure  # noqa: E402
from trader.sim.resample import WEEK_BARS, evaluate_windows  # noqa: E402
from trader.sim.strategies import regime_scaled, static_subset  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--samples", type=int, default=500)
    ap.add_argument("--bar", type=float, default=0.15)
    args = ap.parse_args()
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:  # noqa: BLE001
        pass

    log_ret = {}
    for f in sorted(glob.glob("data/features/*_factor.parquet")):
        sym = os.path.basename(f)[:-len("_factor.parquet")]
        log_ret[sym] = pd.read_parquet(f).set_index("timestamp")["r_alt"]
    returns = np.expm1(pd.DataFrame(log_ret).sort_index())
    liq = {s["symbol"]: (s.get("liq_usd") or 0.0)
           for s in json.load(open("data/selection.json", encoding="utf-8"))}
    top = list(returns.std().sort_values(ascending=False).head(args.k).index)

    btc = load_anchor("BTC/USDT", "1h")
    close = btc.set_index((btc["timestamp"] // 1000))["close"].sort_index()
    btc_ret = close.pct_change().reindex(returns.index, method="ffill").fillna(0.0)

    def aligned(s):
        return s.reindex(returns.index, method="ffill").fillna(0.0)

    variants = [
        ("ungated", None),
        ("trend cash (off0)", trend_exposure(close, 72, off=0.0)),
        ("trend 50% (off.5)", trend_exposure(close, 72, off=0.5)),
        ("trend 50% +band2%", trend_exposure(close, 72, off=0.5, band=0.02)),
        ("stress 50% (-8%/3d)", stress_exposure(close, 72, drop=-0.08, off=0.5)),
        ("stress cash (-10%/3d)", stress_exposure(close, 72, drop=-0.10, off=0.0)),
    ]

    print(f"refined overlay: vol-top{args.k}, {args.samples} windows, contender +{args.bar:.0%}\n")
    print(f"  {'variant':22} {'exp':>5} {'TOURNEY':>8} {'P(DQ)':>6} {'p95':>7} "
          f"{'bearDD':>7} {'bearDQ':>7} {'bullRet':>8}")
    for name, expo in variants:
        if expo is None:
            strat, mean_exp = static_subset(top), 1.0
        else:
            ex = aligned(expo)
            strat, mean_exp = regime_scaled(static_subset(top), ex), float(ex.mean())
        df = evaluate_windows(returns, strat, liq, n_samples=args.samples, seed=11)
        df["btc"] = df["start"].map(lambda s: float((1 + btc_ret.iloc[s:s + WEEK_BARS]).prod() - 1))
        bear, bull = df[df["btc"] < 0], df[df["btc"] >= 0]
        big = df["ret"] > args.bar
        print(f"  {name:22} {mean_exp:>5.2f} {float((big & ~df['dq']).mean()):>7.0%} "
              f"{df['dq'].mean():>5.0%} {df['ret'].quantile(.95):>+6.1%} "
              f"{bear['maxdd'].mean():>6.1%} {bear['dq'].mean():>6.0%} {bull['ret'].mean():>+7.1%}")

    print("\n  Want: TOURNEY near the ungated row, with bearDD/bearDQ cut. exp = mean exposure")
    print("  (1.00 = always invested). Caveat: bull-conditioned sample, no real crash.")


if __name__ == "__main__":
    main()
