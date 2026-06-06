"""IC analysis — does the factor residual-momentum predict forward returns?

Loads the per-alt factor parquets, builds the cross-sectional signal (resid_mom) and
return panels, and reports the Information Coefficient across horizons — alongside a naive
cross-sectional **price-momentum** baseline (does the factor decomposition add anything?).

The cheapest honest gate before a backtest. Run:
  .venv/Scripts/python.exe scripts/ic_analysis.py
"""

from __future__ import annotations

import glob
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from trader.sim.ic import cross_sectional_ic, forward_return, ic_summary  # noqa: E402

HORIZONS = [1, 6, 24, 72, 168]   # hours
MOM_LOOKBACK = 24                # naive momentum baseline: past 24h return


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:  # noqa: BLE001
        pass

    sig, ret = {}, {}
    for f in sorted(glob.glob("data/features/*_factor.parquet")):
        sym = os.path.basename(f)[:-len("_factor.parquet")]
        df = pd.read_parquet(f).set_index("timestamp")
        sig[sym], ret[sym] = df["resid_mom"], df["r_alt"]
    if not sig:
        print("no factor features found — run scripts/build_factor_features.py first")
        return

    resid_mom = pd.DataFrame(sig).sort_index()
    returns = pd.DataFrame(ret).sort_index()
    momentum = returns.rolling(MOM_LOOKBACK).sum()        # naive cross-sectional momentum
    print(f"IC analysis: {resid_mom.shape[1]} alts, {resid_mom.shape[0]} hourly bars\n")

    signals = {"resid_mom": resid_mom, f"momentum_{MOM_LOOKBACK}h": momentum}
    print(f"  {'signal':14} {'horizon':>8} {'n':>6} {'n_indep':>8} {'mean_IC':>9} "
          f"{'IC_IR':>7} {'t_stat':>7} {'hit%':>6}")
    for name, s in signals.items():
        for k in HORIZONS:
            fwd = returns.apply(lambda c: forward_return(c, k))
            r = ic_summary(cross_sectional_ic(s, fwd), horizon_bars=k)
            print(f"  {name:14} {f'{k}h':>8} {r['n']:>6} {r['n_indep']:>8} "
                  f"{r['mean_ic']:>9.4f} {r['ic_ir']:>7.3f} {r['t_stat']:>7.2f} "
                  f"{r['hit_rate']*100:>5.0f}%")
        print()

    print("  reading it: |mean_IC| ~0.02-0.05 with |t_stat| > 2 = a real (if small) edge;")
    print("  mean_IC ~0 / |t_stat| < 2 = no edge. Compare resid_mom vs the momentum baseline.")


if __name__ == "__main__":
    main()
