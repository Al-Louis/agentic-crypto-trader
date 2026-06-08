"""Regime overlay validation — does gating the vol tilt to risk-on insure the bear case?

Holds vol-top8 when BTC is risk-on (above its trailing EMA) and rotates to cash/stables when
risk-off. Compared to the ungated tilt and the all-20 baseline over resampled 7-day windows —
overall, and **conditioned on each window's BTC return** (the bull-heavy sample masks the
overlay's value; the bear-window split is where it should earn its keep).

Run:  .venv/Scripts/python.exe scripts/regime_overlay.py [--ema-span 72 --samples 500]
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
from trader.features.regime import btc_risk_on  # noqa: E402
from trader.sim.resample import WEEK_BARS, evaluate_windows  # noqa: E402
from trader.sim.strategies import regime_gated, static_subset  # noqa: E402


def _tail(df: pd.DataFrame, bar: float) -> str:
    big = df["ret"] > bar
    return (f"med {df['ret'].median():>+6.1%}  p95 {df['ret'].quantile(.95):>+6.1%}  "
            f"P(>{bar:.0%}) {big.mean():>4.0%}  P(DQ) {df['dq'].mean():>4.0%}  "
            f"TOURNEY {float((big & ~df['dq']).mean()):>4.0%}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ema-span", type=int, default=72)
    ap.add_argument("--samples", type=int, default=500)
    ap.add_argument("--bar", type=float, default=0.15)
    ap.add_argument("--k", type=int, default=8)
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

    # vol-top-k subset + BTC risk-on signal aligned to the alt grid
    top = list(returns.std().sort_values(ascending=False).head(args.k).index)
    btc = load_anchor("BTC/USDT", "1h")
    btc_close = btc.set_index((btc["timestamp"] // 1000))["close"].sort_index()
    risk_on = btc_risk_on(btc_close, args.ema_span).reindex(returns.index, method="ffill").fillna(False)
    btc_ret = btc_close.pct_change().reindex(returns.index, method="ffill").fillna(0.0)
    pct_on = float(risk_on.mean())
    print(f"regime overlay: vol-top{args.k} {top}\n  BTC risk-on {pct_on:.0%} of bars "
          f"(EMA {args.ema_span}h), {args.samples} windows\n")

    ungated = evaluate_windows(returns, static_subset(top), liq, n_samples=args.samples, seed=11)
    gated = evaluate_windows(returns, regime_gated(static_subset(top), risk_on), liq,
                             n_samples=args.samples, seed=11)
    baseline = evaluate_windows(returns, static_subset(list(returns.columns)), liq,
                                n_samples=args.samples, seed=11)

    print("  OVERALL")
    for name, df in [("all-20", baseline), (f"vol-top{args.k} ungated", ungated),
                     (f"vol-top{args.k} GATED", gated)]:
        print(f"    {name:20} {_tail(df, args.bar)}")

    # condition each window on its BTC return (same seed -> same window starts)
    bw = {s: float((1 + btc_ret.iloc[s:s + WEEK_BARS]).prod() - 1) for s in ungated["start"]}
    for df in (ungated, gated):
        df["btc"] = df["start"].map(bw)
    print("\n  BY WINDOW REGIME (BTC return over the week)")
    for label, mask_fn in [("BEAR weeks (BTC<0)", lambda d: d["btc"] < 0),
                           ("BULL weeks (BTC>=0)", lambda d: d["btc"] >= 0)]:
        u, g = ungated[mask_fn(ungated)], gated[mask_fn(gated)]
        print(f"    {label} (n~{len(u)}):")
        print(f"      ungated  ret {u['ret'].mean():>+6.1%}  maxDD {u['maxdd'].mean():>5.1%}  "
              f"P(DQ) {u['dq'].mean():>4.0%}")
        print(f"      GATED    ret {g['ret'].mean():>+6.1%}  maxDD {g['maxdd'].mean():>5.1%}  "
              f"P(DQ) {g['dq'].mean():>4.0%}")

    print("\n  The gate should cut bear-week drawdown/DQ (insurance) while keeping most bull-week")
    print("  upside. If it barely triggers (high risk-on %) it's bull-conditioned — note that.")


if __name__ == "__main__":
    main()
