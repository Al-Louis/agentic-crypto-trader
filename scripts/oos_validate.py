"""Out-of-sample validation of the volatility-tilt selection.

The sweep picked `vol-top8` using *full-sample* volatility → in-sample selection bias. Honest
test: pick the high-vol subset on a **train** split, evaluate the tournament metrics on a
held-out **test** split, and check whether the volatility *ranking persists* train→test (if
high-vol tokens stay high-vol, the tilt generalizes; if vol rank is random across periods, it
was a fluke). Compares the train-selected subset to the all-20 baseline and to the
test-selected subset (the in-sample-on-test ceiling).

Caveat: one chronological split = one OOS regime; walk-forward would be more robust.

Run:  .venv/Scripts/python.exe scripts/oos_validate.py [--train-frac 0.6 --samples 400]
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

from trader.sim.resample import evaluate_windows  # noqa: E402
from trader.sim.strategies import static_subset  # noqa: E402


def _tourney(df: pd.DataFrame, bar: float) -> dict:
    big = df["ret"] > bar
    return {"tourney": float((big & ~df["dq"]).mean()), "p_big": float(big.mean()),
            "p_dq": float(df["dq"].mean()), "p95": float(df["ret"].quantile(0.95)),
            "med": float(df["ret"].median())}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-frac", type=float, default=0.6)
    ap.add_argument("--samples", type=int, default=400)
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
    if not log_ret:
        print("no factor features — run scripts/build_factor_features.py first")
        return
    returns = np.expm1(pd.DataFrame(log_ret).sort_index())
    liq = {s["symbol"]: (s.get("liq_usd") or 0.0)
           for s in json.load(open("data/selection.json", encoding="utf-8"))}

    n = len(returns)
    split = int(n * args.train_frac)
    train, test = returns.iloc[:split], returns.iloc[split:]
    vol_tr, vol_te = train.std(), test.std()
    rank_corr = vol_tr.rank().corr(vol_te.rank())                 # Spearman = Pearson on ranks

    top8_tr = list(vol_tr.sort_values(ascending=False).head(8).index)
    top5_tr = list(vol_tr.sort_values(ascending=False).head(5).index)
    top8_te = list(vol_te.sort_values(ascending=False).head(8).index)
    overlap = len(set(top8_tr) & set(top8_te))

    print(f"OOS validation: train {len(train)} bars / test {len(test)} bars "
          f"(frac {args.train_frac}), {args.samples} test windows\n")
    print(f"  volatility-rank persistence (Spearman train→test): {rank_corr:+.2f}")
    print(f"  vol-top8 overlap train vs test: {overlap}/8")
    print(f"    train top8: {top8_tr}")
    print(f"    test  top8: {top8_te}\n")

    candidates = [
        ("all-20 (baseline)", list(returns.columns)),
        ("vol-top8 (train-sel)", top8_tr),
        ("vol-top5 (train-sel)", top5_tr),
        ("vol-top8 (test-sel = ceiling)", top8_te),
    ]
    print(f"  evaluated on the TEST split:")
    print(f"  {'portfolio':30} {'med':>7} {'p95':>7} {f'P(>{args.bar:.0%})':>8} {'P(DQ)':>6} {'TOURNEY':>8}")
    for name, subset in candidates:
        df = evaluate_windows(test, static_subset(subset), liq, rebalance_every=24,
                              n_samples=args.samples, capital=10_000.0, seed=7)
        t = _tourney(df, args.bar)
        print(f"  {name:30} {t['med']:>+6.1%} {t['p95']:>+6.1%} {t['p_big']:>7.0%} "
              f"{t['p_dq']:>5.0%} {t['tourney']:>7.0%}")

    print("\n  Real if: rank persistence is positive/high, train-sel vol-top8 still beats all-20")
    print("  on TOURNEY out-of-sample, and train-sel is close to the test-sel ceiling. (One split.)")


if __name__ == "__main__":
    main()
