"""Synthetic-crash stress test — does `stress50`/`trend50` cap the tail the sample can't show?

Splices a synthetic crash week (BTC drops; high-vol alts amplify via a stress beta) after real
warmup and runs the candidate's overlay variants through the cost-aware backtester. Reports, per
crash severity × shape, the mean weekly drawdown / return and P(DQ>30%) for ungated vs the gates.

Run:  .venv/Scripts/python.exe scripts/crash_test.py [--beta 1.5 --seeds 20]
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

from trader.sim.backtest import run_xs_backtest  # noqa: E402
from trader.sim.crash import crash_path, simulate_crash_panel  # noqa: E402
from trader.sim.metrics import PerformanceMetrics  # noqa: E402
from trader.strategy import build_candidate, select_vol_tokens  # noqa: E402

W = 168                                              # warmup bars + crash-week bars
SEVERITIES = [-0.15, -0.25, -0.35, -0.50]
SHAPES = ["linear", "sharp"]
VARIANTS = [("ungated", "none"), ("stress50", "stress50"), ("trend50", "trend50")]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--beta", type=float, default=1.5, help="alt stress beta to BTC in the crash")
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--k", type=int, default=8)
    args = ap.parse_args()
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:  # noqa: BLE001
        pass

    log_ret, resid_std = {}, {}
    for f in sorted(glob.glob("data/features/*_factor.parquet")):
        sym = os.path.basename(f)[:-len("_factor.parquet")]
        df = pd.read_parquet(f).set_index("timestamp")
        log_ret[sym] = df["r_alt"]
        resid_std[sym] = float(df["residual"].std())
    returns = np.expm1(pd.DataFrame(log_ret).sort_index())
    liq = {s["symbol"]: (s.get("liq_usd") or 0.0)
           for s in json.load(open("data/selection.json", encoding="utf-8"))}
    top = select_vol_tokens(returns, args.k)

    btc = pd.read_parquet("data/anchor/BTC_USDT/1h.parquet")
    btc_close = (btc["close"]).reset_index(drop=True)
    warm_alt = returns[top].iloc[-W:].reset_index(drop=True)
    warm_btc = btc_close.iloc[-W:].reset_index(drop=True)

    print(f"crash stress test: vol-top{args.k}, stress beta {args.beta}, {args.seeds} seeds/cell\n"
          f"  {'scenario':18} {'ungated':>22} {'stress50':>22} {'trend50':>22}")
    print(f"  {'':18} {'maxDD  ret  P(DQ)':>22} {'maxDD  ret  P(DQ)':>22} {'maxDD  ret  P(DQ)':>22}")

    for shape in SHAPES:
        for sev in SEVERITIES:
            btc_crash = crash_path(W, sev, shape)
            crash_close = warm_btc.to_numpy()[-1] * np.cumprod(1 + btc_crash)
            btc_full = pd.Series(np.concatenate([warm_btc.to_numpy(), crash_close]),
                                 index=range(2 * W))
            cells = []
            for _, overlay in VARIANTS:
                dds, rets, dqs = [], [], []
                for seed in range(args.seeds):
                    crash_alt = simulate_crash_panel(top, btc_crash, args.beta, resid_std, seed)
                    alt = pd.concat([warm_alt, crash_alt], ignore_index=True)
                    alt.index = range(2 * W)
                    wf = build_candidate(alt, btc_full, overlay=overlay, tokens=top)
                    out = run_xs_backtest(alt, wf, liq, warmup=W, rebalance_every=24, capital=10_000)
                    eq = out["equity"].to_numpy()
                    dd = PerformanceMetrics._max_drawdown(eq)
                    dds.append(dd)
                    rets.append(eq[-1] / 10_000 - 1)
                    dqs.append(dd > 0.30)
                cells.append(f"{np.mean(dds):>5.0%} {np.mean(rets):>+5.0%} {np.mean(dqs):>5.0%}")
            print(f"  {shape+' BTC'+f'{sev:.0%}':18} {cells[0]:>22} {cells[1]:>22} {cells[2]:>22}")

    print("\n  P(DQ) = fraction of seeds with weekly drawdown > 30%. The gate is validated if it")
    print("  cuts maxDD/P(DQ) hard where ungated breaches the gate. (Synthetic; stress-beta assumed.)")


if __name__ == "__main__":
    main()
