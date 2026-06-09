"""Walk-forward (multi-window) threshold sweep for rung 0 — the robust way to spend the risk budget.

The single-window sweep overfit (val +167% -> test -17%/44%). This instead scores each config across
many random **7-day windows** (the competition's unit) drawn from train+val, and selects by the
**tournament objective** — maximize P(a big positive week) subject to a low P(DQ) *across windows* —
so a single window's noise can't win. The pick is then verified on **frozen-test** windows.

Fresh state machine per window (stateful strategy), fixed universe (no drift). Deterministic, no
training. Run in background (~thousands of small backtests).

    python scripts/sweep_rung0_wf.py
"""
from __future__ import annotations

import itertools
import sys

sys.path.insert(0, "scripts")
sys.path.insert(0, "src")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from train_rl import load_data, time_split  # noqa: E402
from trader.sim.backtest import run_xs_backtest  # noqa: E402
from trader.sim.metrics import PerformanceMetrics  # noqa: E402
from trader.sim.resample import WEEK_BARS, sample_window_starts  # noqa: E402
from trader.strategy.candidate import build_candidate, select_vol_tokens  # noqa: E402
from trader.strategy.rung0 import build_rung0  # noqa: E402

CAP, N_WIN, TOURNEY, DQ_GATE, MAX_PDQ = 10_000.0, 120, 0.15, 0.30, 0.05


def eval_windows(returns, build_fn, liq, n=N_WIN, seed=0):
    """Per-window weekly distribution for a (re-built-per-window) strategy."""
    rng = np.random.default_rng(seed)
    starts = sample_window_starts(len(returns), WEEK_BARS, WEEK_BARS, n, rng)
    rets, mdds, dqs = [], [], []
    for s in starts:
        sl = returns.iloc[s - WEEK_BARS: s + WEEK_BARS]
        out = run_xs_backtest(sl, build_fn(sl), liq, capital=CAP, warmup=WEEK_BARS, rebalance_every=24)
        eq = out["equity"].to_numpy()
        rets.append(eq[-1] / CAP - 1.0)
        mdds.append(PerformanceMetrics._max_drawdown(eq))
        dqs.append(mdds[-1] > DQ_GATE or out["n_rebalances"] < 7)
    rets, mdds, dqs = np.array(rets), np.array(mdds), np.array(dqs, dtype=bool)
    return {"tourney": float((rets > TOURNEY).mean()), "p_dq": float(dqs.mean()),
            "ret_med": float(np.median(rets)), "ret_p95": float(np.quantile(rets, 0.95)),
            "maxdd_p95": float(np.quantile(mdds, 0.95)), "p_profit": float((rets > 0).mean())}


def main():
    returns, btc, anchor, liq = load_data()
    train_r, val_r, test_r = time_split(returns)
    sel = pd.concat([train_r, val_r]).sort_index()
    universe = select_vol_tokens(sel, 8)
    print(f"selection windows from train+val ({len(sel)} bars); test {len(test_r)} bars")
    print(f"universe = {universe}\n")

    grid = {"stop_k": [0.10, 0.15, 0.20, 0.30], "max_weight": [0.25, 0.40, 0.60, 1.0],
            "cooldown": [0, 1, 2], "breakout_n": [24, 48, 72]}
    keys = list(grid)
    results = []
    for combo in itertools.product(*grid.values()):
        kw = dict(zip(keys, combo))
        m = eval_windows(sel, lambda sl, kw=kw: build_rung0(sl, tokens=universe, **kw), liq)
        results.append((kw, m))

    legal = sorted((r for r in results if r[1]["p_dq"] < MAX_PDQ),
                   key=lambda r: (-r[1]["tourney"], -r[1]["ret_med"]))
    print(f"{len(results)} configs; {len(legal)} gate-safe across windows (P(DQ) < {MAX_PDQ:.0%})\n")
    print(f"  top by tournament rate P(week > +{TOURNEY:.0%}), selection windows:")
    print(f"  {'stop_k':>7}{'maxW':>6}{'cool':>5}{'brkN':>5}{'tourney':>9}{'P(DQ)':>7}{'retMed':>8}{'ret95':>8}")
    for kw, m in legal[:8]:
        print(f"  {kw['stop_k']:>7.2f}{kw['max_weight']:>6.2f}{kw['cooldown']:>5}{kw['breakout_n']:>5}"
              f"{m['tourney']*100:>8.0f}%{m['p_dq']*100:>6.0f}%{m['ret_med']*100:>+7.1f}%{m['ret_p95']*100:>+7.1f}%")

    pick = legal[0][0] if legal else None
    print(f"\nPICK (best gate-safe tournament rate): {pick}")
    print(f"\n  TEST-window verification (frozen, same windows for all):")
    print(f"  {'strategy':20}{'tourney':>9}{'P(DQ)':>7}{'retMed':>8}{'ret95':>8}{'maxdd95':>9}")
    cands = [("rung0 default", lambda sl: build_rung0(sl, tokens=universe)),
             ("vol-top8 none", lambda sl: build_candidate(sl, tokens=universe, overlay="none")),
             ("vol-top8 trend50",
              lambda sl: build_candidate(sl, btc.reindex(sl.index), tokens=universe, overlay="trend50"))]
    if pick:
        cands.insert(0, ("rung0 PICK", lambda sl: build_rung0(sl, tokens=universe, **pick)))
    for nm, bf in cands:
        m = eval_windows(test_r, bf, liq, seed=1)
        print(f"  {nm:20}{m['tourney']*100:>8.0f}%{m['p_dq']*100:>6.0f}%{m['ret_med']*100:>+7.1f}%"
              f"{m['ret_p95']*100:>+7.1f}%{m['maxdd_p95']*100:>8.1f}%")


if __name__ == "__main__":
    main()
