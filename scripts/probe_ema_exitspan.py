"""P-EMA-EXITSPAN — the CLEAN test of the user's hypothesis: keep the ENTRY trend-EMA at 72 (ignitions
unchanged) and ONLY lengthen the EXIT weakness-EMA (the price<ema cut), so a multi-day consolidation's
shallow noise-dips don't trip the exit (the FF Apr-9 -0.1% break). Isolates the exit from the entry
degradation the global ema_span sweep suffered. Cold-weekly grader, val (decision) + test (confirm) +
FF week, return AND worst-week DD (DQ). Torch-free. exit_ema_span=None == baseline (byte-identical).
"""
from __future__ import annotations
import sys; sys.path[:0] = ["scripts", "src"]
import numpy as np
import pandas as pd
from train_rl import load_data, build_volume_panel, time_split
from trader.train import weekly_eval as we
from trader.strategy.rung0 import build_rung0, run_rung0

WARMUP = 168
EXIT_SPANS = [None, 100, 168, 200, 240]   # None = exit-EMA == entry-EMA (72): the baseline
FF_WEEK = int(pd.Timestamp("2026-04-06", tz="UTC").timestamp())


def ci(v, n=2000, seed=0):
    v = np.asarray(v, float)
    r = np.random.default_rng(seed)
    m = v[r.integers(0, v.size, (n, v.size))].mean(1)
    return v.mean(), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


returns, btc, _a, liq = load_data()
tr, va, te = time_split(returns)
val_start, test_start = int(va.index[0]), int(te.index[0])

res = {N: {"val": [], "test": [], "ff": None} for N in EXIT_SPANS}
for ws, win in we.cold_week_windows(returns):
    sp = we.split_label(ws, val_start, test_start)
    if sp == "train":
        continue
    uni = we.causal_voltop_universe(win, k=10)
    sigvol = build_volume_panel(list(win.columns), win.index)
    for N in EXIT_SPANS:
        sig = build_rung0(win, tokens=uni, volume=sigvol, vol_mult=2.0, ema_span=72, exit_ema_span=N)
        eq, _, _ = run_rung0(win, sig, liq, warmup=WARMUP)
        eq = eq.iloc[WARMUP:]
        r = float(eq.iloc[-1] / eq.iloc[0] - 1.0); dd = abs(float((eq / eq.cummax() - 1.0).min()))
        res[N][sp].append((r, dd))
        if abs(ws - FF_WEEK) < 3 * 3600:
            res[N]["ff"] = (r, dd)

print("entry-EMA fixed 72 | EXIT-EMA sweep | cold-weekly rung-0 (vol_mult 2.0, causal top-10) val+test")
print(f"{'exitEMA':>9}{'VAL mean[CI]':>22}{'valDDw':>8}{'TEST mean':>10}{'testDDw':>9}{'DQwk':>5}{'FF-wk':>8}")
base_val = None
for N in EXIT_SPANS:
    vr = [x[0] for x in res[N]["val"]]; vd = [x[1] for x in res[N]["val"]]
    te_r = [x[0] for x in res[N]["test"]]; td = [x[1] for x in res[N]["test"]]
    m, lo, hi = ci(vr); ff = res[N]["ff"]
    if N is None:
        base_val = m
    lbl = "72(base)" if N is None else str(N)
    print(f"{lbl:>9}{f'{m*100:+.1f}[{lo*100:+.0f},{hi*100:+.0f}]':>22}{max(vd)*100:>7.0f}%"
          f"{np.mean(te_r)*100:>9.1f}%{max(td)*100:>8.0f}%{sum(d > 0.30 for d in vd + td):>5}"
          f"{(ff[0] * 100 if ff else float('nan')):>7.0f}%")
print("\nREAD: a longer EXIT-EMA wins iff VAL mean rises above 72(base) AND worst-week DD does NOT worsen "
      "(stays <30%), confirmed on TEST + a better FF week. If DD climbs (holding losers longer) or test "
      "contradicts val, detuning the exit just trades FF-class noise-breaks for fatter losers.")
