"""P-EMA-SPAN — is the rung-0 trend-EMA span (72h = ~3 days) simply TOO TIGHT? Sweep it through the
DEPLOYMENT-HONEST cold-weekly grader and read return AND worst-week max-DD (the DQ object), val (decision)
+ test (confirm), + the FF week specifically. A longer EMA cuts false consolidation-breaks (fewer noise
exits, holds FF into its rip) BUT lags real breakdowns (cuts later -> fatter losers -> DQ risk). This
measures the net tradeoff. Torch-free; reuses build_rung0 (it takes ema_span) + run_rung0 + weekly_eval.

NOTE: ema_span changes BOTH the ignition (cushion>0/ema_up) and the exit (price<ema) — a global sweep
(net effect). If a longer span wins net, the next step is an EXIT-ONLY isolation (entry 72 / exit longer).
"""
from __future__ import annotations
import sys; sys.path[:0] = ["scripts", "src"]
import numpy as np
import pandas as pd
from train_rl import load_data, build_volume_panel, time_split
from trader.train import weekly_eval as we
from trader.strategy.rung0 import build_rung0, run_rung0

WARMUP = 168
SPANS = [50, 72, 100, 168, 200]
FF_WEEK = int(pd.Timestamp("2026-04-06", tz="UTC").timestamp())


def wkret(win, uni, span):
    sig = build_rung0(win, tokens=uni, volume=None, vol_mult=2.0, ema_span=span)
    eq, _, _ = run_rung0(win, sig, liq, warmup=WARMUP)
    eq = eq.iloc[WARMUP:]
    return float(eq.iloc[-1] / eq.iloc[0] - 1.0), abs(float((eq / eq.cummax() - 1.0).min()))


def ci(v, n=2000, seed=0):
    v = np.asarray(v, float)
    r = np.random.default_rng(seed)
    m = v[r.integers(0, v.size, (n, v.size))].mean(1)
    return v.mean(), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


returns, btc, _a, liq = load_data()
vol = build_volume_panel(list(returns.columns), returns.index)
tr, va, te = time_split(returns)
val_start, test_start = int(va.index[0]), int(te.index[0])

res = {N: {"val": [], "test": [], "ff": None} for N in SPANS}
for ws, win in we.cold_week_windows(returns):
    sp = we.split_label(ws, val_start, test_start)
    if sp == "train":
        continue
    uni = we.causal_voltop_universe(win, k=10)
    sigvol = build_volume_panel(list(win.columns), win.index)  # vol panel for this window
    for N in SPANS:
        sig = build_rung0(win, tokens=uni, volume=sigvol, vol_mult=2.0, ema_span=N)
        eq, _, _ = run_rung0(win, sig, liq, warmup=WARMUP)
        eq = eq.iloc[WARMUP:]
        r = float(eq.iloc[-1] / eq.iloc[0] - 1.0); dd = abs(float((eq / eq.cummax() - 1.0).min()))
        res[N][sp].append((r, dd))
        if abs(ws - FF_WEEK) < 3 * 3600:
            res[N]["ff"] = (r, dd)

print(f"cold-weekly rung-0, vol_mult 2.0, causal top-10 | EMA-span sweep | val+test OOS")
print(f"{'span':>5}{'VAL mean[CI]':>22}{'valDDworst':>11}{'TEST mean':>10}{'testDDworst':>12}"
      f"{'DQwk':>5}{'FF-week ret':>12}")
for N in SPANS:
    vr = [x[0] for x in res[N]["val"]]; vd = [x[1] for x in res[N]["val"]]
    te_r = [x[0] for x in res[N]["test"]]; td = [x[1] for x in res[N]["test"]]
    alldd = vd + td
    m, lo, hi = ci(vr)
    ffr = res[N]["ff"]
    tag = "  <- rule" if N == 72 else ""
    print(f"{N:>5}{f'{m*100:+.1f}[{lo*100:+.0f},{hi*100:+.0f}]':>22}{max(vd)*100:>10.0f}%"
          f"{np.mean(te_r)*100:>9.1f}%{max(td)*100:>11.0f}%{sum(d>0.30 for d in alldd):>5}"
          f"{(ffr[0]*100 if ffr else float('nan')):>10.0f}%{tag}")
print("\nREAD: a longer span WINS iff VAL mean rises (CI-lo ideally >72's) AND worst-week DD does NOT "
      "worsen / stays <30% (DQwk=0), confirmed on TEST. If DD blows past 30% or test contradicts val, "
      "the longer EMA just trades noise-breaks for DQ-sized losers.")
