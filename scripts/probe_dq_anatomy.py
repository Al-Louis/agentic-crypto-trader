"""P-DQ-ANATOMY — when a longer EXIT-EMA pushes a cold week past the 30% DD gate, is it (a) RIDING A
LOSER DOWN (real capital loss, week ends down) or (b) CATCHING A PUMP then GIVING IT BACK (week ends up
/ flat, big peak-to-trough but profitable)? Drawdown = peak/cummax can't tell them apart; this does.

Per OOS week at exit-EMA 72 (base) vs 168: end-return, PEAK equity multiple, max-DD, and the universe's
biggest within-week token move (was a pump even available). pump-giveback => peak_mult high AND end >~0;
loser => peak_mult ~1 AND end < 0. Torch-free.
"""
from __future__ import annotations
import sys; sys.path[:0] = ["scripts", "src"]
import numpy as np
from datetime import datetime, timezone
from train_rl import load_data, build_volume_panel, time_split
from trader.train import weekly_eval as we
from trader.strategy.rung0 import build_rung0, run_rung0

WARMUP = 168


def d(ws): return datetime.fromtimestamp(int(ws), timezone.utc).strftime("%Y-%m-%d")


returns, btc, _a, liq = load_data()
tr, va, te = time_split(returns)
val_start, test_start = int(va.index[0]), int(te.index[0])

print(f"{'exitEMA':>7} {'week':11}{'split':5}{'endRet':>8}{'peakX':>7}{'maxDD':>7}{'peak@h':>7}"
      f"{'topTokMove':>12}  classification")
for N in (None, 168):
    rows = []
    for ws, win in we.cold_week_windows(returns):
        sp = we.split_label(ws, val_start, test_start)
        if sp == "train":
            continue
        uni = we.causal_voltop_universe(win, k=10)
        sigvol = build_volume_panel(list(win.columns), win.index)
        sig = build_rung0(win, tokens=uni, volume=sigvol, vol_mult=2.0, ema_span=72, exit_ema_span=N)
        eq, _, _ = run_rung0(win, sig, liq, warmup=WARMUP)
        eq = eq.iloc[WARMUP:].to_numpy()
        if len(eq) < 2:
            continue
        end_ret = eq[-1] / eq[0] - 1.0
        peak_mult = eq.max() / eq[0]
        peak_h = int(np.argmax(eq))
        dd = abs(float((eq / np.maximum.accumulate(eq) - 1.0).min()))
        # biggest within-week token PEAK move in the universe (was a pump even available)
        px = (1.0 + win.fillna(0.0)).cumprod()
        peakmoves = [(t, float(px[t].iloc[WARMUP:].max() / px[t].iloc[WARMUP] - 1.0)) for t in uni]
        bt, bm = max(peakmoves, key=lambda x: x[1])
        if dd > 0.28:
            cls = ("PUMP-GIVEBACK (profitable, gave back)" if peak_mult > 1.25 and end_ret > -0.05
                   else "rode a LOSER down" if peak_mult < 1.12 and end_ret < 0
                   else "mixed")
            rows.append((d(ws), sp, end_ret, peak_mult, dd, peak_h, bt, bm, cls))
    lbl = "72" if N is None else str(N)
    if not rows:
        print(f"{lbl:>7}  (no week with DD>28%)")
    for (wk, sp, er, pm, dd, ph, bt, bm, cls) in rows:
        print(f"{lbl:>7} {wk:11}{sp:5}{er*100:>+7.0f}%{pm:>6.2f}x{dd*100:>6.0f}%{ph:>6}h"
              f"{f'{bt} +{bm*100:.0f}%':>12}  {cls}")
