"""Evaluate the user's 2026-06-19 proposal: drop the 7 BTC-"correlated" majors -> a FIXED 13-token
universe (no weekly causal re-pick), so high-vol spikes (the FF Apr 9-10 roundtrip) are never missed.

Four grounded, torch-free questions:
  1. Are the 7 removals actually BTC-correlated? (multi-horizon corr: 1h / 4h / 1d, full + val + test)
  2. What axis ARE they on? (trailing vol rank -> are the 7 just the 7 lowest-vol?)
  3. Does a fixed 13 actually keep FF/the big movers in view each week? (coverage vs causal top-k)
  4. Does it improve the DEPLOYMENT-HONEST substrate? (cold-weekly rung-0 RULE: top10-of-20 causal
     vs top10-of-13 causal vs FIXED-all-13), incl. a lookahead control (is KEEP13 ~ causal-top-13?).
"""
from __future__ import annotations
import os, sys; sys.path[:0] = ["scripts", "src"]
import numpy as np, pandas as pd
from datetime import datetime, timezone

from train_rl import load_data, build_volume_panel, time_split
from trader.train.weekly_eval import cold_week_windows, causal_voltop_universe, split_label, WARMUP
from trader.strategy.rung0 import build_rung0, run_rung0

VOL_MULT = 2.0   # match the ef2 substrate (k=10, vol_mult=2.0) so the comparison is apples-to-apples
def d(ts): return datetime.fromtimestamp(int(ts), timezone.utc).strftime("%Y-%m-%d")

returns, btc, anchor, liq = load_data()
vol = build_volume_panel(list(returns.columns), returns.index)
cols = list(returns.columns)
tr, va, te = time_split(returns)
val_start, test_start = int(va.index[0]), int(te.index[0])

REMOVE = ["ADA", "SFP", "XRP", "BabyDoge", "LINK", "LTC", "XAUt"]
KEEP13 = [c for c in cols if c not in REMOVE]
print(f"POOL={len(cols)}  REMOVE={len(REMOVE)}  KEEP13={len(KEEP13)}")
print("KEEP13:", sorted(KEEP13))
missing = [t for t in REMOVE if t not in cols]
if missing:
    print("!! NOT IN POOL:", missing)

# ---- BTC series: returns or price? normalize to returns ----
b = btc if isinstance(btc, pd.Series) else pd.Series(np.asarray(btc).reshape(-1)[:len(returns)], index=returns.index)
b = b.reindex(returns.index)
if b.abs().mean() > 0.2:           # looks like price levels -> convert
    b = b.pct_change()
print(f"\nBTC series: mean|.|={b.abs().mean():.4f} std={b.std():.4f}  (treated as returns)")

# ---- Q1: multi-horizon BTC correlation ----
def block_corr(r, bb, h):
    lr, blr = np.log1p(r.fillna(0.0)), np.log1p(bb.fillna(0.0))
    g = np.arange(len(lr)) // h
    A, B = lr.groupby(g).sum(), blr.groupby(g).sum()
    return {c: float(A[c].corr(B)) for c in cols}

print("\n=== Q1: BTC correlation by horizon (FULL sample) ===")
ch = {h: block_corr(returns, b, h) for h in (1, 4, 24)}
print(f"  {'token':11}{'1h':>8}{'4h':>8}{'1d':>8}   {'REMOVE?':>8}")
for c in sorted(cols, key=lambda x: -ch[24][x]):
    print(f"  {c:11}{ch[1][c]:+8.3f}{ch[4][c]:+8.3f}{ch[24][c]:+8.3f}   {'<-- cut' if c in REMOVE else '':>8}")

# ---- Q2: trailing-vol rank (are the 7 the 7 lowest-vol?) ----
print("\n=== Q2: full-sample hourly vol rank (low -> high) ===")
volr = {c: float(returns[c].std()) for c in cols}
for i, c in enumerate(sorted(cols, key=lambda x: volr[x]), 1):
    print(f"  {i:2}. {c:11} vol={volr[c]*100:5.2f}%   {'<-- cut' if c in REMOVE else ''}")

# ---- lookahead control: causal top-13 picked ONCE at val open vs KEEP13 ----
win_val = pd.concat([tr.tail(WARMUP), va])
causal13_atval = set(causal_voltop_universe(win_val, k=13))
print(f"\n=== lookahead control: causal-top-13 @ VAL OPEN ({d(val_start)}) vs KEEP13 ===")
print(f"  causal13@valopen: {sorted(causal13_atval)}")
print(f"  KEEP13         : {sorted(KEEP13)}")
print(f"  overlap={len(causal13_atval & set(KEEP13))}/13  "
      f"in KEEP13 not causal: {sorted(set(KEEP13)-causal13_atval)}  "
      f"in causal not KEEP13: {sorted(causal13_atval-set(KEEP13))}")

# ---- Q3 + Q4: per-week coverage + cold-weekly rung-0 over 3 universe schemes ----
def wk_rung0(win, uni):
    sig = build_rung0(win, tokens=uni, volume=vol, vol_mult=VOL_MULT)
    eq, records, _ = run_rung0(win, sig, liq, warmup=WARMUP)
    eq = eq.iloc[WARMUP:]
    ret = float(eq.iloc[-1] / eq.iloc[0] - 1.0)
    dd = abs(float((eq / eq.cummax() - 1.0).min()))
    return ret, dd

schemes = {"A_top10of20": [], "B_top10of13": [], "C_fixed13": []}
print("\n=== Q3/Q4: per-week (val+test) coverage + rung-0 return ===")
print(f"  {'week':12}{'split':5} {'bigmover(wk%)':>16} {'inA':>4}{'inB':>4}{'inC':>4}"
      f"   {'retA':>7}{'retB':>7}{'retC':>7}")
for ws, win in cold_week_windows(returns):
    if ws < val_start:
        continue
    sp = split_label(ws, val_start, test_start)
    uniA = causal_voltop_universe(win, k=10)
    uniB = causal_voltop_universe(win[KEEP13], k=10)
    uniC = list(KEEP13)
    # biggest within-week mover
    px = (1.0 + win.fillna(0.0)).cumprod()
    mv = {c: float(px[c].iloc[-1] / px[c].iloc[WARMUP] - 1.0) for c in cols}
    big = max(cols, key=lambda x: mv[x])
    rA, ddA = wk_rung0(win, uniA); rB, ddB = wk_rung0(win, uniB); rC, ddC = wk_rung0(win, uniC)
    schemes["A_top10of20"].append((rA, ddA)); schemes["B_top10of13"].append((rB, ddB)); schemes["C_fixed13"].append((rC, ddC))
    print(f"  {d(ws):12}{sp:5} {big+f'({mv[big]*100:+.0f}%)':>16} "
          f"{('Y' if big in uniA else '-'):>4}{('Y' if big in uniB else '-'):>4}{('Y' if big in uniC else '-'):>4}"
          f"   {rA*100:+6.1f}%{rB*100:+6.1f}%{rC*100:+6.1f}%")

print("\n=== Q4 aggregate (val+test cold weeks, rung-0 RULE, vol_mult=2.0) ===")
print(f"  {'scheme':14}{'n':>3}{'mean':>8}{'median':>8}{'winrate':>8}{'worstDD':>8}{'DQwk':>5}")
for name, rows in schemes.items():
    r = np.array([x[0] for x in rows]); dd = np.array([x[1] for x in rows])
    print(f"  {name:14}{len(r):3}{r.mean()*100:+7.1f}%{np.median(r)*100:+7.1f}%"
          f"{(r>0).mean()*100:6.0f}% {dd.max()*100:6.1f}% {int((dd>0.30).sum()):4}")
