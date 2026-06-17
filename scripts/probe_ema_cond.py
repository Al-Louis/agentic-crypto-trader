"""P-EMA-COND — is the rung-0 EXIT's EMA-break a profit-giveback or a stop-loss? Should it be P&L-gated?

THE RUNG-0 EXIT (rung0.run_rung0:121 / event_env._scan_bar:394-397) fires on:
    price < peak*(1-stop_k)            # TRAILING stop off the running peak  (the disaster-stop)
    OR  price < ema  (i.e. cushion<0)  # the EMA-BREAK clause                (fires REGARDLESS of P&L)
The second clause is unconditional: it sells a WINNER or a BREAKEVEN position on an ordinary
below-EMA pullback, giving back profit it would otherwise keep. The ZEC forensic is the case:
ZEC's Apr-3 EMA-break sells were only -0.9% / -1.9% below the BUY price (NOT deep losers), and the
token then ran +16%; the unconditional EMA-break harvested those before the run-up.

THE USER'S PROPOSED CONDITIONAL — gate ONLY the EMA-break clause on the position's P&L:
    price < peak*(1-stop_k)                       # trailing stop: UNCONDITIONAL (kept)
    OR ( price < ema  AND  unreal < -T )          # EMA-break now only fires if >T underwater
  where  unreal = price/entry_px - 1   (current price vs the BUY price; causal, no lookahead).
This turns the EMA-break from an unconditional profit-giveback into a STOP-LOSS: deep losers
(>T underwater) are still cut at the EMA-break; winners/breakeven are HELD through it, relying on
the UNCONDITIONAL trailing stop / loss_floor / tp / (later) the agent's learned exit to protect.
The user's specific rule = T=0.05 ("only sell at the EMA cross if price is >5% below the buy").

WHY THIS IS A NEW SLICE (precise): the prior P-EMABREAK probe tested hold-through-ALL EMA-breaks
(net ~0 because it ALSO held the losing trend-breaks) and a surge-at-break discriminator (no edge).
This conditional is NARROWER: hold through an EMA-break ONLY when NOT >T-underwater, while STILL
cutting the >T-underwater losers. That P&L-gated slice was never isolated.

THE TRAPS (the skeptic MUST hunt these — they are the whole point):
  1. ALPHA-vs-BETA (capacity-probe trap, MIRRORED). Holding winners longer through EMA-breaks = MORE
     time in market = MORE bull BETA. The "edge" may be beta, not alpha. We report Sharpe-like
     mean/std AND Calmar (mean/worstDD) per T, the per-REGIME edge (bull vs flat/down), and
     corr(weekly edge, market-EW direction). Invariant Sharpe/Calmar + market-correlated + bull-only
     => BETA (would reverse in a down week), NOT alpha.
  2. DD. Longer holds let a winner give back MORE before the trailing stop catches it => potentially
     HIGHER cold-weekly PORTFOLIO max-DD (the real ~30% DQ object). We measure the within-week
     PORTFOLIO maxDD + DQ-breach weeks per T. (Baseline rung-0 already BREACHES at 35.53% on the
     2026-04-13 cold week — production-confirmed; that week is in TRAIN+VAL here.)
  3. REALIZABILITY. Judge on REALIZED cold-weekly PORTFOLIO return (the grader), NOT run-up.

METHOD: a thin conditional-EMA-break counterfactual executor `run_rung0_cf_ema` (mirror of
probe_capacity.run_rung0_cf, which is itself VALIDATED bar-identical to production run_rung0) that
adds `ema_break_min_loss=T` gating ONLY the cush<0 clause; the trailing stop and loss_floor stay
unconditional. The T=baseline (unconditional) cell is VALIDATED bar-identical to production
run_rung0 before any sweep. Graded on the cold-weekly PORTFOLIO grader over TRAIN+VAL cold weeks
(TEST 5 weeks FROZEN; the one boundary VAL week 2026-04-20 pulls 48 TEST-region bars exactly as
production does — verified non-distorting in the capacity probe, noted here). PAIRED weekly
bootstrap vs baseline (same weeks cancel common market variance). n=23 cold weeks,
worst-week-dominated; per-week reported. Costs (AMM fee + impact + gas) applied EQUALLY across T.

  .venv\\Scripts\\python.exe scripts\\probe_ema_cond.py [--boot 5000]

Does NOT modify production. Does NOT commit.
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "scripts")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

WARMUP = 168
STOP_K = 0.25
COOLDOWN = 48
RULE_EF = 0.20
WEEK_SECS = 7 * 24 * 3600

# baseline sentinel: T = None => unconditional EMA-break (the production rung-0 RULE).
BASELINE = None
# sweep grid of the loss threshold T (price must be > T below the BUY price for the EMA-break to fire)
T_GRID = [BASELINE, 0.0, 0.03, 0.05, 0.08, 0.10, 0.20]


def t_label(T):
    return "baseline(uncond)" if T is BASELINE else f"T={T:.2f}"


# =====================================================================================================
# Conditional-EMA-break counterfactual executor.
#
# Line-for-line the SAME as probe_capacity.run_rung0_cf (which is validated bar-identical to
# production rung0.run_rung0) EXCEPT for the exit test. The only change vs production is:
#     EMA-break clause `price < ema`  ->  gated on `unreal < -T`  when T is not None.
# The trailing stop (`price < peak*(1-stop_k)`) and loss_floor stay UNCONDITIONAL. `unreal` is the
# current price vs the BUY price (st[t]["entry_px"], recorded causally at the funded buy — no
# scale-in/blend, matching the rung-0 RULE which has no scale-in). When T is None the gate collapses
# to the bare `price < ema` and the cell is bit-identical to production (validated below).
# =====================================================================================================

def run_rung0_cf_ema(returns, signal_fn, liquidity, *, ema_break_min_loss=BASELINE,
                     capital=10_000.0, warmup=WARMUP, entry_frac=RULE_EF, stop_k=STOP_K,
                     cooldown=COOLDOWN, rotate=True, lp_fee_bps=None, gas_usd=None,
                     track_token=None):
    """Counterfactual rung-0 executor with a P&L-gated EMA-break. Returns
    (equity Series, records, total_fees, n_buys, tok_realized). `track_token` (optional) accumulates
    that token's REALIZED PnL (sells minus buys incl. costs) for the ZEC anchor."""
    from trader.sim.broker import DEFAULT_GAS_USD, DEFAULT_LP_FEE_BPS, amm_cost_usd
    lp_fee_bps = DEFAULT_LP_FEE_BPS if lp_fee_bps is None else lp_fee_bps
    gas_usd = DEFAULT_GAS_USD if gas_usd is None else gas_usd
    T = ema_break_min_loss
    syms = list(returns.columns)
    pos = pd.Series(0.0, index=syms)
    cash, fees, bar = float(capital), 0.0, 0
    n_buys = 0
    tok_realized = 0.0                                           # realized PnL of track_token (cash flows)
    # st adds `entry_px` (the BUY price, for unreal) on top of run_rung0's state — nothing else changes.
    st = {s: {"held": False, "origin": None, "peak": None, "exit_reb": -10 ** 9,
              "prior_origin": None, "entry_px": None} for s in syms}
    eq = np.empty(len(returns))
    records: list = []
    for i in range(len(returns)):
        r = returns.iloc[i].reindex(syms).fillna(0.0).to_numpy()
        pos = pd.Series(pos.to_numpy() * (1.0 + r), index=syms)
        equity = float(pos.sum() + cash)
        tu, tf = {}, {}

        def trade(t, delta):
            nonlocal cash, fees, tok_realized
            c = amm_cost_usd(delta, liquidity.get(t, 0.0), lp_fee_bps, gas_usd)
            cash -= delta + c
            pos[t] += delta
            fees += c
            tu[t], tf[t] = tu.get(t, 0.0) + delta, tf.get(t, 0.0) + c
            if t == track_token:
                tok_realized -= (delta + c)                     # buy: cash out (neg); sell: cash in (pos)

        if i >= warmup and equity > 1.0:
            sig = signal_fn(returns.iloc[: i + 1])
            for t in [s for s in sig if st[s]["held"]]:
                s, d = st[t], sig[t]
                s["peak"] = max(s["peak"], d["price"])
                stop_hit = d["price"] < s["peak"] * (1.0 - stop_k)        # UNCONDITIONAL trailing stop
                ema_break = d["price"] < d["ema"]                         # the cush<0 clause
                if T is not None:                                         # P&L-gate ONLY the EMA-break
                    unreal = d["price"] / s["entry_px"] - 1.0             # causal: current px vs BUY px
                    ema_break = ema_break and (unreal < -T)               # only if >T underwater
                if stop_hit or ema_break:
                    v = float(pos[t])
                    if abs(v) >= 1.0:
                        trade(t, -v)
                    s.update(held=False, prior_origin=s["origin"], exit_reb=bar)
            cands = [t for t in sig if sig[t]["ignite"] and not st[t]["held"]
                     and (bar - st[t]["exit_reb"]) >= cooldown
                     and (st[t]["prior_origin"] is None or sig[t]["price"] > st[t]["prior_origin"])]
            cands.sort(key=lambda t: sig[t]["cushion"], reverse=True)
            for t in cands:
                if min(entry_frac * equity, cash) < 1.0 and rotate:
                    held = [h for h in sig if st[h]["held"]]
                    if held:
                        weak = min(held, key=lambda h: sig[h]["cushion"])
                        if sig[weak]["cushion"] < sig[t]["cushion"]:
                            v = float(pos[weak])
                            if abs(v) >= 1.0:
                                trade(weak, -v)
                            st[weak].update(held=False, prior_origin=st[weak]["origin"], exit_reb=bar)
                            equity = float(pos.sum() + cash)
                size = min(entry_frac * equity, cash)
                if size >= 1.0:
                    trade(t, size)
                    st[t].update(held=True, origin=sig[t]["price"], peak=sig[t]["price"],
                                 entry_px=sig[t]["price"])     # record the BUY price for unreal
                    n_buys += 1
            bar += 1
        eq[i] = float(pos.sum() + cash)
        if tu or (i >= warmup and i % 24 == 0):
            e = eq[i] if eq[i] > 0 else 1.0
            records.append({"time": int(returns.index[i]),
                            "weights": {s: float(pos[s] / e) for s in syms if pos[s] > 1e-6},
                            "trades_usd": tu, "trade_fees": tf})
    return pd.Series(eq, index=returns.index), records, fees, n_buys, tok_realized


def validate_baseline(returns, liq, vol):
    """T=baseline (unconditional EMA-break) MUST be bar-identical to production run_rung0 — the whole
    sweep rests on this. Compares equity curves + fees over the full TRAIN split."""
    from trader.strategy.rung0 import build_rung0, run_rung0
    from trader.train.weekly_eval import causal_voltop_universe
    uni = causal_voltop_universe(returns.iloc[:WARMUP + 200], k=8, warmup=WARMUP)
    sig = build_rung0(returns, tokens=uni, volume=vol)
    eq_prod, _, f_prod = run_rung0(returns, sig, liq, warmup=WARMUP)
    eq_cf, _, f_cf, _, _ = run_rung0_cf_ema(returns, sig, liq, ema_break_min_loss=BASELINE, warmup=WARMUP)
    max_abs = float(np.max(np.abs(eq_prod.to_numpy() - eq_cf.to_numpy())))
    ok = max_abs < 1e-6 and abs(f_prod - f_cf) < 1e-9
    print(f"\n  [validate] run_rung0_cf_ema(T=baseline) vs production run_rung0: "
          f"max|eq diff|={max_abs:.2e}  fee diff={abs(f_prod - f_cf):.2e}  -> "
          f"{'IDENTICAL (sweep valid)' if ok else 'MISMATCH (PROBE INVALID)'}")
    return ok


def grade_week_ema(ws, win, liq, vol, *, ema_break_min_loss=BASELINE, k=8, warmup=WARMUP):
    """Grade ONE cold week under the conditional-EMA-break — the EXACT weekly_eval.grade_week_baselines
    recipe (same universe pick, same B&H), but the rule runs through run_rung0_cf_ema. Returns
    (week_return, PORTFOLIO maxDD, trade_days, n_buys, buyhold_ret, regime_label, market_ew)."""
    from trader.strategy.rung0 import build_rung0
    from trader.train.weekly_eval import (_trade_days, buyhold_return, causal_voltop_universe,
                                          risk_parity_caps, week_regime)
    uni = causal_voltop_universe(win, k=k, warmup=warmup)
    sig = build_rung0(win, tokens=uni, volume=vol)
    eq, records, _, n_buys, _ = run_rung0_cf_ema(win, sig, liq, ema_break_min_loss=ema_break_min_loss,
                                                 warmup=warmup)
    eq = eq.iloc[warmup:]                                        # the COLD week only (drop the prepad)
    ret = float(eq.iloc[-1] / eq.iloc[0] - 1.0)
    dd = abs(float((eq / eq.cummax() - 1.0).min()))             # within-week PORTFOLIO drawdown
    tdays = len(_trade_days(records, ws))
    caps = risk_parity_caps(win, uni, 0.005, 0.02, warmup=warmup)
    bh = buyhold_return(win, liq, uni, caps, warmup=warmup)
    label, ew = week_regime(win, uni, warmup=warmup)
    return ret, dd, tdays, n_buys, bh, label, ew


# ---- paired weekly bootstrap (same weeks cancel common market variance) -----------------------------

def paired_boot(edge, n_boot, rng):
    """Percentile bootstrap CI for the MEAN of a paired weekly edge vector (T_week - baseline_week)."""
    e = np.asarray(edge, dtype=float)
    if e.size == 0:
        return 0.0, 0.0, 0.0
    means = e[rng.integers(0, e.size, size=(n_boot, e.size))].mean(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return float(e.mean()), float(lo), float(hi)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--boot", type=int, default=5000)
    args = p.parse_args()
    from train_rl import build_volume_panel, load_data, time_split
    from trader.train.weekly_eval import DD_GATE, cold_week_windows, split_label

    returns, btc, _anchor, liq = load_data()
    train_r, val_r, _test_r = time_split(returns)
    val_start = int(val_r.index[0])
    test_start = int(_test_r.index[0])
    vol = build_volume_panel(list(returns.columns), returns.index)
    rng = np.random.default_rng(0)

    print("=" * 100)
    print("P-EMA-COND — P&L-gated EMA-break exit (gate ONLY cush<0 on unreal<-T; trailing stop kept)")
    print("=" * 100)

    # ---- 0) VALIDATE the baseline cell reproduces production exactly ----
    if not validate_baseline(train_r, liq, vol):
        print("  ABORT: baseline cell is not bar-identical to production run_rung0.")
        return

    # ---- collect TRAIN+VAL cold weeks (TEST frozen) ----
    weeks = [(ws, win) for ws, win in cold_week_windows(returns)
             if split_label(ws, val_start, test_start) in ("train", "val")]
    split_counts = {}
    for ws, _ in weeks:
        lbl = split_label(ws, val_start, test_start)
        split_counts[lbl] = split_counts.get(lbl, 0) + 1
    n_weeks = len(weeks)
    print(f"\n  cold weeks (train+val, TEST {sum(1 for ws,_ in cold_week_windows(returns) if split_label(ws,val_start,test_start)=='test')} weeks FROZEN): "
          f"n={n_weeks}  {split_counts}")
    print(f"  DD object = within-week PORTFOLIO maxDD (fresh $10k/week). DQ gate = {DD_GATE:.0%}. "
          f"Costs equal across T.")
    boundary = [ws for ws, _ in weeks if ws >= test_start - WEEK_SECS and ws < test_start]
    if boundary:
        print(f"  NB boundary VAL week(s) {boundary} pull TEST-region bars exactly as production does "
              f"(non-distorting; verified in capacity probe).")

    # ---- 1+2) sweep T, grade every cold week, collect per-T weekly vectors ----
    per_t = {}      # T -> dict of np arrays: rets, dds, tdays, buys (aligned by week order)
    # baseline week metadata (regime + market EW) — same across T (universe pick is T-independent)
    bh_rets = np.empty(n_weeks)
    regimes = []
    market_ew = np.empty(n_weeks)
    for T in T_GRID:
        rets, dds, tdays, buys = [], [], [], []
        for wi, (ws, win) in enumerate(weeks):
            ret, dd, td, nb, bh, lbl, ew = grade_week_ema(ws, win, liq, vol, ema_break_min_loss=T)
            rets.append(ret); dds.append(dd); tdays.append(td); buys.append(nb)
            if T is BASELINE:
                bh_rets[wi] = bh; regimes.append(lbl); market_ew[wi] = ew
        per_t[T] = {"rets": np.array(rets), "dds": np.array(dds),
                    "tdays": np.array(tdays), "buys": np.array(buys)}
    regimes = np.array(regimes)
    base_rets = per_t[BASELINE]["rets"]

    # ---- per-T headline table ----
    print(f"\n########## SWEEP — cold-weekly PORTFOLIO grade per T (n={n_weeks} weeks) ##########")
    print(f"  {'T':16s} {'mean':>7s} {'median':>7s} {'std':>6s} {'worstWk':>8s} {'worstDD':>8s} "
          f"{'DQwk':>5s} {'trd/day':>8s} {'buys/wk':>8s}")
    for T in T_GRID:
        d = per_t[T]
        r, dd = d["rets"], d["dds"]
        worst_dd = float(dd.max())
        dq = int((dd > DD_GATE).sum())
        print(f"  {t_label(T):16s} {r.mean():+7.2%} {np.median(r):+7.2%} {r.std(ddof=1):6.2%} "
              f"{r.min():+8.2%} {worst_dd:8.2%} {dq:5d} {np.mean(d['tdays'])/7.0:8.2f} "
              f"{np.mean(d['buys']):8.2f}")

    # ---- paired-bootstrap weekly edge vs baseline ----
    print(f"\n########## PAIRED weekly edge vs baseline (same weeks; n={n_weeks}; boot={args.boot}) ##########")
    print(f"  {'T':16s} {'edge_mean':>10s} {'95% CI':>22s}  verdict")
    edges = {}
    for T in T_GRID:
        if T is BASELINE:
            continue
        edge = per_t[T]["rets"] - base_rets
        edges[T] = edge
        m, lo, hi = paired_boot(edge, args.boot, rng)
        tag = ("edge>0 (CI-lo>0)" if lo > 0 else
               "edge<0 (CI-hi<0)" if hi < 0 else
               "edge>0 but CI straddles 0" if m > 0 else
               "edge<=0 (CI straddles 0)")
        print(f"  {t_label(T):16s} {m:+10.3%} [{lo:+.3%}, {hi:+.3%}]   {tag}")

    # ---- 3) ALPHA vs BETA ----
    print(f"\n########## ALPHA vs BETA (the capacity-probe trap, MIRRORED) ##########")
    bull = regimes == "bull"
    flatdown = ~bull
    nb, nfd = int(bull.sum()), int(flatdown.sum())
    print(f"  regime split among the {n_weeks} cold weeks: bull={nb}  flat/down={nfd}")
    print(f"\n  --- Sharpe-like (mean/std) & Calmar (mean/worstDD) per T ---")
    print(f"  {'T':16s} {'mean':>7s} {'std':>6s} {'Sharpe':>7s} {'worstDD':>8s} {'Calmar':>7s}")
    sharpe, calmar = {}, {}
    for T in T_GRID:
        r, dd = per_t[T]["rets"], per_t[T]["dds"]
        s = float(r.mean() / r.std(ddof=1)) if r.std(ddof=1) > 0 else float("nan")
        wdd = float(dd.max())
        c = float(r.mean() / wdd) if wdd > 0 else float("nan")
        sharpe[T], calmar[T] = s, c
        print(f"  {t_label(T):16s} {r.mean():+7.2%} {r.std(ddof=1):6.2%} {s:+7.3f} {wdd:8.2%} {c:+7.3f}")

    print(f"\n  --- per-REGIME paired edge vs baseline (bull vs flat/down weeks) ---")
    print(f"  {'T':16s} {'bull_edge':>10s} {'n':>3s} {'flatdown_edge':>14s} {'n':>3s}")
    for T in T_GRID:
        if T is BASELINE:
            continue
        e = edges[T]
        be = float(e[bull].mean()) if nb else float("nan")
        fe = float(e[flatdown].mean()) if nfd else float("nan")
        print(f"  {t_label(T):16s} {be:+10.3%} {nb:3d} {fe:+14.3%} {nfd:3d}")

    print(f"\n  --- corr(weekly edge, market-EW direction) per T ---")
    print(f"  (market_ew = the universe-EW move per week; positive corr => edge concentrates in up weeks => BETA)")
    print(f"  {'T':16s} {'corr':>7s}")
    edge_mkt_corr = {}
    for T in T_GRID:
        if T is BASELINE:
            continue
        e = edges[T]
        if np.std(e) > 0 and np.std(market_ew) > 0:
            c = float(np.corrcoef(e, market_ew)[0, 1])
        else:
            c = float("nan")
        edge_mkt_corr[T] = c
        print(f"  {t_label(T):16s} {c:+7.3f}")

    # alpha/beta verdict at the user's T=0.05
    T05 = 0.05
    base_sharpe, base_calmar = sharpe[BASELINE], calmar[BASELINE]
    print(f"\n  --- ALPHA vs BETA verdict at the user's T={T05:.2f} ---")
    e05 = edges[T05]
    be05 = float(e05[bull].mean()) if nb else float("nan")
    fe05 = float(e05[flatdown].mean()) if nfd else float("nan")
    sharpe_improves = sharpe[T05] > base_sharpe + 1e-4
    calmar_improves = calmar[T05] > base_calmar + 1e-4
    helps_flatdown = nfd > 0 and fe05 > 1e-4
    mkt_corr = edge_mkt_corr[T05]
    print(f"    Sharpe: base {base_sharpe:+.3f} -> T05 {sharpe[T05]:+.3f}  ({'IMPROVES' if sharpe_improves else 'flat/worse'})")
    print(f"    Calmar: base {base_calmar:+.3f} -> T05 {calmar[T05]:+.3f}  ({'IMPROVES' if calmar_improves else 'flat/worse'})")
    print(f"    bull edge {be05:+.3%} (n={nb})   flat/down edge {fe05:+.3%} (n={nfd})  "
          f"({'helps flat/down' if helps_flatdown else 'does NOT help flat/down'})")
    print(f"    corr(edge, market_ew) = {mkt_corr:+.3f}  ({'market-correlated (beta-like)' if mkt_corr > 0.3 else 'weak/neg corr'})")
    is_alpha = (sharpe_improves or calmar_improves) and helps_flatdown
    is_beta = (not helps_flatdown) and (mkt_corr > 0.3) and (be05 > 1e-4)
    verdict_ab = ("ALPHA-LIKE (risk-adj improves AND helps flat/down)" if is_alpha else
                  "BETA-LIKE (bull-only, market-correlated, risk-adj not improved)" if is_beta else
                  "INERT / mixed (read the rows)")
    print(f"    => {verdict_ab}")

    # ---- 4) ZEC anchor ----
    print(f"\n########## ZEC ANCHOR — does T=0.05 hold ZEC's Apr-3 EMA-breaks through the run-up? ##########")
    zec = next((c for c in returns.columns if c.upper().startswith("ZEC")), None)
    if zec is None:
        print("  ZEC not in the universe columns — anchor skipped.")
        zec_line = "ZEC not found in returns columns; anchor not run."
    else:
        from trader.strategy.rung0 import build_rung0
        from trader.train.weekly_eval import causal_voltop_universe
        # The cold-weekly grader RE-PICKS the voltopk-8 universe PER WEEK (causal). ZEC's Apr-3 cycle
        # is captured in whichever cold week(s) ZEC is in THAT week's universe — the production-honest
        # frame (a single full-window universe pick would NOT match how the grader trades ZEC). So
        # walk every TRAIN+VAL cold week, find the weeks where ZEC IS in the week's universe, and sum
        # ZEC's realized cash-flow PnL under baseline vs T=0.05 within each such cold week.
        zec_base_tot, zec_t05_tot, zec_weeks = 0.0, 0.0, []
        for ws, win in weeks:
            uni = causal_voltop_universe(win, k=8, warmup=WARMUP)
            if zec not in uni:
                continue
            sigw = build_rung0(win, tokens=uni, volume=vol)
            _, _, _, _, zb = run_rung0_cf_ema(win, sigw, liq, ema_break_min_loss=BASELINE,
                                              warmup=WARMUP, track_token=zec)
            _, _, _, _, zt = run_rung0_cf_ema(win, sigw, liq, ema_break_min_loss=T05,
                                              warmup=WARMUP, track_token=zec)
            zec_base_tot += zb; zec_t05_tot += zt
            if abs(zb) > 1e-6 or abs(zt) > 1e-6:
                zec_weeks.append((ws, zb, zt))
        diff = zec_t05_tot - zec_base_tot
        # also the contiguous TRAIN+VAL run (full-window universe) as a cross-check on the motivating case
        tv = returns.loc[:test_start - 1] if test_start in returns.index else returns.iloc[:len(train_r) + len(val_r)]
        uni_fw = causal_voltop_universe(tv.iloc[:WARMUP + 200], k=8, warmup=WARMUP) if len(tv) > WARMUP + 200 \
            else list(tv.columns)
        zec_in_fw = zec in uni_fw
        cont = ""
        if zec_in_fw:
            sigc = build_rung0(tv, tokens=uni_fw, volume=vol)
            _, _, _, _, cb = run_rung0_cf_ema(tv, sigc, liq, ema_break_min_loss=BASELINE,
                                              warmup=WARMUP, track_token=zec)
            _, _, _, _, ct = run_rung0_cf_ema(tv, sigc, liq, ema_break_min_loss=T05,
                                              warmup=WARMUP, track_token=zec)
            cont = f"; contiguous-window cross-check: baseline {cb:+.2f} vs T=0.05 {ct:+.2f} (diff {ct-cb:+.2f})"
        import datetime as _dt
        def _d(ws):
            return _dt.datetime.fromtimestamp(int(ws), _dt.timezone.utc).strftime("%Y-%m-%d")
        print(f"  ZEC col = {zec}.  cold weeks where ZEC IS in the week's voltopk-8 universe: "
              f"{len(zec_weeks)} (of {n_weeks})   in full-window universe: {zec_in_fw}")
        print(f"  (the motivating 'Apr-3' EMA-break cycle falls in the cold week starting 2026-04-06)")
        for ws, zb, zt in zec_weeks:
            print(f"    week {_d(ws)} ({ws}): ZEC realized PnL  baseline {zb:+8.2f}  T=0.05 {zt:+8.2f}  "
                  f"diff {zt-zb:+8.2f}")
        print(f"  ZEC realized PnL summed over cold weeks (cash flows, costs incl.):  "
              f"baseline {zec_base_tot:+.2f}  T=0.05 {zec_t05_tot:+.2f}  diff {diff:+.2f}")
        zec_line = (f"ZEC ({zec}) in {len(zec_weeks)}/{n_weeks} cold-week universes; summed realized PnL "
                    f"baseline {zec_base_tot:+.2f} vs T=0.05 {zec_t05_tot:+.2f} (diff {diff:+.2f}){cont}")
        if not zec_weeks:
            zec_line += " — NOTE: ZEC was never funded in any TRAIN+VAL cold week's universe (the Apr-3 cycle " \
                        "sits in a different week/split or ZEC didn't make voltopk-8 those weeks)"

    # ---- n & power ----
    print(f"\n########## n & POWER ##########")
    print(f"  n = {n_weeks} cold weeks (train+val; TEST 5 weeks frozen). Worst-week-dominated: the DQ")
    print(f"  object is the single worst week's PORTFOLIO maxDD, so a 23-week paired test has LOW power")
    print(f"  to certify a small mean edge and the worstDD/DQ read turns on 1-2 weeks. Treat any")
    print(f"  CI-straddling edge as NOT established; treat DQ breaches as load-bearing.")

    print(f"\n  [done] script: scripts/probe_ema_cond.py — no production modified, no commit.")


if __name__ == "__main__":
    main()
