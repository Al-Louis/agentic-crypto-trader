"""P-EXIT-REWARD — is the EXIT-TIMING gap LEARNABLE from CAUSAL obs?

THE FINDING THIS ARM ATTACKS (triple-confirmed): THE EXIT IS THE ALPHA. rung-0 ignitions carry
+11-16% available run-up to the local high, but EVERY mechanical exit (4 entry filters, two tp-rung
ladders, a full trailing-stop sweep) caps REALIZED capture at ~BREAKEVEN. P-REIGNITE just refuted
the entry-SELECTION premise (held re-ignitions are no better entries than fresh), so the binding
constraint is REWARD / exit-timing, NOT entry information.

QUESTION: define a giveback-from-peak reward R_peak (a clean PURE FUNCTION of a trade's
entry/exit/peak), then measure two gap-closed numbers on the rung-0 trade book:
  - ORACLE gap  : a clairvoyant exit AT THE TRUE causal-window PEAK. UPPER BOUND ONLY (closes 100%
                  by construction). *** DRIFT ALARM: the oracle number is NEVER the success bar. ***
  - SURROGATE gap (THE LOAD-BEARING METRIC): a SIMPLE obs-only exit policy FIT ON TRAIN, evaluated
                  OOS ON VAL with NO refitting. The OOS-val fraction of the rule->peak gap it closes.
The oracle-minus-surrogate DELTA is the honest answer:
  clearly POSITIVE surrogate OOS gap  -> the signal is LEARNABLE -> green-light an exit-reward sweep.
  near-zero surrogate gap (only oracle closes it) -> hindsight-only, NOT trainable -> no desktop slot.

HONEST GATE (the project's, restated): the RL agent must beat the rung-0 RULE OOS + survive ~30%
max-DD DQ + >=1 trade/day on COLD-WEEKLY eval; B&H/Random reported only. THIS probe does not touch
that gate — it is a go/no-go on whether to SPEND a desktop slot training an exit reward. The honest
metric here is the SURROGATE OOS-VAL gap closed (causal, no refit), NOT the oracle.

METHODOLOGY (leakage-prone — rigorous):
  * The trade book = the rung-0 RULE's ACTUAL trades, replicated VERBATIM from `_rule_equity_curve`
    (event_env.py:689-744). Each trade = (entry_bar, exit_bar, token). The rule's realized capture
    is what its trailing-stop/EMA exit actually banks within the causal window.
  * CAUSAL peak-horizon denominator: peak run-up within [entry, entry+H], H fixed, IDENTICAL for
    oracle / surrogate / rule.
  * Every surrogate obs at bar b uses ONLY data <= b. Surrogate FIT ON TRAIN ONLY, applied to VAL
    with NO refit. Oracle clearly separated (it reads the future; labelled upper bound).
  * INCREMENTAL VALUE: the policy already carries the `giveback` obs slot. The surrogate must BEAT a
    giveback-ONLY exit OOS, or its 'signal' is just re-deriving an obs the policy already has.
  * PREMATURE-SELL margin X: PRE-REGISTERED below before reading results.
  * DD scope: PER-TRADE worst intra-trade drawdown (NOT the cold-weekly PORTFOLIO DD the DQ
    measures) — explicitly scoped.
  * Move-clustered / token-clustered bootstrap (ignitions cluster in time; per-bar n is
    pseudo-replicated). n ~= 952 train / ~321 val ignitions. TEST split FROZEN — never touched.

Torch-free, laptop-local. Reuses the EventRungEnv-replication pattern from probe_reignite.py /
probe_wick.py.  Run:  .venv\\Scripts\\python.exe scripts\\probe_exit_reward.py [--boot 5000]

DO NOT modify production src/trader/strategy or event_reward.py. DO NOT commit.
"""
from __future__ import annotations

import argparse
import json
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
H_PEAK = 24           # CAUSAL peak horizon (bars) — the forward window peak run-up is measured over.
#                       Matches the fwd_horizon=24 the entry_forward reward already uses, and the
#                       +11-16% run-up framing of the exit-is-alpha probe. Identical for all 3 exits.

# ---- KAPPA (giveback-penalty weight in R_peak). Stated up front. ------------------------------
KAPPA = 1.0           # R_peak = captured_runup - kappa * giveback_from_peak. kappa=1 weights a $ of
#                       giveback equally to a $ of captured run-up (symmetric). Reported across a
#                       small kappa grid too, but the headline gap-closed metric is kappa-FREE
#                       (it is defined on realized capture vs the peak, see gap_closed()).

# ---- PRE-REGISTERED premature-sell margin X (set BEFORE reading any result). ------------------
# A "premature sell" = the exit fires at bar b but the price would have run MATERIALLY higher
# (>= PREMATURE_DELTA more run-up) later within the causal window [b, entry+H]. The surrogate's
# premature-sell rate must NOT exceed the rung-0 RULE's missed-continuation rate by more than X pts.
PREMATURE_DELTA = 0.05   # "materially further" = >=5pp additional run-up left on the table after exit
PREMATURE_X_PTS = 10.0   # PRE-REGISTERED margin: surrogate premature-sell rate may exceed the rule's
#                          missed-continuation rate by at most +10.0 percentage points. (set up front)


# ================================================================================================
# 1) R_peak — the PURE FUNCTION (credit captured run-up, penalize giveback-from-within-move-peak).
# ================================================================================================
def r_peak(entry_px: float, exit_px: float, peak_px: float, kappa: float = KAPPA) -> float:
    """Pure function of a trade's (entry, exit, within-move peak) prices.

        captured  = exit_px / entry_px - 1.0           # realized run-up the exit banked
        peak_run  = peak_px / entry_px - 1.0           # the within-move peak run-up (>= captured)
        giveback  = peak_px  / exit_px  - 1.0          # surrendered from the peak (>= 0)
        R_peak    = captured - kappa * giveback

    A clairvoyant exit AT the peak has giveback==0 and captured==peak_run -> R_peak == peak_run
    (the max attainable). Selling early banks less captured AND, if price keeps rising, eats a
    positive giveback penalty. Selling after a reversal banks the reversal in captured AND a large
    giveback. PURE: depends only on the three prices + kappa. (peak_px is the running peak over the
    HOLD path [entry, exit]; for the within-move-peak variant used by the oracle/gap it is the
    causal-window peak — both are passed in explicitly, never read globally.)"""
    if entry_px <= 0 or exit_px <= 0 or peak_px <= 0:
        return 0.0
    captured = exit_px / entry_px - 1.0
    giveback = peak_px / exit_px - 1.0
    return captured - kappa * max(giveback, 0.0)


# ================================================================================================
# 2) gap-closed metric (kappa-FREE: realized CAPTURE vs the causal-window peak), identical 3 ways.
# ================================================================================================
def captured(entry_px: float, exit_px: float) -> float:
    return exit_px / entry_px - 1.0 if entry_px > 0 and exit_px > 0 else 0.0


def gap_closed(policy_caps, rule_caps, peak_caps):
    """Fraction of the rule->peak gap a policy closes, AGGREGATE over trades (sum of $ captured,
    not a mean of ratios — robust to tiny-denominator trades).

        closed = (sum policy_capture - sum rule_capture) / (sum peak_capture - sum rule_capture)

    peak_capture = the ORACLE's capture (sell at the causal-window peak) — the same denominator for
    every policy. 1.0 = closed the whole gap (== oracle). 0.0 = matched the rule. <0 = worse than
    the rule. Identical definition for oracle (==1.0 by construction), surrogate, giveback-only."""
    num = float(np.sum(policy_caps) - np.sum(rule_caps))
    den = float(np.sum(peak_caps) - np.sum(rule_caps))
    if abs(den) < 1e-9:
        return float("nan")
    return num / den


# ================================================================================================
# Build the env + the rung-0 RULE's ACTUAL trade book (verbatim mirror of _rule_equity_curve).
# ================================================================================================
def build_env(r, btc, liq, vol, k=8):
    from trader.train.event_env import EventRungEnv
    env = EventRungEnv(r, btc, liq, volume=vol, episode_bars=len(r) - WARMUP - 1,
                       k=k, warmup=WARMUP, universe_mode="voltopk", seed=0)
    env.reset(start=WARMUP)
    return env


def rule_trades(env):
    """Replicate `_rule_equity_curve` VERBATIM and emit one record per CLOSED rule trade:
    (token, entry_bar, exit_bar). The rule exits on a trailing-stop off the running peak OR an
    EMA-break (cush<0); enters strongest fundable ignitions; rotates losers. This IS the rule's
    book, the exact trades whose exit timing we are scoring. Open trades at end are recorded with
    exit_bar = end (the rule's mark-to-end), matching the equity-curve treatment."""
    from trader.sim.broker import amm_cost_usd
    px, cush, ig, cix = env._px, env._cush, env._ignite, env.col_ix
    sk, cd, ef = STOP_K, COOLDOWN, RULE_EF
    fee, gas, liq = env.lp_fee_bps, env.gas_usd, env.liquidity
    start, end = env.start, env.end
    cash, pos = env.capital, {}
    cool = {t: -10 ** 9 for t in env.universe}
    prior = {t: None for t in env.universe}
    trades = []           # (tok, entry_bar, exit_bar)

    def value(t, bar):
        p = pos[t]
        j = cix[t]
        return p["usd"] * px[bar, j] / px[p["entry_bar"], j]

    for bar in range(start, end + 1):
        equity = cash + sum(value(t, bar) for t in pos)
        if equity > 1.0:
            for t in list(pos):                                   # 1) exits (stop off the peak / EMA)
                j = cix[t]
                p = pos[t]
                p["peak_px"] = max(p["peak_px"], px[bar, j])
                if px[bar, j] < p["peak_px"] * (1.0 - sk) or cush[bar, j] < 0.0:
                    v = value(t, bar)
                    cash += v - amm_cost_usd(-v, liq.get(t, 0.0), fee, gas)
                    trades.append((t, p["entry_bar"], bar))
                    cool[t], prior[t] = bar, p["origin"]
                    del pos[t]
            cands = [t for t in env.universe if ig[bar, cix[t]] and t not in pos
                     and (bar - cool[t]) >= cd
                     and (prior[t] is None or px[bar, cix[t]] > prior[t])]
            cands.sort(key=lambda t: cush[bar, cix[t]], reverse=True)
            for t in cands:                                       # 2) fund strongest; rotate losers
                if min(ef * equity, cash) < 1.0 and pos:
                    weak = min(pos, key=lambda h: cush[bar, cix[h]])
                    if cush[bar, cix[weak]] < cush[bar, cix[t]]:
                        v = value(weak, bar)
                        cash += v - amm_cost_usd(-v, liq.get(weak, 0.0), fee, gas)
                        trades.append((weak, pos[weak]["entry_bar"], bar))
                        cool[weak], prior[weak] = bar, pos[weak]["origin"]
                        del pos[weak]
                        equity = cash + sum(value(tt, bar) for tt in pos)
                size = min(ef * equity, cash)
                if size >= 1.0:
                    j = cix[t]
                    cash -= size + amm_cost_usd(size, liq.get(t, 0.0), fee, gas)
                    pos[t] = {"usd": size, "entry_bar": bar, "peak_px": px[bar, j], "origin": px[bar, j]}
    for t in list(pos):                                           # open at end -> mark to end
        trades.append((t, pos[t]["entry_bar"], end))
    return trades


# ================================================================================================
# Causal per-bar obs (data <= b) for the surrogate's HOLD path, + the causal-window peak.
# ================================================================================================
def trade_path(env, tok, entry_bar, rule_exit_bar):
    """For one rule trade build the per-bar HOLD-PATH features and the causal-window peak.

    Returns dict with arrays indexed by hold-bar offset h (b = entry_bar + h), h in [0, H] capped at
    the panel end. Every feature at b uses ONLY data <= b (causal). Also returns the causal-window
    peak price/bar (max px over [entry_bar, entry_bar+H], capped at panel end) — the IDENTICAL
    denominator for oracle/surrogate/rule.

    Features (all causal, all available in the env's obs vector at bar b):
      giveback   = px[b]/running_peak[b] - 1   (<=0; drawdown from the peak SEEN SO FAR on the hold)
      unreal     = px[b]/entry_px - 1          (unrealized gain)
      surge      = env._surge[b]               (clipped volume surge)
      surge_decay= surge_now / max(surge so far on hold)   (<=1; momentum cooling)
      cush       = env._cush[b]                (price vs EMA)
      accel      = (px[b]/px[b-1]) - (px[b-1]/px[b-2])  (2nd diff of 1-bar returns; velocity change)
      held_frac  = h / H                       (fraction of the horizon held)
    """
    px, surge, cush, cix, n = env._px, env._surge, env._cush, env.col_ix, env.n_bars
    j = cix[tok]
    entry_px = px[entry_bar, j]
    last_b = min(entry_bar + H_PEAK, n - 1)
    bars = list(range(entry_bar, last_b + 1))
    running_peak = -np.inf
    max_surge = 1e-12
    feats, prices = [], []
    peak_px, peak_bar = -np.inf, entry_bar
    for b in bars:
        p = px[b, j]
        running_peak = max(running_peak, p)
        if p > peak_px:
            peak_px, peak_bar = p, b
        s = float(surge[b, j])
        max_surge = max(max_surge, s)
        giveback = p / running_peak - 1.0 if running_peak > 0 else 0.0
        unreal = p / entry_px - 1.0 if entry_px > 0 else 0.0
        surge_decay = s / max_surge if max_surge > 0 else 0.0
        cu = float(cush[b, j])
        if b - 2 >= 0 and px[b - 1, j] > 0 and px[b - 2, j] > 0:
            accel = (px[b, j] / px[b - 1, j]) - (px[b - 1, j] / px[b - 2, j])
        else:
            accel = 0.0
        held_frac = (b - entry_bar) / H_PEAK
        feats.append([giveback, unreal, s, surge_decay, cu, accel, held_frac])
        prices.append(p)
    return {
        "tok": tok, "entry_bar": entry_bar, "entry_px": entry_px,
        "bars": np.array(bars), "prices": np.array(prices),
        "feats": np.array(feats, dtype=float),           # [n_hold, 7]
        "peak_px": peak_px, "peak_bar": peak_bar,
        # the RULE's realized exit price within the causal window: the rule may exit AFTER the window
        # (it has no horizon) — for an apples-to-apples gap we cap the rule exit at the window end so
        # all three policies are scored on the SAME [entry, entry+H] interval. The rule's WITHIN-
        # WINDOW exit = min(rule_exit_bar, last_b).
        "rule_exit_bar": min(rule_exit_bar, last_b),
        "rule_exit_px": px[min(rule_exit_bar, last_b), j],
        "last_b": last_b,
    }


FEAT_NAMES = ["giveback", "unreal", "surge", "surge_decay", "cush", "accel", "held_frac"]


# ================================================================================================
# Surrogate exit policies. Each maps a trade-path -> an exit-bar index (offset into path.bars).
#   - oracle      : argmax price (the causal-window peak). UPPER BOUND.
#   - rule        : the rule's within-window exit bar.
#   - giveback    : sell the FIRST bar giveback <= -tau (single-threshold; the obs the policy HAS).
#   - surrogate   : a small logistic on the 7 causal feats predicting "sell now"; sell the first bar
#                   p(sell) >= thresh. Fit on TRAIN (coef + thresh chosen to MAXIMIZE train gap), no
#                   refit on VAL.
# ================================================================================================
def exit_oracle(path):
    k = int(np.argmax(path["prices"]))
    return path["bars"][k], path["prices"][k]


def exit_rule(path):
    return path["rule_exit_bar"], path["rule_exit_px"]


def exit_giveback(path, tau):
    """Sell the first hold-bar whose giveback (px/running_peak-1) <= -tau. Else hold to window end."""
    gb = path["feats"][:, 0]                              # giveback column
    hit = np.where(gb <= -tau)[0]
    k = int(hit[0]) if len(hit) else len(path["bars"]) - 1
    return path["bars"][k], path["prices"][k]


def _logit(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))


def exit_surrogate(path, w, b0, thresh):
    """Sell the first hold-bar where sigma(w.feat + b0) >= thresh. Else hold to window end.
    NEVER sells at h=0 (entry bar) — an exit needs at least one bar held (matches the env: exits are
    prompted on held positions on subsequent bars). Pure linear-in-causal-feats threshold rule."""
    F = path["feats"]
    z = F @ w + b0
    pr = _logit(z)
    pr[0] = 0.0                                           # cannot exit on the entry bar itself
    hit = np.where(pr >= thresh)[0]
    k = int(hit[0]) if len(hit) else len(path["bars"]) - 1
    return path["bars"][k], path["prices"][k]


# ----- fit the surrogate on TRAIN -----
def fit_surrogate(paths, seed=0):
    """Fit a logistic 'sell now' classifier on TRAIN trade-paths. LABEL (causal-window peak, used
    only to DEFINE the supervised target on TRAIN — the FEATURES are strictly causal, and the fitted
    policy applied OOS reads NO future): y=1 at the bar that IS the within-window peak (sell here),
    y=0 elsewhere. Standardize feats (mean/std from TRAIN only), fit by simple gradient descent, then
    CHOOSE the probability threshold that MAXIMIZES the TRAIN aggregate gap-closed. Returns the
    coefficients in RAW feature space (fold standardization into w,b0) + the chosen threshold."""
    X, Y = [], []
    for p in paths:
        F = p["feats"]
        peak_h = int(np.argmax(p["prices"]))
        y = np.zeros(len(F));
        if peak_h > 0:                                    # never label the entry bar as a sell
            y[peak_h] = 1.0
        X.append(F); Y.append(y)
    X = np.vstack(X); Y = np.concatenate(Y)
    mu, sd = X.mean(0), X.std(0)
    sd[sd < 1e-9] = 1.0
    Xs = (X - mu) / sd
    rng = np.random.default_rng(seed)
    w = rng.normal(0, 0.01, Xs.shape[1])
    b = 0.0
    lr, l2 = 0.5, 1e-3
    pos_w = max((len(Y) - Y.sum()) / max(Y.sum(), 1.0), 1.0)   # class-weight the rare peak bars
    sample_w = np.where(Y > 0, pos_w, 1.0)
    for _ in range(4000):
        z = Xs @ w + b
        pr = _logit(z)
        g = (pr - Y) * sample_w
        gw = Xs.T @ g / len(Y) + l2 * w
        gb = g.mean()
        w -= lr * gw
        b -= lr * gb
    # fold standardization back: sigma(w.((x-mu)/sd)+b) = sigma((w/sd).x + (b - w.mu/sd))
    w_raw = w / sd
    b_raw = b - float(np.sum(w * mu / sd))
    # choose the threshold maximizing TRAIN aggregate gap-closed
    rule_caps = np.array([captured(p["entry_px"], p["rule_exit_px"]) for p in paths])
    peak_caps = np.array([captured(p["entry_px"], p["peak_px"]) for p in paths])
    best_thr, best_gap = 0.5, -1e9
    for thr in np.linspace(0.05, 0.95, 19):
        caps = np.array([captured(p["entry_px"], exit_surrogate(p, w_raw, b_raw, thr)[1])
                         for p in paths])
        gc = gap_closed(caps, rule_caps, peak_caps)
        if np.isfinite(gc) and gc > best_gap:
            best_gap, best_thr = gc, thr
    return {"w": w_raw, "b0": b_raw, "thresh": float(best_thr), "mu": mu, "sd": sd,
            "train_gap": float(best_gap), "coef_std": (w).tolist()}


# ----- fit the giveback-only threshold on TRAIN (the incremental-value floor) -----
def fit_giveback(paths):
    rule_caps = np.array([captured(p["entry_px"], p["rule_exit_px"]) for p in paths])
    peak_caps = np.array([captured(p["entry_px"], p["peak_px"]) for p in paths])
    best_tau, best_gap = 0.10, -1e9
    for tau in np.linspace(0.02, 0.30, 29):
        caps = np.array([captured(p["entry_px"], exit_giveback(p, tau)[1]) for p in paths])
        gc = gap_closed(caps, rule_caps, peak_caps)
        if np.isfinite(gc) and gc > best_gap:
            best_gap, best_tau = gc, tau
    return {"tau": float(best_tau), "train_gap": float(best_gap)}


# ================================================================================================
# premature-sell rate (PRE-REGISTERED definition) + per-trade DD scope.
# ================================================================================================
def premature_rate(paths, exit_fn):
    """Fraction of trades where the exit fires but >= PREMATURE_DELTA additional run-up was still
    available LATER in the causal window (px would have risen materially further). For the rule this
    is its 'missed-continuation rate'. exit_fn(path) -> (exit_bar, exit_px)."""
    cnt = prem = 0
    for p in paths:
        eb, epx = exit_fn(p)
        # max price AFTER the exit bar within the window
        bars, prices = p["bars"], p["prices"]
        k = int(np.searchsorted(bars, eb))
        if k >= len(bars) - 1:
            cnt += 1
            continue                                      # exited at/after window end -> no continuation missed
        future_max = prices[k + 1:].max()
        addl = future_max / epx - 1.0 if epx > 0 else 0.0
        prem += 1 if addl >= PREMATURE_DELTA else 0
        cnt += 1
    return prem / cnt if cnt else float("nan"), cnt


def per_trade_dd(paths, exit_fn):
    """PER-TRADE worst drawdown from the entry's running peak UP TO the exit bar (NOT portfolio DD).
    Scope note: this is intra-trade, not the cold-weekly PORTFOLIO max-DD the ~30% DQ measures."""
    dds = []
    for p in paths:
        eb, _ = exit_fn(p)
        bars, prices = p["bars"], p["prices"]
        k = int(np.searchsorted(bars, eb))
        seg = prices[:k + 1]
        run = np.maximum.accumulate(seg)
        dd = (seg / run - 1.0).min() if len(seg) else 0.0
        dds.append(dd)
    return float(np.mean(dds)), float(np.min(dds))


# ================================================================================================
# Move-clustered (token-clustered) bootstrap on the surrogate OOS gap-closed.
# ================================================================================================
def cluster_bootstrap_gap(paths, exit_fn, n_boot, rng):
    """Resample TOKENS (clusters) with replacement; recompute aggregate gap-closed per resample.
    Ignitions cluster in time within a token -> per-trade samples are pseudo-replicated, so we
    resample at the token level (the move/cluster unit). Returns (point, lo, hi) at 95%."""
    by_tok = {}
    for i, p in enumerate(paths):
        by_tok.setdefault(p["tok"], []).append(i)
    toks = list(by_tok)
    rule_caps = np.array([captured(p["entry_px"], p["rule_exit_px"]) for p in paths])
    peak_caps = np.array([captured(p["entry_px"], p["peak_px"]) for p in paths])
    pol_caps = np.array([captured(p["entry_px"], exit_fn(p)[1]) for p in paths])
    point = gap_closed(pol_caps, rule_caps, peak_caps)
    boots = np.empty(n_boot)
    for bi in range(n_boot):
        pick = rng.integers(0, len(toks), len(toks))
        idx = np.concatenate([by_tok[toks[k]] for k in pick])
        gc = gap_closed(pol_caps[idx], rule_caps[idx], peak_caps[idx])
        boots[bi] = gc
    boots = boots[np.isfinite(boots)]
    lo, hi = np.percentile(boots, [2.5, 97.5]) if len(boots) else (float("nan"), float("nan"))
    return float(point), float(lo), float(hi)


# ================================================================================================
# main
# ================================================================================================
def build_paths(env):
    trades = rule_trades(env)
    paths = []
    for tok, eb, xb in trades:
        if env._px[eb, env.col_ix[tok]] <= 0:
            continue
        p = trade_path(env, tok, eb, xb)
        if len(p["bars"]) < 3:                            # need >=2 hold bars for accel + a sell bar
            continue
        paths.append(p)
    return paths


def report_split(name, paths, fitted_sur, fitted_gb, n_boot, rng, is_train):
    n = len(paths)
    ntok = len({p["tok"] for p in paths})
    rule_caps = np.array([captured(p["entry_px"], p["rule_exit_px"]) for p in paths])
    peak_caps = np.array([captured(p["entry_px"], p["peak_px"]) for p in paths])

    sur_fn = lambda p: exit_surrogate(p, fitted_sur["w"], fitted_sur["b0"], fitted_sur["thresh"])
    gb_fn = lambda p: exit_giveback(p, fitted_gb["tau"])

    sur_caps = np.array([captured(p["entry_px"], sur_fn(p)[1]) for p in paths])
    gb_caps = np.array([captured(p["entry_px"], gb_fn(p)[1]) for p in paths])

    oracle_gc = gap_closed(peak_caps, rule_caps, peak_caps)         # == 1.0 by construction
    sur_gc = gap_closed(sur_caps, rule_caps, peak_caps)
    gb_gc = gap_closed(gb_caps, rule_caps, peak_caps)

    print(f"\n########## {name} ##########  rule trades={n}  tokens={ntok}  H_peak={H_PEAK}")
    print(f"  rule    mean capture {rule_caps.mean():+7.2%}   total ${rule_caps.sum():+.3f}/unit")
    print(f"  ORACLE  mean capture {peak_caps.mean():+7.2%}   total ${peak_caps.sum():+.3f}/unit"
          f"   gap-closed = {oracle_gc:+.1%}  <<< UPPER BOUND ONLY (clairvoyant peak)")
    print(f"  giveback-only(tau={fitted_gb['tau']:.2f}) mean {gb_caps.mean():+7.2%}"
          f"   gap-closed = {gb_gc:+.1%}   (incremental-value FLOOR — the obs the policy already has)")
    print(f"  SURROGATE(thr={fitted_sur['thresh']:.2f}) mean {sur_caps.mean():+7.2%}"
          f"   gap-closed = {sur_gc:+.1%}   <<< LOAD-BEARING" + ("  [TRAIN — in-sample]" if is_train
          else "  [VAL — OOS, NO refit]"))
    incr = sur_gc - gb_gc
    print(f"  oracle - surrogate delta = {oracle_gc - sur_gc:+.1%}    "
          f"surrogate - giveback-only (incremental) = {incr:+.1%}")

    # premature-sell rates (pre-registered)
    rule_prem, _ = premature_rate(paths, exit_rule)
    sur_prem, _ = premature_rate(paths, sur_fn)
    gb_prem, _ = premature_rate(paths, gb_fn)
    margin = (sur_prem - rule_prem) * 100.0
    verdict_prem = "PASS" if margin <= PREMATURE_X_PTS else "FAIL"
    print(f"  premature-sell (>= {PREMATURE_DELTA:.0%} addl run-up left): "
          f"rule(missed-cont) {rule_prem:.1%}  surrogate {sur_prem:.1%}  giveback {gb_prem:.1%}")
    print(f"     surrogate - rule = {margin:+.1f}pts  vs pre-registered X=+{PREMATURE_X_PTS:.0f}pts"
          f"  -> {verdict_prem}")

    # per-trade DD scope (NOT portfolio DD)
    sur_dd_mean, sur_dd_worst = per_trade_dd(paths, sur_fn)
    rule_dd_mean, rule_dd_worst = per_trade_dd(paths, exit_rule)
    print(f"  PER-TRADE DD (scope: intra-trade, NOT cold-weekly portfolio DQ): "
          f"rule mean {rule_dd_mean:+.2%}/worst {rule_dd_worst:+.2%}  "
          f"surrogate mean {sur_dd_mean:+.2%}/worst {sur_dd_worst:+.2%}")

    # bootstrap CI on the surrogate gap (move/token-clustered)
    pt, lo, hi = cluster_bootstrap_gap(paths, sur_fn, n_boot, rng)
    ci_sig = "CI-low > 0 (POSITIVE)" if lo > 0 else "CI straddles 0" if lo <= 0 <= hi else "CI-high < 0 (NEG)"
    print(f"  surrogate gap-closed bootstrap (token-clustered, {n_boot} resamples): "
          f"{pt:+.1%}  95%CI [{lo:+.1%}, {hi:+.1%}]  -> {ci_sig}")

    return {
        "n": n, "ntok": ntok,
        "rule_mean_capture": float(rule_caps.mean()), "oracle_mean_capture": float(peak_caps.mean()),
        "oracle_gap": oracle_gc, "surrogate_gap": sur_gc, "giveback_gap": gb_gc,
        "oracle_minus_surrogate": oracle_gc - sur_gc, "surrogate_minus_giveback": incr,
        "rule_premature": rule_prem, "surrogate_premature": sur_prem, "giveback_premature": gb_prem,
        "premature_margin_pts": margin, "premature_verdict": verdict_prem,
        "sur_dd_mean": sur_dd_mean, "sur_dd_worst": sur_dd_worst,
        "rule_dd_mean": rule_dd_mean, "rule_dd_worst": rule_dd_worst,
        "boot_point": pt, "boot_lo": lo, "boot_hi": hi,
    }


def main():
    global H_PEAK
    ap = argparse.ArgumentParser()
    ap.add_argument("--boot", type=int, default=5000)
    ap.add_argument("--horizon", type=int, default=0,
                    help="causal peak horizon (bars); robustness axis (0 = use module default)")
    args = ap.parse_args()
    if args.horizon:
        H_PEAK = int(args.horizon)
    from train_rl import build_volume_panel, load_data, time_split
    returns, btc, anchor, liq = load_data()
    train_r, val_r, test_r = time_split(returns)
    vol = build_volume_panel(list(returns.columns), returns.index)
    rng = np.random.default_rng(0)

    print("=" * 96)
    print("P-EXIT-REWARD — is the exit-timing gap LEARNABLE from CAUSAL obs?  (torch-free probe)")
    print(f"  R_peak = captured - kappa*giveback   (kappa={KAPPA}; gap-closed metric is kappa-FREE)")
    print(f"  CAUSAL peak horizon H = {H_PEAK} bars (identical for oracle/surrogate/rule)")
    print(f"  PRE-REGISTERED premature-sell margin X = +{PREMATURE_X_PTS:.0f}pts "
          f"(delta={PREMATURE_DELTA:.0%}); set BEFORE reading results")
    print("  *** DRIFT ALARM: the ORACLE gap is an UPPER BOUND, NEVER the success bar (exp1->exp5). ***")
    print("  HONEST go/no-go metric = the SURROGATE OOS-VAL gap closed (fit on train, no refit).")
    print("=" * 96)

    # FIT on TRAIN only
    env_tr = build_env(train_r, btc, liq, vol)
    paths_tr = build_paths(env_tr)
    print(f"\n[fit] TRAIN universe(voltopk-8) = {env_tr.universe}")
    fitted_sur = fit_surrogate(paths_tr)
    fitted_gb = fit_giveback(paths_tr)
    print(f"[fit] surrogate logit coefs (standardized): "
          f"{dict(zip(FEAT_NAMES, [round(c, 3) for c in fitted_sur['coef_std']]))}")
    print(f"[fit] surrogate thresh={fitted_sur['thresh']:.2f} (train gap {fitted_sur['train_gap']:+.1%}); "
          f"giveback tau={fitted_gb['tau']:.2f} (train gap {fitted_gb['train_gap']:+.1%})")

    res = {}
    res["TRAIN"] = report_split("TRAIN (in-sample sanity)", paths_tr, fitted_sur, fitted_gb,
                                args.boot, rng, is_train=True)

    # APPLY OOS on VAL (no refit)
    env_va = build_env(val_r, btc, liq, vol)
    paths_va = build_paths(env_va)
    print(f"\n[oos] VAL universe(voltopk-8) = {env_va.universe}")
    res["VAL"] = report_split("VAL (OOS — the headline)", paths_va, fitted_sur, fitted_gb,
                              args.boot, rng, is_train=False)

    v = res["VAL"]
    print("\n" + "=" * 96)
    print("VERDICT INPUTS (VAL / OOS):")
    print(f"  surrogate OOS-val gap closed   : {v['surrogate_gap']:+.1%}  "
          f"95%CI [{v['boot_lo']:+.1%},{v['boot_hi']:+.1%}]")
    print(f"  oracle (upper bound)           : {v['oracle_gap']:+.1%}")
    print(f"  oracle - surrogate delta       : {v['oracle_minus_surrogate']:+.1%}")
    print(f"  surrogate - giveback-only (incr): {v['surrogate_minus_giveback']:+.1%}")
    print(f"  premature margin vs rule       : {v['premature_margin_pts']:+.1f}pts "
          f"(X=+{PREMATURE_X_PTS:.0f}) -> {v['premature_verdict']}")
    print("=" * 96)
    # machine-readable tail for the structured return
    print("RESULT_JSON " + json.dumps({"H_peak": H_PEAK, "kappa": KAPPA,
          "premature_X_pts": PREMATURE_X_PTS, "premature_delta": PREMATURE_DELTA,
          "n_train": res["TRAIN"]["n"], "n_val": res["VAL"]["n"], **{f"val_{k}": v[k] for k in v},
          "train_surrogate_gap": res["TRAIN"]["surrogate_gap"]}, default=float))


if __name__ == "__main__":
    main()
