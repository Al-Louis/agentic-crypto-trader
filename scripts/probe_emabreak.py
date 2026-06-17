"""P-EMABREAK — is the rung-0 EMA-BREAK exit a HAIR-TRIGGER that systematically sells shallow
below-EMA pullbacks (giving up run-up), and can a CAUSAL discriminator tell a premature cut from a
real trend-break OOS?

THE FORENSIC THAT MOTIVATES THIS (wsi-s3 ZEC, data-confirmed): the rung-0 exit that cut the ZEC
winner was the EMA-BREAK (ema_hit = cushion<0, event_env._scan_bar L396; re-fired every
exit_commit=12 bars), NOT the -25% trailing stop, NOT a tp, NOT rotation, NOT the loss_floor. ZEC was
~0.9-1.9% from its blended cost and ~8% off peak (a hair below EMA = an ordinary breather), and at the
full-exit bar SURGE WAS RISING (1.75) -> the ignition was not dead -> yet it sold, then ZEC ran +60%
on cost over 9 days. HYPOTHESIS: the EMA-break is a hair-trigger that systematically sells shallow
below-EMA pullbacks (which [[ignition-edge-is-contrarian-not-strength]] says OUTPERFORM).

THE QUESTION (population-level — the one-seed/one-week forensic demands it): across ALL valid
ignitions (train+val), HYPOTHETICALLY HELD from entry (NOT just the rule's ~20 funded trades — use
the broad population for POWER), at the FIRST EMA-break (first cushion<0 after entry):
  1. RESUMER BASE RATE: what fraction of EMA-breaks would, if HELD past the break, RESUME to a NEW
     HIGH (above the causal pre-break running peak) within H=24/48 bars = a PREMATURE cut? (vs a real
     TREND-BREAK that does not resume / continues down.)
  2. COST: how much forward run-up do premature exits give up (realized-at-break vs resume)?
  3. DISCRIMINATOR: does a CAUSAL feature set at the break bar — shallow giveback-from-peak +
     surge-still-alive (and/or surge_decay, cush magnitude, bars-since-entry, unreal) — SEPARATE
     resumers from real trend-breaks? FIT on TRAIN, evaluate OOS on VAL (no refit). It MUST beat a
     giveback-ONLY discriminator (the obs the agent already carries) — surge-alive is the candidate
     NEW signal the policy ignored in the forensic.

HONEST GATE (the project's, restated): the RL agent must beat the rung-0 RULE OOS + survive ~30%
max-DD DQ + >=1 trade/day on COLD-WEEKLY eval; B&H/Random reported only. THIS probe does NOT touch
that gate — it is a go/no-go on whether the EMA-break is systematically premature AND a causal
discriminator (beating giveback-only) separates resumers OOS, which would justify a SINGLE-VARIABLE
add on the wkw base (a giveback-penalty reward and/or a shallow-dip-vs-trend-break EXIT OBS), then
validated COLD-WEEKLY on the honest gate. If NOT -> the EMA-break is doing its job; don't change the
substrate.

RECONCILE with P-EXIT-REWARD (NO-GO): that arm measured TIMING THE PEAK on the rule's ~20 trades
(underpowered, giveback-dominated). P-EMABREAK is NARROWER (not selling too early at the EMA-break
specifically) on a BROADER population (all ignitions -> hypothetical EMA-break, ~hundreds) for power.
surge-rising-at-exit is the specific discriminator P-EXIT-REWARD did not isolate.

METHODOLOGY (leakage-prone — rigorous):
  * Population = every valid in-universe ignition (broad), hypothetically HELD from entry; one
    hypothetical hold per FRESH ignition (no overlapping double-count: a token already inside an
    active hold does not open a second hold until that hold's break is found).
  * The FIRST EMA-break = first bar b>entry with cush[b]<0. We ALSO record whether the trailing stop
    (px < running_peak*(1-STOP_K)) would have fired first; an event is a clean "EMA-break" only when
    the EMA-break is the first trigger (the trailing-stop ones are reported separately, not the
    headline — the forensic is specifically about the EMA-break hair-trigger).
  * RESUME label legitimately uses FORWARD bars (it is the OUTCOME). DISCRIMINATOR features are
    STRICTLY CAUSAL (<= break bar). Discriminator fit on TRAIN only, applied to VAL with NO refit.
  * The 'new high' reference is the CAUSAL pre-break running peak (peak over [entry..break]), NOT a
    global/forward max.
  * Universe via trailing _std (the env's `_pick_universe`, NOT the full-window vol-rank that once
    peeked at late pumpers).
  * Move/token-clustered bootstrap (resumers cluster within a token's pump). Pre-registered n-floor
    -> INCONCLUSIVE below it. TEST split FROZEN — never touched.
  * WATCH: resume-then-CRASH (is 'resume to a new high' realizable as PnL, or a touch-then-roundtrip?
    -> report the run-up GIVEN UP, which is the realizable upper bound, and a held-to-H realized
    number). Is the resumer rate a BULL-regime artifact (report the split's regime; everything
    resumes in a bull)? Does the discriminator survive token-leave-one-out? Threshold-fishing only on
    train.

Torch-free, laptop-local. Reuses the EventRungEnv-replication pattern from probe_reignite.py /
probe_exit_reward.py / probe_wick.py.  Run:
  .venv\\Scripts\\python.exe scripts\\probe_emabreak.py [--boot 5000]

DO NOT modify production src/trader. DO NOT commit.
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
STOP_K = 0.25          # the rule's trailing stop (used ONLY to flag which trigger fires first)
COOLDOWN = 48

# ---- PRE-REGISTERED n-floor (set BEFORE reading results) --------------------------------------
# Below this many clean EMA-break events on a split, that split's verdict is INCONCLUSIVE. The
# discriminator OOS read needs both classes present; require >= MIN_PER_CLASS in each of resumer /
# trend-break on VAL too, else the OOS separation is INCONCLUSIVE.
N_FLOOR = 60
MIN_PER_CLASS = 15

# ---- discriminator feature set (all STRICTLY CAUSAL at the break bar) -------------------------
#   giveback     = px[break]/causal_peak - 1            (<=0; how deep below the pre-break peak)
#   surge        = _surge[break]                        (volume surge still alive?)
#   surge_decay  = surge[break] / max(surge over hold)  (<=1; momentum cooling)
#   cush         = _cush[break]                         (<0 by definition of the break; HOW far below EMA)
#   bars_held    = break - entry                        (staleness of the move)
#   unreal       = px[break]/entry_px - 1               (unrealized gain at the break)
FEAT_NAMES = ["giveback", "surge", "surge_decay", "cush", "bars_held", "unreal"]
# the giveback-ONLY discriminator (the incremental-value floor): the agent ALREADY carries the
# `giveback` obs slot, so a discriminator that is "just giveback" adds no NEW information. surge (+
# surge_decay) is the candidate NEW signal the policy ignored in the forensic.
GIVEBACK_ONLY = ["giveback"]


# ================================================================================================
# Env construction (verbatim probe pattern) — gives the EXACT voltopk-8 universe + causal signals.
# ================================================================================================
def build_env(r, btc, liq, vol, k=8):
    from trader.train.event_env import EventRungEnv
    env = EventRungEnv(r, btc, liq, volume=vol, episode_bars=len(r) - WARMUP - 1,
                       k=k, warmup=WARMUP, universe_mode="voltopk", seed=0)
    env.reset(start=WARMUP)
    return env


# ================================================================================================
# Build the EMA-break event population: every valid ignition -> hypothetical hold -> first break.
# ================================================================================================
def build_events(env):
    """For each token in the env's voltopk universe, walk bars; at each FRESH valid ignition open a
    hypothetical hold (if not already inside one). Roll forward to the FIRST EMA-break (cush<0). Record
    the causal break features + the counterfactual forward outcome. Avoid double-counting: while a
    token is inside an active hold (entry..break) its intervening ignitions do NOT open new holds.

    Returns a list of event dicts. STRICTLY CAUSAL features; the resume label uses forward bars (the
    outcome). Both H=24 and H=48 outcomes computed; events whose H-window runs off-panel are dropped
    for that H (tracked per-H), so each H has a clean, unbiased denominator.
    """
    px, cush, surge, ig, cix, n = env._px, env._cush, env._surge, env._ignite, env.col_ix, env.n_bars
    start, end = env.start, env.end
    events = []
    for t in env.universe:
        j = cix[t]
        b = max(start, WARMUP)
        while b < min(end, n):
            if not ig[b, j] or px[b, j] <= 0:
                b += 1
                continue
            # ignition at b: open a hypothetical hold from b. Find the FIRST EMA-break b_x>b (cush<0).
            entry_bar = b
            entry_px = px[b, j]
            running_peak = px[b, j]
            peak_bar = b
            max_surge = max(float(surge[b, j]), 1e-12)
            b_x = None
            stop_first = False           # did the trailing stop fire at/before the EMA-break?
            bb = b + 1
            while bb < min(end, n):
                p = px[bb, j]
                # update the CAUSAL running peak over [entry..bb] BEFORE testing the break at bb (the
                # peak the rule's exit logic tracks: peak includes the current bar's high; here px is a
                # close index so peak is over closes [entry..bb])
                if p > running_peak:
                    running_peak = p
                    peak_bar = bb
                max_surge = max(max_surge, float(surge[bb, j]))
                stop_hit = p < running_peak * (1.0 - STOP_K)
                ema_hit = cush[bb, j] < 0.0
                if stop_hit and not ema_hit:
                    # the trailing stop would cut FIRST and the EMA hasn't broken — not an EMA-break
                    # event. End this hold here (the rule would be flat); the trade is a stop event.
                    b_x = bb
                    stop_first = True
                    break
                if ema_hit:
                    b_x = bb
                    stop_first = stop_hit   # both fired same bar -> note it, still an EMA-break
                    break
                bb += 1
            if b_x is None:
                # never broke EMA before the split/panel end -> no event; advance past this ignition
                b += 1
                continue
            # --- causal break features (<= b_x) ---
            # the CAUSAL pre-break running peak is over [entry .. b_x] (peak the position had SEEN;
            # px[b_x] is below EMA so it is at-or-below this peak). 'new high' = strictly > this peak.
            pre_break_peak = running_peak
            break_px = px[b_x, j]
            gb = break_px / pre_break_peak - 1.0 if pre_break_peak > 0 else 0.0   # <=0
            s_now = float(surge[b_x, j])
            s_decay = s_now / max_surge if max_surge > 0 else 0.0
            cu = float(cush[b_x, j])
            bars_held = b_x - entry_bar
            unreal = break_px / entry_px - 1.0 if entry_px > 0 else 0.0

            ev = {
                "tok": t, "entry_bar": entry_bar, "break_bar": b_x, "entry_px": entry_px,
                "pre_break_peak": pre_break_peak, "break_px": break_px,
                "stop_first": bool(stop_first),
                "feat": {"giveback": gb, "surge": s_now, "surge_decay": s_decay,
                         "cush": cu, "bars_held": float(bars_held), "unreal": unreal},
            }
            # --- counterfactual outcome: held PAST the break, does px resume to a NEW HIGH within H? ---
            for H in (24, 48):
                last = b_x + H
                if last >= n:
                    ev[f"resume_{H}"] = None       # window off-panel -> dropped for this H
                    ev[f"runup_given_up_{H}"] = None
                    ev[f"realized_{H}"] = None
                    ev[f"crash_{H}"] = None
                    continue
                fwd = px[b_x + 1: last + 1, j]
                fwd_max = float(fwd.max())
                resumed = fwd_max > pre_break_peak   # strictly ABOVE the causal pre-break peak
                ev[f"resume_{H}"] = bool(resumed)
                # run-up GIVEN UP by exiting at the break vs the best forward price (realizable upper
                # bound an exit-override could capture): max(0, fwd_max/break_px - 1)
                ev[f"runup_given_up_{H}"] = max(0.0, fwd_max / break_px - 1.0) if break_px > 0 else 0.0
                # held-to-H REALIZED (not the peak): px at the end of the window vs the break price —
                # the realizability check (does the resume round-trip back down by H?)
                ev[f"realized_{H}"] = float(px[last, j] / break_px - 1.0) if break_px > 0 else 0.0
                # resume-then-CRASH: it touched a new high but ended the window BELOW the break price
                ev[f"crash_{H}"] = bool(resumed and px[last, j] < break_px)
            events.append(ev)
            # advance: the hold ended at b_x; the next FRESH ignition for this token can open a new
            # hold only on/after b_x (cooldown-agnostic, broad population). Jump the cursor to b_x so we
            # don't re-open holds on ignitions that fired DURING this hold (no overlapping double-count).
            b = b_x
        # (the while-loop's b += 1 paths handle non-ignition / unbroken cases)
    return events


# ================================================================================================
# Regime: per-split BTC + universe-breadth context (is the resumer rate a bull artifact?).
# ================================================================================================
def regime(env):
    """Report the split's regime so a high resumer rate can be read against it (everything resumes in
    a bull). BTC start->end, and the mean fraction of the universe above its EMA (breadth)."""
    btc = env.btc.to_numpy()
    btc_ret = float(btc[env.end] / btc[env.start] - 1.0) if btc[env.start] > 0 else 0.0
    # universe basket return (equal-weight, the alts decouple from BTC) over the split
    cols = [env.col_ix[t] for t in env.universe]
    px = env._px
    alt_rets = [float(px[env.end, j] / px[env.start, j] - 1.0) for j in cols if px[env.start, j] > 0]
    alt_ret = float(np.mean(alt_rets)) if alt_rets else 0.0
    breadth = float(np.mean(env._cush[env.start:env.end, cols] > 0.0))
    return {"btc_ret": btc_ret, "alt_basket_ret": alt_ret, "mean_breadth": breadth}


# ================================================================================================
# Discriminator: a simple logistic on causal break feats predicting RESUMER (y=1). Fit on TRAIN,
# applied OOS on VAL with NO refit. Compared to a giveback-ONLY discriminator.
# ================================================================================================
def _logit(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))


def fit_logistic(X, y, seed=0, iters=6000, lr=0.3, l2=1e-3):
    """Standardize (mean/std from TRAIN only) + fit a logistic by GD, class-weighted. Returns coefs
    folded back into RAW feature space + the train-standardization so VAL is scored identically."""
    mu, sd = X.mean(0), X.std(0)
    sd[sd < 1e-9] = 1.0
    Xs = (X - mu) / sd
    rng = np.random.default_rng(seed)
    w = rng.normal(0, 0.01, Xs.shape[1])
    b = 0.0
    npos, nneg = max(y.sum(), 1.0), max(len(y) - y.sum(), 1.0)
    sw = np.where(y > 0, nneg / npos, 1.0)         # balance classes
    sw = sw / sw.mean()
    for _ in range(iters):
        pr = _logit(Xs @ w + b)
        g = (pr - y) * sw
        w -= lr * (Xs.T @ g / len(y) + l2 * w)
        b -= lr * g.mean()
    w_raw = w / sd
    b_raw = b - float(np.sum(w * mu / sd))
    return {"w": w_raw, "b0": b_raw, "coef_std": w.tolist()}


def score_logistic(X, model):
    return _logit(X @ model["w"] + model["b0"])


def auc(scores, y):
    """Mann-Whitney AUC (P(score_resumer > score_trendbreak)). 0.5 = no separation."""
    pos = scores[y > 0]
    neg = scores[y <= 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    order = np.argsort(np.concatenate([pos, neg]))
    ranks = np.empty(len(order), dtype=float)
    ranks[order] = np.arange(1, len(order) + 1)
    # average ranks for ties
    s = np.concatenate([pos, neg])
    su = np.argsort(s, kind="mergesort")
    s_sorted = s[su]
    r = np.arange(1, len(s) + 1, dtype=float)
    i = 0
    while i < len(s_sorted):
        jx = i
        while jx + 1 < len(s_sorted) and s_sorted[jx + 1] == s_sorted[i]:
            jx += 1
        if jx > i:
            r[i:jx + 1] = (i + 1 + jx + 1) / 2.0
        i = jx + 1
    ranks2 = np.empty(len(s), dtype=float)
    ranks2[su] = r
    rank_pos = ranks2[:len(pos)].sum()
    return float((rank_pos - len(pos) * (len(pos) + 1) / 2.0) / (len(pos) * len(neg)))


def feat_matrix(events, names, H):
    """Stack the causal feature matrix + resumer labels for events with a valid H-outcome."""
    rows, ys, toks, idx = [], [], [], []
    for i, e in enumerate(events):
        if e.get(f"resume_{H}") is None:
            continue
        rows.append([e["feat"][k] for k in names])
        ys.append(1.0 if e[f"resume_{H}"] else 0.0)
        toks.append(e["tok"])
        idx.append(i)
    return (np.array(rows, dtype=float), np.array(ys, dtype=float),
            np.array(toks), np.array(idx, dtype=int))


# ================================================================================================
# Bucket separation: the forensic's hypothesis as a concrete, readable contrast.
#   shallow-giveback + surge-alive  vs  deep-giveback + dead-surge.
# Thresholds chosen on TRAIN ONLY (median split), applied to VAL with NO refit.
# ================================================================================================
def fit_bucket_thresholds(events_tr):
    """TRAIN medians for giveback and surge_decay (the surge-alive proxy). 'shallow' = giveback above
    (closer to 0 than) the median; 'surge-alive' = surge_decay above the median (momentum less cooled).
    No outcome used in choosing thresholds (medians of the causal feature distribution)."""
    gb = np.array([e["feat"]["giveback"] for e in events_tr])
    sd = np.array([e["feat"]["surge_decay"] for e in events_tr])
    return {"gb_med": float(np.median(gb)), "sd_med": float(np.median(sd))}


def bucket_rates(events, thr, H):
    """Resumer rate in the shallow-giveback+surge-alive bucket vs the deep-giveback+dead-surge bucket,
    using TRAIN thresholds. The separation = shallow/alive rate - deep/dead rate."""
    shallow_alive, deep_dead = [], []
    for e in events:
        if e.get(f"resume_{H}") is None:
            continue
        gb, sd = e["feat"]["giveback"], e["feat"]["surge_decay"]
        y = 1.0 if e[f"resume_{H}"] else 0.0
        if gb >= thr["gb_med"] and sd >= thr["sd_med"]:
            shallow_alive.append(y)
        elif gb < thr["gb_med"] and sd < thr["sd_med"]:
            deep_dead.append(y)
    sa = float(np.mean(shallow_alive)) if shallow_alive else float("nan")
    dd = float(np.mean(deep_dead)) if deep_dead else float("nan")
    return sa, dd, len(shallow_alive), len(deep_dead)


# ================================================================================================
# Token-clustered bootstrap on the headline OOS separation (AUC of the full vs giveback-only model).
# ================================================================================================
def cluster_bootstrap_auc(X, y, toks, model, n_boot, rng):
    by = {}
    for i, tk in enumerate(toks):
        by.setdefault(tk, []).append(i)
    tk_list = list(by)
    point = auc(score_logistic(X, model), y)
    boots = []
    for _ in range(n_boot):
        pick = rng.integers(0, len(tk_list), len(tk_list))
        idx = np.concatenate([by[tk_list[k]] for k in pick])
        if len(np.unique(y[idx])) < 2:
            continue
        boots.append(auc(score_logistic(X[idx], model), y[idx]))
    boots = np.array([b for b in boots if np.isfinite(b)])
    lo, hi = (np.percentile(boots, [2.5, 97.5]) if len(boots) else (float("nan"), float("nan")))
    return float(point), float(lo), float(hi)


def cluster_bootstrap_delta(X_full, X_gb, y, toks, m_full, m_gb, n_boot, rng):
    """Bootstrap the AUC DELTA (full - giveback-only) — the incremental of surge-alive — token-
    clustered. CI-low > 0 means surge adds OOS separation beyond the obs the agent already carries."""
    by = {}
    for i, tk in enumerate(toks):
        by.setdefault(tk, []).append(i)
    tk_list = list(by)
    point = auc(score_logistic(X_full, m_full), y) - auc(score_logistic(X_gb, m_gb), y)
    boots = []
    for _ in range(n_boot):
        pick = rng.integers(0, len(tk_list), len(tk_list))
        idx = np.concatenate([by[tk_list[k]] for k in pick])
        if len(np.unique(y[idx])) < 2:
            continue
        d = auc(score_logistic(X_full[idx], m_full), y[idx]) - auc(score_logistic(X_gb[idx], m_gb), y[idx])
        boots.append(d)
    boots = np.array([b for b in boots if np.isfinite(b)])
    lo, hi = (np.percentile(boots, [2.5, 97.5]) if len(boots) else (float("nan"), float("nan")))
    return float(point), float(lo), float(hi)


# ================================================================================================
# Token-leave-one-out: refit the discriminator dropping each token in turn (TRAIN), score that
# token's VAL events — does the separation survive removing any single token's leverage?
# (Applied on the COMBINED train+val token set for the leave-one-out; reported as a robustness note.)
# ================================================================================================
def token_loo_auc(events_tr, events_va, names, H, seed=0):
    """For each token present in VAL: refit on TRAIN events EXCLUDING that token, score that token's
    VAL events. Report the min AUC across tokens with >= MIN_PER_CLASS resumers+trendbreaks (so a
    single token's pump can't carry the result). Coarse robustness, not the headline."""
    va_toks = sorted({e["tok"] for e in events_va if e.get(f"resume_{H}") is not None})
    results = {}
    for tk in va_toks:
        tr_sub = [e for e in events_tr if e["tok"] != tk]
        va_sub = [e for e in events_va if e["tok"] == tk]
        Xtr, ytr, _, _ = feat_matrix(tr_sub, names, H)
        Xva, yva, _, _ = feat_matrix(va_sub, names, H)
        if len(np.unique(ytr)) < 2 or len(np.unique(yva)) < 2:
            continue
        npos = int(yva.sum()); nneg = int(len(yva) - yva.sum())
        if npos < MIN_PER_CLASS or nneg < MIN_PER_CLASS:
            continue
        m = fit_logistic(Xtr, ytr, seed=seed)
        results[tk] = (auc(score_logistic(Xva, m), yva), npos, nneg)
    return results


# ================================================================================================
# Per-split reporting.
# ================================================================================================
def runup_given_up(events, H, resumers_only=True):
    """Mean forward run-up GIVEN UP by exiting at the break (realizable upper bound). If resumers_only,
    restricted to events that DID resume (premature cuts) — the cost of the hair-trigger."""
    vals = []
    for e in events:
        if e.get(f"resume_{H}") is None:
            continue
        if resumers_only and not e[f"resume_{H}"]:
            continue
        vals.append(e[f"runup_given_up_{H}"])
    return (float(np.mean(vals)), float(np.median(vals)), len(vals)) if vals else (float("nan"), float("nan"), 0)


def report_split(name, events, env, thr, fitted_full, fitted_gb, n_boot, rng, is_train):
    clean = [e for e in events if not e["stop_first"]]   # the headline = EMA-break-FIRST events
    stop_first = [e for e in events if e["stop_first"]]
    reg = regime(env)
    print(f"\n########## {name} ##########  universe(voltopk-8)={env.universe}")
    print(f"  REGIME: BTC {reg['btc_ret']:+.1%} over split | alt-basket(equal-wt) {reg['alt_basket_ret']:+.1%}"
          f" | mean breadth {reg['mean_breadth']:.0%}  <<< read resumer rate against this (bull artifact watch)")
    print(f"  events: {len(events)} total ignition-holds | {len(clean)} EMA-break-FIRST (headline) | "
          f"{len(stop_first)} trailing-stop-first (reported separately)")

    out = {"name": name, "regime": reg, "n_events": len(events), "n_clean": len(clean),
           "n_stop_first": len(stop_first)}

    for H in (24, 48):
        sub = [e for e in clean if e.get(f"resume_{H}") is not None]
        if not sub:
            print(f"  H={H}: no events with a full forward window")
            continue
        res = np.array([1.0 if e[f"resume_{H}"] else 0.0 for e in sub])
        base = float(res.mean())
        # cost: run-up given up by PREMATURE cuts (resumers); and crash/round-trip realizability
        gu_mean, gu_med, n_gu = runup_given_up(sub, H, resumers_only=True)
        # realized-at-H among resumers (does the new-high round-trip back below the break?)
        realized = np.array([e[f"realized_{H}"] for e in sub if e[f"resume_{H}"]])
        crash = np.array([1.0 if e[f"crash_{H}"] else 0.0 for e in sub if e[f"resume_{H}"]])
        realized_mean = float(realized.mean()) if len(realized) else float("nan")
        crash_rate = float(crash.mean()) if len(crash) else float("nan")
        # bucket separation (TRAIN thresholds), the forensic hypothesis made concrete
        sa, dd, n_sa, n_dd = bucket_rates(sub, thr, H)
        print(f"\n  --- H={H} (n={len(sub)} clean EMA-break events) ---")
        print(f"    RESUMER base rate (resume to a NEW HIGH > causal pre-break peak): {base:.1%}  "
              f"(n_resumer={int(res.sum())} / n_trendbreak={int(len(res)-res.sum())})")
        print(f"    COST: run-up GIVEN UP by premature cuts (resumers): mean {gu_mean:+.2%} "
              f"median {gu_med:+.2%} (n={n_gu})  <<< realizable upper bound an override could capture")
        print(f"    REALIZABILITY: resumers' held-to-H={H} realized vs break px: mean {realized_mean:+.2%}"
              f"  | resume-then-CRASH rate (touched new high but ended < break): {crash_rate:.0%}")
        print(f"    BUCKET separation (TRAIN-median thresholds): "
              f"shallow-giveback+surge-alive resumer {sa:.1%} (n={n_sa})  vs  "
              f"deep-giveback+dead-surge resumer {dd:.1%} (n={n_dd})")
        sep = (sa - dd) if (np.isfinite(sa) and np.isfinite(dd)) else float("nan")
        print(f"      -> bucket separation (shallow/alive - deep/dead) = {sep:+.1%}")

        out[f"H{H}"] = {"n": len(sub), "resumer_base": base,
                        "n_resumer": int(res.sum()), "n_trendbreak": int(len(res) - res.sum()),
                        "runup_given_up_mean": gu_mean, "runup_given_up_median": gu_med,
                        "resumer_realized_at_H_mean": realized_mean, "resume_then_crash_rate": crash_rate,
                        "bucket_shallow_alive_resumer": sa, "bucket_deep_dead_resumer": dd,
                        "bucket_separation": sep}

        # --- DISCRIMINATOR (AUC of the fitted models) — full vs giveback-only ---
        Xf, yf, tkf, _ = feat_matrix(sub, FEAT_NAMES, H)
        Xg, yg, tkg, _ = feat_matrix(sub, GIVEBACK_ONLY, H)
        npos, nneg = int(yf.sum()), int(len(yf) - yf.sum())
        if npos < MIN_PER_CLASS or nneg < MIN_PER_CLASS:
            print(f"    DISCRIMINATOR: INCONCLUSIVE for H={H} — class floor not met "
                  f"(resumer={npos}, trendbreak={nneg}; need >= {MIN_PER_CLASS} each)")
            out[f"H{H}"]["discriminator"] = "INCONCLUSIVE_class_floor"
            continue
        m_full = fitted_full[H]
        m_gb = fitted_gb[H]
        auc_full = auc(score_logistic(Xf, m_full), yf)
        auc_gb = auc(score_logistic(Xg, m_gb), yg)
        tag = "[TRAIN in-sample]" if is_train else "[VAL OOS — NO refit]"
        print(f"    DISCRIMINATOR {tag}: AUC full({'+'.join(FEAT_NAMES)}) = {auc_full:.3f}  | "
              f"AUC giveback-only = {auc_gb:.3f}  | incremental (full-gb) = {auc_full - auc_gb:+.3f}")
        # bootstrap CIs (token-clustered)
        pt_f, lo_f, hi_f = cluster_bootstrap_auc(Xf, yf, tkf, m_full, n_boot, rng)
        pt_d, lo_d, hi_d = cluster_bootstrap_delta(Xf, Xg, yf, tkf, m_full, m_gb, n_boot, rng)
        sig_f = "CI-low>0.5 (separates)" if lo_f > 0.5 else "CI straddles 0.5"
        sig_d = "CI-low>0 (incremental REAL)" if lo_d > 0 else "CI straddles 0 (no incremental)"
        print(f"      AUC full bootstrap (token-clustered, {n_boot}): {pt_f:.3f} 95%CI [{lo_f:.3f},{hi_f:.3f}] -> {sig_f}")
        print(f"      incremental AUC (full - giveback) bootstrap: {pt_d:+.3f} 95%CI [{lo_d:+.3f},{hi_d:+.3f}] -> {sig_d}")
        out[f"H{H}"]["discriminator"] = {
            "auc_full": auc_full, "auc_giveback_only": auc_gb, "incremental_auc": auc_full - auc_gb,
            "auc_full_ci": [lo_f, hi_f], "incremental_auc_ci": [lo_d, hi_d],
            "coef_std_full": dict(zip(FEAT_NAMES, [round(c, 3) for c in m_full["coef_std"]])),
        }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--boot", type=int, default=5000)
    args = ap.parse_args()
    from train_rl import build_volume_panel, load_data, time_split
    returns, btc, anchor, liq = load_data()
    train_r, val_r, test_r = time_split(returns)
    vol = build_volume_panel(list(returns.columns), returns.index)
    rng = np.random.default_rng(0)

    print("=" * 100)
    print("P-EMABREAK — is the rung-0 EMA-break a hair-trigger? + can a causal discriminator (beating")
    print("             giveback-only) separate premature cuts (resumers) from trend-breaks OOS?")
    print(f"  n-floor (per split) = {N_FLOOR}; class floor (per class, per H) = {MIN_PER_CLASS}")
    print("  RESUME label uses forward bars (the OUTCOME); DISCRIMINATOR feats are STRICTLY CAUSAL (<=break).")
    print("  Discriminator FIT ON TRAIN, applied to VAL with NO refit. TEST split FROZEN.")
    print("  *** DRIFT ALARM: this probe is a SUBSTRATE go/no-go, NOT the honest gate. The fix it could")
    print("      justify (giveback-penalty reward / surge-at-break exit obs) must still be validated")
    print("      COLD-WEEKLY: beat rung-0 OOS + survive ~30% DD DQ + >=1 trade/day. ***")
    print("=" * 100)

    # --- build event populations ---
    env_tr = build_env(train_r, btc, liq, vol)
    env_va = build_env(val_r, btc, liq, vol)
    ev_tr = build_events(env_tr)
    ev_va = build_events(env_va)
    clean_tr = [e for e in ev_tr if not e["stop_first"]]
    clean_va = [e for e in ev_va if not e["stop_first"]]
    print(f"\n[pop] TRAIN clean EMA-break events = {len(clean_tr)} ; VAL = {len(clean_va)}")
    if len(clean_tr) < N_FLOOR:
        print(f"[pop] TRAIN below n-floor ({N_FLOOR}) -> TRAIN fit INCONCLUSIVE")
    if len(clean_va) < N_FLOOR:
        print(f"[pop] VAL below n-floor ({N_FLOOR}) -> VAL verdict INCONCLUSIVE")

    # --- FIT discriminators on TRAIN ONLY (per H) ---
    thr = fit_bucket_thresholds(clean_tr)
    fitted_full, fitted_gb = {}, {}
    for H in (24, 48):
        Xf, yf, _, _ = feat_matrix(clean_tr, FEAT_NAMES, H)
        Xg, yg, _, _ = feat_matrix(clean_tr, GIVEBACK_ONLY, H)
        if len(yf) and len(np.unique(yf)) == 2:
            fitted_full[H] = fit_logistic(Xf, yf, seed=0)
            fitted_gb[H] = fit_logistic(Xg, yg, seed=0)
            print(f"[fit] H={H} full coefs(std): "
                  f"{dict(zip(FEAT_NAMES, [round(c,3) for c in fitted_full[H]['coef_std']]))}")
        else:
            fitted_full[H] = {"w": np.zeros(len(FEAT_NAMES)), "b0": 0.0, "coef_std": [0]*len(FEAT_NAMES)}
            fitted_gb[H] = {"w": np.zeros(1), "b0": 0.0, "coef_std": [0]}
    print(f"[fit] bucket thresholds (TRAIN medians): giveback>={thr['gb_med']:+.3f} (shallow), "
          f"surge_decay>={thr['sd_med']:.3f} (alive)")

    res = {}
    res["TRAIN"] = report_split("TRAIN (in-sample sanity)", ev_tr, env_tr, thr,
                                fitted_full, fitted_gb, args.boot, rng, is_train=True)
    res["VAL"] = report_split("VAL (OOS — the headline)", ev_va, env_va, thr,
                              fitted_full, fitted_gb, args.boot, rng, is_train=False)

    # --- token-leave-one-out robustness (VAL), H=48 ---
    print("\n########## TOKEN-LEAVE-ONE-OUT (VAL, H=48; refit TRAIN excl. token, score that token) ##########")
    loo = token_loo_auc(clean_tr, clean_va, FEAT_NAMES, 48)
    if loo:
        aucs = [v[0] for v in loo.values()]
        for tk, (a, npos, nneg) in sorted(loo.items(), key=lambda kv: kv[1][0]):
            print(f"  {tk:8s} AUC {a:.3f}  (resumer={npos}, trendbreak={nneg})")
        print(f"  -> min token AUC {min(aucs):.3f} | mean {np.mean(aucs):.3f}  "
              f"(survives if min materially > 0.5)")
        res["loo_min_auc"] = float(min(aucs))
        res["loo_mean_auc"] = float(np.mean(aucs))
    else:
        print("  no token cleared the per-token class floor -> LOO INCONCLUSIVE")
        res["loo_min_auc"] = None

    # --- machine-readable tail ---
    print("\nRESULT_JSON " + json.dumps(res, default=lambda o: float(o) if isinstance(o, (np.floating,)) else o))


if __name__ == "__main__":
    main()
