"""P-CAPACITY — how much of the +EV ignition stream can the rung-0 RULE SAFELY participate in?

The meta-finding two prior probes (P-REIGNITE, P-EXIT-REWARD) independently hit: the rung-0 RULE
funds only ~5-12% of in-universe ignitions (39 TRAIN / 20 VAL closed trades of ~952/321 ignitions).
Both the entry-SELECTION edge (refuted) and the exit-REWARD edge (data-starved) sit DOWNSTREAM of a
more basic limit: HOW MUCH of the +EV ignition stream is reachable WITHOUT breaching the ~30% DQ.

Torch-free, laptop-local. Loads data via train_rl.load_data / build_volume_panel (the SAME path the
rule + env use). Reuses the production rung-0 RULE (run_rung0) and the cold-weekly PORTFOLIO grader
(weekly_eval) verbatim. Does NOT modify production code.

=====================================================================================================
PART A — CHARACTERIZE the unfunded ignitions (which gate rejects the most +EV; money on the table?).

For every in-universe VALID ignition bar (a bar the rule would actually consider — in the split's
voltopk-8 universe, with px[b]>0, fwd window on-panel), replay the rule's EXACT fund/skip decision
(a verbatim mirror of run_rung0 / _rule_equity_curve) and tag the ignition by the FIRST gate that
binds it, in the rule's own evaluation order:

  funded            — the rule opened a NEW position on this flat token this bar (won a slot/rotation).
  already-held      — t is currently in the rule's book (the scale-in class; entry prompts skip it).
  cooldown          — flat, but within `cooldown` bars of a prior exit on this token.
  not-reclaimed     — flat & cooled, but px[b] <= the prior cycle's runup origin (dead-zone re-entry).
  capacity-no-rot   — flat, cooled, reclaimed, cleared the candidate set, but no cash AND no holding
                      to rotate (book empty / cash short with nothing to free).
  rotation-rejected — flat, cooled, reclaimed, cash short, has holdings, but the WEAKEST holding's
                      cushion >= this candidate's cushion (the swap-weak-for-strong guard refused).

Attribution is to the FIRST binding gate (the one that would have to change to fund it) — exits and
funding are replayed bar-by-bar so a candidate that loses a slot to a STRONGER co-ignition that bar
is correctly 'rotation-rejected' / 'capacity', not mis-tagged. Outcome per ignition = causal forward
max-run-up max(px[b..b+H])/px[b]-1 (H=24, H=48). DELIVERABLE: the binding gate (most +EV rejected) and
whether unfunded +EV ~= funded +EV (money on the table) vs < funded +EV (the gate does useful
selection). Contrast funded-vs-each-class via a per-week-block + token-clustered bootstrap (ignitions
cluster in time within a token). N-FLOOR = 30 events/class for a conclusive class verdict.

=====================================================================================================
PART B — DQ-AWARE COUNTERFACTUAL (return-vs-DD Pareto on the COLD-WEEKLY PORTFOLIO).

Re-run the rung-0 RULE with LOOSER capacity settings on the SAME cold-weekly windows the deployment
grader uses, and for each setting report the COLD-WEEKLY PORTFOLIO return AND the PORTFOLIO max-DD
(the real ~30% DQ object — within-week, fresh $10k, NOT per-trade DD) AND activity (trades/day). The
binding knobs from A are swept: cooldown {48,24,12,0}, reclaimed gate {on,off}, and per-position size
entry_frac (concurrent-slot count: bigger frac => fewer concurrent names; smaller => more). The
reclaimed toggle is the ONLY knob run_rung0 doesn't expose, so a thin, faithful counterfactual
executor (`run_rung0_cf`) mirrors run_rung0 line-for-line and adds a `reclaimed_gate` flag (and
re-exposes cooldown/entry_frac); it is VALIDATED bar-identical to production run_rung0 at the rule's
default knobs before any sweep runs. Costs (AMM fee + impact + gas) applied EQUALLY everywhere.

THE TRAP (P-EXIT-REWARD scoping limit, do NOT repeat): the DD object is the COLD-WEEKLY PORTFOLIO
max-DD, NOT per-trade intra-trade DD and NOT one continuous-window DD. More participation => more
CONCURRENT positions => potentially higher portfolio DD; the capacity lever only helps if it improves
the return-vs-DD frontier INSIDE the DQ. DELIVERABLE: is there a DQ-safe Pareto point that captures
materially more of the +EV stream, and how much return does DQ-safe extra participation buy?

TRAIN + VAL only. The TEST split (1025 bars / 5 cold weeks) is FROZEN and never touched here.

  .venv\\Scripts\\python.exe scripts\\probe_capacity.py [--boot 5000] [--nfloor 30]
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
RULE_EF = 0.20            # the rung-0 RULE's fixed entry frac (run_rung0 default == EventRungEnv.rule_entry_frac)
WEEK_SECS = 7 * 24 * 3600


# =====================================================================================================
# PART A — replay the rule's fund/skip and attribute the FIRST binding gate per ignition
# =====================================================================================================

def build_env(r, btc, liq, vol):
    """One full-window episode reset at WARMUP — gives the EXACT voltopk-8 universe + the precomputed
    causal signal arrays (px / cushion / surge / ignite) over the whole split. Same pattern as
    probe_reignite.build_env."""
    from trader.train.event_env import EventRungEnv
    env = EventRungEnv(r, btc, liq, volume=vol, episode_bars=len(r) - WARMUP - 1,
                       k=8, warmup=WARMUP, universe_mode="voltopk", seed=0)
    env.reset(start=WARMUP)
    return env


def rule_replay_attribution(env):
    """Replay the rung-0 RULE loop VERBATIM (mirror of run_rung0 / _rule_equity_curve), and at EVERY
    in-universe valid ignition bar attribute the rule's decision to the FIRST binding gate.

    Returns `attrib[(tok, bar)] -> reason`. Reasons:
      funded / already-held / cooldown / not-reclaimed / capacity-no-rot / rotation-rejected.

    The attribution is computed INSIDE the same bar-by-bar replay that drives funding, so a candidate
    that clears its own gates but loses its slot to a stronger co-igniting token that bar is tagged by
    the funding outcome (rotation-rejected / capacity), never mis-attributed. To get this exactly
    right we attribute the flat-cooled-reclaimed candidates AFTER the bar's funding loop has run, by
    recording which of them actually got funded and which were displaced/blocked."""
    px, cush, ig, cix = env._px, env._cush, env._ignite, env.col_ix
    sk, cd, ef = STOP_K, COOLDOWN, RULE_EF
    from trader.sim.broker import amm_cost_usd
    fee, gas, liq = env.lp_fee_bps, env.gas_usd, env.liquidity
    start, end = env.start, env.end
    cash, pos = env.capital, {}
    cool = {t: -10 ** 9 for t in env.universe}
    prior = {t: None for t in env.universe}

    def value(t, bar):
        p = pos[t]
        j = cix[t]
        return p["usd"] * px[bar, j] / px[p["entry_bar"], j]

    attrib = {}
    for bar in range(start, end + 1):
        equity = cash + sum(value(t, bar) for t in pos)
        if equity <= 1.0:
            continue
        for t in list(pos):                                       # 1) exits (trailing stop OR EMA-break)
            j = cix[t]
            p = pos[t]
            p["peak_px"] = max(p["peak_px"], px[bar, j])
            if px[bar, j] < p["peak_px"] * (1.0 - sk) or cush[bar, j] < 0.0:
                v = value(t, bar)
                cash += v - amm_cost_usd(-v, liq.get(t, 0.0), fee, gas)
                cool[t], prior[t] = bar, p["origin"]
                del pos[t]
        # held-book snapshot AFTER exits, BEFORE new entries — the state each ignition this bar sees.
        held_now = set(pos)
        # pre-classify the gate each IGNITING token hits, in the rule's evaluation order.
        # gate 1: already-held; gate 2: cooldown; gate 3: not-reclaimed. cands = passed all three.
        cands = []
        for t in env.universe:
            if not ig[bar, cix[t]]:
                continue
            if t in held_now:
                attrib[(t, bar)] = "already-held"
            elif (bar - cool[t]) < cd:
                attrib[(t, bar)] = "cooldown"
            elif not (prior[t] is None or px[bar, cix[t]] > prior[t]):
                attrib[(t, bar)] = "not-reclaimed"
            else:
                cands.append(t)                                   # cleared 1-3: a fundable candidate
        # 2) the rule's funding loop: fund strongest first, loser-funded rotation. Tag each candidate
        # by its ACTUAL outcome this bar (funded / rotation-rejected / capacity-no-rot).
        cands.sort(key=lambda t: cush[bar, cix[t]], reverse=True)
        for t in cands:
            funded = False
            reason = None
            if min(ef * equity, cash) < 1.0:                      # cash short -> try loser-funded rotation
                if pos:
                    weak = min(pos, key=lambda h: cush[bar, cix[h]])
                    if cush[bar, cix[weak]] < cush[bar, cix[t]]:  # swap weak for strong
                        v = value(weak, bar)
                        cash += v - amm_cost_usd(-v, liq.get(weak, 0.0), fee, gas)
                        cool[weak], prior[weak] = bar, pos[weak]["origin"]
                        del pos[weak]
                        equity = cash + sum(value(tt, bar) for tt in pos)
                    else:
                        reason = "rotation-rejected"              # weakest holding stronger than candidate
                else:
                    reason = "capacity-no-rot"                    # no cash AND no holding to free
            size = min(ef * equity, cash)
            if reason is None and size >= 1.0:
                j = cix[t]
                cash -= size + amm_cost_usd(size, liq.get(t, 0.0), fee, gas)
                pos[t] = {"usd": size, "entry_bar": bar, "peak_px": px[bar, j], "origin": px[bar, j]}
                funded = True
            attrib[(t, bar)] = "funded" if funded else (reason or "capacity-no-rot")
    return attrib


def fwd_runup(px_col, b, H, n):
    """Causal forward run-up over [b, b+H]: max(px[b..b+H])/px[b]-1. None if window runs off-panel."""
    if b + H >= n:
        return None
    p0 = px_col[b]
    if p0 <= 0:
        return None
    return float(px_col[b: b + H + 1].max() / p0 - 1.0)


def classify_rows(env, attrib):
    """Walk every in-universe valid ignition bar and emit a row with its gate reason, surge, week
    block, and causal forward run-ups (H=24/48). Drops bars whose fwd-48 window runs off-panel (so
    funded vs unfunded see the same horizon discipline — no truncated-window bias)."""
    px, ig, surge, cix = env._px, env._ignite, env._surge, env.col_ix
    n = env.n_bars
    start, end = env.start, env.end
    idx = env.returns.index.to_numpy()
    idx_s = (idx // 1000 if idx.max() > 1e12 else idx).astype(np.int64)
    rows = []
    for t in env.universe:
        j = cix[t]
        pxc = px[:, j]
        for b in range(max(start, WARMUP), min(end + 1, n)):
            if not ig[b, j] or pxc[b] <= 0:
                continue
            f24 = fwd_runup(pxc, b, 24, n)
            f48 = fwd_runup(pxc, b, 48, n)
            if f24 is None or f48 is None:
                continue
            rows.append({"tok": t, "bar": b, "surge": float(surge[b, j]),
                         "reason": attrib.get((t, b), "MISSING"),
                         "week": int(idx_s[b] // WEEK_SECS), "f24": f24, "f48": f48})
    return rows


# ---- bootstrap (per-week-block + token-clustered) ---------------------------------------------------

def block_bootstrap_diff(g_vals, g_keys, h_vals, h_keys, n_boot, rng):
    """Bootstrap mean(g) - mean(h) by resampling the CLUSTER KEY (here (token, week) blocks) with
    replacement within each group — ignitions cluster in time within a token AND within a week, so
    per-bar samples are NOT independent. Returns (point, ci_lo, ci_hi) at 95%, or (None,None,None) if
    either group is empty."""
    def by_cluster(vals, keys):
        d = {}
        for v, k in zip(vals, keys):
            d.setdefault(k, []).append(v)
        return d
    G, H = by_cluster(g_vals, g_keys), by_cluster(h_vals, h_keys)
    Gk, Hk = list(G), list(H)
    if not Gk or not Hk:
        return None, None, None
    point = float(np.mean(g_vals) - np.mean(h_vals))
    diffs = np.empty(n_boot)
    for i in range(n_boot):
        gs = np.concatenate([G[Gk[k]] for k in rng.integers(0, len(Gk), len(Gk))])
        hs = np.concatenate([H[Hk[k]] for k in rng.integers(0, len(Hk), len(Hk))])
        diffs[i] = gs.mean() - hs.mean()
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return point, float(lo), float(hi)


REASONS = ("funded", "already-held", "cooldown", "not-reclaimed", "capacity-no-rot", "rotation-rejected")


def part_a(rows, label, n_boot, nfloor, rng):
    print(f"\n########## PART A — {label} ##########")
    print(f"  in-universe valid ignitions classified (fwd-48 on-panel): {len(rows)}")
    miss = [r for r in rows if r["reason"] == "MISSING"]
    if miss:
        print(f"  WARNING: {len(miss)} ignitions unattributed (MISSING) — replay/classify bar mismatch")
    funded = [r for r in rows if r["reason"] == "funded"]
    print(f"\n  --- per-gate breakdown (n, tokens, fwd-runup) ---")
    summary = {}
    for reason in REASONS:
        sub = [r for r in rows if r["reason"] == reason]
        if not sub:
            print(f"    {reason:18s}: n=0")
            summary[reason] = None
            continue
        f24 = np.array([r["f24"] for r in sub])
        f48 = np.array([r["f48"] for r in sub])
        ntok = len({r["tok"] for r in sub})
        summary[reason] = {"n": len(sub), "f24": f24, "f48": f48}
        print(f"    {reason:18s}: n={len(sub):4d}  tokens={ntok}  share={len(sub)/len(rows):5.1%}")
        print(f"        fwd24 mean {f24.mean():+7.2%}  median {np.median(f24):+7.2%}  win {np.mean(f24>0):4.0%}")
        print(f"        fwd48 mean {f48.mean():+7.2%}  median {np.median(f48):+7.2%}  win {np.mean(f48>0):4.0%}")
    # binding gate by VOLUME of +EV rejected: sum of positive fwd-48 run-up across each unfunded class
    print(f"\n  --- which gate rejects the most +EV (sum of fwd-48 run-up over the class) ---")
    ev_by_gate = {}
    for reason in REASONS:
        if reason == "funded" or summary.get(reason) is None:
            continue
        f48 = summary[reason]["f48"]
        ev_by_gate[reason] = float(f48.sum())          # total run-up the class carries (the table-money)
        print(f"    {reason:18s}: n={summary[reason]['n']:4d}  Sum(fwd48)={f48.sum():+8.2f}  "
              f"mean {f48.mean():+7.2%}")
    binding = max(ev_by_gate, key=ev_by_gate.get) if ev_by_gate else None
    print(f"    => BINDING gate (most total +EV rejected): {binding}")
    # money-on-the-table contrast: funded vs each unfunded class (per (token,week)-block bootstrap)
    print(f"\n  --- money on the table: funded vs each unfunded class (block bootstrap; n-floor={nfloor}) ---")
    if not funded:
        print("    no funded ignitions in this split — cannot contrast")
        return summary, binding
    fkeys = [(r["tok"], r["week"]) for r in funded]
    verdicts = {}
    for reason in REASONS:
        if reason == "funded" or summary.get(reason) is None:
            continue
        sub = [r for r in rows if r["reason"] == reason]
        nf = summary["funded"]["n"]
        conclusive = (len(sub) >= nfloor and nf >= nfloor)
        for H, key in (("H24", "f24"), ("H48", "f48")):
            uv = [r[key] for r in sub]
            ukeys = [(r["tok"], r["week"]) for r in sub]
            fv = summary["funded"][key]
            pt, lo, hi = block_bootstrap_diff(uv, ukeys, list(fv), fkeys, n_boot, rng)
            tag = ("INCONCLUSIVE (n<floor)" if not conclusive else
                   "unfunded > funded (money on table)" if lo is not None and lo > 0 else
                   "unfunded < funded (useful selection)" if hi is not None and hi < 0 else
                   "~= funded (CI straddles 0)")
            print(f"    {reason:18s} {H}: unfunded {np.mean(uv):+7.2%}  funded {fv.mean():+7.2%}  "
                  f"diff {pt:+7.2%}  95%CI [{lo:+.2%},{hi:+.2%}]  -> {tag}")
            verdicts[(reason, H)] = tag
    return summary, binding


# =====================================================================================================
# PART B — DQ-aware counterfactual: the cold-weekly PORTFOLIO return-vs-DD Pareto
# =====================================================================================================

def run_rung0_cf(returns, signal_fn, liquidity, *, capital=10_000.0, warmup=WARMUP,
                 entry_frac=RULE_EF, stop_k=STOP_K, cooldown=COOLDOWN, reclaimed_gate=True,
                 rotate=True, lp_fee_bps=None, gas_usd=None):
    """Counterfactual executor — a FAITHFUL mirror of production run_rung0 that re-exposes the binding
    capacity knobs (cooldown, entry_frac) AND adds the one knob production doesn't expose: a
    `reclaimed_gate` toggle (the dead-zone re-entry block). Everything else is line-for-line identical
    to src/trader/strategy/rung0.run_rung0. VALIDATED bar-identical to production at the default knobs
    (cooldown=48, reclaimed_gate=True, entry_frac=0.20, rotate=True) before any sweep runs.

    Does NOT touch production code. Returns (equity Series, records, total_fees)."""
    from trader.sim.broker import DEFAULT_GAS_USD, DEFAULT_LP_FEE_BPS, amm_cost_usd
    lp_fee_bps = DEFAULT_LP_FEE_BPS if lp_fee_bps is None else lp_fee_bps
    gas_usd = DEFAULT_GAS_USD if gas_usd is None else gas_usd
    syms = list(returns.columns)
    pos = pd.Series(0.0, index=syms)
    cash, fees, bar = float(capital), 0.0, 0
    n_buys = 0                                                    # NEW-position opens (participation proxy)
    st = {s: {"held": False, "origin": None, "peak": None, "exit_reb": -10 ** 9,
              "prior_origin": None} for s in syms}
    eq = np.empty(len(returns))
    records: list = []
    for i in range(len(returns)):
        r = returns.iloc[i].reindex(syms).fillna(0.0).to_numpy()
        pos = pd.Series(pos.to_numpy() * (1.0 + r), index=syms)
        equity = float(pos.sum() + cash)
        tu, tf = {}, {}

        def trade(t, delta):
            nonlocal cash, fees
            c = amm_cost_usd(delta, liquidity.get(t, 0.0), lp_fee_bps, gas_usd)
            cash -= delta + c
            pos[t] += delta
            fees += c
            tu[t], tf[t] = tu.get(t, 0.0) + delta, tf.get(t, 0.0) + c

        if i >= warmup and equity > 1.0:
            sig = signal_fn(returns.iloc[: i + 1])
            for t in [s for s in sig if st[s]["held"]]:
                s, d = st[t], sig[t]
                s["peak"] = max(s["peak"], d["price"])
                if d["price"] < s["peak"] * (1.0 - stop_k) or d["price"] < d["ema"]:
                    v = float(pos[t])
                    if abs(v) >= 1.0:
                        trade(t, -v)
                    s.update(held=False, prior_origin=s["origin"], exit_reb=bar)
            cands = [t for t in sig if sig[t]["ignite"] and not st[t]["held"]
                     and (bar - st[t]["exit_reb"]) >= cooldown
                     and ((not reclaimed_gate) or st[t]["prior_origin"] is None
                          or sig[t]["price"] > st[t]["prior_origin"])]
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
                    st[t].update(held=True, origin=sig[t]["price"], peak=sig[t]["price"])
                    n_buys += 1
            bar += 1
        eq[i] = float(pos.sum() + cash)
        if tu or (i >= warmup and i % 24 == 0):
            e = eq[i] if eq[i] > 0 else 1.0
            records.append({"time": int(returns.index[i]),
                            "weights": {s: float(pos[s] / e) for s in syms if pos[s] > 1e-6},
                            "trades_usd": tu, "trade_fees": tf})
    return pd.Series(eq, index=returns.index), records, fees, n_buys


def validate_cf(returns, liq, vol):
    """Confirm run_rung0_cf at default knobs is bar-identical to production run_rung0 — the whole Part
    B DD object rests on this. Compares equity curves over the full TRAIN split."""
    from trader.strategy.rung0 import build_rung0, run_rung0
    from trader.train.weekly_eval import causal_voltop_universe
    uni = causal_voltop_universe(returns.iloc[:WARMUP + 200], k=8, warmup=WARMUP)
    sig = build_rung0(returns, tokens=uni, volume=vol)
    eq_prod, _, f_prod = run_rung0(returns, sig, liq, warmup=WARMUP)
    eq_cf, _, f_cf, _ = run_rung0_cf(returns, sig, liq, warmup=WARMUP)
    max_abs = float(np.max(np.abs(eq_prod.to_numpy() - eq_cf.to_numpy())))
    ok = max_abs < 1e-6 and abs(f_prod - f_cf) < 1e-9
    print(f"\n  [validate] run_rung0_cf vs production run_rung0 (default knobs): "
          f"max|eq diff|={max_abs:.2e}  fee diff={abs(f_prod - f_cf):.2e}  -> "
          f"{'IDENTICAL' if ok else 'MISMATCH (Part B INVALID)'}")
    return ok


def grade_week_cf(ws, win, liq, vol, *, cooldown=COOLDOWN, reclaimed_gate=True, entry_frac=RULE_EF,
                  k=8, warmup=WARMUP):
    """Grade ONE cold week under counterfactual knobs — the EXACT grade_week_baselines recipe, but the
    rule runs through run_rung0_cf with the swept knobs. Returns (week-return, PORTFOLIO max-DD,
    trade-days). The DD is the WITHIN-WEEK PORTFOLIO drawdown (fresh $10k, the real DQ object), NOT
    per-trade DD — computed identically to weekly_eval.grade_week_baselines."""
    from trader.strategy.rung0 import build_rung0
    from trader.train.weekly_eval import causal_voltop_universe, _trade_days
    uni = causal_voltop_universe(win, k=k, warmup=warmup)
    sig = build_rung0(win, tokens=uni, volume=vol)
    eq, records, _, n_buys = run_rung0_cf(win, sig, liq, warmup=warmup, cooldown=cooldown,
                                          reclaimed_gate=reclaimed_gate, entry_frac=entry_frac)
    eq = eq.iloc[warmup:]                                          # the COLD week only (drop the prepad)
    ret = float(eq.iloc[-1] / eq.iloc[0] - 1.0)
    dd = abs(float((eq / eq.cummax() - 1.0).min()))               # WITHIN-WEEK PORTFOLIO drawdown
    tdays = len(_trade_days(records, ws))
    return ret, dd, tdays, n_buys


def part_b(returns, val_start, test_start, liq, vol, n_boot, rng):
    from trader.train.weekly_eval import (DD_GATE, bootstrap_mean_ci, cold_week_windows, split_label)
    print(f"\n########## PART B — DQ-aware counterfactual (cold-weekly PORTFOLIO; DQ={DD_GATE:.0%}) ##########")
    # collect TRAIN+VAL cold weeks (TEST frozen)
    weeks = [(ws, win) for ws, win in cold_week_windows(returns)
             if split_label(ws, val_start, test_start) in ("train", "val")]
    splits = {}
    for ws, _ in weeks:
        splits[split_label(ws, val_start, test_start)] = splits.get(split_label(ws, val_start, test_start), 0) + 1
    print(f"  cold weeks (train+val, TEST frozen): n={len(weeks)}  {splits}")
    print(f"  DD object = within-week PORTFOLIO max-DD (fresh $10k/week), costs applied equally.\n")

    # Focused sweep. The 2x2 CONTROL isolates the two things a 'looser' knob bundles together:
    # PARTICIPATION (cooldown->0 + reclaimed OFF: open more positions) vs PER-NAME SIZING (entry_frac:
    # smaller frac => smaller per-position bet => lower portfolio DD, more concurrent slots). The
    # baseline (cd48, reclaimed on, ef.20) is the rung-0 RULE the deployment gate already grades.
    #   baseline      = cd48 recl  ef.20   (reference)
    #   +participation= cd0  NOrecl ef.20   (loosen capacity gates ONLY — isolate participation's DD cost)
    #   +sizing       = cd48 recl  ef.125  (shrink per-name bet ONLY — isolate sizing's DD relief)
    #   +both         = cd0  NOrecl ef.125  (loosen gates AND shrink size)
    # plus the cooldown ladder at ef.20 to confirm cooldown alone is not the binding lever.
    settings = [
        ("baseline cd48 recl ef.20",   dict(cooldown=48, reclaimed_gate=True,  entry_frac=0.20)),
        ("cd24 recl ef.20",            dict(cooldown=24, reclaimed_gate=True,  entry_frac=0.20)),
        ("cd0  recl ef.20",            dict(cooldown=0,  reclaimed_gate=True,  entry_frac=0.20)),
        ("+PART cd0 NOrecl ef.20",     dict(cooldown=0,  reclaimed_gate=False, entry_frac=0.20)),
        ("+SIZE cd48 recl ef.125",     dict(cooldown=48, reclaimed_gate=True,  entry_frac=0.125)),
        ("+BOTH cd0 NOrecl ef.125",    dict(cooldown=0,  reclaimed_gate=False, entry_frac=0.125)),
        ("+SIZE cd48 recl ef.10",      dict(cooldown=48, reclaimed_gate=True,  entry_frac=0.10)),
        ("+BOTH cd0 NOrecl ef.10",     dict(cooldown=0,  reclaimed_gate=False, entry_frac=0.10)),
    ]
    print(f"  {'setting':26s} {'mean':>7s} {'ci_lo':>7s} {'median':>7s} {'worstDD':>8s} {'dqWk':>5s} "
          f"{'trd/day':>8s} {'buys/wk':>8s} {'DQ-safe':>8s}")
    results = []
    base_rets = None
    for name, kw in settings:
        rets, dds, tdays, buys = [], [], [], []
        for ws, win in weeks:
            ret, dd, td, nb = grade_week_cf(ws, win, liq, vol, **kw)
            rets.append(ret); dds.append(dd); tdays.append(td); buys.append(nb)
        rets = np.array(rets); dds = np.array(dds)
        lo, hi, mean = bootstrap_mean_ci(rets, n_boot=n_boot)
        worst_dd = float(dds.max())
        dq_weeks = int((dds > DD_GATE).sum())
        dq_safe = worst_dd < DD_GATE
        trd_per_day = float(np.mean(tdays)) / 7.0
        buys_per_wk = float(np.mean(buys))
        if name.startswith("baseline"):
            base_rets = rets.copy()
        results.append({"name": name, "rets": rets, "dds": dds, "mean": mean, "ci_lo": lo,
                        "worst_dd": worst_dd, "dq_weeks": dq_weeks, "dq_safe": dq_safe,
                        "trd_per_day": trd_per_day, "buys_per_wk": buys_per_wk})
        print(f"  {name:26s} {mean:+7.2%} {lo:+7.2%} {np.median(rets):+7.2%} {worst_dd:8.2%} "
              f"{dq_weeks:5d} {trd_per_day:8.2f} {buys_per_wk:8.2f} {'YES' if dq_safe else 'NO':>8s}")

    # Pareto: among DQ-SAFE settings, which dominate baseline on return without worse DD?
    print(f"\n  --- return-vs-DD Pareto (DQ-safe settings only; paired-vs-baseline edge) ---")
    base = next(r for r in results if r["name"].startswith("baseline"))
    pareto = []
    for r in results:
        if not r["dq_safe"]:
            print(f"    {r['name']:28s}: BREACHES DQ (worstDD {r['worst_dd']:.1%}) — excluded")
            continue
        if r["name"].startswith("baseline"):
            continue
        # paired weekly edge vs baseline (same weeks) — cancels the common market variance
        edge = r["rets"] - base_rets
        lo, hi, mean_e = bootstrap_mean_ci(edge, n_boot=n_boot)
        better_ret = mean_e > 0
        not_worse_dd = r["worst_dd"] <= base["worst_dd"] + 1e-9
        is_pareto = better_ret and not_worse_dd
        sig = "edge>0 (CI-lo>0)" if lo > 0 else "edge>0 but CI straddles 0" if mean_e > 0 else "edge<=0"
        print(f"    {r['name']:28s}: ret-edge vs base {mean_e:+6.2%} CI[{lo:+.2%},{hi:+.2%}] {sig}; "
              f"DD {r['worst_dd']:.1%} vs base {base['worst_dd']:.1%}"
              + ("  <= PARETO (more participation, DQ-safe, no worse DD)" if is_pareto else ""))
        if is_pareto:
            pareto.append((r["name"], mean_e, lo, r["worst_dd"]))
    if not pareto:
        print(f"    => NO DQ-safe setting dominates baseline on return without worse portfolio DD.")
    else:
        best = max(pareto, key=lambda x: x[1])
        print(f"    => PARETO point(s): {[p[0] for p in pareto]}; best ret-edge {best[1]:+.2%} "
              f"(CI-lo {best[2]:+.2%}), worstDD {best[3]:.1%} (base {base['worst_dd']:.1%})")

    # 2x2 decomposition: is the DQ relief from PARTICIPATION (gates) or per-name SIZING (entry_frac)?
    def get(n):
        return next((r for r in results if r["name"].startswith(n)), None)
    b, part, size, both = get("baseline"), get("+PART"), get("+SIZE cd48 recl ef.125"), get("+BOTH cd0 NOrecl ef.125")
    print(f"\n  --- decomposition: what actually moves the portfolio DD? (worstDD; buys/wk) ---")
    if all((b, part, size, both)):
        print(f"    baseline          worstDD {b['worst_dd']:6.2%}  buys/wk {b['buys_per_wk']:.2f}")
        print(f"    +PARTICIPATION    worstDD {part['worst_dd']:6.2%}  buys/wk {part['buys_per_wk']:.2f}  "
              f"(loosen gates: {'RAISES' if part['worst_dd']>b['worst_dd'] else 'lowers'} DD, "
              f"{'+' if part['buys_per_wk']>b['buys_per_wk'] else ''}{part['buys_per_wk']-b['buys_per_wk']:.2f} buys)")
        print(f"    +SIZING (ef.125)  worstDD {size['worst_dd']:6.2%}  buys/wk {size['buys_per_wk']:.2f}  "
              f"(shrink bet: {'lowers' if size['worst_dd']<b['worst_dd'] else 'RAISES'} DD by "
              f"{b['worst_dd']-size['worst_dd']:+.2%})")
        print(f"    +BOTH             worstDD {both['worst_dd']:6.2%}  buys/wk {both['buys_per_wk']:.2f}")
        print(f"    => The DQ relief is driven by PER-NAME SIZING (entry_frac), NOT by extra participation."
              if size['worst_dd'] < b['worst_dd'] - 0.03 and part['worst_dd'] >= b['worst_dd'] - 0.03
              else "    => Participation and sizing both move DD; read the rows above.")
    return results, base, pareto


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--boot", type=int, default=5000)
    p.add_argument("--nfloor", type=int, default=30)
    args = p.parse_args()
    from train_rl import build_volume_panel, load_data, time_split
    returns, btc, _anchor, liq = load_data()
    train_r, val_r, _test_r = time_split(returns)
    val_start = int(val_r.index[0])
    test_start = int(_test_r.index[0])
    vol = build_volume_panel(list(returns.columns), returns.index)
    rng = np.random.default_rng(0)

    # ---- PART A: characterize on TRAIN and VAL (TEST frozen) ----
    a_summ = {}
    for name, r in (("TRAIN", train_r), ("VAL", val_r)):
        env = build_env(r, btc, liq, vol)
        print(f"\n===== {name} SPLIT =====  bars={env.n_bars}  universe(voltopk-8)={env.universe}")
        attrib = rule_replay_attribution(env)
        rows = classify_rows(env, attrib)
        a_summ[name] = part_a(rows, name, args.boot, args.nfloor, rng)

    # ---- PART B: DQ-aware counterfactual on the cold-weekly grader (train+val weeks; TEST frozen) ----
    if not validate_cf(train_r, liq, vol):
        print("  ABORT Part B: counterfactual executor failed the identity check.")
        return
    part_b(returns, val_start, test_start, liq, vol, args.boot, rng)


if __name__ == "__main__":
    main()
