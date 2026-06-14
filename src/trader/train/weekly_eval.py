"""Cold weekly-session evaluation — the DEPLOYMENT-honest grader (the 2026-06-14 FORK;
[[AI Training]] §"the train/deploy STRUCTURE mismatch", [[Experiment Log]] §2026-06-14).

The competition is ONE Mon-00:00-UTC week traded from a fresh $10k. Grading a strategy on one
long *continuous* episode (`train_event.evaluate_and_gate`) FLATTERS it: positions ride across
weeks on warm context, which is exactly what made s0 look like a star (ZEC +$2,747) and hid that
it skips the same setups cold. This grades the way the agent actually runs — many INDEPENDENT
cold weeks, each from $10k, no cross-week holds, the universe re-picked causally before each week.

Torch-free: the rung-0 RULE and Buy&Hold baselines need no policy, so the deployment BAR is
measurable on the laptop. A trained policy is graded by passing its `predict_fn` (desktop, torch).

Mirrors `scripts/simulate_weekly.py`'s week-slicing exactly (the dashboard producer), but its job
is the GATE numbers (per-week return / within-week maxDD / activity floor / regime), not the chart.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

WEEK_SECS = 7 * 24 * 3600
MONDAY_PHASE = 345600          # t % WEEK_SECS at 00:00 UTC Monday (the unix epoch was a Thursday)
HOUR = 3600
WARMUP = 168
WEEK_BARS = 168
START_CAPITAL = 10_000.0
DD_GATE = 0.30                 # the competition's hard max-drawdown disqualifier
MIN_WEEK_BARS = 150            # skip gappy/short weeks (matches simulate_weekly)


def cold_week_windows(returns: pd.DataFrame, warmup: int = WARMUP, week_bars: int = WEEK_BARS,
                      min_week_bars: int = MIN_WEEK_BARS):
    """Yield `(ws, win)` for every 00:00-UTC-Monday week that has a full `warmup` prepad behind it
    and >= `min_week_bars` of in-week data. `win` = the warmup prepad + that week's bars (contiguous
    real time, so the week is causally tradeable from its first bar). Faithful to simulate_weekly."""
    idx = [int(t) for t in returns.index]
    pos = {t: i for i, t in enumerate(idx)}
    for ws in idx:
        if ws % WEEK_SECS != MONDAY_PHASE:
            continue
        i0 = pos[ws]
        if i0 < warmup:
            continue                                           # need a full warmup before the week
        we = ws + WEEK_SECS
        win = returns.iloc[i0 - warmup: i0 - warmup + warmup + week_bars]
        in_week = [t for t in idx[i0: i0 + week_bars] if t < we]
        if len(win) < warmup + min_week_bars or len(in_week) < min_week_bars:
            continue
        yield ws, win


def split_label(ws: int, val_start: int, test_start: int) -> str:
    """Which split a week's START falls in. 'train' = in-sample (trained on); 'val' = the tuning
    window (held-out but tuned-against); 'test' = the never-touched OOS window. The honest gate is
    judged on val+test (unseen); 'train' weeks are reported for context only."""
    return "test" if ws >= test_start else "val" if ws >= val_start else "train"


def causal_voltop_universe(win: pd.DataFrame, k: int = 8, warmup: int = WARMUP) -> list[str]:
    """The k highest trailing-vol tokens at the week open — the rung-0 universe, picked CAUSALLY
    (rolling-warmup std at bar `warmup-1`, NOT full-window std, which would peek at late pumpers;
    matches `train_event.rung0_baseline` / `EventRungEnv._pick_universe`)."""
    std = win.rolling(warmup, min_periods=8).std().to_numpy()
    order = np.argsort(np.nan_to_num(std[warmup - 1], nan=-1.0))[::-1][:k]
    return [win.columns[j] for j in order]


def _trade_days(records: list, ws: int) -> set[int]:
    """Distinct day-of-week indices (0..6 from the Monday open) on which the rule actually traded —
    a record with a non-empty `trades_usd`. Markerless daily snapshots (empty tu) don't count."""
    days = set()
    for rec in records:
        if rec.get("trades_usd"):
            days.add(int((int(rec["time"]) - ws) // (24 * HOUR)))
    return {d for d in days if 0 <= d <= 6}


def risk_parity_caps(win, universe, vol_target, cap_floor, max_entry_frac=0.34, warmup=WARMUP):
    """Per-token weight caps ∝ vol_target/trailing_vol, clipped [cap_floor, max_entry_frac] — the
    env's risk-parity formula, computed directly (torch-free). Trailing vol = rolling-warmup return
    std at the week open. Buy&Hold normalizes by Σcap, so only the relative weights matter."""
    std_row = win.rolling(warmup, min_periods=8).std().to_numpy()[warmup - 1]
    cix = {t: j for j, t in enumerate(win.columns)}
    caps = {}
    for t in universe:
        v = std_row[cix[t]]
        cap = max_entry_frac if not np.isfinite(v) or v <= 0 else vol_target / v
        caps[t] = float(np.clip(cap, cap_floor, max_entry_frac))
    return caps


def buyhold_return(win, liq, universe, caps, warmup=WARMUP, capital=START_CAPITAL):
    """Buy&Hold of the universe, risk-parity weighted, fully invested at the week open, one entry AMM
    cost via the same broker — the honest passive bar and the long-default-overlay floor."""
    from trader.sim.broker import DEFAULT_GAS_USD, DEFAULT_LP_FEE_BPS, amm_cost_usd
    px = (1.0 + win.fillna(0.0)).cumprod().to_numpy()
    cix = {t: i for i, t in enumerate(win.columns)}
    wsum = sum(caps[t] for t in universe) or 1.0
    eq_end = 0.0
    for t in universe:
        alloc = (caps[t] / wsum) * capital
        j = cix[t]
        p0 = px[warmup, j]
        if p0 <= 0:
            eq_end += alloc
            continue
        invested = alloc - amm_cost_usd(alloc, liq.get(t, 0.0), DEFAULT_LP_FEE_BPS, DEFAULT_GAS_USD)
        eq_end += invested * px[-1, j] / p0
    return eq_end / capital - 1.0


def week_regime(win, universe, warmup=WARMUP):
    """bull/bear/flat over the basket for this week (universe-EW move open->close) — descriptive only;
    the gate judges the random-week distribution, not regime buckets ([[AI Training]] §the-fork)."""
    px = (1.0 + win.fillna(0.0)).cumprod()
    ew = float(np.mean([px[t].iloc[-1] / px[t].iloc[warmup] - 1.0 for t in universe]))
    return ("bull" if ew > 0.10 else "bear" if ew < -0.10 else "flat"), ew


def grade_week_baselines(ws, win, liq, vol, k=8, vol_target=0.005, cap_floor=0.02, warmup=WARMUP):
    """Grade the rung-0 RULE + risk-parity Buy&Hold over one cold week. Torch-free (no policy), so the
    deployment BAR is laptop-measurable. Returns a WeekResult (regime/split filled by the caller)."""
    from trader.strategy.rung0 import build_rung0, run_rung0
    uni = causal_voltop_universe(win, k=k, warmup=warmup)
    sig = build_rung0(win, tokens=uni, volume=vol)
    eq, records, _fees = run_rung0(win, sig, liq, warmup=warmup)
    eq = eq.iloc[warmup:]                                       # the COLD week only (drop the prepad)
    ret = float(eq.iloc[-1] / eq.iloc[0] - 1.0)
    dd = abs(float((eq / eq.cummax() - 1.0).min()))
    tdays = _trade_days(records, ws)
    caps = risk_parity_caps(win, uni, vol_target, cap_floor, warmup=warmup)
    bh = buyhold_return(win, liq, uni, caps, warmup=warmup)
    label, ew = week_regime(win, uni, warmup=warmup)
    return WeekResult(ws=ws, split="", regime=label, rung0_ret=ret, rung0_dd=dd,
                      rung0_trade_days=len(tdays), rung0_active_ok=(len(tdays) >= 7),
                      buyhold_ret=bh, universe_ew_ret=ew)


@dataclass
class WeekResult:
    ws: int
    split: str
    regime: str
    rung0_ret: float
    rung0_dd: float
    rung0_trade_days: int        # of 7; competition needs >=1 trade EVERY day
    rung0_active_ok: bool        # all 7 days traded
    buyhold_ret: float
    universe_ew_ret: float       # the week's market move over the agent's basket


@dataclass
class Aggregate:
    n_weeks: int
    splits: dict
    rung0_mean: float
    rung0_median: float
    rung0_winrate: float         # fraction of weeks with positive return
    rung0_best: float
    rung0_worst: float
    rung0_worst_dd: float        # worst single-week drawdown (the cold-session DQ axis)
    rung0_dq_weeks: int          # weeks breaching the 30% gate
    buyhold_mean: float
    buyhold_winrate: float
    bull_gap_mean: float         # mean (buyhold - rung0) over BULL weeks — the substrate-decision number
    n_bull: int
    activity_fail_weeks: int     # weeks the rule misses >=1 trade on some day (Rule-1 DQ risk)
    weeks: list = field(default_factory=list)


def bootstrap_mean_ci(values, n_boot: int = 2000, alpha: float = 0.05, seed: int = 0):
    """Percentile bootstrap CI for the MEAN of a small weekly-return sample. The deployment gate
    judges a config on its weekly-return DISTRIBUTION, not a point estimate — a single +92% week
    (the kind that flattered s0) must not crown a config, so the gate reads the CI LOWER BOUND, not
    the raw mean. Deterministic (fixed seed; Date/random are unavailable to scripts anyway)."""
    v = np.asarray(values, dtype=float)
    if v.size == 0:
        return 0.0, 0.0, 0.0
    rng = np.random.default_rng(seed)
    means = v[rng.integers(0, v.size, size=(n_boot, v.size))].mean(axis=1)
    lo, hi = np.percentile(means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi), float(v.mean())


def weekly_gate(policy_rets, policy_dds, bh_rets, rung0_rets, activity_ok,
                dd_gate: float = DD_GATE, seed: int = 0, require_activity: bool = False) -> dict:
    """The DEPLOYMENT-honest gate in the random-week frame (the 2026-06-14 fork). A config passes iff,
    over the held-out cold weeks:
      1. SURVIVES — worst single-week maxDD < `dd_gate` (each cold week is a fresh-$10k session, so DD
         is within-week, exactly as the competition scores).
      2. BEATS HOLDING — the PAIRED weekly edge (policy_week - bh_week) has a CI lower bound > 0. Pairing
         is essential: policy and B&H see the SAME weeks, so the difference cancels the common (huge)
         market variance and isolates the policy's edge — an unpaired policy-CI-vs-B&H-mean test is
         swamped by one +85% week. B&H is the long-default-basket floor, the bar the skeleton failed.
      3. BEATS THE RULE — the paired edge (policy_week - rung0_week) has a CI lower bound > 0.
      4. ACTIVE (`require_activity`, default OFF / informational) — >=1-trade/day every week. This is a
         UNIVERSAL daily requirement (passive B&H trips it too), satisfied at deploy by a forced minimal
         daily rebalance — a guardrail, not a strategy discriminator — so it's reported, not binding,
         unless explicitly required. `activity_fail_weeks` is always reported.
    Returns the verdict packet (the binding constraint named on failure)."""
    pol = np.asarray(policy_rets, dtype=float)
    bh = np.asarray(bh_rets, dtype=float)
    r0 = np.asarray(rung0_rets, dtype=float)
    _, _, mean = bootstrap_mean_ci(pol, seed=seed)
    bh_lo, bh_hi, bh_edge = bootstrap_mean_ci(pol - bh, seed=seed) if pol.size else (0.0, 0.0, 0.0)
    r0_lo, r0_hi, r0_edge = bootstrap_mean_ci(pol - r0, seed=seed + 1) if pol.size else (0.0, 0.0, 0.0)
    worst_dd = float(max(policy_dds)) if len(policy_dds) else 0.0
    activity_pass = all(activity_ok) if len(activity_ok) else False
    checks = {
        "survives_dq": worst_dd < dd_gate,
        "beats_buyhold": bh_lo > 0.0,                          # paired weekly edge robustly positive
        "beats_rung0": r0_lo > 0.0,
    }
    if require_activity:
        checks["activity_floor"] = activity_pass
    passed = all(checks.values())
    binding = None if passed else next(k for k in checks if not checks[k])
    return {"pass": passed, "binding": binding,
            "policy_mean": mean, "worst_week_dd": worst_dd,
            "buyhold_mean": float(bh.mean()) if bh.size else 0.0,
            "rung0_mean": float(r0.mean()) if r0.size else 0.0,
            "edge_vs_buyhold": bh_edge, "edge_buyhold_ci": [bh_lo, bh_hi],
            "edge_vs_rung0": r0_edge, "edge_rung0_ci": [r0_lo, r0_hi],
            "activity_fail_weeks": int(sum(not a for a in activity_ok)), "checks": checks}


def aggregate(weeks: list[WeekResult], oos_only: bool = True) -> Aggregate:
    """Distil per-week results into the deployment-honest read. `oos_only` restricts the headline
    numbers to val+test weeks (the unseen regimes the gate cares about)."""
    sel = [w for w in weeks if (w.split in ("val", "test")) or not oos_only]
    if not sel:
        sel = list(weeks)
    r = np.array([w.rung0_ret for w in sel])
    bh = np.array([w.buyhold_ret for w in sel])
    bull = [w for w in sel if w.regime == "bull"]
    gaps = np.array([w.buyhold_ret - w.rung0_ret for w in bull]) if bull else np.array([0.0])
    splits = {}
    for w in weeks:
        splits[w.split] = splits.get(w.split, 0) + 1
    return Aggregate(
        n_weeks=len(sel), splits=splits,
        rung0_mean=float(r.mean()), rung0_median=float(np.median(r)),
        rung0_winrate=float((r > 0).mean()), rung0_best=float(r.max()), rung0_worst=float(r.min()),
        rung0_worst_dd=float(max(w.rung0_dd for w in sel)),
        rung0_dq_weeks=int(sum(w.rung0_dd > DD_GATE for w in sel)),
        buyhold_mean=float(bh.mean()), buyhold_winrate=float((bh > 0).mean()),
        bull_gap_mean=float(gaps.mean()), n_bull=len(bull),
        activity_fail_weeks=int(sum(not w.rung0_active_ok for w in sel)),
        weeks=sel,
    )
