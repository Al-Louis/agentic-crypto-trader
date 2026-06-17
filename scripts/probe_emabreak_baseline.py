"""SKEPTIC CHECK for P-EMABREAK (regime-confound + power lens). NOT production. NOT committed.

The P-EMABREAK headline: 22-55% of ignition EMA-breaks would resume to a new high if held -> the
EMA-break is "leaky". The lens question the original probe did NOT answer: is that resumer rate a
HAIR-TRIGGER PATHOLOGY (ignition breaks resume MORE than ordinary dips) or just BETA (in a rising
alt regime everything below-EMA bounces)?

Two baselines, computed with the EXACT same causal-peak + forward-resume machinery as the headline
probe, on the SAME voltopk-8 universe / same splits / same H:

  (A) RANDOM-BELOW-EMA baseline: every bar where cush crosses below 0 (the SAME ema_hit trigger)
      that is NOT inside an ignition hold. 'Causal pre-break peak' = trailing peak over a matched
      lookback window (the median ignition hold length per split) ending at the dip bar. Resume =
      forward max within H strictly above that trailing peak. This is "a below-EMA dip with no
      ignition pedigree" — the null the headline must beat to be a pathology.

  (B) ALL-BARS placeholder resume rate: from any random bar, does px make a new high vs a matched
      trailing peak within H? Pure beta floor.

If ignition-EMA-break resume rate ~= random-below-EMA resume rate -> NOT a hair-trigger; it is the
regime. If ignition >> random -> the original "leaky" read survives the confound.

Reuses build_env / build_events from probe_emabreak (same env replication).
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, "scripts")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from probe_emabreak import build_env, build_events, WARMUP  # noqa: E402

import train_rl  # noqa: E402


def matched_holdlen(events):
    """Median (entry->break) hold length among clean ignition events — the lookback the random
    baseline uses for its trailing causal peak, so the 'new high' bar is comparable."""
    lens = [e["break_bar"] - e["entry_bar"] for e in events if not e["stop_first"]]
    return int(np.median(lens)) if lens else 24


def ignition_hold_mask(env, events):
    """Bars (per token col) that lie inside an ignition hold [entry..break] — excluded from the
    random baseline so we compare against NON-ignition dips only."""
    n = env._px.shape[0]
    mask = {}
    for e in events:
        j = env.col_ix[e["tok"]]
        s = mask.setdefault(j, np.zeros(n, dtype=bool))
        s[e["entry_bar"]:e["break_bar"] + 1] = True
    return mask


def random_belowema_events(env, lookback, hold_mask, H_list=(24, 48)):
    """Every bar b where cush crosses below 0 (the SAME ema_hit trigger), NOT inside an ignition
    hold, with px>0. Causal pre-break peak = trailing max of px over [b-lookback .. b]. Resume =
    forward max in (b, b+H] strictly above that trailing peak. Mirrors the headline outcome exactly
    but with NO ignition pedigree."""
    px, cush, n = env._px, env._cush, env._px.shape[0]
    start, end = env.start, env.end
    evs = []
    for t in env.universe:
        j = env.col_ix[t]
        hm = hold_mask.get(j)
        for b in range(max(start, WARMUP) + lookback, min(end, n)):
            if px[b, j] <= 0:
                continue
            # cross BELOW ema this bar (cush[b]<0 and cush[b-1]>=0) == a fresh below-ema dip
            if not (cush[b, j] < 0.0 and cush[b - 1, j] >= 0.0):
                continue
            if hm is not None and hm[b]:
                continue                       # inside an ignition hold -> not a 'random' dip
            peak = float(px[b - lookback:b + 1, j].max())
            ev = {"tok": t, "bar": b}
            for H in H_list:
                last = b + H
                if last >= n:
                    ev[f"resume_{H}"] = None
                    continue
                fwd_max = float(px[b + 1:last + 1, j].max())
                ev[f"resume_{H}"] = bool(fwd_max > peak)
            evs.append(ev)
    return evs


def all_bar_resume(env, lookback, H_list=(24, 48)):
    """Pure-beta floor: from EVERY bar, does px make a new high vs a matched trailing peak within H?
    No EMA condition at all — the unconditional 'things go up' rate in this regime."""
    px, n = env._px, env._px.shape[0]
    start, end = env.start, env.end
    out = {H: [] for H in H_list}
    for t in env.universe:
        j = env.col_ix[t]
        for b in range(max(start, WARMUP) + lookback, min(end, n)):
            if px[b, j] <= 0:
                continue
            peak = float(px[b - lookback:b + 1, j].max())
            for H in H_list:
                last = b + H
                if last >= n:
                    continue
                out[H].append(1.0 if float(px[b + 1:last + 1, j].max()) > peak else 0.0)
    return {H: (float(np.mean(v)), len(v)) for H, v in out.items()}


def rate(evs, H):
    v = [1.0 if e[f"resume_{H}"] else 0.0 for e in evs if e.get(f"resume_{H}") is not None]
    return (float(np.mean(v)), len(v)) if v else (float("nan"), 0)


def main():
    returns, btc, anchor, liq = train_rl.load_data()
    train_r, val_r, test_r = train_rl.time_split(returns)
    vol = train_rl.build_volume_panel(list(returns.columns), returns.index)

    print("=" * 96)
    print("SKEPTIC BASELINE: ignition EMA-break resume rate  vs  random below-EMA dip resume rate")
    print("  (same universe / split / causal-peak / forward-H machinery). If ~equal -> the resumer")
    print("  rate is BETA (regime), not an ignition hair-trigger pathology.")
    print("=" * 96)

    for nm, r in (("TRAIN", train_r), ("VAL", val_r)):
        env = build_env(r, btc, liq, vol)
        ev_ig = build_events(env)
        clean_ig = [e for e in ev_ig if not e["stop_first"]]
        lb = matched_holdlen(ev_ig)
        hm = ignition_hold_mask(env, ev_ig)
        ev_rand = random_belowema_events(env, lb, hm)
        allbar = all_bar_resume(env, lb)

        print(f"\n########## {nm} ##########   matched lookback (median ignition hold) = {lb} bars")
        for H in (24, 48):
            ig_rate, ig_n = rate(clean_ig, H)
            rd_rate, rd_n = rate(ev_rand, H)
            ab_rate, ab_n = allbar[H]
            lift_vs_rand = ig_rate - rd_rate
            lift_vs_all = ig_rate - ab_rate
            print(f"  H={H}:")
            print(f"    ignition EMA-break  resume rate = {ig_rate:5.1%}  (n={ig_n})")
            print(f"    random below-EMA dip resume rate = {rd_rate:5.1%}  (n={rd_n})   "
                  f"<<< the NULL the pathology must beat")
            print(f"    all-bar (pure beta) resume rate  = {ab_rate:5.1%}  (n={ab_n})")
            print(f"    LIFT ignition - random-below-EMA = {lift_vs_rand:+5.1%}   "
                  f"| ignition - all-bar = {lift_vs_all:+5.1%}")
            verdict = ("PATHOLOGY-CONSISTENT (ignition resumes MORE than random dips)"
                       if lift_vs_rand > 0.05 else
                       "BETA/REGIME (ignition ~= random dip; resume rate is not ignition-specific)")
            print(f"    -> {verdict}")


if __name__ == "__main__":
    main()
