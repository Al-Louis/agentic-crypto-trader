"""Probe the EXTENSION / ANTI-CHASE entry brake (user, 2026-06-19): the current ignition fires on
`surge>=mult & rising>0 & cush>0 & ema_up` — every term REQUIRES price to already be up, and the
re-entry gate `reclaimed = px>prior_origin` demands a HIGHER price than last cycle. So the agent can
re-ignite on a SECOND pump leg and buy the top (s1 W21: ZEC #2 bought 04-10 01:00 @371 ~5h after a
310->377 rip, then -1.6%; the FF->ZEC rotation funded it and BOTH legs lost). The contrarian-edge
note ([[ignition-edge-is-contrarian-not-strength]]) says strength underperforms — so blocking the
MOST-EXTENDED ignitions should be NET-POSITIVE. This is the SUBTRACTIVE complement to
probe_pullback_entry.py (which tests ADDING below-EMA entries).

Among the bars the agent ACTUALLY buys (the CURRENT trigger), bucket by an extension metric and report
forward 24h/48h return NET of ~1% round-trip cost, PER SPLIT (train/val/test). If the top-extension
bucket is net-negative while cheaper buckets are positive AND it persists OOS, an extension brake is a
real +EV lever (not a bull-window artifact). Reuses env precomputes (`_px`/`_cush`/`_surge`).

  python scripts/probe_extension_entry.py [--surge 2.5] [--cost 0.01] [--vol-mult 2.0]

Extension metrics (all measured AT the trigger bar):
  cush  = px/ema - 1            (how far price sits ABOVE its 72h EMA — the env's own cushion)
  ru12  = px[b]/px[b-12] - 1    (run-up over the prior 12h — "already pumped")
  ru24  = px[b]/px[b-24] - 1    (run-up over the prior 24h; the trigger already forces ru24>0)
  ru48  = px[b]/px[b-48] - 1    (run-up over the prior 48h — full second-leg chase, the ZEC case)
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, "scripts")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

WARMUP = 168


def trigger_mask(env, surge_min):
    """The CURRENT ignition trigger as a [bar x token] bool, derived from the env's own arrays."""
    px, cush, surge = env._px, env._cush, env._surge
    ema = px / (1.0 + cush)
    ema_up = np.zeros_like(px, dtype=bool); ema_up[4:] = ema[4:] >= ema[:-4]
    rising = np.zeros_like(px, dtype=bool); rising[24:] = px[24:] > px[:-24]
    return surge >= surge_min, (surge >= surge_min) & rising & (cush > 0) & ema_up


def ext_metrics(env):
    """Extension metrics [bar x token] at the trigger bar."""
    px, cush = env._px, env._cush
    def ru(n):
        out = np.full_like(px, np.nan)
        out[n:] = px[n:] / np.where(px[:-n] > 0, px[:-n], np.nan) - 1.0
        return out
    return {"cush": cush, "ru12": ru(12), "ru24": ru(24), "ru48": ru(48)}


def fwd(px, b, j, h):
    return px[b + h, j] / px[b, j] - 1.0 if px[b, j] > 0 else np.nan


def run_split(name, r, btc, liq, vol, surge_min, cost, vol_mult):
    from trader.train.event_env import EventRungEnv
    env = EventRungEnv(r, btc, liq, volume=vol, episode_bars=len(r) - WARMUP - 1,
                       k=8, warmup=WARMUP, universe_mode="voltopk", seed=0, vol_mult=vol_mult)
    px = env._px
    n = env.n_bars
    _, trig = trigger_mask(env, surge_min)
    mets = ext_metrics(env)
    # gather every trigger event with its metric values + forward returns
    ev = []
    for b in range(WARMUP, n - 48):
        for j in np.where(trig[b])[0]:
            if px[b, j] <= 0:
                continue
            row = {"f24": fwd(px, b, j, 24), "f48": fwd(px, b, j, 48)}
            for mn, mv in mets.items():
                row[mn] = mv[b, j]
            ev.append(row)
    print(f"\n=== {name}  (n triggers = {len(ev)}) ===")
    if not ev:
        return
    f24 = np.clip(np.array([e["f24"] for e in ev]), -5, 5)
    f48 = np.clip(np.array([e["f48"] for e in ev]), -5, 5)
    print(f"  ALL              f24={f24.mean():+6.2%} (NET {f24.mean()-cost:+6.2%}) "
          f"f48={f48.mean():+6.2%} win24={np.mean(f24>0):4.0%}")
    for mn in ("cush", "ru12", "ru24", "ru48"):
        vals = np.array([e[mn] for e in ev])
        ok = ~np.isnan(vals)
        v, a24, a48 = vals[ok], f24[ok], f48[ok]
        if len(v) < 8:
            continue
        qs = np.quantile(v, [0.25, 0.5, 0.75])
        labels = ["Q1 low ", "Q2     ", "Q3     ", "Q4 high"]
        idx = np.digitize(v, qs)
        print(f"  -- by {mn:5} (Q4=most extended) --")
        for q in range(4):
            m = idx == q
            if not m.any():
                continue
            flag = "  <NET-NEG" if a24[m].mean() - cost < 0 else ""
            print(f"     {labels[q]} n={m.sum():4d}  range[{v[m].min():+.2f},{v[m].max():+.2f}]  "
                  f"f24={a24[m].mean():+6.2%} NET={a24[m].mean()-cost:+6.2%}  f48={a48[m].mean():+6.2%}  "
                  f"win24={np.mean(a24[m]>0):4.0%}{flag}")
        # the actual lever: block top quartile
        hi = idx == 3
        if hi.any():
            kept = ~hi
            print(f"     >> BLOCK Q4: removes {hi.sum()} entries (mean f24 {a24[hi].mean():+.2%}); "
                  f"KEPT mean f24 {a24[kept].mean():+.2%} vs ALL {a24.mean():+.2%}")


def run(surge_min, cost, vol_mult):
    from train_rl import build_volume_panel, load_data, time_split
    returns, btc, anchor, liq = load_data()
    vol = build_volume_panel(list(returns.columns), returns.index)
    tr, va, te = time_split(returns)
    print(f"surge>={surge_min} vol_mult={vol_mult} | round-trip cost {cost:.0%} | "
          f"forward returns of CURRENT-trigger entries, bucketed by extension")
    for name, rr in (("TRAIN", tr), ("VAL", va), ("TEST", te)):
        run_split(name, rr, btc, liq, vol, surge_min, cost, vol_mult)
    print("\nREAD: a real extension brake needs Q4 (most-extended) NET24<0 while Q1/Q2 NET24>0, AND "
          "the BLOCK-Q4 'KEPT mean' to beat ALL, PERSISTING across TRAIN+VAL+TEST. Else chasing is fine.")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--surge", type=float, default=2.5)
    p.add_argument("--cost", type=float, default=0.01)
    p.add_argument("--vol-mult", type=float, default=2.0)
    a = p.parse_args()
    run(a.surge, a.cost, a.vol_mult)


if __name__ == "__main__":
    main()
