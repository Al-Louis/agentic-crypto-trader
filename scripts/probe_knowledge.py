"""Probe the two ranked KNOWLEDGE additions (quant consult 2026-06-12) before any build:

  [1] CROSS-SECTIONAL RANK — does this ignition's cush RANK among the same bar's candidates add
    OOS IC over absolute [cush, surge, btcT]? (exp5 proved the mechanism in-env; this gates the
    explicit obs form.)
  [2] PER-TOKEN CYCLE MEMORY — do bars-since-this-token's-prior-ignition and ret-since-prior-
    ignition (did the last signal already pay?) add OOS IC? (Targets re-buying-the-bleed and
    missed-re-ignition — policy-independent forms so the probe needs no agent.)

Method: in-universe ignitions on TRAIN (voltop8), OLS on the early 60%, OOS Spearman IC of the
prediction vs fwd-24h return on the late 40%. GATE: incremental OOS IC > +0.02 over the baseline
(the bar lever-2's harvest probe cleared at +0.063).

  python scripts/probe_knowledge.py
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, "scripts")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

WARMUP = 168
H = 24


def spearman(a, b):
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    return float(np.corrcoef(ra, rb)[0, 1])


def oos_ic(X, y, frac=0.6):
    n = len(y)
    k = int(n * frac)
    Xt, yt = X[:k], y[:k]
    coef, *_ = np.linalg.lstsq(np.c_[np.ones(len(Xt)), Xt], yt, rcond=None)
    pred = np.c_[np.ones(n - k), X[k:]] @ coef
    return spearman(pred, y[k:])


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--split", default="train", choices=["train", "val", "test"])
    args = p.parse_args()
    from train_rl import build_volume_panel, load_data, time_split
    from trader.train.event_env import EventRungEnv

    returns, btc, anchor, liq = load_data()
    splits = dict(zip(("train", "val", "test"), time_split(returns)))
    train_r = splits[args.split]
    vol = build_volume_panel(list(returns.columns), returns.index)
    env = EventRungEnv(train_r, btc, liq, volume=vol, episode_bars=len(train_r) - WARMUP - 1,
                       k=8, warmup=WARMUP, universe_mode="voltopk", seed=0)
    env.reset(start=WARMUP)
    uni_ix = [env.col_ix[t] for t in env.universe]
    px, ig, cush, surge = env._px, env._ignite, env._cush, env._surge
    btc_s = env.btc.to_numpy()
    btc_ema = env.btc_ema.to_numpy()
    n = len(train_r)

    # collect events in TIME order (the temporal holdout needs chronology)
    events = []          # (bar, j)
    for b in range(WARMUP, n - H):
        for j in uni_ix:
            if ig[b, j] and px[b, j] > 0:
                events.append((b, j))
    last_ig_bar = {}     # per-token prior-ignition tracking (causal: built in time order)
    rows = []
    for b, j in events:
        fwd = px[b + H, j] / px[b, j] - 1.0
        btc_t = btc_s[b] / btc_ema[b] - 1.0 if btc_ema[b] else 0.0
        # [1] cross-sectional context on this bar
        cands = [jj for jj in uni_ix if ig[b, jj]]
        cs = [cush[b, jj] for jj in cands]
        rank = (sorted(cs).index(cush[b, j]) / max(len(cs) - 1, 1)) if len(cs) > 1 else 0.5
        demean = cush[b, j] - float(np.mean(cs))
        # [2] per-token cycle memory (policy-independent forms)
        prior = last_ig_bar.get(j)
        bars_since = min((b - prior), 672) / 672.0 if prior is not None else 1.0
        ret_since = (px[b, j] / px[prior, j] - 1.0) if prior is not None else 0.0
        last_ig_bar[j] = b
        rows.append((cush[b, j], surge[b, j], btc_t,            # baseline
                     rank, demean, float(len(cs)),               # [1]
                     bars_since, np.clip(ret_since, -1, 5),      # [2]
                     fwd))
    A = np.array(rows)
    X, y = A[:, :-1], A[:, -1]
    base = oos_ic(X[:, :3], y)
    xsec = oos_ic(X[:, :6], y)
    cyc = oos_ic(X[:, [0, 1, 2, 6, 7]], y)
    both = oos_ic(X[:, :8], y)
    print(f"events: {len(y)}   (train, voltop8, fwd-{H}h, 60/40 temporal holdout)")
    print(f"baseline [cush,surge,btcT]   OOS IC = {base:+.3f}")
    print(f"+ [1] cross-sectional rank     OOS IC = {xsec:+.3f}   incremental {xsec - base:+.3f}"
          f"   gate(+0.02): {'PASS' if xsec - base > 0.02 else 'FAIL'}")
    print(f"+ [2] per-token cycle memory   OOS IC = {cyc:+.3f}   incremental {cyc - base:+.3f}"
          f"   gate(+0.02): {'PASS' if cyc - base > 0.02 else 'FAIL'}")
    print(f"+ both                       OOS IC = {both:+.3f}   incremental {both - base:+.3f}")
    # the quant's specific split test: ignitions where the prior signal ALREADY PAID vs not
    paid = A[:, 7] > 0.10
    if paid.sum() > 20 and (~paid).sum() > 20:
        print(f"\nprior-signal-already-paid (ret_since>10%): n={int(paid.sum())}  "
              f"fwd24 {y[paid].mean():+.2%}  vs not-paid n={int((~paid).sum())}  fwd24 {y[~paid].mean():+.2%}")


if __name__ == "__main__":
    main()
