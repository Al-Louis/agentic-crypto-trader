"""Horizon-credit PROBE (torch-free, no training) — does the long-episode curriculum have a lever?

OVERLAY-1 learned a DEFENSIVE basin: it trims held basket names on rung-0 "weakness" prompts
(price below trailing peak by stop_k, or below its EMA), which wins in bear but gives back the
bull (per-week gap vs B&H −27..−31% in bull weeks). The credit-assignment hypothesis: in a 1-week
(168-bar) episode the reward for trimming a dip arrives in-episode and positive, while the cost —
missing the multi-week run — is truncated at the episode boundary and never credited.

This probe measures, on the TRAIN split, the FORWARD RETURN of a held name from each weakness bar
over horizons H ∈ {168, 336, 672}. If holding-through pays MATERIALLY more over longer horizons
(mean fwd return more positive at 672 than 168 — a sign flip, or magnitude ≥2×, esp. in bull
windows), then training on long episodes first gives the agent the signal to stop trimming the
bull -> the horizon curriculum has a real lever. If it's flat across horizons, no lever -> DO NOT
launch the curriculum sweep; re-route to the reward-shape path. See [[Experiment Log]] §OVERLAY-1.
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, "scripts")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from trader import config  # noqa: E402

WARMUP = 168
STOP_K = 0.25          # rung-0 trailing-stop (the overlay default)
EMA_SPAN = 72          # rung-0 trend EMA
K = 8                  # vol-top-k basket
HORIZONS = [168, 336, 672]
STRIDE = 72            # sample an episode start every 3 days (sparse -> distinct baskets)


def main() -> None:
    config.load_dotenv()
    from train_rl import load_data, time_split

    returns, _btc, _anchor, _liq = load_data()
    train, _val, _test = time_split(returns)
    px = (1.0 + train.fillna(0.0)).cumprod()
    ema = px.ewm(span=EMA_SPAN, adjust=False).mean()
    std = train.rolling(WARMUP, min_periods=8).std()
    pxn, eman, stdn = px.to_numpy(), ema.to_numpy(), std.to_numpy()
    n = len(train)
    maxH = max(HORIZONS)

    # data-availability guard (risk #3): distinct valid longest-horizon episode starts on train.
    valid_672 = len(range(WARMUP, n - maxH))
    print(f"train: {n} bars, {train.shape[1]} tokens | distinct valid {maxH}-bar starts = {valid_672} "
          f"(sampled every {STRIDE} -> {len(range(WARMUP, n - maxH, STRIDE))} starts)")

    # accumulators: fwd returns from weakness bars, by horizon and by window-regime
    allf = {h: [] for h in HORIZONS}
    byreg = {h: {"bull": [], "flat": [], "bear": []} for h in HORIZONS}
    nstarts = 0
    for s in range(WARMUP, n - maxH, STRIDE):
        row = np.nan_to_num(stdn[s - 1], nan=-1.0)
        uni = np.argsort(row)[::-1][:K]                       # causal vol-top-8 (the held basket)
        ew = float(np.mean([pxn[s + maxH, j] / pxn[s, j] - 1.0 for j in uni if pxn[s, j] > 0]))
        reg = "bull" if ew > 0.10 else "bear" if ew < -0.10 else "flat"
        nstarts += 1
        bmax = min(s + maxH, n - maxH)                        # ensure every b has b+maxH in-range
        for j in uni:
            if pxn[s, j] <= 0:
                continue
            peak = pxn[s, j]
            for b in range(s, bmax):
                p = pxn[b, j]
                peak = max(peak, p)
                if p <= 0:
                    continue
                if p < peak * (1.0 - STOP_K) or p < eman[b, j]:   # rung-0 weakness (a trim prompt)
                    for h in HORIZONS:
                        fwd = pxn[b + h, j] / p - 1.0
                        allf[h].append(fwd)
                        byreg[h][reg].append(fwd)

    def stats(xs):
        a = np.array(xs)
        return (np.mean(a), np.median(a), float((a > 0).mean()), len(a)) if a.size else (0, 0, 0, 0)

    print(f"\nstarts sampled: {nstarts}  |  weakness-bar samples @168: {len(allf[168])}")
    print(f"\n{'horizon':>8} {'mean_fwd':>9} {'median':>8} {'hold-pays%':>10} {'n':>8}   "
          f"(fwd return holding a name from a weakness bar)")
    for h in HORIZONS:
        m, md, pos, nn = stats(allf[h])
        print(f"{h:>8} {m:>+8.2%} {md:>+7.2%} {pos:>9.0%} {nn:>8}")

    print(f"\n  by window regime (mean fwd return):")
    print(f"  {'regime':>6} " + " ".join(f"{('H'+str(h)):>10}" for h in HORIZONS) + "   trim->hold sign")
    for reg in ("bull", "flat", "bear"):
        ms = [stats(byreg[h][reg])[0] for h in HORIZONS]
        ns = stats(byreg[HORIZONS[0]][reg])[3]
        print(f"  {reg:>6} " + " ".join(f"{m:>+9.2%}" for m in ms) + f"   (n={ns})")

    # verdict: lever exists if mean fwd at 672 is materially more positive than at 168 (sign flip or ≥2×)
    m168, m672 = stats(allf[168])[0], stats(allf[672])[0]
    b168, b672 = stats(byreg[168]["bull"])[0], stats(byreg[672]["bull"])[0]
    grew = (m672 > 0 and m168 <= 0) or (abs(m672) >= 2 * abs(m168) and m672 > m168)
    bull_grew = (b672 > 0 and b168 <= 0) or (abs(b672) >= 2 * abs(b168) and b672 > b168)
    print(f"\nVERDICT: overall fwd {m168:+.2%}->{m672:+.2%} (168->672); bull {b168:+.2%}->{b672:+.2%}")
    if grew or bull_grew:
        print("  [YES] LEVER EXISTS — holding-through pays materially more at long horizon "
              "(esp. bull) -> the horizon curriculum can teach the agent to stop trimming the bull. LAUNCH.")
    else:
        print("  [NO] FLAT across horizons — horizon is NOT the operative lever. DO NOT launch the "
              "curriculum sweep; re-route to the reward-shape (regime-conditioned) path.")
    if valid_672 < 100:
        print(f"  [WARN] only {valid_672} valid {maxH}-bar starts — drop the longest phase to 504 (3wk).")


if __name__ == "__main__":
    main()
