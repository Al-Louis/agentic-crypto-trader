"""Probe the TOKEN-PERSONALITY theory (user, 2026-06-12): low-cap/MM-controlled tokens have
stable per-token indicator affinities measurable over a trailing window.

  A) PERSISTENCE — for each token x indicator FAMILY (trend / reversion / breakout / volume),
     does the trailing-30d efficacy (rolling IC of the family signal vs fwd-24h return) predict
     the NEXT 7 days' efficacy? Pooled Spearman across weekly checkpoints. This is the same
     persistence test that validated the vol tilt (+0.66). No persistence = the "personality"
     is in-window noise (multiple-comparisons trap).
  B) PAYOFF — at the in-universe ignition events, do efficacy-weighted family signals add
     OOS IC > +0.02 over the [cush, surge, btcT] baseline?

Train split only. Causal throughout (trailing windows, shifted extrema).

  python scripts/probe_personality.py
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, "scripts")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from probe_knowledge import oos_ic, spearman  # noqa: E402

WARMUP = 168
H = 24
EFF_WIN = 720        # trailing 30d efficacy window
CKPT = 168           # weekly checkpoints
FAMS = ("trend", "reversion", "breakout", "volume")


def family_signals(px_df, vol_df):
    """Per-family CAUSAL signal panels (same shape as px). Continuous, sign = direction bet."""
    ema_f = px_df.ewm(span=12, adjust=False).mean()
    ema_s = px_df.ewm(span=48, adjust=False).mean()
    trend = (ema_f / ema_s - 1.0)
    sma24 = px_df.rolling(24, min_periods=4).mean()
    reversion = -(px_df / sma24 - 1.0)                  # stretched-down -> expect bounce
    rmax = px_df.rolling(72, min_periods=8).max().shift(1)
    breakout = (px_df / rmax - 1.0).clip(-0.5, 0.2)
    vrec = vol_df.rolling(4, min_periods=1).mean()
    vbase = vol_df.shift(4).rolling(164, min_periods=1).mean()
    surge = (vrec / vbase.replace(0.0, np.nan)).fillna(0.0).clip(0, 10)
    rising = (px_df / px_df.shift(24) - 1.0)
    volume = surge * np.sign(rising)
    return {"trend": trend.to_numpy(), "reversion": reversion.to_numpy(),
            "breakout": breakout.to_numpy(), "volume": volume.to_numpy()}


def window_ic(sig, fwd, lo, hi):
    s, f = sig[lo:hi], fwd[lo:hi]
    m = np.isfinite(s) & np.isfinite(f)
    if m.sum() < 24 or np.std(s[m]) == 0:
        return None
    return spearman(s[m], f[m])


def main():
    from train_rl import build_volume_panel, load_data, time_split
    from trader.train.event_env import EventRungEnv

    returns, btc, anchor, liq = load_data()
    train_r, _, _ = time_split(returns)
    vol = build_volume_panel(list(returns.columns), returns.index).reindex(train_r.index).fillna(0.0)
    px_df = (1.0 + train_r.fillna(0.0)).cumprod()
    sigs = family_signals(px_df, vol)
    px = px_df.to_numpy()
    n, ntok = px.shape
    fwd = np.full((n, ntok), np.nan)
    fwd[: n - H] = px[H:] / px[: n - H] - 1.0

    # --- A) persistence: trailing-30d IC vs next-7d IC, pooled per family ---
    print(f"A) PERSISTENCE (train, {ntok} tokens, weekly checkpoints, eff_win={EFF_WIN}h)")
    overall_pairs = []
    for fam in FAMS:
        pairs = []
        for j in range(ntok):
            for t in range(EFF_WIN, n - CKPT - H, CKPT):
                past = window_ic(sigs[fam][:, j], fwd[:, j], t - EFF_WIN, t)
                nxt = window_ic(sigs[fam][:, j], fwd[:, j], t, t + CKPT)
                if past is not None and nxt is not None:
                    pairs.append((past, nxt))
        if len(pairs) > 30:
            a = np.array(pairs)
            rho = spearman(a[:, 0], a[:, 1])
            hit = float(np.mean(np.sign(a[:, 0]) == np.sign(a[:, 1])))
            print(f"  {fam:10}: n={len(pairs):4d}  persistence rho={rho:+.3f}  sign-agree={hit:.0%}")
            overall_pairs += pairs
    a = np.array(overall_pairs)
    rho_all = spearman(a[:, 0], a[:, 1])
    print(f"  {'ALL':10}: n={len(a):4d}  persistence rho={rho_all:+.3f}   "
          f"(vol-tilt validation bar was +0.66; >+0.20 = real personality)")

    # --- B) payoff at ignition events: efficacy-weighted signals, incremental OOS IC ---
    env = EventRungEnv(train_r, btc, liq, volume=vol, episode_bars=len(train_r) - WARMUP - 1,
                       k=8, warmup=WARMUP, universe_mode="voltopk", seed=0)
    env.reset(start=WARMUP)
    uni_ix = [env.col_ix[t] for t in env.universe]
    cush, surge_env, ig = env._cush, env._surge, env._ignite
    btc_s, btc_e = env.btc.to_numpy(), env.btc_ema.to_numpy()
    rows = []
    eff_cache: dict = {}
    for b in range(max(WARMUP, EFF_WIN), n - H):
        for j in uni_ix:
            if not (ig[b, j] and px[b, j] > 0):
                continue
            ck = (b // CKPT) * CKPT                       # refresh efficacy weekly (cheap + causal)
            feats = []
            for fam in FAMS:
                key = (fam, j, ck)
                if key not in eff_cache:
                    eff_cache[key] = window_ic(sigs[fam][:, j], fwd[:, j], ck - EFF_WIN, ck) or 0.0
                feats.append(eff_cache[key] * sigs[fam][b, j])   # efficacy-weighted current signal
            btc_t = btc_s[b] / btc_e[b] - 1.0 if btc_e[b] else 0.0
            rows.append((cush[b, j], surge_env[b, j], btc_t, *feats, fwd[b, j]))
    A2 = np.array(rows)
    X, y = A2[:, :-1], A2[:, -1]
    base = oos_ic(X[:, :3], y)
    pers = oos_ic(X, y)
    inc = pers - base
    print(f"\nB) PAYOFF at ignitions (n={len(y)}): baseline OOS IC {base:+.3f}  "
          f"+personality {pers:+.3f}  incremental {inc:+.3f}  gate(+0.02): "
          f"{'PASS' if inc > 0.02 else 'FAIL'}")


if __name__ == "__main__":
    main()
