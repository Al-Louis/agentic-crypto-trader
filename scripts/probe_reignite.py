"""P-REIGNITE — does ADDING to a held, in-profit winner on a FRESH ignition carry positive
forward run-up vs a matched single-leg (fresh flat) ignition? Tests the PREMISE of the new
`scale_in` env flag BEFORE we trust the desktop sweep currently running on it.

Torch-free, laptop-local. Loads data via train_rl.load_data / build_volume_panel (the SAME path
the env uses) and replicates the EXACT rung-0 ignition + exit definitions from
`src/trader/train/event_env.py`. The held-set is produced by re-running the rule mirror
(`_rule_equity_curve`) loop verbatim and recording, at every bar, which tokens the rule holds and
their entry bar `b0` — so bucket membership IS the rule's actual book, not a re-derivation.

Buckets at each in-universe VALID ignition bar `b` (b that the rule would actually consider —
in the env's voltopk universe for the split, with `_px[b]>0`):

  B = held under the rule AND in-profit:  px[b] > px[b0]   (single-entry px[b0] is a DISCLOSED
      conservative proxy for the blended cost_px: a real scale-in raises cost_px above px[b0],
      so px[b]>px[b0] OVER-classifies in-profit -> a PASS is trustworthy; an underwater-by-blend
      add can't sneak into B and inflate it).
  A = matched single-leg control: a fresh ignition on a FLAT (not-held) token that the rule would
      actually fund-consider (cooled & reclaimed). Matched B<->A by surge bucket so the contrast
      isolates the held-re-ignition effect, not a surge-distribution difference.
  C = held but UNDERWATER (px[b] < px[b0]) — reported directional-only.

Outcome = forward run-up max(_px[b..b+H])/_px[b]-1 for H=24 and H=48 (the available run-up an add
could capture, matching the prior probe's "run-up" framing). Strictly causal: only px[b..b+H].

Contrast B-A via a TOKEN-CLUSTERED bootstrap (ignitions cluster in time within a token; per-bar
samples are NOT independent). N-floor: if bucket B < 30 on VAL the val verdict is INCONCLUSIVE.
TRAIN and VAL run separately; TEST is FROZEN and never touched here.

  .venv\\Scripts\\python.exe scripts\\probe_reignite.py [--surge-match] [--boot 5000]
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
RULE_EF = 0.20            # the rule's fixed entry frac (matches EventRungEnv.rule_entry_frac)


def build_env(name, r, btc, liq, vol):
    """Construct the env over the WHOLE split (one full-window episode), reset at WARMUP — the same
    pattern probe_wick.py uses. Gives us the EXACT voltopk-8 universe + precomputed signal arrays."""
    from trader.train.event_env import EventRungEnv
    env = EventRungEnv(r, btc, liq, volume=vol, episode_bars=len(r) - WARMUP - 1,
                       k=8, warmup=WARMUP, universe_mode="voltopk", seed=0)
    env.reset(start=WARMUP)
    return env


def rule_holds(env):
    """Re-run the rung-0 RULE mirror loop VERBATIM from `_rule_equity_curve` (event_env.py:689-744),
    but instead of equity record, per bar, the rule's held book: {tok: entry_bar b0}. The returned
    `held[bar]` is exactly the set of tokens the rule is holding at the START of `bar`'s decisions
    (after that bar's exits, before its new entries) — the state a re-ignition would scale into.

    Also returns `funded[bar]` = the set of FLAT tokens the rule actually FUNDED a fresh entry on at
    `bar` (passed cooled+reclaimed AND won a slot), so the bucket-A control is the rule's real
    single-leg entries, and `considered[bar]` = flat ignition tokens that cleared cooled+reclaimed
    (the fundable candidate set) for a looser A definition."""
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

    held = {}            # bar -> {tok: entry_bar}
    funded = {}          # bar -> set(tok) freshly funded this bar
    considered = {}      # bar -> set(tok) flat ignition candidates that cleared cooled+reclaimed
    for bar in range(start, end + 1):
        equity = cash + sum(value(t, bar) for t in pos)
        if equity > 1.0:
            for t in list(pos):                                   # 1) exits (trailing stop OR EMA-break)
                j = cix[t]
                p = pos[t]
                p["peak_px"] = max(p["peak_px"], px[bar, j])
                if px[bar, j] < p["peak_px"] * (1.0 - sk) or cush[bar, j] < 0.0:
                    v = value(t, bar)
                    cash += v - amm_cost_usd(-v, liq.get(t, 0.0), fee, gas)
                    cool[t], prior[t] = bar, p["origin"]
                    del pos[t]
        # snapshot the held book AFTER exits, BEFORE new entries — the state a scale-in sees
        held[bar] = {t: pos[t]["entry_bar"] for t in pos}
        fset, cset = set(), set()
        if equity > 1.0:
            cands = [t for t in env.universe if ig[bar, cix[t]] and t not in pos
                     and (bar - cool[t]) >= cd
                     and (prior[t] is None or px[bar, cix[t]] > prior[t])]
            cset = set(cands)
            cands.sort(key=lambda t: cush[bar, cix[t]], reverse=True)
            for t in cands:                                       # 2) fund strongest; rotate losers
                if min(ef * equity, cash) < 1.0 and pos:
                    weak = min(pos, key=lambda h: cush[bar, cix[h]])
                    if cush[bar, cix[weak]] < cush[bar, cix[t]]:
                        v = value(weak, bar)
                        cash += v - amm_cost_usd(-v, liq.get(weak, 0.0), fee, gas)
                        cool[weak], prior[weak] = bar, pos[weak]["origin"]
                        del pos[weak]
                        equity = cash + sum(value(tt, bar) for tt in pos)
                size = min(ef * equity, cash)
                if size >= 1.0:
                    j = cix[t]
                    cash -= size + amm_cost_usd(size, liq.get(t, 0.0), fee, gas)
                    pos[t] = {"usd": size, "entry_bar": bar, "peak_px": px[bar, j], "origin": px[bar, j]}
                    fset.add(t)
        funded[bar] = fset
        considered[bar] = cset
    return held, funded, considered


def fwd_runup(px_col, b, H, n):
    """Forward run-up over [b, b+H]: max(px[b..b+H])/px[b]-1. Causal. None if window runs off-panel."""
    if b + H >= n:
        return None
    p0 = px_col[b]
    if p0 <= 0:
        return None
    window = px_col[b: b + H + 1]
    return float(window.max() / p0 - 1.0)


def classify(env, held, funded, considered):
    """Walk every in-universe ignition bar in [WARMUP, end] and tag each as bucket A/B/C with its
    surge and forward run-ups. Returns a list of dicts."""
    px, ig, surge, cix = env._px, env._ignite, env._surge, env.col_ix
    n = env.n_bars
    start, end = env.start, env.end
    rows = []
    for t in env.universe:
        j = cix[t]
        pxc = px[:, j]
        for b in range(max(start, WARMUP), min(end, n)):
            if not ig[b, j] or pxc[b] <= 0:
                continue
            f24 = fwd_runup(pxc, b, 24, n)
            f48 = fwd_runup(pxc, b, 48, n)
            if f24 is None or f48 is None:
                continue
            hb = held.get(b, {})
            rec = {"tok": t, "bar": b, "surge": float(surge[b, j]),
                   "f24": f24, "f48": f48}
            if t in hb:                                  # held under the rule
                b0 = hb[t]
                rec["b0"] = b0
                if pxc[b] > pxc[b0]:
                    rec["bucket"] = "B"                  # held + in-profit (vs single-entry proxy)
                else:
                    rec["bucket"] = "C"                  # held + underwater
            else:                                        # flat token
                # A = the rule's real single-leg fresh entries (funded this bar). The looser
                # variant (considered) is reported as a sensitivity check.
                rec["funded"] = (t in funded.get(b, set()))
                rec["considered"] = (t in considered.get(b, set()))
                rec["bucket"] = "A"
            rows.append(rec)
    return rows


def surge_bucket(s):
    if s < 3.5:
        return 0
    if s < 5.0:
        return 1
    if s < 8.0:
        return 2
    return 3


def token_cluster_bootstrap(B_vals, B_toks, A_vals, A_toks, n_boot, rng):
    """Bootstrap the B-A mean difference by RESAMPLING TOKENS (clusters), not individual bars.
    Resample tokens with replacement within each bucket; pool that resample's events; take the mean.
    Returns (point_estimate, ci_lo, ci_hi) at 95%."""
    B_by = {}
    for v, tk in zip(B_vals, B_toks):
        B_by.setdefault(tk, []).append(v)
    A_by = {}
    for v, tk in zip(A_vals, A_toks):
        A_by.setdefault(tk, []).append(v)
    Bk, Ak = list(B_by), list(A_by)
    if not Bk or not Ak:
        return None, None, None
    point = float(np.mean(B_vals) - np.mean(A_vals))
    diffs = np.empty(n_boot)
    for i in range(n_boot):
        bs = [B_by[Bk[k]] for k in rng.integers(0, len(Bk), len(Bk))]
        as_ = [A_by[Ak[k]] for k in rng.integers(0, len(Ak), len(Ak))]
        bpool = np.concatenate(bs) if bs else np.array([0.0])
        apool = np.concatenate(as_) if as_ else np.array([0.0])
        diffs[i] = bpool.mean() - apool.mean()
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return point, float(lo), float(hi)


def summarize(rows, label):
    print(f"\n=== {label} ===  total in-universe ignitions classified: {len(rows)}")
    for bk in ("A", "B", "C"):
        sub = [r for r in rows if r["bucket"] == bk]
        if not sub:
            print(f"  {bk}: n=0")
            continue
        f24 = np.array([r["f24"] for r in sub])
        f48 = np.array([r["f48"] for r in sub])
        ntok = len({r["tok"] for r in sub})
        extra = ""
        if bk == "A":
            nf = sum(1 for r in sub if r.get("funded"))
            extra = f"  [funded(real single-leg)={nf}]"
        print(f"  {bk}: n={len(sub):4d}  tokens={ntok}{extra}")
        print(f"       fwd24  mean {f24.mean():+7.2%}  median {np.median(f24):+7.2%}  win {np.mean(f24>0):4.0%}")
        print(f"       fwd48  mean {f48.mean():+7.2%}  median {np.median(f48):+7.2%}  win {np.mean(f48>0):4.0%}")
    return rows


def contrast(rows, label, n_boot, rng, use_funded_A, surge_match):
    B = [r for r in rows if r["bucket"] == "B"]
    A_all = [r for r in rows if r["bucket"] == "A"]
    A = [r for r in A_all if r.get("funded")] if use_funded_A else A_all
    a_desc = "funded single-leg" if use_funded_A else "all flat ignitions"
    if surge_match and B and A:
        # weight A to B's surge-bucket distribution by RESAMPLING A within each surge bucket to
        # match B's per-bucket counts (keeps A's per-token clustering for the bootstrap caveat note,
        # but equalizes the surge mix so the contrast isn't a surge-distribution artifact).
        from collections import Counter
        bdist = Counter(surge_bucket(r["surge"]) for r in B)
        A_by_sb = {}
        for r in A:
            A_by_sb.setdefault(surge_bucket(r["surge"]), []).append(r)
        matched = []
        for sb, cnt in bdist.items():
            pool = A_by_sb.get(sb, [])
            if not pool:
                continue
            idx = rng.integers(0, len(pool), cnt)
            matched.extend(pool[k] for k in idx)
        A = matched
        a_desc += " (surge-matched to B)"
    print(f"\n  --- B - A contrast [{label}] (A = {a_desc}) ---")
    if len(B) == 0 or len(A) == 0:
        print(f"     n_B={len(B)} n_A={len(A)} — cannot contrast")
        return
    for H, key in (("H=24", "f24"), ("H=48", "f48")):
        Bv = [r[key] for r in B]
        Av = [r[key] for r in A]
        pt, lo, hi = token_cluster_bootstrap(Bv, [r["tok"] for r in B], Av, [r["tok"] for r in A],
                                             n_boot, rng)
        sig = "CI-low > 0 (positive)" if lo is not None and lo > 0 else \
              "CI straddles 0" if lo is not None and hi is not None and lo <= 0 <= hi else \
              "CI-high < 0 (negative)"
        print(f"     {H}: B mean {np.mean(Bv):+7.2%}  A mean {np.mean(Av):+7.2%}  "
              f"B-A {pt:+7.2%}  95%CI [{lo:+.2%}, {hi:+.2%}]  -> {sig}")


def zec_anchor(env, rows, held):
    """Locate ZEC Apr-9 ~16:00 (the +16.2% re-ignition the agent could not take) and confirm it
    lands in bucket B with a large forward run-up. Uses the probe_wick searchsorted locate pattern."""
    if "ZEC" not in env.col_ix:
        print("\n  ZEC anchor: ZEC not in panel/universe for this split — skipped")
        return
    target = int(pd.Timestamp("2026-04-09T16:00", tz="UTC").timestamp())
    idx = env.returns.index.to_numpy()
    idx_s = idx // 1000 if idx.max() > 1e12 else idx
    b = int(np.searchsorted(idx_s, target))
    j = env.col_ix["ZEC"]
    # search a small window around the target for the actual ignition bar (timestamp granularity)
    found = None
    for bb in range(max(b - 2, 0), min(b + 3, env.n_bars)):
        if env._ignite[bb, j]:
            found = bb
            break
    if found is None:
        print(f"\n  ZEC anchor: no ZEC ignition within +-2h of 2026-04-09 16:00 "
              f"(bar {b}); ZEC may be out of this split's window")
        return
    rec = next((r for r in rows if r["tok"] == "ZEC" and r["bar"] == found), None)
    hb = held.get(found, {})
    print(f"\n  ZEC anchor: ignition at bar {found} ({pd.to_datetime(idx_s[found], unit='s', utc=True)})")
    if rec is None:
        in_uni = "ZEC" in env.universe
        print(f"     not in classified rows (ZEC in split voltopk universe: {in_uni}; "
              f"held now: {'ZEC' in hb}) — likely the fwd-48 window runs off-panel or ZEC isn't top-8 vol")
        return
    print(f"     bucket={rec['bucket']}  surge={rec['surge']:.2f}  "
          f"fwd24={rec['f24']:+.2%}  fwd48={rec['f48']:+.2%}"
          + (f"  (entry b0={rec.get('b0')})" if 'b0' in rec else ""))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--boot", type=int, default=5000)
    p.add_argument("--surge-match", action="store_true",
                   help="weight bucket A to B's surge-bucket distribution")
    args = p.parse_args()
    from train_rl import build_volume_panel, load_data, time_split
    returns, btc, anchor, liq = load_data()
    train_r, val_r, test_r = time_split(returns)
    vol = build_volume_panel(list(returns.columns), returns.index)
    rng = np.random.default_rng(0)

    for name, r in (("TRAIN", train_r), ("VAL", val_r)):     # TEST split FROZEN — not touched
        env = build_env(name, r, btc, liq, vol)
        print(f"\n########## {name} SPLIT ##########  bars={env.n_bars}  universe(voltopk-8)={env.universe}")
        held, funded, considered = rule_holds(env)
        rows = classify(env, held, funded, considered)
        summarize(rows, name)
        contrast(rows, name, args.boot, rng, use_funded_A=True, surge_match=args.surge_match)
        contrast(rows, name, args.boot, rng, use_funded_A=False, surge_match=args.surge_match)
        zec_anchor(env, rows, held)


if __name__ == "__main__":
    main()
