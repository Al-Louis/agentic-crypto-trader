"""PROBE 3 — WALLET-COHORT LEAD (pre-registered, 2026-06-12).

Hypothesis ([[Trading Strategies]] PARKED note): wallet-attributed flow leads
price by actionable hours — NEW-wallet accumulation precedes pumps,
AGED-wallet distribution precedes local dumps, and the resident MM wallets
going QUIET precedes personality breaks / detonations.

Attribution: the swap `recipient` (the wallet receiving the out-side of the
final hop; the user on simple router swaps, noisy on multi-hop — addresses
seen as recipients across >= --router-tokens of our pools are dropped as
router/aggregator infrastructure). Per-swap wallet token delta = -a_token
(pool out = wallet in). HONEST FRAMING: this is a recipient-proxy, not
tx.from; the probe gates whether tx.from enrichment is worth the calls.

Cohorts (rolling, per token):
  NEW   — wallet first seen in this pool < 7d ago
  AGED  — first seen >= 28d ago
  MM    — top --mm-k wallets by cumulative volume with two-sided flow
          (|net| / volume < 0.5), recomputed weekly from trailing history

Reported per split (train/val; test frozen):
  IC of trailing-24h cohort net flow (quote units, /pool size) vs fwd
  {24,48,72}h returns; MM-quiet hours (trailing-24h MM volume share z < -1.5)
  -> P(detonation within 48h) vs base + fwd worst trough.

  python scripts/probe_wallet_cohort.py [--mm-k 5] [--router-tokens 4]
"""
from __future__ import annotations

import argparse
import os
import sys
from collections import defaultdict

import numpy as np
import pandas as pd

sys.path.insert(0, "scripts")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

WARMUP = 168
HORIZONS = (24, 48, 72)
NEW_H, AGED_H = 7 * 24 * 3600, 28 * 24 * 3600


def spearman(x, y):
    """Spearman rho = Pearson on ranks (avoids a scipy dependency)."""
    xs, ys = pd.Series(x).rank(), pd.Series(y).rank()
    return xs.corr(ys)


def _swaps(symbol: str, registry: dict) -> pd.DataFrame:
    """Swap rows with interpolated ts and token-side wallet delta."""
    from trader.chain.collector import load_block_index, load_pool_logs
    from trader.chain.panels import interpolate_ts
    df = load_pool_logs(symbol)
    df = df[df["event"] == "swap"].copy()
    if df.empty:
        return df
    bix = load_block_index()
    df["ts"] = interpolate_ts(df["block"].to_numpy(), bix)
    df["hour"] = (df["ts"] // 3600).astype("int64") * 3600
    side = registry[symbol]["token_side"]
    tok, qt = ("a0", "a1") if side == 0 else ("a1", "a0")
    df["wallet_token"] = -df[tok]                  # pool out = wallet in
    df["wallet_quote"] = -df[qt]
    df["vol_quote"] = df[qt].abs()
    return df[["hour", "recipient", "wallet_token", "wallet_quote", "vol_quote"]]


def cohort_panels(sw: pd.DataFrame, index: pd.Index, mm_k: int) -> pd.DataFrame:
    """Hourly cohort net flows + MM volume share, aligned to `index`."""
    first_seen = sw.groupby("recipient")["hour"].min()
    sw = sw.assign(first=sw["recipient"].map(first_seen))
    age = sw["hour"] - sw["first"]
    new_m, aged_m = age < NEW_H, age >= AGED_H

    # MM set: recomputed weekly from trailing history (volume + two-sidedness)
    weeks = ((sw["hour"] - sw["hour"].min()) // (168 * 3600)).astype(int)
    mm_flags = np.zeros(len(sw), dtype=bool)
    hist = None
    for w in sorted(weeks.unique()):
        m = weeks < w
        if m.sum():
            g = sw[m].groupby("recipient").agg(vol=("vol_quote", "sum"),
                                               net=("wallet_quote", "sum"))
            g = g[g["vol"] > 0]
            g["two_sided"] = (g["net"].abs() / g["vol"]) < 0.5
            hist = set(g[g["two_sided"]].nlargest(mm_k, "vol").index)
        if hist:
            mm_flags |= (weeks == w).to_numpy() & sw["recipient"].isin(hist).to_numpy()

    out = pd.DataFrame(index=index)
    for label, m in (("new", new_m.to_numpy()), ("aged", aged_m.to_numpy()),
                     ("mm", mm_flags)):
        sub = sw[m]
        out[f"{label}_net_quote"] = sub.groupby("hour")["wallet_quote"].sum().reindex(index).fillna(0.0)
        out[f"{label}_vol"] = sub.groupby("hour")["vol_quote"].sum().reindex(index).fillna(0.0)
    out["all_vol"] = sw.groupby("hour")["vol_quote"].sum().reindex(index).fillna(0.0)
    return out


def detonation_mask(r, vol):
    v = vol.reindex(r.index).fillna(0.0)
    pxf = (1.0 + r.fillna(0.0)).cumprod()
    rising = (pxf / pxf.shift(24) - 1.0).to_numpy()
    vrec = v.rolling(4, min_periods=1).mean()
    vbase = v.shift(4).rolling(164, min_periods=1).mean()
    surge = (vrec / vbase.replace(0.0, np.nan)).fillna(0.0).to_numpy()
    return (surge >= 8.0) & (rising <= -0.15)


def run_split(name, r, vol, cohorts, det):
    pxf = (1.0 + r.fillna(0.0)).cumprod()
    n = len(r)
    recs = defaultdict(list)        # cohort -> (signal, fwd24, fwd48, fwd72)
    quiet_rows, base_rows = [], []  # (det48, worst48)
    for s, co in cohorts.items():
        if s not in r.columns:
            continue
        j = r.columns.get_loc(s)
        c = co.reindex(r.index)
        px = pxf[s].to_numpy()
        roll_vol = c["all_vol"].rolling(24, min_periods=1).sum()
        for label in ("new", "aged"):
            sig = (c[f"{label}_net_quote"].rolling(24, min_periods=1).sum()
                   / roll_vol.replace(0.0, np.nan)).to_numpy()
            for b in range(WARMUP, n - max(HORIZONS)):
                if px[b] > 0 and np.isfinite(sig[b]) and roll_vol.iloc[b] > 0:
                    recs[label].append((sig[b], px[b + 24] / px[b] - 1,
                                        px[b + 48] / px[b] - 1, px[b + 72] / px[b] - 1))
        share = (c["mm_vol"].rolling(24, min_periods=1).sum()
                 / roll_vol.replace(0.0, np.nan))
        mu, sd = share.expanding(168).mean(), share.expanding(168).std()
        quiet = ((share - mu) / sd.replace(0.0, np.nan) < -1.5).to_numpy()
        for b in range(WARMUP, n - 48):
            if not (px[b] > 0):
                continue
            row = (det[b + 1: b + 49, j].any(), px[b + 1: b + 49].min() / px[b] - 1)
            (quiet_rows if quiet[b] else base_rows).append(row)

    print(f"\n=== {name} ===")
    for label in ("new", "aged"):
        a = np.array(recs[label])
        if not len(a):
            print(f"  {label}: no obs")
            continue
        line = f"  {label.upper():4} flow (n={len(a):,}): "
        for h, col in zip(HORIZONS, (1, 2, 3)):
            ic = spearman(a[:, 0], a[:, col])
            line += f"IC(fwd{h}) {ic:+.4f}  "
        print(line + f"(noise ~{2 / np.sqrt(len(a)):.4f})")
    q, base = np.array(quiet_rows, dtype=float), np.array(base_rows, dtype=float)
    if len(q) and len(base):
        print(f"  MM-QUIET hours: n={len(q):,} ({len(q) / (len(q) + len(base)):.1%})  "
              f"P(det<=48h) {q[:, 0].mean():.2%} vs base {base[:, 0].mean():.2%} "
              f"(lift x{q[:, 0].mean() / max(base[:, 0].mean(), 1e-9):.1f})  "
              f"worst48 {q[:, 1].mean():+.2%} vs {base[:, 1].mean():+.2%}")
    else:
        print(f"  MM-QUIET: insufficient obs (quiet={len(q)}, base={len(base)})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mm-k", type=int, default=5)
    ap.add_argument("--router-tokens", type=int, default=4,
                    help="recipients seen in >= this many of our pools = infrastructure, dropped")
    args = ap.parse_args()

    from train_rl import build_volume_panel, load_data, time_split
    from trader.chain.registry import load_registry

    registry = {p["symbol"]: p for p in load_registry()}
    returns, _btc, _anchor, _liq = load_data()
    vol = build_volume_panel(list(returns.columns), returns.index)

    raw = {}
    seen_in = defaultdict(set)
    for s in registry:
        sw = _swaps(s, registry)
        if len(sw):
            raw[s] = sw
            for a in sw["recipient"].unique():
                seen_in[a].add(s)
    routers = {a for a, toks in seen_in.items() if len(toks) >= args.router_tokens}
    print(f"dropping {len(routers)} cross-pool recipient addresses as infrastructure")
    cohorts = {}
    for s, sw in raw.items():
        sw = sw[~sw["recipient"].isin(routers)]
        cohorts[s] = cohort_panels(sw, returns.index, args.mm_k)

    det = detonation_mask(returns, vol)
    train_r, val_r, _test = time_split(returns)
    a, b = len(train_r), len(train_r) + len(val_r)
    for name, rr, dd in (("train", train_r, det[:a]), ("val", val_r, det[a:b])):
        run_split(name, rr, vol, cohorts, dd)


if __name__ == "__main__":
    main()
