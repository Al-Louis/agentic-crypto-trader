"""Gecko-vs-CMC SIGNAL parity: instantiate the real EventRungEnv on each feed's features+volume and
diff the deterministic triggers it precomputes — EMA-break eligibility (cush<0) and the ignition entry
signal (_ignite) — per token per bar over the live window. This is the laptop-runnable proxy for "does
the agent trade differently"; it uses the env's OWN computation (no re-implementation). The full LSTM
replay (cut/hold modulation) is a separate desktop job (needs torch + weights).

Run: PYTHONPATH=src python scripts/parity_signals.py
"""
from __future__ import annotations

import glob
import json
import os
import datetime as dt

import numpy as np
import pandas as pd

from trader.train.event_env import EventRungEnv

WINDOW_START = 1782086400   # 2026-06-22T00:00:00Z


def load_returns(feat_dir: str) -> pd.DataFrame:
    ret = {}
    for f in sorted(glob.glob(os.path.join(feat_dir, "*_factor.parquet"))):
        sym = os.path.basename(f)[:-len("_factor.parquet")]
        ret[sym] = pd.read_parquet(f).set_index("timestamp")["r_alt"]
    returns = np.expm1(pd.DataFrame(ret).sort_index())
    for col in returns.columns:                      # zero each token's first valid return (artifact)
        fv = returns[col].first_valid_index()
        if fv is not None:
            returns.loc[fv, col] = 0.0
    return returns


def volume_panel(ohlcv_root: str, tokens, index) -> pd.DataFrame:
    cols = {}
    for t in tokens:
        dirs = glob.glob(os.path.join(ohlcv_root, "hour_1", f"{t}_*"))
        if not dirs:
            continue
        df = pd.concat([pd.read_parquet(f) for f in glob.glob(os.path.join(dirs[0], "*.parquet"))],
                       ignore_index=True).drop_duplicates("timestamp").sort_values("timestamp")
        ts = df["timestamp"].to_numpy()
        ts = (ts // 1000) if ts.max() > 1e12 else ts
        cols[t] = pd.Series(df["volume"].to_numpy(), index=ts).reindex(index).fillna(0.0)
    return pd.DataFrame(cols, index=index)


def btc_close(anchor_root: str) -> pd.Series:
    a = pd.read_parquet(os.path.join(anchor_root, "BTC_USDT", "1h.parquet")).set_index("timestamp").sort_index()
    if a.index.max() > 1e12:
        a.index = (a.index // 1000).astype("int64")
    return a["close"]


def build_env(feat_dir: str, ohlcv_root: str, anchor_root: str, liq: dict) -> EventRungEnv:
    returns = load_returns(feat_dir)
    btc = btc_close(anchor_root)
    vol = volume_panel(ohlcv_root, returns.columns, returns.index)
    return EventRungEnv(returns, btc, liq, volume=vol, ema_span=72, warmup=168, k=8)


def signals(env: EventRungEnv) -> dict:
    """{token: DataFrame[index=ts, cush, ignite]} from the env's precomputed arrays."""
    idx = list(env.returns.index)
    out = {}
    for j, t in enumerate(env.returns.columns):
        out[t] = pd.DataFrame({"cush": env._cush[:, j], "ignite": env._ignite[:, j].astype(bool)}, index=idx)
    return out


def main() -> None:
    liq = {s["symbol"]: (s.get("liq_usd") or 0.0)
           for s in json.load(open("data/selection.json", encoding="utf-8"))}
    g = signals(build_env("data/features_gecko", "data/ohlcv_gecko", "data/anchor_cmc", liq))
    c = signals(build_env("data/features_cmc", "data/ohlcv_cmc", "data/anchor_cmc", liq))

    toks = [t for t in g if t in c]
    print(f"tokens compared: {len(toks)} | window from {dt.datetime.fromtimestamp(WINDOW_START, dt.timezone.utc):%m-%d %H:%M}Z\n")
    tot_cells = ema_flip = ig_flip = 0
    per_tok = []
    for t in toks:
        gj, cj = g[t], c[t]
        idx = [ts for ts in gj.index if ts in cj.index and ts >= WINDOW_START
               and not np.isnan(gj.loc[ts, "cush"]) and not np.isnan(cj.loc[ts, "cush"])]
        if not idx:
            continue
        gb = (gj.loc[idx, "cush"] < 0)          # EMA-break eligible (price below EMA)
        cb = (cj.loc[idx, "cush"] < 0)
        ef = int((gb.values != cb.values).sum())
        igf = int((gj.loc[idx, "ignite"].values != cj.loc[idx, "ignite"].values).sum())
        tot_cells += len(idx); ema_flip += ef; ig_flip += igf
        if ef or igf:
            per_tok.append((t, len(idx), ef, igf))
    print("=== EMA-break eligibility (cush<0) + ignition flips over the live window ===")
    print(f"  total (token,bar) cells: {tot_cells}")
    print(f"  EMA-break SIGN flips:    {ema_flip}  ({ema_flip/max(tot_cells,1)*100:.2f}% of cells)")
    print(f"  ignition flips:          {ig_flip}  ({ig_flip/max(tot_cells,1)*100:.2f}% of cells)")
    print("  per-token (only tokens with >=1 flip):  token  bars  ema_flips  ig_flips")
    for t, n, ef, igf in sorted(per_tok, key=lambda x: -(x[2] + x[3])):
        print(f"     {t:<10} {n:>4}  {ef:>4}  {igf:>4}")

    # TAC detail around the phantom exit
    if "TAC" in toks:
        print("\n=== TAC cush by bar (live window) — sign flip = feed changes the EMA-break decision ===")
        gj, cj = g["TAC"], c["TAC"]
        idx = [ts for ts in gj.index if ts in cj.index and ts >= WINDOW_START]
        for ts in idx:
            gc, cc = gj.loc[ts, "cush"], cj.loc[ts, "cush"]
            if np.isnan(gc) or np.isnan(cc):
                continue
            flag = "  <== FLIP (gecko breaks, cmc holds)" if (gc < 0) != (cc < 0) else ""
            print(f"  {dt.datetime.fromtimestamp(ts, dt.timezone.utc):%m-%d %H:%M}  gecko {gc:+.4f}  cmc {cc:+.4f}{flag}")


if __name__ == "__main__":
    main()
