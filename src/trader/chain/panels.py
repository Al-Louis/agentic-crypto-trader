"""Hourly per-token panels derived from raw pool events.

Aggregates each pool's decoded log rows into an hourly frame aligned (by
epoch-second hour timestamps) to the recorded OHLCV/returns index used by
every prior probe — that alignment is the whole point: the liquidity/flow
probes run on the SAME historical window as the price-only probes did.

Sign conventions (from ``trader.chain.events``: positive = into pool):
  net_token_in / net_quote_in   net swap flow from the POOL's perspective;
                                a net *buying* hour has net_quote_in > 0 and
                                net_token_in < 0.
  lp_remove_*                   positive magnitudes (how much was withdrawn).

Per-row timestamps are interpolated from the sampled (block, ts) index —
BSC block time moved 3s -> 0.45s across the window, so interpolation is
piecewise between samples a few thousand blocks apart (seconds of error,
against hourly buckets).
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd

from trader.chain.collector import DEFAULT_ROOT, load_block_index, load_pool_logs
from trader.chain.registry import load_registry

PANEL_COLUMNS = [
    "n_swaps", "vol_token", "vol_quote", "net_token_in", "net_quote_in",
    "n_mints", "n_burns", "n_collects",
    "lp_add_token", "lp_add_quote", "lp_remove_token", "lp_remove_quote",
    "liquidity_end", "reserve_token_end", "reserve_quote_end", "price_end",
    "unique_swappers",
]


def interpolate_ts(blocks: np.ndarray, index: pd.DataFrame) -> np.ndarray:
    """Per-log epoch-second timestamps from the sampled block index."""
    return np.interp(blocks, index["block"].to_numpy(), index["ts"].to_numpy())


def build_panel(symbol: str, root: str = DEFAULT_ROOT) -> pd.DataFrame:
    pools = load_registry(os.path.join(root, "_pools.json"))
    p = next(pp for pp in pools if pp["symbol"] == symbol)
    df = load_pool_logs(symbol, root)
    if df.empty:
        return pd.DataFrame(columns=PANEL_COLUMNS,
                            index=pd.Index([], name="timestamp", dtype="int64"))
    bix = load_block_index(root)
    df["ts"] = interpolate_ts(df["block"].to_numpy(), bix)
    df["hour"] = (df["ts"] // 3600).astype("int64") * 3600

    side = p["token_side"]
    tok, qt = ("a0", "a1") if side == 0 else ("a1", "a0")

    sw = df[df["event"] == "swap"]
    mi = df[df["event"] == "mint"]
    bu = df[df["event"] == "burn"]
    co = df[df["event"] == "collect"]

    g = pd.DataFrame(index=pd.Index(sorted(df["hour"].unique()), name="timestamp"))
    gs = sw.groupby("hour")
    g["n_swaps"] = gs.size()
    g["vol_token"] = gs[tok].apply(lambda s: s.abs().sum())
    g["vol_quote"] = gs[qt].apply(lambda s: s.abs().sum())
    g["net_token_in"] = gs[tok].sum()
    g["net_quote_in"] = gs[qt].sum()
    g["unique_swappers"] = gs["recipient"].nunique()
    g["n_mints"] = mi.groupby("hour").size()
    g["n_burns"] = bu.groupby("hour").size()
    g["n_collects"] = co.groupby("hour").size()
    g["lp_add_token"] = mi.groupby("hour")[tok].sum()
    g["lp_add_quote"] = mi.groupby("hour")[qt].sum()
    g["lp_remove_token"] = -bu.groupby("hour")[tok].sum()
    g["lp_remove_quote"] = -bu.groupby("hour")[qt].sum()

    # state at hour end: liquidity + reserves + pool price (token in quote units)
    if p["version"] == "v2":
        sy = df[df["event"] == "sync"]
        rt, rq = ("r0", "r1") if side == 0 else ("r1", "r0")
        last = sy.groupby("hour").last()
        r0v = pd.to_numeric(last["r0"])      # object dtype from None-mixed parquet cols
        r1v = pd.to_numeric(last["r1"])
        g["reserve_token_end"] = r0v if side == 0 else r1v
        g["reserve_quote_end"] = r1v if side == 0 else r0v
        g["liquidity_end"] = np.sqrt(r0v * r1v)
        g["price_end"] = (r1v / r0v) if side == 0 else (r0v / r1v)
    else:
        last = sw.groupby("hour").last()
        price10 = pd.to_numeric(last["price1per0"])      # token1 per token0
        liq = pd.to_numeric(last["liquidity"])           # normalized L
        sqrt_p = np.sqrt(price10)
        r0v, r1v = liq / sqrt_p, liq * sqrt_p            # virtual in-range reserves
        g["reserve_token_end"] = r0v if side == 0 else r1v
        g["reserve_quote_end"] = r1v if side == 0 else r0v
        g["liquidity_end"] = liq
        g["price_end"] = price10 if side == 0 else 1.0 / price10

    counts = ["n_swaps", "n_mints", "n_burns", "n_collects", "unique_swappers"]
    flows = ["vol_token", "vol_quote", "net_token_in", "net_quote_in",
             "lp_add_token", "lp_add_quote", "lp_remove_token", "lp_remove_quote"]
    g[counts] = g[counts].fillna(0).astype("int64")
    g[flows] = g[flows].fillna(0.0)
    # state columns: forward-fill within the panel (no events -> unchanged pool)
    state = ["liquidity_end", "reserve_token_end", "reserve_quote_end", "price_end"]
    g[state] = g[state].ffill()
    return g[PANEL_COLUMNS]


def build_all(root: str = DEFAULT_ROOT, out_dir: str | None = None,
              logger=print) -> dict[str, pd.DataFrame]:
    out_dir = out_dir or os.path.join(root, "panels", "hour")
    os.makedirs(out_dir, exist_ok=True)
    pools = load_registry(os.path.join(root, "_pools.json"))
    panels = {}
    for p in pools:
        g = build_panel(p["symbol"], root)
        panels[p["symbol"]] = g
        g.reset_index().to_parquet(os.path.join(out_dir, f"{p['symbol']}.parquet"),
                                   index=False)
        logger(f"  {p['symbol']:10} {len(g):6} hours "
               f"({'empty' if g.empty else str(pd.to_datetime(g.index.min(), unit='s').date()) + ' -> ' + str(pd.to_datetime(g.index.max(), unit='s').date())})")
    return panels


def load_panel(symbol: str, root: str = DEFAULT_ROOT) -> pd.DataFrame:
    path = os.path.join(root, "panels", "hour", f"{symbol}.parquet")
    return pd.read_parquet(path).set_index("timestamp")
