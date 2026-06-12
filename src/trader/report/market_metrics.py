"""Market-structure metrics for the volatility / correlation dashboard.

A pure (numpy/pandas, torch-free, laptop-testable) computation that turns the OHLCV return panel
into the static ``market_metrics.json`` the frontend renders — a top-level artifact published
alongside ``leaderboard.json`` (see [[Apentic Data Contract]]). It captures the three things the
agent's risk picture hinges on:

  1. **per-token realized volatility** — the alts span ~8x the median; a few monsters (HUMA, SIREN,
     SKYAI) carry multi-thousand-percent excursions while a calm tail (gold, XRP) behaves normally;
  2. **each token's correlation + beta to BTC** — they DECOUPLE: alts pump on their own volume
     dynamics while BTC bleeds (train: BTC -31% while the basket +26%);
  3. **the full token x token correlation matrix** — near-zero average pairwise corr (+0.13, just
     +0.035 among the monsters) is what makes risk-parity sizing collapse portfolio drawdown.

Pure function ⇒ deterministic given inputs (pass ``generated`` to pin the timestamp in tests).
"""

from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pandas as pd

from trader.report.apentic import _slug, _to_secs

HOURS_PER_YEAR = 24 * 365
# the full lookback ladder (user frontend toggle, 2026-06-12): how token volatility EVOLVES, and
# per-window top-8 rankings — the evolving-pool view (would ZEC fall out of the tradeable 8 by
# day 3 while a candidate rises?). Old keys kept stable for the existing frontend.
DEFAULT_WINDOWS = {"24h": 24, "7d": 168, "30d": 720, "90d": 2160, "180d": 4320}


def _ann_vol(r: pd.Series, hpy: int = HOURS_PER_YEAR) -> float:
    s = float(r.std())
    return s * np.sqrt(hpy) if np.isfinite(s) else 0.0


def compute_market_metrics(returns: pd.DataFrame, btc_close: pd.Series, *,
                           windows: dict[str, int] | None = None, vol_spark_window: int = 168,
                           excursion_window: int = 336, spark_points: int = 150,
                           hours_per_year: int = HOURS_PER_YEAR,
                           generated: str | None = None) -> dict:
    """Return the ``market_metrics.json`` payload (see module docstring for the three metric groups).

    ``returns`` is the simple-return panel [time x token]; ``btc_close`` the BTC anchor close.
    ``vol_spark_window`` = rolling-vol lookback for each token's sparkline (downsampled to
    ``spark_points``); ``excursion_window`` = window for the max runup / drawdown (default 14d).
    """
    windows = windows or DEFAULT_WINDOWS
    R = returns.sort_index().fillna(0.0)
    idx = R.index
    btc_ret = btc_close.reindex(idx).ffill().bfill().pct_change().fillna(0.0)
    btc_np = btc_ret.to_numpy()
    var_btc = float(btc_ret.var())
    px = (1.0 + R).cumprod()

    tokens: list[dict] = []
    for t in R.columns:
        r = R[t]
        p = px[t]
        roll = p / p.shift(excursion_window) - 1.0                 # rolling excursion over the window
        rv = (r.rolling(vol_spark_window, min_periods=8).std() * np.sqrt(hours_per_year)).dropna()
        step = max(len(rv) // spark_points, 1)
        spark = [{"time": _to_secs(ts), "ann_vol": round(float(v), 4)} for ts, v in rv.iloc[::step].items()]
        corr_btc = float(r.corr(btc_ret)) if r.std() > 0 and btc_ret.std() > 0 else 0.0
        beta_btc = float(np.cov(r.to_numpy(), btc_np)[0, 1] / var_btc) if var_btc > 0 else 0.0
        tokens.append({
            "symbol": t, "slug": _slug(t),
            "ann_vol": round(_ann_vol(r, hours_per_year), 4),
            "ret_window": round(float(p.iloc[-1] / p.iloc[0] - 1.0), 4),
            "vol_by_window": {k: round(_ann_vol(r.iloc[-w:], hours_per_year), 4)
                              for k, w in windows.items()},
            "max_runup": round(float(roll.max()) if roll.notna().any() else 0.0, 4),
            "max_drawdown": round(float(roll.min()) if roll.notna().any() else 0.0, 4),
            "corr_btc": round(corr_btc if np.isfinite(corr_btc) else 0.0, 4),
            "beta_btc": round(beta_btc if np.isfinite(beta_btc) else 0.0, 4),
            "vol_series": spark,
        })

    tokens.sort(key=lambda d: d["ann_vol"], reverse=True)           # most volatile first (display order)

    # full correlation matrix incl. BTC (for the heatmap), in the same display order + BTC last
    order = [d["symbol"] for d in tokens]
    M = R[order].copy()
    M["BTC"] = btc_ret
    labels = order + ["BTC"]
    C = M.corr().reindex(index=labels, columns=labels).fillna(0.0)
    matrix = [[round(float(C.iloc[i, j]), 4) for j in range(len(labels))] for i in range(len(labels))]

    Ctok = C.loc[order, order]                                     # token-only block for peer averages
    for d in tokens:
        peers = Ctok.loc[d["symbol"]].drop(d["symbol"])
        d["avg_corr_peers"] = round(float(peers.mean()) if len(peers) else 0.0, 4)

    # per-window vol RANKINGS + top-8 pools — the evolving-universe view. A window longer than the
    # available history falls back to the full panel (rankings converge to ann_vol order).
    vol_rankings = {}
    for k in windows:
        ranked = sorted(tokens, key=lambda d: d["vol_by_window"][k], reverse=True)
        vol_rankings[k] = {
            "ranked": [{"symbol": d["symbol"], "ann_vol": d["vol_by_window"][k],
                        "rank": i + 1} for i, d in enumerate(ranked)],
            "top8": [d["symbol"] for d in ranked[:8]],
        }

    n = len(tokens)
    offdiag = Ctok.to_numpy()[np.triu_indices(n, 1)] if n > 1 else np.array([0.0])
    uni_ew = float(np.mean([d["ret_window"] for d in tokens])) if tokens else 0.0
    label = "bull" if uni_ew > 0.10 else "bear" if uni_ew < -0.10 else "flat"

    return {
        "generated": generated or datetime.now(timezone.utc).isoformat(),
        "window": {"start": _to_secs(idx[0]), "end": _to_secs(idx[-1]), "bars": int(len(idx)),
                   "hours_per_year": hours_per_year, "vol_window": vol_spark_window,
                   "excursion_window": excursion_window},
        "btc": {"ret_window": round(float((1.0 + btc_ret).prod() - 1.0), 4),
                "ann_vol": round(_ann_vol(btc_ret, hours_per_year), 4)},
        "tokens": tokens,
        "vol_rankings": vol_rankings,
        "correlation": {"symbols": labels, "matrix": matrix},
        "summary": {"n_tokens": n,
                    "avg_pairwise_corr": round(float(offdiag.mean()), 4),
                    "median_pairwise_corr": round(float(np.median(offdiag)), 4),
                    "max_pairwise_corr": round(float(offdiag.max()), 4),
                    "universe_ew_return": round(uni_ew, 4), "regime_label": label},
    }
