"""Factor model — the "Bitcoin-is-King" residual / divergence features.

For each alt, a causal **two-factor** model

    r_alt = alpha + beta_btc * r_btc + beta_bnb * r_bnb + eps

with time-varying (rolling) betas. The residual **eps** is the idiosyncratic return — a
positive residual while the market bleeds is *hidden strength* (accumulation); its
**cross-sectional rank** across the universe is the selection signal (vault "Trading
Strategies"). **R²** measures how factor-driven (vs dev-controlled) a token is.

**Causal by construction:** betas at time *t* are fit on the trailing window `[t-W, t-1]`
and applied to predict *t*; the residual at *t* uses only data ≤ *t*. **Sparse** alt series
are reindexed onto the **dense anchor grid** (forward-filled), so alt returns align 1:1 with
BTC/BNB instead of silently spanning gaps.

Inputs must share a timestamp unit — the build script normalizes the ms anchor to the
seconds-based alt grid before calling here.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def log_returns(close: pd.Series) -> pd.Series:
    return np.log(close / close.shift(1))


def align_returns(alt: pd.DataFrame, btc: pd.DataFrame, bnb: pd.DataFrame) -> pd.DataFrame:
    """Align alt/BTC/BNB closes onto the dense anchor (BTC) grid over their overlap.

    The sparse alt series is forward-filled onto the grid (an untraded bar carries the last
    price → 0 return), so all three return series share one timeline.
    Returns `[timestamp, r_alt, r_btc, r_bnb]` (NaN warmup dropped).
    """
    def closes(df: pd.DataFrame) -> pd.Series:
        return df.set_index("timestamp")["close"].sort_index()

    a, b, n = closes(alt), closes(btc), closes(bnb)
    if a.empty or b.empty or n.empty:
        return pd.DataFrame(columns=["timestamp", "r_alt", "r_btc", "r_bnb"])

    lo = max(a.index.min(), b.index.min(), n.index.min())
    hi = min(a.index.max(), b.index.max(), n.index.max())
    grid = b.index[(b.index >= lo) & (b.index <= hi)]          # dense BTC grid over overlap

    a = a.reindex(grid, method="ffill")
    n = n.reindex(grid, method="ffill")
    b = b.reindex(grid)

    out = pd.DataFrame({
        "timestamp": np.asarray(grid),
        "r_alt": log_returns(a).to_numpy(),
        "r_btc": log_returns(b).to_numpy(),
        "r_bnb": log_returns(n).to_numpy(),
    })
    return out.dropna().reset_index(drop=True)


def rolling_factor(ret: pd.DataFrame, window: int) -> pd.DataFrame:
    """Causal rolling two-factor OLS.

    For each row *i* with enough history, betas are fit on `[i-window, i-1]` and applied to
    predict row *i*; `residual[i] = r_alt[i] - predicted[i]`. Adds columns
    `alpha, beta_btc, beta_bnb, r2, predicted, residual` (NaN until warmup).
    """
    y = ret["r_alt"].to_numpy(float)
    X = np.column_stack([np.ones(len(ret)),
                         ret["r_btc"].to_numpy(float), ret["r_bnb"].to_numpy(float)])
    n = len(ret)
    alpha = np.full(n, np.nan)
    bbtc = np.full(n, np.nan)
    bbnb = np.full(n, np.nan)
    r2 = np.full(n, np.nan)
    pred = np.full(n, np.nan)
    resid = np.full(n, np.nan)

    for i in range(window, n):
        Xw, yw = X[i - window:i], y[i - window:i]               # trailing window, excludes i
        coef, *_ = np.linalg.lstsq(Xw, yw, rcond=None)
        alpha[i], bbtc[i], bbnb[i] = coef
        ss_res = float(np.sum((yw - Xw @ coef) ** 2))
        ss_tot = float(np.sum((yw - yw.mean()) ** 2))
        r2[i] = 1.0 - ss_res / ss_tot if ss_tot > 1e-18 else np.nan
        pred[i] = float(X[i] @ coef)                            # predict current from past betas
        resid[i] = y[i] - pred[i]

    out = ret.copy()
    out["alpha"], out["beta_btc"], out["beta_bnb"] = alpha, bbtc, bbnb
    out["r2"], out["predicted"], out["residual"] = r2, pred, resid
    return out


def residual_momentum(residual: pd.Series, span: int) -> pd.Series:
    """EWMA of the residual — persistent idiosyncratic strength (causal)."""
    return residual.ewm(span=span, adjust=False, min_periods=span).mean()


def compute_factor_features(alt: pd.DataFrame, btc: pd.DataFrame, bnb: pd.DataFrame,
                            window: int = 168, mom_span: int = 24) -> pd.DataFrame:
    """Full per-alt pipeline: align → rolling two-factor OLS → residual momentum."""
    fac = rolling_factor(align_returns(alt, btc, bnb), window)
    fac["resid_mom"] = residual_momentum(fac["residual"], mom_span)
    return fac


def cross_sectional_zscore(panel: dict[str, pd.Series]) -> pd.DataFrame:
    """Per-timestamp standardization across symbols — the cross-sectional selection signal.

    `panel`: `{symbol: series indexed by timestamp}`. Returns a timestamp × symbol z-score
    matrix; a high z = idiosyncratic strength relative to the universe *right now*.
    """
    wide = pd.DataFrame(panel)
    mu = wide.mean(axis=1)
    sd = wide.std(axis=1).replace(0, np.nan)
    return wide.sub(mu, axis=0).div(sd, axis=0)
