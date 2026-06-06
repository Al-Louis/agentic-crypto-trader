"""Information Coefficient (IC) — the cheap honest test of a cross-sectional signal.

IC = the per-timestamp rank correlation between a signal across the universe and *forward*
returns across the universe. A mean IC ≈ 0 (t-stat < 2) means no edge — the cheapest gate
to run *before* building a backtest (vault "Simulated Market" honesty discipline; the
TradeSim post-mortem's "gate every signal" lesson).

Honesty note: overlapping forward windows autocorrelate the IC series and inflate the
t-stat, so significance is computed on a **non-overlapping** subsample.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def forward_return(returns: pd.Series, k: int) -> pd.Series:
    """Sum of the next k (log) returns: `fwd[t] = r[t+1] + ... + r[t+k]`. Tail → NaN."""
    return returns.rolling(k).sum().shift(-k)


def cross_sectional_ic(signal: pd.DataFrame, fwd: pd.DataFrame,
                       method: str = "spearman", min_names: int = 8) -> pd.Series:
    """Per-timestamp rank correlation between `signal` and `fwd` across columns (symbols).

    Both are timestamp × symbol. Rows with fewer than `min_names` paired values are skipped.
    """
    idx = signal.index.intersection(fwd.index)
    out: dict = {}
    for ts in idx:
        s, f = signal.loc[ts], fwd.loc[ts]
        m = s.notna() & f.notna()
        if int(m.sum()) < min_names:
            continue
        sm, fm = s[m], f[m]
        # Spearman == Pearson on ranks — avoids a scipy dependency for pandas' spearman.
        out[ts] = (sm.rank().corr(fm.rank()) if method == "spearman"
                   else sm.corr(fm, method=method))
    return pd.Series(out, dtype=float).dropna()


def ic_summary(ic: pd.Series, horizon_bars: int) -> dict:
    """Aggregate IC stats; the t-stat uses a non-overlapping subsample (honest significance)."""
    ic = ic.dropna()
    n = len(ic)
    if n == 0:
        return {"n": 0, "n_indep": 0, "mean_ic": float("nan"), "ic_ir": float("nan"),
                "t_stat": float("nan"), "hit_rate": float("nan")}
    mean, std = float(ic.mean()), float(ic.std())
    sub = ic.iloc[::max(horizon_bars, 1)]                       # non-overlapping samples
    t = (float(sub.mean()) / (float(sub.std()) / np.sqrt(len(sub)))
         if len(sub) > 1 and sub.std() > 0 else float("nan"))
    return {"n": n, "n_indep": len(sub), "mean_ic": mean,
            "ic_ir": mean / std if std > 0 else float("nan"),
            "t_stat": t, "hit_rate": float((ic > 0).mean())}
