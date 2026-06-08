"""Baseline cross-sectional weight functions for the backtester.

Each takes the trailing returns panel (`hist`, causal — through the decision bar) and
returns target weights. The honest bar to beat is Buy & Hold; the rest probe whether any
simple cross-sectional tilt survives AMM costs.
"""

from __future__ import annotations

import pandas as pd


def _available(hist: pd.DataFrame) -> list[str]:
    """Symbols with a valid most-recent return (tradeable at the decision bar)."""
    last = hist.iloc[-1]
    return [c for c in hist.columns if pd.notna(last[c])]


def equal_weight(hist: pd.DataFrame) -> pd.Series:
    """Equal weight across currently-tradeable symbols (Buy&Hold if rebalanced once)."""
    syms = _available(hist)
    n = len(syms)
    return pd.Series({s: 1.0 / n for s in syms}) if n else pd.Series(dtype=float)


def _ranked(hist: pd.DataFrame, lookback: int, k: int, ascending: bool) -> pd.Series:
    score = hist.iloc[-lookback:].sum().dropna().sort_values(ascending=ascending)
    picks = score.head(k).index
    return pd.Series({s: 1.0 / len(picks) for s in picks}) if len(picks) else pd.Series(dtype=float)


def xs_momentum(hist: pd.DataFrame, lookback: int = 24, k: int = 5) -> pd.Series:
    """Long the top-k by trailing `lookback`-bar return (continuation)."""
    return _ranked(hist, lookback, k, ascending=False)


def xs_reversal(hist: pd.DataFrame, lookback: int = 24, k: int = 5) -> pd.Series:
    """Long the bottom-k by trailing return (the IC-suggested mean reversion)."""
    return _ranked(hist, lookback, k, ascending=True)


def static_subset(symbols):
    """Weights-fn factory: equal weight over a **fixed** subset, held (ignores history).

    The low-turnover way to take a concentration / volatility / beta tilt — entry-timing
    re-ranking is dead here (it churns thin pools), so the lever is *which fixed set to hold*.
    """
    sset = list(dict.fromkeys(symbols))

    def weights(hist: pd.DataFrame) -> pd.Series:
        last = hist.iloc[-1]
        avail = [c for c in sset if c in hist.columns and pd.notna(last.get(c))]
        n = len(avail)
        return pd.Series({s: 1.0 / n for s in avail}) if n else pd.Series(dtype=float)

    return weights


def regime_gated(base_fn, risk_on: pd.Series):
    """Gate a base weights-fn by a risk-on series (indexed by timestamp).

    Risk-on → the base weights (e.g. the vol tilt); risk-off → **cash** (empty weights). The
    daily rebalance still occurs (the agent forces a daily ping trade), so ≥1-trade/day holds
    even in a sustained risk-off; the realized cost in cash is ~gas (negligible).
    """
    def weights(hist: pd.DataFrame) -> pd.Series:
        t = hist.index[-1]
        on = bool(risk_on.get(t, False))            # default risk-off if the bar is unknown
        return base_fn(hist) if on else pd.Series(dtype=float)

    return weights


def regime_scaled(base_fn, exposure: pd.Series):
    """Scale the base weights by a continuous exposure series ∈ [0,1] (the rest → cash).

    The refined overlay: exposure 1 = full tilt, 0.5 = half tilt + half cash, 0 = all cash.
    Daily rebalance still satisfies ≥1 trade/day.
    """
    def weights(hist: pd.DataFrame) -> pd.Series:
        e = float(exposure.get(hist.index[-1], 0.0))
        return base_fn(hist) * e

    return weights
