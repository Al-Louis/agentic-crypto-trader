"""Market regime signal — a crude, explainable risk-on/off gate from the BTC anchor.

Per the TradeSim post-mortem (favor robust, explainable buckets over a tuned classifier),
risk-on = BTC trading **above its trailing EMA** (a classic trend filter). Causal. Feeds the
regime overlay (vault "Market Conditions" / "Trading Strategies") that holds the volatility
tilt in risk-on weeks and rotates to stables in risk-off — insuring the (under-sampled) bear
case while keeping the validated upside.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def btc_risk_on(close: pd.Series, ema_span: int = 72) -> pd.Series:
    """Risk-on (True) when BTC close > its trailing EMA(`ema_span`); warmup → False (risk-off).

    A 72-bar (≈3-day) EMA on hourly data responds within a weekly window; longer = slower.
    """
    ema = close.ewm(span=ema_span, adjust=False, min_periods=ema_span).mean()
    return (close > ema).fillna(False)


def trend_exposure(close: pd.Series, ema_span: int = 72, off: float = 0.5,
                   band: float = 0.0) -> pd.Series:
    """Continuous exposure ∈ {off, 1}: full above EMA·(1−band), `off` below (warmup → off).

    The refined trend gate: `off=0.0` = full cash (the blunt original), `off=0.5` = half de-risk
    (keep half the upside in false-off periods); `band` adds a dead-zone to cut whipsaws.
    """
    ema = close.ewm(span=ema_span, adjust=False, min_periods=ema_span).mean()
    below = (close < ema * (1.0 - band)).fillna(True)
    return pd.Series(np.where(below, off, 1.0), index=close.index)


def stress_exposure(close: pd.Series, window: int = 72, drop: float = -0.08,
                    off: float = 0.0) -> pd.Series:
    """Extreme-stress-only gate: stay **fully invested** unless BTC fell more than `drop` over
    the trailing `window` bars, then cut to `off`. De-risks only in genuine stress, so it keeps
    the upside the trend gate sacrifices."""
    stressed = (close.pct_change(window) < drop).fillna(False)
    return pd.Series(np.where(stressed, off, 1.0), index=close.index)


def severity_exposure(close: pd.Series, window: int = 72, soft: float = -0.05,
                      hard: float = -0.20, floor: float = 0.0) -> pd.Series:
    """Severity-scaled gate — the design the crash test pointed to.

    Exposure scales **smoothly** with BTC's trailing `window`-bar return: full (1.0) while the
    drop is shallower than `soft`, then ramps down linearly to `floor` (default full cash) at
    `hard`. This fixes both flaws of the binary gates at once — **dormant in calm** (keeps the
    upside the trend gate sacrifices) yet **reaches full cash in a deep crash** (which the
    half-exposure gates can't survive). Warmup → fully invested. (vault "Trading Strategies".)
    """
    trail = close.pct_change(window)
    raw = ((trail - hard) / (soft - hard)).clip(lower=0.0, upper=1.0)
    return (floor + (1.0 - floor) * raw).fillna(1.0)
