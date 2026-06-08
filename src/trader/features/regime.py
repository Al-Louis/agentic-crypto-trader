"""Market regime signal — a crude, explainable risk-on/off gate from the BTC anchor.

Per the TradeSim post-mortem (favor robust, explainable buckets over a tuned classifier),
risk-on = BTC trading **above its trailing EMA** (a classic trend filter). Causal. Feeds the
regime overlay (vault "Market Conditions" / "Trading Strategies") that holds the volatility
tilt in risk-on weeks and rotates to stables in risk-off — insuring the (under-sampled) bear
case while keeping the validated upside.
"""

from __future__ import annotations

import pandas as pd


def btc_risk_on(close: pd.Series, ema_span: int = 72) -> pd.Series:
    """Risk-on (True) when BTC close > its trailing EMA(`ema_span`); warmup → False (risk-off).

    A 72-bar (≈3-day) EMA on hourly data responds within a weekly window; longer = slower.
    """
    ema = close.ewm(span=ema_span, adjust=False, min_periods=ema_span).mean()
    return (close > ema).fillna(False)
