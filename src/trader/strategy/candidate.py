"""The validated strategy candidate — daily-rebalanced volatility tilt + regime overlay.

Assembled from the full research loop (vault "Trading Strategies" / "Token Universe"):

  - **Universe:** equal-weight the `k` highest-realized-volatility eligible tokens. The
    volatility tilt is OOS-validated (it ~doubles the tournament contender rate vs the passive
    baseline on held-out windows; vol-rank persists Spearman +0.66). Volatility tilt ≫ beta tilt.
  - **Overlay:** scale exposure by a BTC regime gate. The sweep frontier:
      * `none`        — pure bull bet (best raw tournament rate; no bear insurance).
      * `stress50`    — **default/best design:** extreme-stress-only (de-risk to 50% only on a
                        deep trailing BTC drop). Keeps ~full upside and insures the crash tail —
                        but the crash protection is **UNVALIDATED** in the crash-free sample
                        (pending a synthetic-crash stress test).
      * `trend50`     — visible/validated hedge: half de-risk below the BTC trend EMA. Costs
                        ~6 pts of tournament rate but demonstrably drives bear-week DQ to 0%.
      * `stresscash` / `trendcash` — full-cash variants (more insurance, more upside cost).
  - **Compliant:** the backtester rebalances daily → satisfies the ≥1-trade/day activity rule
    (buy-and-hold would be disqualified). Daily rebalancing also trims drawdown.

This is the decision core behind the strategy interface; execution, custody, and guardrails
stay separate (vault "Security and Encryption"). `build_candidate` returns a `weights_fn` ready
for `trader.sim.backtest` / live wiring.
"""

from __future__ import annotations

import pandas as pd

from trader.features.regime import stress_exposure, trend_exposure
from trader.sim.strategies import regime_scaled, static_subset

OVERLAYS = ("none", "stress50", "stresscash", "trend50", "trendcash")
DEFAULT_OVERLAY = "stress50"


def select_vol_tokens(returns: pd.DataFrame, k: int = 8) -> list[str]:
    """The `k` highest realized-volatility tokens — the validated tilt.

    Live, compute this from recent *pre-competition* data (vol-rank is persistent).
    """
    return list(returns.std().sort_values(ascending=False).head(k).index)


def _exposure(btc_close: pd.Series, overlay: str) -> pd.Series:
    if overlay == "stress50":
        return stress_exposure(btc_close, window=72, drop=-0.08, off=0.5)
    if overlay == "stresscash":
        return stress_exposure(btc_close, window=72, drop=-0.10, off=0.0)
    if overlay == "trend50":
        return trend_exposure(btc_close, ema_span=72, off=0.5)
    if overlay == "trendcash":
        return trend_exposure(btc_close, ema_span=72, off=0.0)
    raise ValueError(f"unknown overlay {overlay!r}; choose from {OVERLAYS}")


def build_candidate(returns: pd.DataFrame, btc_close: pd.Series | None = None,
                    k: int = 8, overlay: str = DEFAULT_OVERLAY,
                    tokens: list[str] | None = None):
    """Return a backtester `weights_fn` for the candidate strategy.

    Args:
        returns: alt returns panel (used to rank volatility, unless `tokens` is given).
        btc_close: BTC close series, indexed in the **same timestamp units as `returns`**
            (required for any overlay other than `none`).
        k: number of highest-volatility tokens to hold (default 8).
        overlay: one of `OVERLAYS`.
        tokens: optional explicit token list (overrides the volatility selection).
    """
    if overlay not in OVERLAYS:
        raise ValueError(f"unknown overlay {overlay!r}; choose from {OVERLAYS}")
    base = static_subset(tokens if tokens is not None else select_vol_tokens(returns, k))
    if overlay == "none":
        return base
    if btc_close is None:
        raise ValueError(f"overlay {overlay!r} requires btc_close")
    exposure = _exposure(btc_close, overlay).reindex(returns.index, method="ffill").fillna(0.0)
    return regime_scaled(base, exposure)
