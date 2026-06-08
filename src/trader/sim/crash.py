"""Synthetic-crash stress test — validate the overlay's tail protection.

The bull-conditioned sample has no real crash, so the stress overlay sits dormant and its
protection is unvalidated. This constructs a crash week — BTC drops `total_drop`, and the
high-volatility alts **amplify** it via a stress beta + idiosyncratic noise (correlations
spike toward 1 in a real selloff) — spliced after real warmup so the regime gate has genuine
pre-crash context. The script then checks whether `stress50` / `trend50` cap the drawdown/DQ
where the ungated tilt blows the 30% gate.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def crash_path(n_bars: int, total_drop: float, shape: str = "linear") -> np.ndarray:
    """Per-bar simple returns producing ~`total_drop` over `n_bars`.

    shape: 'linear' (a steady bleed — the harder case for a trailing-return gate) or
    'sharp' (the whole drop front-loaded into the first day, then flat).
    """
    if shape == "sharp":
        out = np.zeros(n_bars)
        d = max(1, n_bars // 7)                                   # whole drop over ~day 1
        out[:d] = (1.0 + total_drop) ** (1.0 / d) - 1.0
        return out
    return np.full(n_bars, (1.0 + total_drop) ** (1.0 / n_bars) - 1.0)


def simulate_crash_panel(tokens, btc_returns, beta: float, resid_std: dict,
                         seed: int = 0) -> pd.DataFrame:
    """Alt returns over a crash = `beta`·BTC + idiosyncratic noise per token.

    A single stress `beta` for all (in a crash, idiosyncratic structure collapses and
    everything sells off together); `resid_std` adds token-specific noise.
    """
    rng = np.random.default_rng(seed)
    btc = np.asarray(btc_returns, dtype=float)
    return pd.DataFrame({s: beta * btc + rng.normal(0, resid_std.get(s, 0.01), len(btc))
                         for s in tokens})
