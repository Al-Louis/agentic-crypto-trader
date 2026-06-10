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


def inject_crash(returns: pd.DataFrame, *, at: int, duration: int = 8, total_drop: float = -0.6,
                 beta: float = 1.4, resid_std: float = 0.03, shape: str = "sharp",
                 tokens=None, seed: int = 0) -> pd.DataFrame:
    """Splice a synthetic ALT-crash into `returns` (a COPY): over `duration` bars starting at bar
    index `at`, the traded alts sell off TOGETHER — in a real liquidation cascade idiosyncratic
    structure collapses and correlations spike toward 1 — as `beta`·(systemic drop path) + per-token
    noise. The bull/flat historical sample has no alt-crash (only BTC fell; the alts pumped), so this
    is the scenario where a regime-adaptive policy's de-risking can finally be MEASURED. The realized
    alt drop ≈ `beta`·`total_drop` (e.g. 1.4·−0.6 ≈ −84%, SIREN-scale). [[AI Training]]"""
    out = returns.copy()
    cols = list(returns.columns) if tokens is None else [t for t in tokens if t in returns.columns]
    n = len(out)
    end = min(at + duration, n)
    idx = out.index[at:end]
    d = len(idx)
    sys_path = crash_path(d, total_drop, shape)               # the shared systemic drop
    rng = np.random.default_rng(seed)
    for t in cols:
        out.loc[idx, t] = beta * sys_path + rng.normal(0, resid_std, d)
    return out


def inject_random_crashes(returns: pd.DataFrame, *, n_crashes: int, rng, min_gap: int = 200,
                          **crash_kw) -> tuple:
    """Inject `n_crashes` non-overlapping crashes at random bars — TRAINING augmentation so the agent
    SEES crashes and can learn to de-risk into low breadth. Returns (augmented_returns, [bar indices])."""
    out = returns.copy()
    n = len(out)
    margin = crash_kw.get("duration", 8) * 3
    placed: list[int] = []
    for _ in range(n_crashes * 50):
        if len(placed) >= n_crashes:
            break
        at = int(rng.integers(margin, max(margin + 1, n - margin)))
        if all(abs(at - p) > min_gap for p in placed):
            out = inject_crash(out, at=at, seed=int(rng.integers(0, 1_000_000)), **crash_kw)
            placed.append(at)
    return out, sorted(placed)
