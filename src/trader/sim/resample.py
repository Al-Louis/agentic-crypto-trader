"""7-day-window resampling — the competition-relevant evaluation.

The contest scores a single **7-day** window, so a strategy's 7-month return/drawdown is the
wrong statistic. This samples many random 7-day windows (each with a trailing warmup so the
strategy's weights stay causal), runs each through the cost-aware backtester, and returns the
**distribution** of weekly return and weekly max-drawdown — most importantly **P(breach the
30% DQ gate)**, the real disqualification risk (vault "Simulated Market": the single-week
variance question).

Windows overlap (≈7 months of data ⇒ only ~30 *independent* weeks), so the distribution is
smooth but autocorrelated — read it as a shape, not n independent draws.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from trader.sim.backtest import run_xs_backtest
from trader.sim.metrics import PerformanceMetrics

WEEK_BARS = 168       # 7 days of hourly bars
DQ_THRESHOLD = 0.30   # competition max-drawdown gate


def sample_window_starts(n_bars: int, window: int, warmup: int, n_samples: int,
                         rng: np.random.Generator) -> np.ndarray:
    """Random eval-start indices `s`; each window uses the slice `[s-warmup, s+window)`."""
    lo, hi = warmup, n_bars - window
    if hi <= lo:
        return np.array([], dtype=int)
    return rng.integers(lo, hi, size=n_samples)


def evaluate_windows(returns: pd.DataFrame, weights_fn, liquidity: dict,
                     window: int = WEEK_BARS, warmup: int = WEEK_BARS,
                     rebalance_every: int = 24, n_samples: int = 300,
                     capital: float = 10_000.0, dq_threshold: float = DQ_THRESHOLD,
                     seed: int = 0) -> pd.DataFrame:
    """Per-window metrics: `[ret, maxdd, dq, profit]` for `n_samples` random 7-day windows."""
    rng = np.random.default_rng(seed)
    starts = sample_window_starts(len(returns), window, warmup, n_samples, rng)
    rows = []
    for s in starts:
        sl = returns.iloc[s - warmup: s + window]
        out = run_xs_backtest(sl, weights_fn, liquidity, capital=capital,
                              warmup=warmup, rebalance_every=rebalance_every)
        eq = out["equity"].to_numpy()
        ret = eq[-1] / capital - 1.0
        mdd = PerformanceMetrics._max_drawdown(eq)        # warmup is flat at capital (1st peak)
        rows.append({"ret": ret, "maxdd": mdd, "dq": mdd > dq_threshold, "profit": ret > 0})
    return pd.DataFrame(rows)


def summarize(df: pd.DataFrame, label: str) -> dict:
    """Distribution summary for one strategy's windows."""
    if df.empty:
        return {"strategy": label, "n": 0}
    survived = df[~df["dq"]]
    return {
        "strategy": label, "n": len(df),
        "ret_med": float(df["ret"].median()),
        "ret_p5": float(df["ret"].quantile(0.05)),
        "ret_p95": float(df["ret"].quantile(0.95)),
        "maxdd_med": float(df["maxdd"].median()),
        "maxdd_p95": float(df["maxdd"].quantile(0.95)),
        "p_dq": float(df["dq"].mean()),
        "p_profit": float(df["profit"].mean()),
        "ret_med_survived": float(survived["ret"].median()) if len(survived) else float("nan"),
    }
