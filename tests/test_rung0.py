"""Rung-0 disciplined trend-hold: the core behavior must be ride-the-runup then stand-aside."""

from __future__ import annotations

import numpy as np
import pandas as pd

from trader.strategy.rung0 import build_rung0


def _runup_then_bleed(n=240):
    """A token that runs up 1.0→~3.0, crashes back below its origin, then chops sideways (dead-zone),
    plus two low-vol fillers so the vol-top-k picks the eventful one."""
    px = np.ones(n)
    for i in range(50, 90):          # runup
        px[i] = px[i - 1] * 1.03
    for i in range(90, 120):         # rollover / crash back below origin
        px[i] = px[i - 1] * 0.95
    for i in range(120, n):          # dead-zone: sideways below the runup origin
        px[i] = px[i - 1] * (1.002 if i % 2 else 0.998)
    idx = pd.RangeIndex(n) * 3600
    rng = np.random.default_rng(0)
    return pd.DataFrame({
        "RUN": pd.Series(px, index=idx).pct_change().fillna(0.0),
        "CALM1": pd.Series(rng.normal(0, 0.002, n), index=idx),
        "CALM2": pd.Series(rng.normal(0, 0.002, n), index=idx),
    })


def test_rung0_rides_runup_then_stands_aside():
    returns = _runup_then_bleed()
    fn = build_rung0(returns, k=1, ema_span=10, breakout_n=10, stop_k=0.1, cooldown=2)
    held = []
    for i in range(30, len(returns), 5):              # mimic the backtester's per-rebalance calls
        w = fn(returns.iloc[: i + 1])
        held.append((i, float(w.get("RUN", 0.0)) if len(w) else 0.0))

    assert any(w > 0 for i, w in held if 50 <= i <= 100), "must hold during the runup (let it run)"
    deadzone = [w for i, w in held if i >= 140]
    assert all(w == 0 for w in deadzone), f"must stand aside in the dead-zone, got {deadzone}"


def test_rung0_caps_weight_and_exits_to_cash():
    returns = _runup_then_bleed()
    fn = build_rung0(returns, k=1, max_weight=0.25)
    seen_hold = seen_cash = False
    for i in range(30, len(returns), 5):
        w = fn(returns.iloc[: i + 1])
        tot = float(w.sum()) if len(w) else 0.0
        assert tot <= 0.25 + 1e-9, "per-token cap (k=1) bounds total weight"
        seen_hold |= tot > 0
        seen_cash |= tot == 0
    assert seen_hold and seen_cash, "should both hold (runup) and sit in cash (dead-zone)"
