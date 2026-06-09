"""Rung-0 disciplined trend-hold: the core behavior must be ride-the-runup then stand-aside."""

from __future__ import annotations

import numpy as np
import pandas as pd

from trader.strategy.rung0 import build_rung0


def _runup_then_bleed(n=240):
    """A token that runs up 1.0→~3.0 (with a volume spike igniting it), crashes back below its origin,
    then chops sideways (dead-zone, no volume spike), plus two low-vol fillers."""
    px = np.ones(n)
    for i in range(50, 90):          # runup
        px[i] = px[i - 1] * 1.03
    for i in range(90, 120):         # rollover / crash back below origin
        px[i] = px[i - 1] * 0.95
    for i in range(120, n):          # dead-zone: sideways below the runup origin
        px[i] = px[i - 1] * (1.002 if i % 2 else 0.998)
    idx = pd.RangeIndex(n) * 3600
    rng = np.random.default_rng(0)
    returns = pd.DataFrame({
        "RUN": pd.Series(px, index=idx).pct_change().fillna(0.0),
        "CALM1": pd.Series(rng.normal(0, 0.002, n), index=idx),
        "CALM2": pd.Series(rng.normal(0, 0.002, n), index=idx),
    })
    vol_run = np.full(n, 100.0)
    vol_run[48:58] = 400.0           # a 4x volume spike igniting the runup; flat (no spike) elsewhere
    volume = pd.DataFrame({"RUN": pd.Series(vol_run, index=idx),
                           "CALM1": pd.Series(100.0, index=idx),
                           "CALM2": pd.Series(100.0, index=idx)})
    return returns, volume


def _fn(returns, volume, **kw):
    return build_rung0(returns, k=1, ema_span=10, volume=volume, vol_mult=2.5, vol_spike=4,
                       vol_base=20, **kw)


def test_rung0_ignites_on_volume_then_stands_aside():
    returns, volume = _runup_then_bleed()
    fn = _fn(returns, volume, stop_k=0.1, cooldown=2)
    held = []
    for i in range(30, len(returns), 5):              # mimic the backtester's per-rebalance calls
        w = fn(returns.iloc[: i + 1])
        held.append((i, float(w.get("RUN", 0.0)) if len(w) else 0.0))

    assert any(w > 0 for i, w in held if 48 <= i <= 100), "must enter on the volume spike + run"
    deadzone = [w for i, w in held if i >= 140]
    assert all(w == 0 for w in deadzone), f"no volume spike in the dead-zone → stand aside, got {deadzone}"


def test_rung0_no_entry_without_volume_spike():
    returns, volume = _runup_then_bleed()
    flat = volume.copy()
    flat["RUN"] = 100.0                               # remove the spike: same runup, no ignition
    fn = _fn(returns, flat, stop_k=0.1, cooldown=2)
    for i in range(30, len(returns), 5):
        w = fn(returns.iloc[: i + 1])
        assert (float(w.get("RUN", 0.0)) if len(w) else 0.0) == 0.0, "no spike → never enters"
