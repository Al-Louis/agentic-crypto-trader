"""Rung-0 disciplined trend-hold: the core behavior must be ride-the-runup then stand-aside.

The strategy splits into a stateless `build_rung0` signal (ignition primitives) and the
`run_rung0` executor (held-state, cash, rotation). We test the signal's ignition directly and
the executor's observable behavior (buys the runup, never re-buys the dead-zone).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from trader.strategy.rung0 import build_rung0, run_rung0


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


def _signal(returns, volume):
    return build_rung0(returns, k=1, ema_span=10, volume=volume, vol_mult=2.5, vol_spike=4,
                       vol_base=20, vol_fast=4)


def _ignite_bars(sig_fn, returns, lo=30):
    return [i for i in range(lo, len(returns)) if sig_fn(returns.iloc[: i + 1]).get("RUN", {}).get("ignite")]


def _run(returns, volume, **kw):
    liq = {c: 1e9 for c in returns.columns}
    return run_rung0(returns, _signal(returns, volume), liq, warmup=30, cooldown=8, **kw)


def _buy_bars(records, token):
    return [int(rec["time"] / 3600) for rec in records if rec["trades_usd"].get(token, 0.0) > 0]


def test_rung0_ignites_on_volume_then_stands_aside():
    returns, volume = _runup_then_bleed()
    bars = _ignite_bars(_signal(returns, volume), returns)
    assert any(48 <= b <= 100 for b in bars), "must ignite on the volume spike + run"
    assert not any(b >= 140 for b in bars), "no spike in the dead-zone → no ignition"


def test_rung0_no_ignition_without_volume_spike():
    returns, volume = _runup_then_bleed()
    flat = volume.copy()
    flat["RUN"] = 100.0                               # remove the spike: same runup, no ignition
    assert not _ignite_bars(_signal(returns, flat), returns), "no spike → never ignites"


def test_run_rung0_buys_runup_then_stands_aside():
    returns, volume = _runup_then_bleed()
    eq, records, fees = _run(returns, volume, stop_k=0.1)
    buys = _buy_bars(records, "RUN")
    assert any(48 <= b <= 100 for b in buys), "should buy RUN on the ignition"
    assert not any(b >= 140 for b in buys), "should not re-buy RUN in the dead-zone"
    assert eq.iloc[-1] != eq.iloc[0], "equity should move once it trades"


def test_run_rung0_phantom_held_fix_only_holds_when_funded():
    """An ignition that can't be funded (no cash, no rotation) must NOT flip the token to held —
    so it stays eligible to trade later instead of phantom-holding."""
    returns, volume = _runup_then_bleed()
    liq = {c: 1e9 for c in returns.columns}
    # capital below the per-trade floor + rotation off: the buy can never fund
    eq, records, fees = run_rung0(returns, _signal(returns, volume), liq, capital=0.5,
                                  warmup=30, cooldown=8, rotate=False)
    assert not any(rec["trades_usd"] for rec in records), "no cash → no funded trade, no phantom hold"
