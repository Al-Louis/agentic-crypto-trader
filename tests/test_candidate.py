"""Tests for the codified strategy candidate."""

import numpy as np
import pandas as pd
import pytest

from trader.strategy import OVERLAYS, build_candidate, select_vol_tokens


def _returns(n=300, seed=0):
    rng = np.random.default_rng(seed)
    # B and D given higher volatility -> should be selected by the vol tilt
    scale = {"A": 0.005, "B": 0.05, "C": 0.005, "D": 0.05, "E": 0.01}
    idx = np.arange(n) * 3600
    return pd.DataFrame({s: rng.normal(0, v, n) for s, v in scale.items()}, index=idx)


def _btc_close(returns):
    # rising BTC over the panel's timestamps
    return pd.Series(np.linspace(100, 200, len(returns)), index=returns.index)


def test_select_vol_tokens_picks_highest_vol():
    picks = select_vol_tokens(_returns(), k=2)
    assert set(picks) == {"B", "D"}


def test_overlay_none_is_plain_subset():
    ret = _returns()
    wf = build_candidate(ret, k=2, overlay="none")
    w = wf(ret)
    assert set(w.index) == {"B", "D"}
    assert abs(w.sum() - 1.0) < 1e-12


def test_overlay_scales_exposure_in_risk_on():
    ret = _returns()
    wf = build_candidate(ret, _btc_close(ret), k=2, overlay="trend50")
    w = wf(ret)                                   # last bar: BTC rising -> risk-on -> full
    assert abs(w.sum() - 1.0) < 1e-9


def test_unknown_overlay_raises():
    with pytest.raises(ValueError):
        build_candidate(_returns(), overlay="bogus")


def test_overlay_requires_btc_close():
    with pytest.raises(ValueError):
        build_candidate(_returns(), overlay="stress50")   # no btc_close


def test_default_overlay_is_in_set():
    from trader.strategy import DEFAULT_OVERLAY
    assert DEFAULT_OVERLAY in OVERLAYS
