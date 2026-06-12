"""market_metrics.json producer — the volatility/correlation dashboard contract. Pins the shape
the frontend reads and the three facts it must surface: per-token vol ordering, token<->BTC
decoupling (corr/beta), and a symmetric token x token correlation matrix incl. BTC."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from trader.report.market_metrics import compute_market_metrics


def _panel(n=400):
    """WILD (high-vol, BTC-independent), BTCISH (tracks BTC), CALM (low-vol) + a BTC price series."""
    idx = 1_700_000_000 + np.arange(n) * 3600
    rng = np.random.default_rng(0)
    btc_ret = rng.normal(0, 0.012, n)
    btc = pd.Series(np.cumprod(1 + btc_ret) * 1e4, index=idx)
    cols = {
        "WILD": pd.Series(rng.normal(0, 0.060, n), index=idx),       # most volatile, decoupled
        "BTCISH": pd.Series(0.9 * btc_ret + rng.normal(0, 0.002, n), index=idx),  # tracks BTC
        "CALM": pd.Series(rng.normal(0, 0.003, n), index=idx),       # least volatile
    }
    return pd.DataFrame(cols), btc


def _metrics():
    r, btc = _panel()
    return compute_market_metrics(r, btc, generated="2026-06-10T00:00:00+00:00")


def test_shape_and_keys():
    m = _metrics()
    assert set(m) == {"generated", "window", "btc", "tokens", "vol_rankings", "correlation", "summary"}
    assert len(m["tokens"]) == 3
    assert m["summary"]["n_tokens"] == 3
    tok = m["tokens"][0]
    assert {"symbol", "slug", "ann_vol", "ret_window", "vol_by_window", "max_runup",
            "max_drawdown", "corr_btc", "beta_btc", "avg_corr_peers", "vol_series"} <= set(tok)


def test_tokens_sorted_by_vol_desc():
    m = _metrics()
    vols = [t["ann_vol"] for t in m["tokens"]]
    assert vols == sorted(vols, reverse=True)
    assert m["tokens"][0]["symbol"] == "WILD"          # the high-vol token leads
    assert m["tokens"][-1]["symbol"] == "CALM"


def test_btc_decoupling_captured():
    m = _metrics()
    by = {t["symbol"]: t for t in m["tokens"]}
    assert by["BTCISH"]["corr_btc"] > 0.7              # the BTC-tracking token is highly correlated
    assert abs(by["WILD"]["corr_btc"]) < 0.3           # the decoupled monster is ~uncorrelated


def test_correlation_matrix_symmetric_unit_diagonal_incl_btc():
    m = _metrics()
    labels, M = m["correlation"]["symbols"], m["correlation"]["matrix"]
    assert labels[-1] == "BTC" and len(labels) == 4    # 3 tokens + BTC
    assert len(M) == 4 and all(len(row) == 4 for row in M)
    for i in range(4):
        assert M[i][i] == 1.0
        for j in range(4):
            assert M[i][j] == M[j][i]                   # symmetric


def test_all_finite_and_json_serializable():
    m = _metrics()
    s = json.dumps(m)                                  # raises if non-finite (inf/nan) present
    reloaded = json.loads(s)
    for t in reloaded["tokens"]:
        for k in ("ann_vol", "corr_btc", "beta_btc", "avg_corr_peers"):
            assert np.isfinite(t[k])


def test_deterministic_given_generated():
    assert _metrics() == _metrics()


def test_summary_regime_and_corr_stats():
    m = _metrics()["summary"]
    assert m["regime_label"] in {"bull", "bear", "flat"}
    assert -1.0 <= m["avg_pairwise_corr"] <= 1.0
    assert m["max_pairwise_corr"] >= m["avg_pairwise_corr"]
