"""Tests for the Apentic bundle exporter (the frontend data contract)."""

from __future__ import annotations

import json
import math

import numpy as np
import pandas as pd
import pytest

from trader.report import apentic as ap
from trader.sim.metrics import PerformanceMetrics

# Required keys on the frontend's MetricsReport / CandleData / EquityPoint / RoundTrip.
FRONTEND_METRICS_KEYS = {
    "total_return_pct", "annualized_return_pct", "sharpe_ratio", "sortino_ratio",
    "calmar_ratio", "max_drawdown_pct", "total_trades", "win_rate", "profit_factor",
    "episodes_evaluated", "episodes_profitable", "avg_episode_return",
}
ROUNDTRIP_KEYS = {
    "id", "episode", "entry_datetime", "entry_price", "entry_portfolio_value",
    "exit_datetime", "exit_price", "exit_portfolio_value", "exit_reason",
    "duration_steps", "pnl_usdt", "pnl_pct", "total_fees",
}


def test_to_secs_normalizes_ms_and_s():
    assert ap._to_secs(1_760_032_800_000) == 1_760_032_800   # ms → s
    assert ap._to_secs(1_765_922_400) == 1_765_922_400       # already s


def test_metrics_to_frontend_has_all_keys_and_no_inf():
    # A 2-point curve makes annualized/Calmar blow up → must serialize as null, not inf.
    m = PerformanceMetrics.compute_all(np.array([100.0, 110.0]), steps_per_year=24 * 365)
    d = ap.metrics_to_frontend(m)
    assert FRONTEND_METRICS_KEYS <= set(d)
    json.dumps(d)  # must be JSON-serializable (no inf/nan)
    for v in d.values():
        assert v is None or not (isinstance(v, float) and not math.isfinite(v))


def test_roundtrips_from_position_folds_one_winning_trip():
    # Price doubles while in position the whole time → one profitable round-trip.
    idx = pd.RangeIndex(6) * 3600 + 1_700_000_000
    prices = pd.Series([10.0, 11, 12, 13, 14, 15], index=idx)
    position = pd.Series([1, 1, 1, 1, 1, 0], index=idx, dtype=float)  # exit on last bar
    # Deep pool → negligible AMM impact, so the +50% ride shows through as profit.
    trips, equity, trades = ap.roundtrips_from_position(
        prices, position, capital=10_000.0, liquidity_usd=1e9)
    assert len(trips) == 1
    assert ROUNDTRIP_KEYS <= set(trips[0])
    assert trips[0]["pnl_usdt"] > 0 and trips[0]["pnl_pct"] > 0
    assert equity.iloc[-1] > equity.iloc[0]
    assert [t.side for t in trades] == ["buy", "sell"]


def test_roundtrips_open_position_marked_at_end():
    idx = pd.RangeIndex(3) * 3600 + 1_700_000_000
    prices = pd.Series([10.0, 11, 12], index=idx)
    position = pd.Series([1, 1, 1], index=idx, dtype=float)  # never exits
    trips, _, _ = ap.roundtrips_from_position(prices, position)
    assert len(trips) == 1 and trips[0]["exit_reason"] == "end of sample"


def test_export_run_writes_bundle_and_upserts_manifest(tmp_path):
    idx = pd.RangeIndex(4) * 3600 + 1_700_000_000
    equity = pd.Series([100.0, 101, 102, 103], index=idx)
    ohlcv = pd.DataFrame({"timestamp": idx, "open": 1.0, "high": 1.1, "low": 0.9,
                          "close": 1.05, "volume": 5.0})
    m = ap.metrics_to_frontend(PerformanceMetrics.compute_all(equity.to_numpy()))

    entry = ap.export_run(
        tmp_path, "run-x", equity=equity, metrics=m, trades=[],
        candles=ap.candles_from_ohlcv(ohlcv), symbol="HUMA",
        model_name="demo", regime="full", timestamp="2026-06-08T00:00:00+00:00")

    run_dir = tmp_path / "run-x"
    for name in ap.BUNDLE_FILES:
        assert (run_dir / name).exists(), name
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert manifest == [entry] and entry["id"] == "run-x"
    candles = json.loads((run_dir / "candles.json").read_text())
    assert candles[0]["time"] == 1_700_000_000 and "close" in candles[0]


def test_upsert_manifest_at_local_uri(tmp_path):
    uri = str(tmp_path / "manifest.json")
    ap.upsert_manifest_at(uri, {"id": "a", "model_name": "v1"})
    items = ap.upsert_manifest_at(uri, {"id": "a", "model_name": "v2"})  # replace same id
    assert items == [{"id": "a", "model_name": "v2"}]


def test_publish_run_local_uploads_and_merges_manifest(tmp_path):
    src = tmp_path / "out" / "run-x"
    src.mkdir(parents=True)
    for f in ap.BUNDLE_FILES:
        (src / f).write_text("[]")
    target = tmp_path / "dash"
    entry = {"id": "run-x", "model_name": "demo", "symbol": "HUMA"}

    ap.publish_run(src, "run-x", entry, str(target))

    for f in ap.BUNDLE_FILES:
        assert (target / "run-x" / f).read_text() == "[]"
    assert json.loads((target / "manifest.json").read_text()) == [entry]


def test_publish_run_local_skips_cloudfront(tmp_path, monkeypatch):
    import remote_train
    monkeypatch.setattr(remote_train, "invalidate_cloudfront",
                        lambda *a, **k: pytest.fail("must not invalidate a local target"))
    src = tmp_path / "out" / "run-y"
    src.mkdir(parents=True)
    for f in ap.BUNDLE_FILES:
        (src / f).write_text("[]")
    # Passing a dist id but a LOCAL target → invalidation guarded off (only s3 targets).
    ap.publish_run(src, "run-y", {"id": "run-y"}, str(tmp_path / "dash"), cloudfront_dist_id="DIST")
    assert (tmp_path / "dash" / "run-y" / "metrics.json").exists()


def test_upsert_manifest_replaces_same_id(tmp_path):
    mpath = tmp_path / "manifest.json"
    ap.upsert_manifest(mpath, {"id": "a", "model_name": "v1"})
    ap.upsert_manifest(mpath, {"id": "b", "model_name": "v1"})
    items = ap.upsert_manifest(mpath, {"id": "a", "model_name": "v2"})  # replace a
    ids = {e["id"]: e["model_name"] for e in items}
    assert ids == {"a": "v2", "b": "v1"} and len(items) == 2
