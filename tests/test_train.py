"""Tests for the training-loop core: config, diagnose gates, experiment registry."""

from __future__ import annotations

from trader.train import config as C
from trader.train.diagnose import diagnose
from trader.train.registry import Registry


# ---- config -----------------------------------------------------------------
def test_demo_config_and_stable_key():
    a = C.demo_config("HUMA", 168, 0.04)
    assert a == {"kind": "demo-heuristic", "token": "HUMA", "ema": 168, "band": 0.04}
    # key is order-independent and stable; different params → different key
    assert C.config_key(a) == C.config_key(dict(reversed(list(a.items()))))
    assert C.config_key(a) != C.config_key(C.demo_config("ZEC", 168, 0.04))


# ---- diagnose ---------------------------------------------------------------
def _metrics(**over):
    base = {"total_return_pct": 0.2, "max_drawdown_pct": 0.1, "sharpe_ratio": 1.5,
            "fees_as_pct_of_pnl": 0.05, "total_trades": 200}
    base.update(over)
    return base


def test_diagnose_all_pass():
    out = diagnose(_metrics(), baseline_return=0.1, days=180)
    assert out["verdict"] == "pass" and out["failed"] == []
    assert {g["name"] for g in out["gates"]} == {
        "drawdown_ok", "positive_sharpe", "fee_drag_ok", "beats_baseline", "activity_ok"}


def test_diagnose_flags_each_failure():
    # the demo's profile: deep drawdown, negative sharpe, loses to baseline, under-trades
    out = diagnose(_metrics(max_drawdown_pct=0.6, sharpe_ratio=-0.4, total_return_pct=-0.43,
                              total_trades=22),
                     baseline_return=-0.18, days=180)
    assert out["verdict"] == "fail"
    assert set(out["failed"]) == {"drawdown_ok", "positive_sharpe", "beats_baseline", "activity_ok"}


def test_diagnose_skips_missing_and_none_metrics():
    out = diagnose({"total_return_pct": 0.1, "max_drawdown_pct": None})  # no baseline/days/sharpe
    names = {g["name"] for g in out["gates"]}
    assert "drawdown_ok" not in names           # None → skipped
    assert "beats_baseline" not in names         # no baseline_return → skipped
    assert "activity_ok" not in names            # no days → skipped


# ---- registry ---------------------------------------------------------------
def test_registry_register_record_get(tmp_path):
    reg = Registry(tmp_path / "experiments")
    exp = reg.register(C.demo_config(), created="2026-06-09T00:00:00Z")
    assert exp.id == "exp-001" and exp.run_id is None
    reg.record(exp.id, run_id="huma-trend-ema168", metrics=_metrics(),
               diagnosis={"verdict": "fail"})
    back = reg.get("exp-001")
    assert back.run_id == "huma-trend-ema168" and back.diagnosis["verdict"] == "fail"
    assert back.metrics["total_trades"] == 200


def test_registry_ids_increment_and_list(tmp_path):
    reg = Registry(tmp_path / "experiments")
    reg.register(C.demo_config("HUMA"))
    reg.register(C.demo_config("ZEC"))
    assert [e.id for e in reg.list()] == ["exp-001", "exp-002"]


def test_registry_lineage_root_first(tmp_path):
    reg = Registry(tmp_path / "experiments")
    a = reg.register(C.demo_config(ema=168))
    b = reg.register(C.demo_config(ema=120), parent_id=a.id)
    c = reg.register(C.demo_config(ema=96), parent_id=b.id)
    assert [e.id for e in reg.lineage(c.id)] == [a.id, b.id, c.id]
