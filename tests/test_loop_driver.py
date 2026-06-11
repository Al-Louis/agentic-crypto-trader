"""The RL-loop driver state machine — full offline cycle with injected deps (no network/ssh)."""

from __future__ import annotations

import pytest

from trader.experiment import driver
from trader.experiment.loop_control import result_from_verdict


def _verdict(margin=-0.05, overall=False, dd=0.10):
    return {"regimes": {"val": {"mean_return": 0.05 + margin, "worst_maxdd": dd,
                                "mean_gate_pass": overall, "binding": None if overall else "Buy&Hold",
                                "bars": {"buyhold": 0.05, "rung0": 0.0, "random": 0.0},
                                "per_seed": []}},
            "overall_pass": overall, "missing": []}


def _deps(verdicts, launched=True, published=4, running=True):
    calls = {"launch": 0, "record": 0}

    def launch(item):
        calls["launch"] += 1
        if not launched:
            return {"launched": False, "refused": "a sweep is already running"}
        return {"launched": True, "plan": {"run_ids": [f"{item.get('prefix') or 'ppo-x'}-abc1234-s{s}"
                                                       for s in (0, 1, 2, 3)]},
                "verify": {"clean": True}}

    def poll(stamped, seeds):
        return {"n_published": published, "running": running}

    def verdict(stamped, seeds):
        return verdicts.pop(0)

    def record():
        calls["record"] += 1
        return {}

    return {"launch": launch, "poll": poll, "verdict": verdict, "record": record,
            "_calls": calls}


def test_full_cycle_launch_run_verdict_continue(tmp_path):
    deps = _deps([_verdict(margin=-0.05)])
    driver.propose({"reward_mode": "relative"}, note="t", state_dir=tmp_path)
    r = driver.step(tmp_path, deps=deps)
    assert r["phase"] == "launched"
    assert r["active"]["stamped"] == "ppo-x-abc1234"        # sha-stamped, seed tail stripped

    deps["poll"] = lambda s, seeds: {"n_published": 2, "running": True}
    assert driver.step(tmp_path, deps=deps)["phase"] == "running"

    deps["poll"] = lambda s, seeds: {"n_published": 4, "running": False}
    r = driver.step(tmp_path, deps=deps)
    assert r["phase"] == "verdict"
    assert r["decision"]["action"] == "continue"
    assert r["needs_proposal"] is True
    assert deps["_calls"]["record"] == 1
    st = driver.load_state(tmp_path)
    assert st["iteration"] == 1 and st["active"] is None and len(st["history"]) == 1
    assert st["history"][0]["margin_vs_buyhold"] == pytest.approx(-0.05)


def test_drift_alarm_halts_after_patience(tmp_path):
    st = driver.load_state(tmp_path)
    st["patience"] = 2
    driver.save_state(st, tmp_path)
    for margin in (-0.05, -0.05, -0.06):                    # first sets best; two stalls follow
        deps = _deps([_verdict(margin=margin)])
        driver.propose({"reward_mode": "relative"}, state_dir=tmp_path)
        driver.step(tmp_path, deps=deps)                    # launch
        deps["poll"] = lambda s, seeds: {"n_published": 4, "running": False}
        r = driver.step(tmp_path, deps=deps)                # verdict
    assert r["decision"]["action"] == "escalate" and r["decision"]["drift_alarm"]
    assert driver.step(tmp_path)["phase"] == "halted"       # halted: no deps needed, no side effects
    driver.reset(tmp_path)
    assert driver.load_state(tmp_path)["halted"] is None    # soft reset keeps history
    assert len(driver.load_state(tmp_path)["history"]) == 3


def test_refused_launch_halts_and_requeues(tmp_path):
    deps = _deps([], launched=False)
    driver.propose({"reward_mode": "relative"}, state_dir=tmp_path)
    r = driver.step(tmp_path, deps=deps)
    assert r["phase"] == "halted" and "refused" in r["reason"]
    assert len(driver.load_state(tmp_path)["queue"]) == 1   # config kept for the retry


def test_dead_sweep_halts(tmp_path):
    deps = _deps([])
    driver.propose({"reward_mode": "relative"}, state_dir=tmp_path)
    driver.step(tmp_path, deps=deps)
    deps["poll"] = lambda s, seeds: {"n_published": 0, "running": False}
    r = driver.step(tmp_path, deps=deps)
    assert r["phase"] == "halted" and "dead at 0/4" in r["reason"]


def test_partial_sweep_death_halts_but_unreachable_box_waits(tmp_path):
    deps = _deps([])
    driver.propose({"reward_mode": "relative"}, state_dir=tmp_path)
    driver.step(tmp_path, deps=deps)
    deps["poll"] = lambda s, seeds: {"n_published": 1, "running": None}    # box unreachable
    assert driver.step(tmp_path, deps=deps)["phase"] == "running"          # wait on CDN, don't kill
    deps["poll"] = lambda s, seeds: {"n_published": 1, "running": False}   # box answers, no trainer
    r = driver.step(tmp_path, deps=deps)
    assert r["phase"] == "halted" and "dead at 1/4" in r["reason"]         # the WSL-close case


def test_result_from_verdict_worst_regime_margin():
    v = {"regimes": {"val": {"mean_return": 0.10, "worst_maxdd": 0.1, "mean_gate_pass": True,
                             "binding": None, "bars": {"buyhold": 0.05}},
                     "crash": {"mean_return": -0.02, "worst_maxdd": 0.1, "mean_gate_pass": False,
                               "binding": "rung-0", "bars": {"buyhold": -0.80}}},
         "overall_pass": False}
    r = result_from_verdict("x", "val", v)
    assert r.margin_vs_buyhold == pytest.approx(0.05)       # min(0.05, +0.78) = the worst regime
    assert r.binding == "crash:rung-0"
    assert not r.honest_gate_pass
