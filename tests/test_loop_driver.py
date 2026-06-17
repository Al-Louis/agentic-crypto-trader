"""The RL-loop driver state machine — full offline cycle with injected deps (no network/ssh)."""

from __future__ import annotations

import pytest

from trader.experiment import driver
from trader.experiment.loop_control import result_from_verdict


def _verdict(margin=-0.05, overall=False, dd=0.10, per_seed=None):
    # `margin` is the policy's margin-vs-Buy&Hold (B&H=0.05). The NORTH STAR is margin-vs-rung-0
    # (rung0=0.0), so margin_vs_rung0 = mean_return = 0.05 + margin.
    return {"regimes": {"val": {"mean_return": 0.05 + margin, "worst_maxdd": dd,
                                "mean_gate_pass": overall, "binding": None if overall else "rung-0",
                                "bars": {"buyhold": 0.05, "rung0": 0.0, "random": 0.0},
                                "per_seed": per_seed or []}},
            "overall_pass": overall, "missing": []}


def _deps(verdicts, launched=True, published=4, running=True):
    calls = {"launch": 0, "record": 0, "publish_leaderboard": []}

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

    def publish_leaderboard(seed_run_id, config_seed_mean, dq_pass):
        calls["publish_leaderboard"].append((seed_run_id, config_seed_mean, dq_pass))
        return {"launched": "launched:123"}

    return {"launch": launch, "poll": poll, "verdict": verdict, "record": record,
            "publish_leaderboard": publish_leaderboard, "_calls": calls}


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
    # north star = margin_vs_rung0 (mean 0.0 − rung0 0.0); margin_vs_buyhold carried as reported-only.
    assert st["history"][0]["margin_vs_rung0"] == pytest.approx(0.0)
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
    # north star = WORST regime's margin-vs-the-rung-0-RULE; margin_vs_buyhold carried as reported.
    v = {"regimes": {"val": {"mean_return": 0.10, "worst_maxdd": 0.1, "mean_gate_pass": True,
                             "binding": None, "bars": {"buyhold": 0.05, "rung0": 0.08}},
                     "crash": {"mean_return": -0.02, "worst_maxdd": 0.1, "mean_gate_pass": False,
                               "binding": "rung-0", "bars": {"buyhold": -0.80, "rung0": 0.01}}},
         "overall_pass": False}
    r = result_from_verdict("x", "val", v)
    assert r.margin_vs_rung0 == pytest.approx(-0.03)        # min(0.10−0.08, −0.02−0.01) = the worst
    assert r.margin_vs_buyhold == pytest.approx(0.05)       # min(0.05, +0.78), reported only
    assert r.binding == "crash:rung-0"
    assert not r.honest_gate_pass


# --- Phase 3: the verdict-phase leaderboard hook --------------------------------------------------

def test_best_seed_and_guard_picks_max_return_and_dq():
    v = {"regimes": {"val": {"per_seed": [
        {"seed": 0, "return": 0.01, "maxdd": 0.10},
        {"seed": 2, "return": 0.16, "maxdd": 0.078},        # best val return
        {"seed": 3, "return": -0.03, "maxdd": 0.35}]}}}      # s3 breaches the 0.30 DQ
    rid, csm, dq = driver._best_seed_and_guard(v, "ppo-x-abc1234")
    assert rid == "ppo-x-abc1234-s2"                         # max-return seed -> the dashboard seed
    assert csm == pytest.approx((0.01 + 0.16 - 0.03) / 3)    # config 4-seed-mean guard (here n=3)
    assert dq is False                                       # one seed's maxdd 0.35 > DQ_MAXDD


def test_best_seed_and_guard_empty_is_none():
    assert driver._best_seed_and_guard({"regimes": {"val": {"per_seed": []}}}, "x") == (None, None, False)
    assert driver._best_seed_and_guard({}, "x") == (None, None, False)


def test_leaderboard_publish_cmd_construction():
    cmd = driver._leaderboard_publish_cmd("ppo-x-s2", 0.045, True, "/venv/py", "/repo")
    assert "simulate_weekly.py --run-id ppo-x-s2" in cmd
    assert "publish_leaderboard.py --run-id ppo-x-s2 --config-seed-mean 0.045 --dq-pass" in cmd
    assert cmd.startswith("nohup bash -c ")              # detached launch
    assert "< /dev/null &" in cmd and cmd.endswith("echo launched:$!")   # MTU-safe ack only
    assert "/tmp/leaderboard_ppo-x-s2.log" in cmd        # output goes to a file, not the reply


def test_leaderboard_publish_cmd_omits_optional_flags():
    cmd = driver._leaderboard_publish_cmd("ppo-x-s0", None, False, "/venv/py", "/repo")
    assert "--config-seed-mean" not in cmd
    assert "--dq-pass" not in cmd
    assert "publish_leaderboard.py --run-id ppo-x-s0" in cmd


def test_verdict_publishes_best_seed_to_leaderboard(tmp_path):
    per_seed = [{"seed": 0, "return": 0.01, "maxdd": 0.10},
                {"seed": 1, "return": 0.05, "maxdd": 0.08},
                {"seed": 2, "return": 0.16, "maxdd": 0.078},     # best
                {"seed": 3, "return": -0.03, "maxdd": 0.12}]
    deps = _deps([_verdict(margin=-0.05, per_seed=per_seed)])
    driver.propose({"reward_mode": "relative"}, note="t", state_dir=tmp_path)
    driver.step(tmp_path, deps=deps)                             # launch -> stamped ppo-x-abc1234
    deps["poll"] = lambda s, seeds: {"n_published": 4, "running": False}
    r = driver.step(tmp_path, deps=deps)                         # verdict
    assert r["phase"] == "verdict"
    assert len(deps["_calls"]["publish_leaderboard"]) == 1       # published exactly once
    rid, csm, dq = deps["_calls"]["publish_leaderboard"][0]
    assert rid == "ppo-x-abc1234-s2"                             # the max-val-return seed
    assert csm == pytest.approx((0.01 + 0.05 + 0.16 - 0.03) / 4)
    assert dq is True                                            # all maxdd <= 0.30
    lb = driver.load_state(tmp_path)["history"][-1]["leaderboard"]
    assert lb["best_seed"] == "ppo-x-abc1234-s2" and lb["launched"] == "launched:123"


def test_leaderboard_publish_failure_does_not_block_verdict(tmp_path):
    deps = _deps([_verdict(per_seed=[{"seed": 0, "return": 0.05, "maxdd": 0.1}])])
    def boom(*a):
        raise RuntimeError("ssh down")
    deps["publish_leaderboard"] = boom
    driver.propose({"reward_mode": "relative"}, state_dir=tmp_path)
    driver.step(tmp_path, deps=deps)
    deps["poll"] = lambda s, seeds: {"n_published": 4, "running": False}
    r = driver.step(tmp_path, deps=deps)
    assert r["phase"] == "verdict"                               # best-effort: NOT halted
    assert "ssh down" in driver.load_state(tmp_path)["history"][-1]["leaderboard_error"]


def test_verdict_without_per_seed_skips_leaderboard(tmp_path):
    deps = _deps([_verdict()])                                   # per_seed defaults to []
    driver.propose({"reward_mode": "relative"}, state_dir=tmp_path)
    driver.step(tmp_path, deps=deps)
    deps["poll"] = lambda s, seeds: {"n_published": 4, "running": False}
    r = driver.step(tmp_path, deps=deps)
    assert r["phase"] == "verdict"
    assert deps["_calls"]["publish_leaderboard"] == []           # nothing to publish, no call
    assert "leaderboard" not in driver.load_state(tmp_path)["history"][-1]
