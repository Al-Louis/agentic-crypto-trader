"""The MCP loop's rd-era surface: the modernized launch builders (full flag whitelist,
sha-stamped run-ids, discrete-aware smoke gate) and the per-regime verdict core."""

from __future__ import annotations

import pytest

from trader.experiment import launch as L
from trader.experiment.diagnostics import regime_verdict

RDL_CONFIG = {  # the full RecurrentPPO config (rdL) — every rd-era knob in one dict
    "reward_mode": "relative", "recurrent": True, "lstm_size": 256, "rule_default": True,
    "exit_commit": 12, "dust_usd": 10.0, "rule_prior": 2.0, "tp_rungs": "0.25,0.5,1.0,2.0",
    "harvest_obs": True, "eval_prepad": True, "loss_floor": 0.2, "det_blacklist": 672,
    "action_mode": "discrete", "n_action_levels": 4, "universe_mode": "voltopk", "k": 8,
    "vol_target": 0.005, "cap_floor": 0.02, "crash_train": 1, "crash_eval": True,
    "norm_reward": True, "dd_lambda": 0.0, "dd_soft": 0.15, "ent_coef": 0.2,
    "lr": 3e-4, "lr_end": 3e-5, "episode_bars": 336,
}


def test_reward_args_accept_the_full_rdl_config():
    args = L.build_reward_args(RDL_CONFIG)
    for flag in ("--recurrent", "--rule-default", "--tp-rungs", "--det-blacklist",
                 "--eval-prepad", "--loss-floor", "--lstm-size"):
        assert flag in args


def test_reward_args_refuse_unknown_key():
    with pytest.raises(ValueError, match="unknown reward_config key"):
        L.build_reward_args({"reward_mode": "relative", "lstm_sise": 256})


OVERLAY_CONFIG = {  # the long-default basket overlay + weekly gate (2026-06-14 fork)
    "reward_mode": "relative", "rule_default": True, "basket_default": True, "no_btc_obs": True,
    "eval_mode": "weekly", "exit_commit": 12, "dust_usd": 10.0, "rule_prior": 2.0,
    "tp_rungs": "0.25,0.5,1.0,2.0", "eval_prepad": True, "loss_floor": 0.2, "action_mode": "discrete",
    "n_action_levels": 4, "universe_mode": "voltopk", "k": 8, "vol_target": 0.005, "cap_floor": 0.02,
    "norm_reward": True, "dd_lambda": 0.0, "ent_coef": 0.2, "lr": 3e-4, "lr_end": 3e-5, "episode_bars": 168,
}


def test_reward_args_accept_the_overlay_config():
    args = L.build_reward_args(OVERLAY_CONFIG)
    for flag in ("--basket-default", "--no-btc-obs", "--eval-mode", "--rule-default"):
        assert flag in args
    assert args[args.index("--eval-mode") + 1] == "weekly"


def test_reward_args_accept_curriculum_horizon():
    args = L.build_reward_args({**OVERLAY_CONFIG, "curriculum_horizon": "672:0.0,336:0.40,168:0.70"})
    assert "--curriculum-horizon" in args
    assert args[args.index("--curriculum-horizon") + 1] == "672:0.0,336:0.40,168:0.70"


def test_smoke_forces_continuous_but_sweep_keeps_weekly():
    """The smoke must run the fast, parseable CONTINUOUS eval (its [eval] line carries the action
    distribution the gate reads); the real sweep keeps --eval-mode weekly."""
    smoke = L.build_smoke_command(python="py", workdir="/w", reward_config=OVERLAY_CONFIG,
                                  split="val", prefix="ppo-event-overlay")
    assert "--eval-mode" not in smoke                       # stripped for the smoke
    assert "--basket-default" in smoke                      # but the substrate stays
    sweep = L.build_sweep_command(python="py", workdir="/w", reward_config=OVERLAY_CONFIG,
                                  seeds=[0], split="val", prefix="ppo-event-overlay")
    assert "--eval-mode weekly" in sweep                    # the sweep grades on the deployment structure


def test_sweep_command_sha_stamps_run_ids_and_sequences():
    cmd = L.build_sweep_command(python="py", workdir="/w", reward_config={"reward_mode": "relative"},
                                seeds=[0, 1], split="val", prefix="ppo-event-x")
    assert "SHA=$(git rev-parse --short HEAD" in cmd       # resolved ON the box
    assert "ppo-event-x-${SHA}-s$s" in cmd                 # the ec1e487 naming convention
    assert "for s in 0 1; do" in cmd                       # sequenced, never parallel
    assert "nohup" in cmd and "echo $!" in cmd


SMOKE_OUT = """[eval] primary=val events=120 action mean=1.520 min=0.000 max=3.000
[train_event] x-smoke: return +1.2%, Sharpe 0.5, maxDD 4.0%, events 120, trades 18
"""


def test_smoke_discrete_mode_passes_healthy_level_policy():
    r = L.parse_smoke(SMOKE_OUT, discrete=True)
    assert r["alive"] and r["straddle"] and r["passed"]
    # the same output FAILS the continuous gate (mean 1.52 > mean_cap) — the stale-gate bug
    assert not L.parse_smoke(SMOKE_OUT)["passed"]


def test_smoke_discrete_single_level_fails():
    pinned = SMOKE_OUT.replace("mean=1.520 min=0.000 max=3.000", "mean=0.000 min=0.000 max=0.000")
    assert not L.parse_smoke(pinned, discrete=True)["passed"]


def _bundle(ret_val, ret_test, dd=0.10, bh=0.05, rule=0.02, rnd=0.0):
    def reg(ret):
        return {"return": ret, "maxdd": dd, "baseline_return": rule, "buyhold_return": bh,
                "random_return": rnd, "gate_pass": None, "gate_binding": None}
    return {"total_return_pct": ret_val, "max_drawdown_pct": dd,
            "regimes": {"val": reg(ret_val), "test": reg(ret_test)},
            "provenance": {"eval_split": "val"}}


def test_regime_verdict_means_and_overall_gate():
    bundles = {"p-s0": _bundle(0.10, 0.08), "p-s1": _bundle(0.06, 0.12)}

    def fetch(url):
        rid = url.rsplit("/", 2)[-2]
        return bundles[rid]

    v = regime_verdict("p", [0, 1], fetch=fetch)
    assert v["regimes"]["val"]["mean_return"] == pytest.approx(0.08)
    assert v["regimes"]["test"]["mean_return"] == pytest.approx(0.10)
    assert v["regimes"]["val"]["mean_gate_pass"] and v["overall_pass"]


def test_regime_verdict_binding_and_missing_seed():
    bundles = {"p-s0": _bundle(0.01, 0.08)}                # val mean 0.01 < buyhold 0.05

    def fetch(url):
        rid = url.rsplit("/", 2)[-2]
        if rid not in bundles:
            raise OSError("404")
        return bundles[rid]

    v = regime_verdict("p", [0, 1], fetch=fetch)
    assert len(v["missing"]) == 1
    assert not v["regimes"]["val"]["mean_gate_pass"]
    assert v["regimes"]["val"]["binding"] == "Buy&Hold"
    assert not v["overall_pass"]


def test_regime_verdict_dq_binds():
    bundles = {"p-s0": _bundle(0.30, 0.30, dd=0.35)}       # great return, DQ'd drawdown

    def fetch(url):
        return bundles[url.rsplit("/", 2)[-2]]

    v = regime_verdict("p", [0], fetch=fetch)
    assert v["regimes"]["val"]["binding"] == "drawdown"
    assert not v["overall_pass"]
