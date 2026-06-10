"""Tests for the laptop-side RL experiment-loop foundation (trader.experiment.*).

Network and SSH are injected (fixture `fetch` / monkeypatched subprocess) so these run offline
and deterministically — the same discipline that lets the MCP tools wrap these cores safely.
"""

from __future__ import annotations

import subprocess
import types

import pytest

from trader.experiment import champion, diagnostics, launch, remote


# ---- remote: status parsing + the MTU / self-match discipline ------------------------------

def test_parse_status_line():
    s = remote.parse_status("drv=1 tr=2 load=6.22")
    assert s == {"driver": 1, "trainers": 2, "load": 6.22, "running": True}
    idle = remote.parse_status("drv=0 tr=0 load=0.11")
    assert idle["running"] is False


def test_status_oneliner_uses_bracket_trick():
    # The patterns must NOT contain their own literal, or pgrep -f counts the status command.
    assert "[r]un_eventrung_sweep" in remote._STATUS_ONELINER
    assert "[t]rain_event.py" in remote._STATUS_ONELINER
    assert "run_eventrung_sweep" not in remote._STATUS_ONELINER.replace("[r]un_eventrung_sweep", "")


def test_run_ssh_rejects_oversize_reply(monkeypatch):
    big = "x" * (remote.MAX_REPLY_BYTES + 1)
    monkeypatch.setattr(subprocess, "run",
                        lambda *a, **k: types.SimpleNamespace(stdout=big, stderr="", returncode=0))
    with pytest.raises(RuntimeError, match="too large"):
        remote.run_ssh("echo big")


def test_run_ssh_raises_on_nonzero(monkeypatch):
    monkeypatch.setattr(subprocess, "run",
                        lambda *a, **k: types.SimpleNamespace(stdout="", stderr="boom", returncode=255))
    with pytest.raises(RuntimeError, match="rc=255"):
        remote.run_ssh("false")


def test_sweep_status_parses_injected_ssh(monkeypatch):
    monkeypatch.setattr(remote, "run_ssh", lambda *a, **k: "drv=1 tr=1 load=5.0")
    assert remote.sweep_status()["running"] is True


# ---- diagnostics: compare_seeds + deviation_alpha over fixture bundles ----------------------

def _metrics(ret, dd, base=0.10):
    return {"total_return_pct": ret, "max_drawdown_pct": dd, "sharpe_ratio": 1.5,
            "total_trades": 20, "baseline_return": base}


def test_compare_seeds_mean_and_skips():
    pub = {"s0": _metrics(0.14, 0.31), "s1": _metrics(0.16, 0.20)}

    def fetch(url):
        for k, v in pub.items():
            if f"-{k}/" in url:
                return v
        raise RuntimeError("not published")

    r = diagnostics.compare_seeds("pfx", [0, 1, 2], fetch=fetch)
    assert r["n"] == 2
    assert r["mean_return"] == pytest.approx(0.15)
    assert r["worst_return"] == pytest.approx(0.14)
    assert r["worst_maxdd"] == pytest.approx(0.31)
    assert r["baseline"] == pytest.approx(0.10)
    assert r["beats_baseline"] is True               # mean 0.15 > baseline 0.10
    assert sum("skip" in p for p in r["per_seed"]) == 1   # s2 not published


def test_deviation_alpha_corr_and_verdict():
    # 4 buys; bigger bets (dev>0) land on the bigger forward move -> positive corr -> capacity-bound.
    def fetch(url):
        if url.endswith("run_info.json"):
            return {"universe": [{"slug": "tok"}]}
        if url.endswith("equity_curve.json"):
            return [{"time": 0, "value": 10000}, {"time": 1_000_000, "value": 10000}]
        if url.endswith("tk_tok_trades.json"):
            return [{"side": "buy", "time": t, "usd": usd}
                    for t, usd in [(0, 1500), (3600, 2000), (7200, 3000), (10800, 4000)]]
        if url.endswith("tk_tok_candles.json"):
            # price jumps after the later (bigger) entries -> bigger bets -> bigger fwd return
            closes = [100, 100, 101, 103, 110] + [120] * 40
            return [{"time": i * 3600, "close": c} for i, c in enumerate(closes)]
        raise RuntimeError("x")

    d = diagnostics.deviation_alpha("pfx", [0], horizon_s=3600 * 3, fetch=fetch)
    assert d["n_entries"] == 4
    assert d["corr"] > 0.1
    assert "capacity-bound" in d["verdict"]


def test_deviation_alpha_degenerate_is_inconclusive():
    def fetch(url):
        if url.endswith("run_info.json"):
            return {"universe": [{"slug": "tok"}]}
        if url.endswith("equity_curve.json"):
            return [{"time": 0, "value": 10000}, {"time": 1_000_000, "value": 10000}]
        if url.endswith("tk_tok_trades.json"):
            return [{"side": "buy", "time": t, "usd": 2000} for t in (0, 3600, 7200, 10800)]
        if url.endswith("tk_tok_candles.json"):
            return [{"time": i * 3600, "close": 100} for i in range(50)]   # flat -> no variance
        raise RuntimeError("x")

    d = diagnostics.deviation_alpha("pfx", [0], fetch=fetch)
    assert "inconclusive" in d["verdict"]


# ---- champion: frozen-test gate + selection + read --------------------------------------

def _man_fetch(configs):
    """Build a fetch that serves manifest + per-run metrics for the given config map."""
    runs = {}
    for label, (split, ret, dd, base) in configs.items():
        for seed in (0, 1):
            runs[f"{label}-s{seed}"] = {
                "provenance": {"eval_split": split, "timesteps": 1_000_000,
                               "reward_mode": "composite", "git_commit": "abc"},
                "total_return_pct": ret + seed * 0.01, "max_drawdown_pct": dd,
                "sharpe_ratio": 2.0, "profit_factor": 1.5, "baseline_return": base,
                "total_trades": 100}

    def fetch(url):
        if url.endswith("manifest.json"):
            return [{"id": rid, "kind": "portfolio", "model_name": "1,000,000 steps"} for rid in runs]
        for rid, m in runs.items():
            if f"/{rid}/metrics.json" in url:
                return m
        raise RuntimeError("404")

    return fetch


def test_champion_requires_frozen_test_pass():
    # A val config with great returns must NOT be champion; only a passing test config qualifies.
    fetch = _man_fetch({
        "val-great": ("val", 1.50, 0.20, 0.30),       # huge but val -> ineligible
        "test-win": ("test", 0.40, 0.25, 0.30),       # test, beats base, DD<gate -> champion
        "test-dd": ("test", 0.90, 0.45, 0.30),         # test but worst DD>gate -> ineligible
    })
    res = champion.rebuild_ledger(fetch=fetch, generated="t")
    assert res["champion"]["config_label"] == "test-win"
    assert res["leaderboard"]["totals"]["configs"] == 3


def test_champion_none_when_nothing_generalizes():
    fetch = _man_fetch({"val-only": ("val", 1.0, 0.20, 0.30)})
    res = champion.rebuild_ledger(fetch=fetch, generated="t")
    assert res["champion"] is None


def test_read_champion_missing(tmp_path):
    out = champion.read_champion(tmp_path)
    assert out["champion"] is None and "note" in out


def test_write_then_read_champion(tmp_path):
    fetch = _man_fetch({"test-win": ("test", 0.40, 0.25, 0.30)})
    res = champion.rebuild_ledger(fetch=fetch, generated="t")
    champion.write_ledger(res, tmp_path)
    back = champion.read_champion(tmp_path)
    assert back["champion"]["config_label"] == "test-win"
    assert (tmp_path / "ledger.jsonl").exists()
    assert (tmp_path / "leaderboard.json").exists()


# ---- server: the verdict packet assembly ------------------------------------------------

def test_rl_diagnose_data_uses_honest_gate(monkeypatch):
    from trader.mcp_server import server
    # Honest gate clears (beats all 3) + DD ok ⇒ gate_pass. Note beats_baseline alone is NOT enough.
    monkeypatch.setattr("trader.experiment.diagnostics.compare_seeds",
                        lambda *a, **k: {"n": 4, "mean_return": 0.15, "spread": 0.02,
                                         "worst_return": 0.13, "best_return": 0.18,
                                         "mean_maxdd": 0.21, "worst_maxdd": 0.28, "baseline": 0.10,
                                         "buyhold": 0.12, "random": 0.05, "regime": {"label": "bull"},
                                         "beats_baseline": True, "beats_buyhold": True,
                                         "gate_pass_mean": True, "gate_binding": None,
                                         "gate_pass_all_seeds": True})
    monkeypatch.setattr("trader.experiment.diagnostics.deviation_alpha",
                        lambda *a, **k: {"n_entries": 40, "corr": 0.01, "verdict": "reward-bound"})
    pkt = server.rl_diagnose_data("pfx", ["0", "1", "2", "3"])
    assert pkt["honest_gate"]["gate_pass"] is True
    assert pkt["regime"] == {"label": "bull"}
    assert pkt["performance"]["buyhold"] == 0.12 and pkt["performance"]["random"] == 0.05


def test_rl_diagnose_fails_gate_when_loses_to_buyhold(monkeypatch):
    from trader.mcp_server import server
    # Beats rung-0 but LOSES to Buy&Hold ⇒ honest gate FAILS even though beats_baseline is True.
    monkeypatch.setattr("trader.experiment.diagnostics.compare_seeds",
                        lambda *a, **k: {"n": 4, "mean_return": -0.047, "worst_maxdd": 0.13,
                                         "baseline": -0.094, "buyhold": 0.07, "random": -0.10,
                                         "regime": {"label": "bull"}, "beats_baseline": True,
                                         "beats_buyhold": False, "gate_pass_mean": False,
                                         "gate_binding": "Buy&Hold", "gate_pass_all_seeds": False})
    monkeypatch.setattr("trader.experiment.diagnostics.deviation_alpha",
                        lambda *a, **k: {"n_entries": 6, "verdict": "reward-bound"})
    pkt = server.rl_diagnose_data("pfx", ["0", "1", "2", "3"])
    assert pkt["honest_gate"]["gate_pass"] is False            # the exp1→exp5 drift, now caught
    assert pkt["honest_gate"]["binding"] == "Buy&Hold"
    assert pkt["honest_gate"]["beats_rung0"] is True


# ---- North-Star Header (Agent Communication Contract) ---------------------------------------

def test_north_star_header_carries_goal_metric_and_state():
    from trader.experiment.contract import north_star_header
    diag = {"prefix": "ppo-event-sel", "performance": {"n": 4, "mean_return": -0.047,
            "buyhold": 0.07, "baseline": -0.094, "random": -0.10, "worst_maxdd": 0.13},
            "regime": {"label": "bull"},
            "honest_gate": {"gate_pass": False, "binding": "Buy&Hold", "dd_ok": True}}
    h = north_star_header("redesign the sizing reward", diag, split="val")
    assert "SUCCESS METRIC" in h and "Buy&Hold" in h and "honest_gate" in h
    assert "policy -4.7% vs B&H +7.0%" in h          # live state, all baselines
    assert "loses to Buy&Hold" in h                  # the binding blocker
    assert "DRIFT ALARM" in h and "RESTATE" in h     # the agent's obligations


def test_north_star_header_handles_no_run():
    from trader.experiment.contract import north_star_header
    h = north_star_header("propose the first experiment", None)
    assert "no completed run yet" in h and "GOAL:" in h


# ---- launch tier: reward-config translation, smoke gate, verify, kill -----------------------

def test_build_reward_args_storetrue_and_unknown():
    args = launch.build_reward_args({"reward_mode": "residual", "r4_beta": 0.8,
                                     "norm_reward": True, "dd_soft": 0.2})
    assert "--reward-mode" in args and "residual" in args
    assert "--norm-reward" in args                    # store_true present ⇒ emitted
    assert launch.build_reward_args({"norm_reward": False}) == []   # falsy bool ⇒ omitted
    with pytest.raises(ValueError, match="unknown reward_config key"):
        launch.build_reward_args({"r4_betaa": 0.8})


def test_reward_args_reproduces_exp5_selector_preset():
    # The freeform knob dict must reproduce run_eventrung_sweep.sh's selector EXTRA exactly.
    exp5 = {"reward_mode": "entry_forward", "ungate": True, "fwd_horizon": 24, "res_gamma": 0.1,
            "norm_reward": True, "dd_lambda": 1.0, "dd_soft": 0.15, "ent_coef": 0.2,
            "lr": 3e-4, "lr_end": 3e-5, "episode_bars": 336}
    got = " ".join(launch.build_reward_args(exp5))
    assert got == ("--reward-mode entry_forward --ungate --fwd-horizon 24 --res-gamma 0.1 "
                   "--norm-reward --dd-lambda 1.0 --dd-soft 0.15 --ent-coef 0.2 "
                   "--lr 0.0003 --lr-end 3e-05 --episode-bars 336")


def test_sweep_command_sequences_seeds_detached():
    cmd = launch.build_sweep_command(python="py", workdir="/w", reward_config={"reward_mode": "absolute"},
                                     seeds=[0, 1, 2, 3], split="val", prefix="pfx")
    assert "nohup bash -c" in cmd and "for s in 0 1 2 3" in cmd   # sequenced, never parallel
    assert "--seed $s" in cmd and "pfx-s$s" in cmd               # $s preserved for inner bash
    assert cmd.rstrip().endswith("echo $!")                      # returns the driver PID


def test_smoke_command_suppresses_publish():
    cmd = launch.build_smoke_command(python="py", workdir="/w", reward_config={"reward_mode": "absolute"},
                                     split="val", prefix="pfx")
    assert "APENTIC_PUBLISH_TARGET=" in cmd          # empty env ⇒ train_event skips publishing
    assert "pfx-smoke" in cmd and "tail -6" in cmd


SMOKE_OK = ("[eval] events=1683 action mean=0.385 min=-1.000 max=1.000\n"
            "[verdict] policy +2.1% (Sh 0.72, DD 9.9%) vs rung-0 rule +29.0% on test -> loses\n"
            "[train_event] ppo-x-smoke: return +2.1%, Sharpe 0.72, maxDD 9.9%, events 1683, trades 184")


def test_parse_smoke_alive_and_straddle():
    s = launch.parse_smoke(SMOKE_OK)
    assert s["alive"] and s["straddle"] and s["passed"]
    assert s["trades"] == 184 and s["events"] == 1683
    assert s["return_pct"] == pytest.approx(0.021)


def test_parse_smoke_dead_zero_trades():
    dead = SMOKE_OK.replace("trades 184", "trades 0")
    s = launch.parse_smoke(dead)
    assert s["alive"] is False and s["passed"] is False


def test_parse_smoke_pinned_no_straddle():
    pinned = ("[eval] events=10 action mean=0.999 min=0.998 max=1.000\n"
              "[train_event] x: return +0.1%, Sharpe 0.1, maxDD 1.0%, events 10, trades 5")
    s = launch.parse_smoke(pinned)
    assert s["alive"] is True and s["straddle"] is False and s["passed"] is False


def test_parse_smoke_unparseable():
    s = launch.parse_smoke("torch exploded\nTraceback ...")
    assert s["passed"] is False and "error" in s


def test_verify_launch_clean_vs_stacked():
    assert launch.verify_launch({"running": True, "load": 7.8, "trainers": 2}, 8)["clean"] is True
    stacked = launch.verify_launch({"running": True, "load": 15.5, "trainers": 4}, 8)
    assert stacked["stacked"] is True and stacked["clean"] is False
    # not running, nothing published ⇒ died (default behaviour preserved)
    dead = launch.verify_launch({"running": False, "load": 0.0, "trainers": 0}, 8)
    assert dead["clean"] is False and dead["completed"] is False
    # not running BUT all seeds published ⇒ a short sweep that self-completed, not a death
    done = launch.verify_launch({"running": False, "load": 0.2, "trainers": 0}, 8,
                                published=2, expected=2)
    assert done["completed"] is True and done["clean"] is True
    # partial publish ⇒ still flagged (died mid-sweep)
    partial = launch.verify_launch({"running": False}, 8, published=1, expected=4)
    assert partial["clean"] is False


def test_parse_preflight_and_kill():
    pf = launch.parse_preflight("HEAD=abc1234 data=20")
    assert pf == {"head": "abc1234", "data_files": 20}
    k = launch.parse_kill("killed=2317 2319 ")
    assert k["killed_pids"] == ["2317", "2319"] and k["n_killed"] == 2


def test_kill_command_never_group_kills_and_self_excludes():
    cmd = launch.build_kill_command()
    assert "kill -- -" not in cmd and "pkill" not in cmd       # specific PIDs, never the group
    assert "[t]rain_event.py" in cmd                           # bracket trick: can't match itself


# ---- rl_train guard flow (injected ssh) -------------------------------------------------

def test_rl_train_refuses_when_a_sweep_is_running(monkeypatch):
    from trader.mcp_server import server
    monkeypatch.setattr(remote, "sweep_status", lambda **k: {"running": True, "load": 7.0})
    out = server.rl_train({"reward_mode": "absolute"}, dry_run=False)
    assert out["launched"] is False and "already running" in out["refused"]


def test_rl_train_smoke_gate_blocks_dud(monkeypatch):
    from trader.mcp_server import server
    monkeypatch.setattr(remote, "sweep_status", lambda **k: {"running": False, "load": 0.0})

    def fake_ssh(cmd, **k):
        if "printf 'HEAD=" in cmd:
            return "HEAD=abc data=20"
        if "train_event.py --timesteps 100000" in cmd:      # the smoke
            return SMOKE_OK.replace("trades 184", "trades 0")   # dead ⇒ gate fails
        return "999"

    monkeypatch.setattr(remote, "run_ssh", fake_ssh)
    out = server.rl_train({"reward_mode": "absolute"})
    assert out["launched"] is False and out["reason"] == "smoke gate failed"
    assert out["smoke"]["alive"] is False
