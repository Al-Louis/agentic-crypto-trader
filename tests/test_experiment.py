"""Tests for the laptop-side RL experiment-loop foundation (trader.experiment.*).

Network and SSH are injected (fixture `fetch` / monkeypatched subprocess) so these run offline
and deterministically — the same discipline that lets the MCP tools wrap these cores safely.
"""

from __future__ import annotations

import subprocess
import types

import pytest

from trader.experiment import champion, diagnostics, remote


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

def test_rl_diagnose_data_packet(monkeypatch):
    from trader.mcp_server import server
    monkeypatch.setattr("trader.experiment.diagnostics.compare_seeds",
                        lambda *a, **k: {"n": 4, "mean_return": 0.15, "spread": 0.02,
                                         "worst_return": 0.13, "best_return": 0.18,
                                         "mean_maxdd": 0.21, "worst_maxdd": 0.28,
                                         "baseline": 0.10, "beats_baseline": True})
    monkeypatch.setattr("trader.experiment.diagnostics.deviation_alpha",
                        lambda *a, **k: {"n_entries": 40, "corr": 0.01, "over_mean": -0.01,
                                         "under_mean": -0.02, "entry_size_min": 0.13,
                                         "entry_size_max": 0.35, "verdict": "reward-bound"})
    pkt = server.rl_diagnose_data("pfx", ["0", "1", "2", "3"])
    assert pkt["gate"]["gate_pass"] is True           # beats base + worst DD 0.28 < 0.30
    assert pkt["reward_capacity"]["verdict"] == "reward-bound"
    assert pkt["performance"]["mean_return"] == 0.15
