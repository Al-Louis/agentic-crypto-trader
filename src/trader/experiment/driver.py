"""The RL-loop DRIVER — the stateful iteration engine behind /rl-loop (vault "MCP Server" 4B/4C).

One `step()` advances the loop one tick through a tiny state machine persisted in
`experiments/loop_state.json`:

    idle + queue  --launch-->  running  --all published-->  verdict -> record -> decide
                                                              |  continue -> idle (needs_proposal)
                                                              |  promote/escalate -> halted

The MECHANICAL steps live here (launch via the guarded rl_train flow, CDN polling, the per-regime
verdict, the ledger append, the loop_control decision). The JUDGMENT step — designing the next
`reward_config` — is deliberately NOT here: when a tick returns `needs_proposal=True`, the driving
agent (the /rl-loop skill, each /loop wake) analyzes the verdict + forensics and calls `propose()`
with ONE new config (single-variable discipline). The frozen test is never spent by the loop:
everything launches on val; `promote` can only fire from a human-triggered final_verdict run.

All side effects are injected via `deps` so the state machine is fully testable offline.
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from trader.experiment.loop_control import ExperimentResult, decide, result_from_verdict

EXPERIMENTS = Path("experiments")
STATE_FILE = "loop_state.json"
DEFAULT_SEEDS = "0 1 2 3"
DEFAULT_TIMESTEPS = 1_000_000
DQ_MAXDD = 0.30   # competition max-drawdown DQ gate (fraction); the leaderboard dq_pass guard

_SEED_TAIL = re.compile(r"-s\d+$")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def load_state(state_dir: Path | str = EXPERIMENTS) -> dict:
    p = Path(state_dir) / STATE_FILE
    if p.exists():
        return json.loads(p.read_text(encoding="utf-8"))
    return {"iteration": 0, "max_iterations": 12, "patience": 3, "active": None,
            "queue": [], "history": [], "halted": None, "last_decision": None, "updated": None}


def save_state(state: dict, state_dir: Path | str = EXPERIMENTS) -> None:
    state["updated"] = _now()
    p = Path(state_dir) / STATE_FILE
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(state, indent=2), encoding="utf-8")


def propose(config: dict, *, note: str = "", timesteps: int = DEFAULT_TIMESTEPS,
            seeds: str = DEFAULT_SEEDS, prefix: str | None = None, sha: str | None = None,
            state_dir: Path | str = EXPERIMENTS) -> dict:
    """Queue the next experiment (the judgment step's output). One config per call."""
    st = load_state(state_dir)
    item = {"config": config, "note": note, "timesteps": int(timesteps), "seeds": seeds,
            "prefix": prefix, "sha": sha, "proposed": _now()}
    st["queue"].append(item)
    save_state(st, state_dir)
    return {"queued": len(st["queue"]), "item": item}


def reset(state_dir: Path | str = EXPERIMENTS, *, hard: bool = False) -> dict:
    """Clear a halt (keep history) or, with `hard`, wipe the loop state entirely."""
    st = {"iteration": 0, "max_iterations": 12, "patience": 3, "active": None, "queue": [],
          "history": [], "halted": None, "last_decision": None,
          "updated": None} if hard else load_state(state_dir)
    if not hard:
        st["halted"] = None
    save_state(st, state_dir)
    return {"reset": "hard" if hard else "soft", "iteration": st["iteration"]}


def _history_results(st: dict) -> list[ExperimentResult]:
    # None-safe on EXISTING loop state: pre-reset entries carry margin_vs_buyhold but no
    # margin_vs_rung0 (DIRECTION RESET 2026-06-15) — read it back as None, never crash.
    return [ExperimentResult(exp_id=h["exp_id"], split=h["split"],
                             honest_gate_pass=h["honest_gate_pass"],
                             margin_vs_rung0=h.get("margin_vs_rung0"),
                             binding=h.get("binding"),
                             margin_vs_buyhold=h.get("margin_vs_buyhold"))
            for h in st["history"]]


def _best_seed_and_guard(verdict: dict, stamped: str) -> tuple[str | None, float | None, bool]:
    """From the val regime's per-seed rows, pick the sweep's best seed (max val return) + the config
    guard (seed-mean return + DQ pass) for the dashboard leaderboard. Computed from the VERDICT the
    driver already has, so the desktop publish needs no (stale) ledger. (None, None, False) if empty."""
    rows = [r for r in ((verdict.get("regimes") or {}).get("val") or {}).get("per_seed", [])
            if isinstance(r.get("return"), (int, float))]
    if not rows:
        return None, None, False
    best = max(rows, key=lambda r: r["return"])
    seed_mean = sum(r["return"] for r in rows) / len(rows)
    dds = [r["maxdd"] for r in rows if isinstance(r.get("maxdd"), (int, float))]
    dq_pass = bool(dds) and all(d <= DQ_MAXDD for d in dds)
    return f"{stamped}-s{best['seed']}", seed_mean, dq_pass


def _leaderboard_publish_cmd(seed_run_id: str, config_seed_mean: float | None, dq_pass: bool,
                             python: str, workdir: str) -> str:
    """Build the fire-and-forget desktop command: `simulate_weekly` then `publish_leaderboard` for one
    seed. PURE so the cross-boundary command string is unit-testable. nohup + `< /dev/null` so the job
    survives the ssh close (we only ever LAUNCH it — never group-kill, which would also drop tailscaled
    per the runbook); the only stdout is the launch ack (tiny — MTU-safe)."""
    import shlex
    csm = f" --config-seed-mean {config_seed_mean}" if config_seed_mean is not None else ""
    dq = " --dq-pass" if dq_pass else ""
    inner = (f"cd {workdir} && {python} scripts/simulate_weekly.py --run-id {seed_run_id} && "
             f"{python} scripts/publish_leaderboard.py --run-id {seed_run_id}{csm}{dq}")
    return (f"nohup bash -c {shlex.quote(inner)} > /tmp/leaderboard_{seed_run_id}.log 2>&1 "
            f"< /dev/null & echo launched:$!")


def real_deps() -> dict[str, Callable[..., Any]]:
    """Production wiring: the same cores the MCP tools call (lazy imports — laptop-side)."""
    def launch(item: dict) -> dict:
        from trader.mcp_server.server import rl_train
        return rl_train(reward_config=item["config"], seeds=item["seeds"], split="val",
                        timesteps=item["timesteps"], prefix=item.get("prefix"),
                        sha=item.get("sha"))

    def poll(stamped: str, seeds: list[str]) -> dict:
        from trader.experiment.diagnostics import compare_seeds
        from trader.experiment.remote import sweep_status
        published = sum(1 for p in compare_seeds(stamped, seeds).get("per_seed", [])
                        if "skip" not in p)
        out = {"n_published": published}
        try:
            out.update(sweep_status())
        except Exception as e:  # noqa: BLE001 — box unreachable: report, don't crash the loop
            out.update({"running": None, "ssh_error": str(e)[:160]})
        return out

    def verdict(stamped: str, seeds: list[str]) -> dict:
        from trader.experiment.diagnostics import regime_verdict
        return regime_verdict(stamped, seeds)

    def record() -> dict:
        # publish=True: every verdict pushes leaderboard.json to the data host (laptop-side
        # creds), so the frontend never trails the loop. The caller treats record as
        # best-effort — a publish failure (offline/creds) notes record_error, never blocks;
        # the local ledger files are written before the publish step inside experiment_record.
        from trader.mcp_server.server import experiment_record
        return experiment_record(sha_only=True, publish=True)

    def publish_leaderboard(seed_run_id: str, config_seed_mean: float | None, dq_pass: bool) -> dict:
        # Auto-publish the sweep's best seed to the simulated-trades leaderboard ([[Dashboard
        # Leaderboard]] §Integration). Fire-and-forget on the DESKTOP (torch for simulate_weekly + the
        # publish creds): the reply is just the launch ack (MTU-safe). The guard is computed laptop-
        # side from the verdict and passed via --config-seed-mean (the desktop's committed ledger is
        # stale and can't resolve it there).
        from trader.experiment.remote import REMOTE_PYTHON, REMOTE_WORKDIR, run_ssh
        cmd = _leaderboard_publish_cmd(seed_run_id, config_seed_mean, dq_pass,
                                       REMOTE_PYTHON, REMOTE_WORKDIR)
        return {"launched": run_ssh(cmd, timeout=20)}

    return {"launch": launch, "poll": poll, "verdict": verdict, "record": record,
            "publish_leaderboard": publish_leaderboard}


def step(state_dir: Path | str = EXPERIMENTS, *, deps: dict | None = None) -> dict:
    """Advance the loop one tick. Returns the phase + everything the driving agent needs."""
    deps = deps or real_deps()
    st = load_state(state_dir)

    if st["halted"]:
        return {"phase": "halted", "reason": st["halted"], "decision": st["last_decision"],
                "hint": "human review — `rl_loop reset` to clear after acting on the reason"}

    if st["active"]:
        a = st["active"]
        seeds = a["seeds"].split()
        poll = deps["poll"](a["stamped"], seeds)
        if poll["n_published"] < len(seeds):
            # running is False ⇒ the box answered and NO driver/trainer exists ⇒ the sweep died
            # mid-flight (e.g. the WSL window closing killed it at 1/4). running None = box
            # unreachable — keep waiting on CDN progress, never declare death blind.
            if poll.get("running") is False:
                st["halted"] = (f"sweep {a['stamped']} dead at {poll['n_published']}/{len(seeds)} "
                                f"published — box answered with no trainer. Requeue the remaining "
                                f"seeds (reset, propose with the missing seeds, step)")
                save_state(st, state_dir)
                return {"phase": "halted", "reason": st["halted"], "poll": poll}
            return {"phase": "running", "active": a, "poll": poll,
                    "hint": "sweep in progress — re-step later (LSTM ~20-30min/seed, MLP ~5)"}

        # all seeds published -> verdict -> record -> decide
        v = deps["verdict"](a["stamped"], seeds)
        res = result_from_verdict(a["stamped"], "val", v)
        st["history"].append({
            "exp_id": res.exp_id, "split": res.split, "iteration": st["iteration"],
            "honest_gate_pass": res.honest_gate_pass, "margin_vs_rung0": res.margin_vs_rung0,
            "margin_vs_buyhold": res.margin_vs_buyhold,   # reported only (DIRECTION RESET 2026-06-15)
            "binding": res.binding, "note": a.get("note", ""), "config": a.get("config"),
            "regimes": {n: {"mean_return": t["mean_return"], "worst_maxdd": t["worst_maxdd"],
                            "mean_gate_pass": t["mean_gate_pass"], "binding": t["binding"]}
                        for n, t in v.get("regimes", {}).items()},
            "completed": _now()})
        st["active"] = None
        try:
            deps["record"]()
        except Exception as e:  # noqa: BLE001 — ledger refresh is best-effort, never blocks
            st["history"][-1]["record_error"] = str(e)[:160]
        # Phase 3: auto-publish the sweep's best seed to the dashboard leaderboard (best-effort,
        # fire-and-forget on the desktop; never blocks the loop — [[Dashboard Leaderboard]]).
        best, csm, dq = _best_seed_and_guard(v, a["stamped"])
        pub_fn = deps.get("publish_leaderboard")
        if best and pub_fn:
            try:
                pub = pub_fn(best, csm, dq)
                st["history"][-1]["leaderboard"] = {"best_seed": best, "config_seed_mean": csm,
                                                    "dq_pass": dq, **pub}
            except Exception as e:  # noqa: BLE001 — leaderboard publish is best-effort, never blocks
                st["history"][-1]["leaderboard_error"] = str(e)[:160]
        decision = decide(_history_results(st), patience=int(st.get("patience", 3)),
                          budget_remaining=st["iteration"] < int(st.get("max_iterations", 12)))
        st["last_decision"] = decision
        if decision["action"] in ("promote", "escalate"):
            st["halted"] = decision["reason"]
        save_state(st, state_dir)
        return {"phase": "verdict", "verdict": v, "result": st["history"][-1],
                "decision": decision,
                "needs_proposal": decision["action"] == "continue" and not st["queue"]}

    if st["queue"]:
        item = st["queue"].pop(0)
        try:
            launch = deps["launch"](item)
        except Exception as e:  # noqa: BLE001 — ssh/transport faults must HALT, not crash unsaved
            st["queue"].insert(0, item)
            st["halted"] = (f"launch transport error: {str(e)[:200]} — CHECK THE BOX (the command "
                            f"may have executed despite the client error) before reset+re-step")
            save_state(st, state_dir)
            return {"phase": "halted", "reason": st["halted"]}
        if not launch.get("launched"):
            st["queue"].insert(0, item)                      # keep it queued for the retry
            st["halted"] = (f"launch refused: {launch.get('refused') or launch.get('reason')}"
                            f" — fix, `rl_loop reset`, re-step")
            save_state(st, state_dir)
            return {"phase": "halted", "reason": st["halted"], "launch": launch}
        run_ids = (launch.get("plan") or {}).get("run_ids") or []
        stamped = _SEED_TAIL.sub("", run_ids[0]) if run_ids else item.get("prefix") or "?"
        st["iteration"] += 1
        st["active"] = {"stamped": stamped, "seeds": item["seeds"], "config": item["config"],
                        "note": item.get("note", ""), "timesteps": item["timesteps"],
                        "launched": _now()}
        save_state(st, state_dir)
        return {"phase": "launched", "active": st["active"], "iteration": st["iteration"],
                "launch_verify": launch.get("verify"), "smoke": launch.get("smoke")}

    return {"phase": "idle", "needs_proposal": True, "iteration": st["iteration"],
            "decision": st["last_decision"],
            "hint": "queue empty — analyze the last verdict (+ forensics) and `propose` ONE config"}
