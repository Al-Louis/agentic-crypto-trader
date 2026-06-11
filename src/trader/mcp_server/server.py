"""trader MCP server (skeleton).

Exposes project operations as MCP tools per the design in the vault note
"BNB Hackathon/MCP Server.md". Phase-1 skeleton: only `health` is real; domain tools are
stubs added per phase. Pure helpers back each tool so they stay unit-testable.
"""

from pathlib import Path

from mcp.server.fastmcp import FastMCP

mcp = FastMCP("trader")

PHASE = "train-loop"

# Where the loop's state + published results live (vault "AI Training" / "MCP Server").
EXPERIMENTS = Path("experiments")
DATA_CDN = "https://data.alexlouis.dev"


def health_status() -> dict:
    """Core health logic (pure, testable)."""
    return {"status": "ok", "server": "trader", "phase": PHASE}


@mcp.tool()
def health() -> dict:
    """Health check — confirm the trader MCP server is alive and report its build phase."""
    return health_status()


@mcp.tool()
def eligible_tokens() -> dict:
    """[STUB] Fixed competition token universe + metadata.

    Not implemented in the skeleton. Design: vault "MCP Server" (Data tier, Phase 2).
    """
    return {"status": "not_implemented", "see": "BNB Hackathon/MCP Server.md"}


# ---- Training loop — analysis tools (🟢 READ tier) ----------------------------------------
# Owner: rl-ml-trainer / quant-analyst · vault "AI Training" / "MCP Server". These only read
# the experiment registry and fetch *published* results from the CDN — no dispatch, no writes.
# Dispatch (start_training, 🟡) is driven via scripts/train_loop.py until it gains a background
# variant for long RL runs.

def list_experiments_data(path: Path = EXPERIMENTS) -> dict:
    """Core logic (pure, testable)."""
    from trader.train.registry import Registry
    return {"experiments": [
        {"id": e.id, "config": e.config, "run_id": e.run_id, "parent_id": e.parent_id,
         "verdict": (e.diagnosis or {}).get("verdict"),
         "failed": (e.diagnosis or {}).get("failed")}
        for e in Registry(path).list()]}


@mcp.tool()
def list_experiments() -> dict:
    """List training experiments: id, config, run id, parent, and the gate verdict."""
    return list_experiments_data()


@mcp.tool()
def experiment(exp_id: str) -> dict:
    """Full record for one experiment (config, metrics, diagnosis) + its lineage (root→exp)."""
    from trader.train.registry import Registry
    reg = Registry(EXPERIMENTS)
    exp = reg.get(exp_id)
    if exp is None:
        return {"error": f"no experiment {exp_id!r}"}
    return {"experiment": exp.to_dict(), "lineage": [e.id for e in reg.lineage(exp_id)]}


@mcp.tool()
def diagnose_run(run_id: str) -> dict:
    """Fetch a published run from data.alexlouis.dev and score it against the honest gates."""
    from trader.train.diagnose import diagnose
    from trader.train.loop import derive_baseline_and_days, fetch_artifact
    metrics = fetch_artifact(DATA_CDN, run_id, "metrics.json")
    baseline, days = derive_baseline_and_days(DATA_CDN, run_id)
    return diagnose(metrics, baseline_return=baseline, days=days)


# ---- RL experiment loop — READ tier (🟢) ---------------------------------------------------
# Owner: rl-ml-trainer / quant-analyst · vault "MCP Server" §"Training / RL experiment loop".
# These wrap the laptop-side cores in `trader.experiment.*`: published-bundle diagnostics (CDN
# reads), the committed champion, and ONE tiny status ssh. No dispatch, no chain writes. The 🟡
# launch tier (rl_train/rl_kill) lands in the next slice.

def _seeds(seeds: str) -> list[str]:
    return seeds.split() if isinstance(seeds, str) else [str(s) for s in seeds]


@mcp.tool()
def rl_status(prefix: str, seeds: str = "0 1 2 3") -> dict:
    """Which sweep seeds have published + desktop liveness (CDN reads + one tiny ssh).

    `prefix` is the run-id stem (e.g. ``ppo-event-res-test``); `seeds` a space-separated list.
    Liveness is best-effort: if the desktop is unreachable, `running` is null with an `ssh_error`.
    """
    from trader.experiment.diagnostics import compare_seeds
    from trader.experiment.remote import sweep_status
    seed_list = _seeds(seeds)
    cmp = compare_seeds(prefix, seed_list, host=DATA_CDN)
    published = [r["run_id"] for r in cmp["per_seed"] if "skip" not in r]
    pending = [r["run_id"] for r in cmp["per_seed"] if "skip" in r]
    out = {"prefix": prefix, "published": published, "pending": pending,
           "n_published": len(published), "n_total": len(seed_list)}
    try:
        out.update(sweep_status())
    except Exception as e:  # noqa: BLE001 — box down / route stalled: still report CDN state
        out.update({"running": None, "ssh_error": str(e)[:200]})
    return out


@mcp.tool()
def rl_compare(prefix: str, seeds: str = "0 1 2 3") -> dict:
    """Per-seed + across-seed mean return / maxDD / Sharpe vs the rung-0 baseline (published bundles)."""
    from trader.experiment.diagnostics import compare_seeds
    return compare_seeds(prefix, _seeds(seeds), host=DATA_CDN)


def rl_diagnose_data(prefix: str, seeds: list[str], *, dd_gate: float = 0.30) -> dict:
    """Assemble the verdict packet the agent/loop reads — judged on the HONEST gate, not beat-rung-0.

    Success = `honest_gate` (vault "Agent Communication Contract"): the seed-mean must beat ALL of
    { rung-0, Buy&Hold, Random } reported per regime, AND clear the drawdown DQ. Reporting
    "beats rung-0" alone is the exact drift that lost exp1→exp5 a day, so the gate here is
    `compare_seeds.gate_pass_mean` (the in-code `honest_gate`) AND the DD guard — never beat-rung-0.
    """
    from trader.experiment.diagnostics import compare_seeds, deviation_alpha
    cmp = compare_seeds(prefix, seeds, host=DATA_CDN)
    dev = deviation_alpha(prefix, seeds, host=DATA_CDN)
    worst_dd = cmp.get("worst_maxdd")
    dd_ok = worst_dd is not None and worst_dd < dd_gate
    honest = bool(cmp.get("gate_pass_mean"))       # beats rung-0 AND Buy&Hold AND Random (per regime)
    gate_pass = cmp.get("n", 0) > 0 and honest and dd_ok
    return {
        "prefix": prefix,
        "reward_capacity": {k: dev.get(k) for k in
                            ("n_entries", "corr", "over_mean", "under_mean",
                             "entry_size_min", "entry_size_max", "verdict")},
        "performance": {k: cmp.get(k) for k in
                        ("n", "mean_return", "spread", "worst_return", "best_return",
                         "mean_maxdd", "worst_maxdd", "baseline", "buyhold", "random")},
        "regime": cmp.get("regime"),
        "honest_gate": {
            "gate_pass": gate_pass,                  # the SINGLE source of truth: honest gate AND DD
            "honest_gate_mean": honest,              # honest_gate on the seed-mean (beats all 3)
            "binding": cmp.get("gate_binding"),      # which baseline it fails: rung-0 / Buy&Hold / Random
            "beats_rung0": cmp.get("beats_baseline"),
            "beats_buyhold": cmp.get("beats_buyhold"),
            "all_seeds_pass": cmp.get("gate_pass_all_seeds"),
            "dd_ok": dd_ok, "worst_maxdd": worst_dd, "dd_gate": dd_gate,
        },
        "note": ("success = honest_gate: beat rung-0 AND Buy&Hold AND Random, per regime, on "
                 "held-out data — NOT beat-rung-0 alone (vault 'Agent Communication Contract')."),
    }


@mcp.tool()
def rl_diagnose(prefix: str, seeds: str = "0 1 2 3") -> dict:
    """Verdict packet: deviation-alpha (reward- vs capacity-bound) + return/DD + the HONEST gate."""
    return rl_diagnose_data(prefix, _seeds(seeds))


@mcp.tool()
def rl_verdict(prefix: str, seeds: str = "0 1 2 3") -> dict:
    """The PER-REGIME verdict table (val / test / crash) — the modern honest gate, per sweep.

    `prefix` is the full sha-stamped run-id stem (e.g. ``ppo-event-rdL-a27e469``). Reads each
    seed bundle's `regimes` block: per-seed rows, seed-mean return, worst-seed maxDD, and the
    mean-level gate per regime; `overall_pass` = every regime passes. This is the exact table
    every manual sweep verdict used — the loop's primary read after rl_status says published.
    """
    from trader.experiment.diagnostics import regime_verdict
    return regime_verdict(prefix, _seeds(seeds), host=DATA_CDN)


@mcp.tool()
def rl_forensics(run_id: str, token: str, times: str = "") -> dict:
    """Trade-level forensics for one token of a published run (the diag_token_events probe).

    Rebuilds the env's ignition/cushion signals locally from the bundle's provenance and
    cross-checks the published trade markers: every entry prompt (BUY / skip / cooldown), the
    component breakdown at each timestamp in `times` (space-separated ISO, e.g.
    "2026-03-22T06:00"), and what the rung-0 rule did with the token. The behavioral
    truth-teller behind every false-flag/veto finding — run it on suspicious tokens before
    proposing rule changes.
    """
    import subprocess
    import sys
    from pathlib import Path
    root = Path(__file__).resolve().parents[3]
    cmd = [sys.executable, str(root / "scripts" / "diag_token_events.py"),
           "--run-id", run_id, "--token", token]
    if times:
        cmd += ["--times", *times.split()]
    proc = subprocess.run(cmd, capture_output=True, text=True, cwd=root, timeout=600)  # noqa: S603
    out = proc.stdout[-8000:]
    return {"run_id": run_id, "token": token, "ok": proc.returncode == 0,
            "report": out or proc.stderr[-2000:]}


@mcp.tool()
def rl_obs_probe(split: str = "train", horizon: int = 24) -> dict:
    """Cheap reward-bound vs capacity-bound check — no training (runs probe_obs_alpha on the desktop).

    SSHes to the keyless box, runs the OOS obs-alpha probe, and parses its one compact `RLPROBE`
    line (IC, per-feature corr, verdict). The probe always uses the TRAIN split for its temporal
    holdout; `split` is accepted for forward-compat and currently informational.
    """
    from trader.experiment.remote import REMOTE_PYTHON, REMOTE_WORKDIR, run_ssh
    # Capture all probe output on the box, then return ONLY the tiny RLPROBE line — or, on
    # failure, the last output line as the reason (e.g. an arg/traceback), never the full spew
    # (MTU). `out=$(...)` keeps the heavy stdout on the remote side; only the final echo returns.
    cmd = (f"cd {REMOTE_WORKDIR} && out=$({REMOTE_PYTHON} scripts/probe_obs_alpha.py "
           f"--json --horizon {int(horizon)} 2>&1); "
           f"echo \"$out\" | grep '^RLPROBE' || echo \"RLPROBE_ERR: $(echo \"$out\" | tail -1)\"")
    try:
        line = run_ssh(cmd, timeout=300.0)
    except Exception as e:  # noqa: BLE001
        return {"error": str(e)[:200], "split": split}
    for ln in line.splitlines():
        if ln.startswith("RLPROBE "):
            import json
            return {"split": split, **json.loads(ln[len("RLPROBE "):])}
    return {"error": line.replace("RLPROBE_ERR:", "").strip()[:200] or "no RLPROBE line",
            "split": split, "hint": "desktop may need the synced probe_obs_alpha.py (--json flag)"}


@mcp.tool()
def experiment_record(sha_only: bool = True, publish: bool = False) -> dict:
    """Rebuild the committed ledger/champion/leaderboard from the published bundles (honest gate).

    Re-derives `experiments/{ledger.jsonl,champion.json,leaderboard.json}` from the CDN manifest —
    the permanent, version-controlled performance trail the loop appends to after each sweep.
    `sha_only` (default) keeps the board to sha-stamped run-ids (the post-ec1e487 valid era).
    `publish=True` also pushes leaderboard.json to the data host (frontend) + CloudFront
    invalidation — laptop-side creds required. The champion stays the honest gate (frozen test).
    """
    from datetime import datetime, timezone
    from trader.experiment.champion import rebuild_ledger, write_ledger
    res = rebuild_ledger(host=DATA_CDN, generated=datetime.now(timezone.utc).isoformat(),
                         sha_only=sha_only)
    write_ledger(res, EXPERIMENTS)
    champ = res["champion"]
    out = {"champion": (champ or {}).get("config_label") if champ else None,
           "n_runs": len(res["rows"]), "n_configs": len(res["summary"]), "sha_only": sha_only}
    if publish:
        import json as _json
        from remote_train.publish import invalidate_cloudfront, put_bytes
        from trader import config as _cfg
        _cfg.load_dotenv()
        target = _cfg.get("APENTIC_PUBLISH_TARGET") or "s3://alexlouis-apentic-data"
        dist = _cfg.get("APENTIC_CLOUDFRONT_DIST_ID") or "E14F268NIY6WLZ"
        put_bytes(f"{target}/leaderboard.json", _json.dumps(res["leaderboard"], indent=2).encode(),
                  "application/json", "no-cache, max-age=0")
        out["published"] = target
        out["invalidation"] = invalidate_cloudfront(dist, ["/leaderboard.json"])
    return out


@mcp.tool()
def experiment_champion() -> dict:
    """The current best config + its exact reproduce command (reads committed experiments/champion.json)."""
    from trader.experiment.champion import read_champion
    return read_champion(EXPERIMENTS)


@mcp.tool()
def rl_north_star(ask: str, prefix: str | None = None, seeds: str = "0 1 2 3") -> dict:
    """Build the North-Star Header to prefix onto an agent/loop consult (Agent Communication Contract).

    Stateless agents are blind, so EVERY consult must carry the goal + the honest-gate success metric
    + live experiment state, or it drifts (exp1→exp5). Pass the sub-problem as `ask`; if `prefix` is
    given, live state is pulled from that sweep's latest bundle (split inferred from a `-test` suffix).
    Returns the ready-to-prepend `header` string — the loop injects this into every agent it spawns.
    """
    from trader.experiment.contract import north_star_header
    diag = rl_diagnose_data(prefix, _seeds(seeds)) if prefix else None
    split = "test" if (prefix and prefix.endswith("-test")) else "val" if prefix else "?"
    return {"header": north_star_header(ask, diag, split=split), "live_diag": diag}


# ---- RL experiment loop — LAUNCH tier (🟡 SIMULATE) ----------------------------------------
# Owner: rl-ml-trainer · vault "MCP Server" §"Training / RL experiment loop". rl_train is THE
# guard tool: it makes the runbook scars structural (launch-once-and-verify ⇒ Vmmem-stacking
# impossible; test-split needs final_verdict ⇒ no meta-overfit; smoke-gate ⇒ no dud sweep).
# rl_kill stops by specific PID, never the process group. Both build their commands from the
# pure builders in trader.experiment.launch, so the logic is unit-tested offline.

import re as _re

_PREFIX_OK = _re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


@mcp.tool()
def rl_train(reward_config: dict, seeds: str = "0 1 2 3", split: str = "val",
             timesteps: int = 1_000_000, n_envs: int = 8, prefix: str | None = None,
             smoke: bool = True, final_verdict: bool = False, sha: str | None = None,
             dry_run: bool = False, verify_wait_s: int = 90) -> dict:
    """Launch a seed sweep on the desktop, SAFELY — the guard tool.

    `reward_config` is the freeform shaping knob dict (e.g. {"reward_mode":"residual","r4_beta":0.8,
    "norm_reward":true,...}); unknown keys are refused. Flow: validate → (dry_run returns the exact
    commands without touching the box) → refuse if a sweep is already running → preflight sync/data
    check → 100k smoke + gate → launch detached, sequenced seeds → wait `verify_wait_s` and confirm
    exactly one clean run (aborts on a stacked run). `split="test"` is refused unless
    `final_verdict=True` (the frozen-test meta-overfit guard — tuning runs go to val).
    """
    from trader.experiment import launch as L
    from trader.experiment.remote import REMOTE_PYTHON, REMOTE_WORKDIR, run_ssh, sweep_status

    seed_list = seeds.split() if isinstance(seeds, str) else [str(s) for s in seeds]
    if split == "test" and not final_verdict:
        return {"refused": "split='test' requires final_verdict=True (frozen-test meta-overfit "
                "guard) — tune on val, spend test only on the final verdict.", "launched": False}
    try:
        L.build_reward_args(reward_config)               # validate early (raises on unknown key)
    except ValueError as e:
        return {"refused": str(e), "launched": False}
    prefix = prefix or L.auto_prefix(reward_config, split)
    if not _PREFIX_OK.match(prefix):
        return {"refused": f"unsafe prefix {prefix!r}", "launched": False}

    common = dict(python=REMOTE_PYTHON, workdir=REMOTE_WORKDIR, reward_config=reward_config,
                  split=split, n_envs=n_envs, prefix=prefix)
    smoke_cmd = L.build_smoke_command(**common)
    sweep_cmd = L.build_sweep_command(**common, seeds=seed_list, timesteps=timesteps)
    preflight_cmd = L.build_preflight_command(workdir=REMOTE_WORKDIR, sha=sha)
    # run-ids are sha-stamped ON the box (`{prefix}-${SHA}-s<seed>`, the ec1e487 convention);
    # the concrete ids are reported after preflight resolves HEAD.
    plan = {"prefix": prefix, "run_id_pattern": f"{prefix}-<sha>-s<seed>", "split": split,
            "timesteps": timesteps, "n_envs": n_envs, "reward_config": reward_config}
    discrete = reward_config.get("action_mode") == "discrete"

    if dry_run:
        return {"dry_run": True, "plan": plan, "launched": False,
                "commands": {"preflight": preflight_cmd, "smoke": smoke_cmd, "sweep": sweep_cmd,
                             "kill": L.build_kill_command()}}

    # 1) launch-once guard: never stack a second sweep onto a running one (the Vmmem incident).
    try:
        status = sweep_status()
    except Exception as e:  # noqa: BLE001
        return {"refused": f"could not reach desktop: {str(e)[:160]}", "launched": False}
    if status.get("running"):
        return {"refused": "a sweep is already running on the desktop — launch-once discipline",
                "status": status, "launched": False}

    # 2) preflight: sync to sha (if given) + confirm HEAD and that market data is present.
    pf = L.parse_preflight(run_ssh(preflight_cmd, timeout=120.0))
    if sha and pf.get("head") != sha:
        return {"refused": f"desktop HEAD {pf.get('head')} != requested {sha}", "preflight": pf,
                "launched": False}
    if not pf.get("data_files"):
        return {"refused": "no market data on desktop (data/ohlcv/hour_1 empty)", "preflight": pf,
                "launched": False}
    stamped = f"{prefix}-{pf.get('head', 'nogit')}"        # the on-box sha stamp, resolved
    plan["run_ids"] = [f"{stamped}-s{s}" for s in seed_list]

    # 3) smoke-gate: a dead/pinned policy never gets a 4-seed sweep. (Discrete policies are judged
    # by levels-used, not the continuous mean-cap; recurrent smokes train slower — longer timeout.)
    smoke_result = None
    if smoke:
        smoke_timeout = 1800.0 if reward_config.get("recurrent") else 900.0
        smoke_result = L.parse_smoke(run_ssh(smoke_cmd, timeout=smoke_timeout), discrete=discrete)
        if not smoke_result.get("passed"):
            return {"launched": False, "reason": "smoke gate failed", "smoke": smoke_result,
                    "preflight": pf, "plan": plan}

    # 4) launch detached, then 5) wait + verify exactly one clean run.
    driver_pid = run_ssh(sweep_cmd, timeout=120.0).splitlines()[-1].strip()
    import time
    time.sleep(max(0, int(verify_wait_s)))
    # A short sweep can self-complete before the check — count published seeds so a COMPLETION
    # isn't misread as a death (a real 1M sweep publishes none in 90 s, so this only helps shorts).
    from trader.experiment.diagnostics import compare_seeds
    published = sum(1 for p in compare_seeds(stamped, seed_list).get("per_seed", []) if "skip" not in p)
    verify = L.verify_launch(sweep_status(), n_envs, published=published, expected=len(seed_list))
    return {"launched": True, "driver_pid": driver_pid, "verify": verify,
            "smoke": smoke_result, "preflight": pf, "plan": plan,
            "warning": ("STACKED OR DEAD — inspect and rl_kill if needed" if not verify["clean"]
                        else None)}


@mcp.tool()
def rl_kill() -> dict:
    """Stop a running sweep by SPECIFIC PID (driver bash + train main) — never the process group.

    The group-kill once took tailscaled down with the job. Returns the killed PIDs and the
    post-kill liveness so the caller can confirm the box is clear.
    """
    from trader.experiment import launch as L
    from trader.experiment.remote import run_ssh, sweep_status
    killed = L.parse_kill(run_ssh(L.build_kill_command(), timeout=30.0))
    try:
        killed["status_after"] = sweep_status()
    except Exception as e:  # noqa: BLE001
        killed["status_after"] = {"error": str(e)[:160]}
    return killed
