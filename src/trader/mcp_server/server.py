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
    """Assemble the verdict packet the agent reads (pure over the two diagnostic cores)."""
    from trader.experiment.diagnostics import compare_seeds, deviation_alpha
    cmp = compare_seeds(prefix, seeds, host=DATA_CDN)
    dev = deviation_alpha(prefix, seeds, host=DATA_CDN)
    worst_dd = cmp.get("worst_maxdd")
    gate_pass = (cmp.get("n", 0) > 0 and bool(cmp.get("beats_baseline"))
                 and worst_dd is not None and worst_dd < dd_gate)
    return {
        "prefix": prefix,
        "reward_capacity": {k: dev.get(k) for k in
                            ("n_entries", "corr", "over_mean", "under_mean",
                             "entry_size_min", "entry_size_max", "verdict")},
        "performance": {k: cmp.get(k) for k in
                        ("n", "mean_return", "spread", "worst_return", "best_return",
                         "mean_maxdd", "worst_maxdd", "baseline", "beats_baseline")},
        "gate": {"dd_gate": dd_gate, "worst_maxdd": worst_dd,
                 "beats_baseline": bool(cmp.get("beats_baseline")), "gate_pass": gate_pass},
    }


@mcp.tool()
def rl_diagnose(prefix: str, seeds: str = "0 1 2 3") -> dict:
    """The verdict packet: deviation-alpha (reward- vs capacity-bound) + return/DD + the honest gate."""
    return rl_diagnose_data(prefix, _seeds(seeds))


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
def experiment_champion() -> dict:
    """The current best config + its exact reproduce command (reads committed experiments/champion.json)."""
    from trader.experiment.champion import read_champion
    return read_champion(EXPERIMENTS)
