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
