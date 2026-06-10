"""The North-Star Header — the goal+metric block injected into every agent/loop task prompt.

Operationalizes vault "Agent Communication Contract": spawned agents are stateless and blind, so
the orchestrator (a human, a workflow, or the MCP train-loop) MUST carry the GOAL + the honest-gate
SUCCESS METRIC + LIVE experiment STATE into every prompt, and the agent must restate it and refuse
a frame disconnected from PnL-vs-Buy&Hold. This module builds that header with live state pulled
from the latest published bundle, so a consult can never be framed as a bare sub-problem — the
exact orchestration failure that lost exp1→exp5 a day optimizing a proxy that stopped paying profit.

`north_star_header(ask, diag)` is pure (takes an already-fetched `rl_diagnose_data` packet); the
MCP tool `rl_north_star` fetches the packet and formats it.
"""

from __future__ import annotations

from typing import Any

# Verbatim from the contract — the parts that never change per task.
GOAL = ("a self-custody RL trading agent that is PROFITABLE and risk-managed on live PnL "
        "(June 22–28), under the ~30% max-drawdown DQ gate. rung-0 is the baseline to BEAT, "
        "never a destination.")
SUCCESS_METRIC = ("on HELD-OUT data, the policy must beat ALL of { rung-0 rule, Buy&Hold of the "
                  "traded universe, Random discretion } — reported PER REGIME (bull/bear/flat). "
                  "This is honest_gate() in scripts/train_event.py. A reward proxy is legitimate "
                  "ONLY with evidence that proxy => this metric.")


def _pct(x: Any) -> str:
    return f"{x * 100:+.1f}%" if isinstance(x, (int, float)) else "n/a"


def format_live_state(diag: dict | None, *, split: str = "?") -> str:
    """One LIVE STATE line from an `rl_diagnose_data` packet (or a no-run placeholder)."""
    if not diag or diag.get("performance", {}).get("n") in (None, 0):
        return "no completed run yet — propose the first experiment from the thesis, not a proxy"
    perf, hg = diag.get("performance", {}), diag.get("honest_gate", {})
    regime = (diag.get("regime") or {}).get("label", "?")
    blocker = ("clears the honest gate" if hg.get("gate_pass")
               else f"loses to {hg.get('binding')}" if hg.get("binding")
               else "fails the drawdown gate" if hg.get("dd_ok") is False else "below the gate")
    return (f"{diag.get('prefix', '?')} | last result: policy {_pct(perf.get('mean_return'))} vs "
            f"B&H {_pct(perf.get('buyhold'))} / rung-0 {_pct(perf.get('baseline'))} / "
            f"Random {_pct(perf.get('random'))} on {split}, regime {regime} | "
            f"worst-DD {_pct(perf.get('worst_maxdd'))} | open blocker: {blocker}")


def north_star_header(ask: str, diag: dict | None = None, *, split: str = "?") -> str:
    """The full North-Star Header block to PREFIX onto any agent/workflow/MCP task prompt."""
    return (
        "## North star\n"
        f"GOAL: {GOAL}\n"
        f"SUCCESS METRIC (non-negotiable): {SUCCESS_METRIC}\n"
        f"LIVE STATE: {format_live_state(diag, split=split)}\n"
        f"THE ASK: {ask}\n"
        "\n"
        "Before proposing anything, RESTATE the success metric + live state in your own words "
        "(that read-back proves the frame landed). If this ask seems disconnected from "
        "PnL-vs-Buy&Hold, SOUND THE DRIFT ALARM and stop — do not optimize a proxy locally.")
