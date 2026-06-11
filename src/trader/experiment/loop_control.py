"""The loop controller — decide continue / promote / drift-alarm-escalate for the autonomous loop.

PURE logic (no I/O), so the autonomous RL loop's continue/stop decision is governed by the
**honest gate** (policy-vs-Buy&Hold), never a proxy reward, and it HALTS → escalates after N
consecutive experiments with no improvement in PnL-vs-Buy&Hold rather than rabbit-holing on a proxy
(the exp1→exp5 failure — vault "Agent Communication Contract" §"For MCP automation"). The workflow
feeds it the experiment history (assembled from rl_diagnose verdicts) and acts on the decision.

The single north-star quantity is `margin_vs_buyhold = policy_mean_return − buyhold_return`. Progress
is a NEW best margin; `patience` consecutive experiments without a new best ⇒ drift alarm.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class ExperimentResult:
    """One experiment's honest-gate verdict (distilled from an rl_diagnose packet)."""
    exp_id: str
    split: str                          # "val" (tuning) | "test" (frozen verdict)
    honest_gate_pass: bool              # beat rung-0 AND Buy&Hold AND Random, DD ok
    margin_vs_buyhold: float | None     # policy_mean_return − buyhold_return (the north star)
    binding: str | None = None          # which baseline it fails (rung-0 / Buy&Hold / Random)


def _trailing_stall(history: list[ExperimentResult]) -> tuple[int, float | None]:
    """Trailing count of experiments that set NO new best margin-vs-Buy&Hold, + the best so far."""
    best = -math.inf
    stall = 0
    for h in history:
        m = h.margin_vs_buyhold
        if m is not None and m > best + 1e-9:
            best, stall = m, 0            # new best ⇒ progress, reset the stall counter
        else:
            stall += 1                    # no improvement (or unmeasured) ⇒ extend the stall
    return stall, (best if best > -math.inf else None)


def decide(history: list[ExperimentResult], *, patience: int = 3,
           budget_remaining: bool = True) -> dict:
    """The loop's next action from the experiment history (most-recent LAST).

    Returns ``{action, reason, ...}`` with action ∈ {promote, escalate, continue}:
      - **promote**: the latest experiment cleared the honest gate on the FROZEN TEST → crown a
        champion and stop (the one place the test split is spent).
      - **escalate**: a drift alarm (no PnL-vs-Buy&Hold improvement in `patience` experiments) or
        the compute budget is exhausted → HALT and hand to a human.
      - **continue**: still below the gate but improving / within patience and budget → keep going.
    """
    if not history:
        return {"action": "continue", "reason": "no experiments yet — run the first from the thesis"}

    last = history[-1]
    if last.honest_gate_pass and last.split == "test":
        return {"action": "promote", "champion": last.exp_id,
                "reason": f"{last.exp_id} cleared the honest gate on the frozen test"}

    stall, best = _trailing_stall(history)
    if stall >= patience:
        best_str = f"{best:+.1%}" if best is not None else "n/a"
        return {"action": "escalate", "drift_alarm": True, "stall": stall, "best_margin": best,
                "reason": (f"DRIFT ALARM: no PnL-vs-Buy&Hold improvement in {stall} experiments "
                           f"(best margin {best_str}) — halt + escalate to a human; do not keep "
                           f"optimizing a proxy")}

    if not budget_remaining:
        return {"action": "escalate", "drift_alarm": False, "best_margin": best,
                "reason": "compute budget exhausted — escalate to a human for the next call"}

    return {"action": "continue", "stall": stall, "best_margin": best,
            "reason": (f"below the gate but within patience ({stall}/{patience}) and budget — "
                       f"propose the next experiment aimed at PnL-vs-Buy&Hold"
                       + (f"; last binding: {last.binding}" if last.binding else ""))}


def result_from_diagnose(exp_id: str, split: str, diag: dict) -> ExperimentResult:
    """Distill an `rl_diagnose` packet into an `ExperimentResult` (the loop's bridge)."""
    perf = diag.get("performance", {})
    hg = diag.get("honest_gate", {})
    mean, bh = perf.get("mean_return"), perf.get("buyhold")
    margin = (mean - bh) if (mean is not None and bh is not None) else None
    return ExperimentResult(exp_id=exp_id, split=split,
                            honest_gate_pass=bool(hg.get("gate_pass")),
                            margin_vs_buyhold=margin, binding=hg.get("binding"))


def result_from_verdict(exp_id: str, split: str, verdict: dict) -> ExperimentResult:
    """Distill a per-regime `rl_verdict` table into an `ExperimentResult` (the modern bridge).

    The north star is the WORST regime's margin-vs-Buy&Hold (the gate demands every regime pass,
    so the binding regime is the one that measures progress); `binding` carries which regime and
    which baseline failed (e.g. ``val:Buy&Hold``)."""
    margins = []
    binding = None
    for name, t in (verdict.get("regimes") or {}).items():
        mean, bh = t.get("mean_return"), (t.get("bars") or {}).get("buyhold")
        if mean is not None and bh is not None:
            margins.append(mean - bh)
        if binding is None and not t.get("mean_gate_pass"):
            binding = f"{name}:{t.get('binding')}"
    return ExperimentResult(exp_id=exp_id, split=split,
                            honest_gate_pass=bool(verdict.get("overall_pass")),
                            margin_vs_buyhold=(min(margins) if margins else None),
                            binding=binding)
