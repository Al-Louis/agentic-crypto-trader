"""Horizon curriculum (2026-06-14) — ramp the RL episode length DOWN over training.

OVERLAY-1 learned a defensive basin (trim-everywhere) because a 1-week (168-bar) episode credits
the immediate reward of trimming a dip but truncates the cost — the missed multi-week run. The
horizon-credit probe (`scripts/probe_horizon_credit.py`) confirmed the lever: in bull windows the
payoff to HOLDING through a weakness bar triples from +10% at 1wk to +30% at 4wk. So train on LONG
(4wk) episodes first — where the missed-run cost is in-episode and creditable, teaching the agent to
ride the bull — then anneal to the 1-week deploy shape (the fork's "curriculum toward" deployment).

This module is torch-free: the SCHEDULE (parse + lookup) lives here and is unit-tested; the SB3
callback that pushes the horizon into the sub-envs (`EventRungEnv.set_episode_bars`) lives in
`scripts/train_event.py` where torch is imported. The env must be constructed at the LARGEST horizon
the schedule names — shrinking is always safe, growing would index past the panel.
"""
from __future__ import annotations


def parse_horizon_schedule(spec: str) -> list[tuple[float, int]]:
    """Parse ``"672:0.0,336:0.40,168:0.70"`` -> ``[(0.0, 672), (0.40, 336), (0.70, 168)]`` (sorted by
    progress threshold). Each pair is ``episode_bars:progress-fraction`` — at training progress >= the
    fraction, the episode length becomes that many bars. Empty string -> ``[]`` (curriculum OFF)."""
    spec = (spec or "").strip()
    if not spec:
        return []
    sched: list[tuple[float, int]] = []
    for part in spec.split(","):
        bars_s, prog_s = part.split(":")
        sched.append((float(prog_s), int(bars_s)))
    sched.sort(key=lambda x: x[0])
    if not sched or sched[0][0] > 0.0:
        raise ValueError(f"horizon schedule must define progress 0.0 (the start); got {spec!r}")
    return sched


def horizon_at(schedule: list[tuple[float, int]], progress: float) -> int:
    """The `episode_bars` for a given training progress in [0,1] — the last phase whose threshold has
    been reached (a step function). `schedule` must be sorted (parse_horizon_schedule guarantees it)."""
    bars = schedule[0][1]
    for thr, b in schedule:
        if progress >= thr:
            bars = b
        else:
            break
    return bars


def max_horizon(schedule: list[tuple[float, int]]) -> int:
    """The largest episode length the schedule uses — the horizon the env must be CONSTRUCTED at
    (its __init__ `_max_start` is the tightest bound; the curriculum only ever shrinks from here)."""
    return max(b for _, b in schedule)
