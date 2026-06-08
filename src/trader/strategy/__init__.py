"""Strategy layer — the swappable decision core behind a stable interface.

The trading strategy is an open design space (momentum, wallet-monitoring, sentiment, RL,
risk overlays). It sits behind one interface so it can be replaced or tuned without
touching execution, custody, or guardrails. The validated candidate
(`build_candidate` — daily-rebalanced vol-tilt + regime overlay) lives in `candidate.py`.
"""

from trader.strategy.candidate import (
    DEFAULT_OVERLAY,
    OVERLAYS,
    build_candidate,
    select_vol_tokens,
)

__all__ = ["build_candidate", "select_vol_tokens", "OVERLAYS", "DEFAULT_OVERLAY"]
