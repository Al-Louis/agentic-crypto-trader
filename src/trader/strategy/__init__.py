"""Strategy layer — the swappable decision core behind a stable interface.

The trading strategy is an open design space (momentum, wallet-monitoring, sentiment, RL,
risk overlays). It sits behind one interface so it can be replaced or tuned without
touching execution, custody, or guardrails.
"""
