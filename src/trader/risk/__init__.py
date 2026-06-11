"""Risk layer — hard guardrails enforced in code, not prompt suggestions.

Allowlist, per-trade and daily caps, slippage protection, and the drawdown stop that keeps
the agent inside the competition's max-drawdown DQ gate. Enforced around the signing call.

Design: vault "TWAK Spike Runbook" §guardrail skeleton spec. `check_trade` is pure; the
caps' state persists in the append-only ledger so a restarted loop stays capped.
"""

from trader.risk.checks import RiskState, TradeIntent, Verdict, check_trade
from trader.risk.policy import SPIKE_POLICY, Policy

__all__ = ["Policy", "RiskState", "SPIKE_POLICY", "TradeIntent", "Verdict", "check_trade"]
