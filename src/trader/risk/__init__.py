"""Risk layer — hard guardrails enforced in code, not prompt suggestions.

Allowlist, per-trade and daily caps, slippage protection, and the drawdown stop that keeps
the agent inside the competition's max-drawdown DQ gate. Enforced around the signing call.
"""
