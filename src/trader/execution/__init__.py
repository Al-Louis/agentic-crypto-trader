"""Execution layer — TWAK self-custody signing and BSC trade submission.

The sole execution layer. Trades are signed locally via the Trust Wallet Agent Kit and
submitted to BSC. Guardrails (see `trader.risk`) are enforced around the signing call:
`execute_trade` is the only path to a signed swap, and it re-checks every cap against the
read-only quote before signing (vault "TWAK Spike Runbook" §guardrail skeleton spec).
"""

from trader.execution.execute import execute_trade

__all__ = ["execute_trade"]
