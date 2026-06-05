"""Execution layer — TWAK self-custody signing and BSC trade submission.

The sole execution layer. Trades are signed locally via the Trust Wallet Agent Kit and
submitted to BSC. Guardrails (see `trader.risk`) are enforced around the signing call.
"""
