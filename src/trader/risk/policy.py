"""Hard risk policy — frozen limits enforced in code around the TWAK signing call.

The policy is a frozen dataclass: out-of-policy trades are **refused with a coded reason,
never negotiated, never auto-adjusted** (vault "TWAK Spike Runbook" §guardrail skeleton,
"Security and Encryption"). `SPIKE_POLICY` pins the Phase-2 dust-trade limits, including
the **$10 lifetime spike ceiling** — the throwaway wallet can never spend past it even
across crashes/restarts, because the caps read the on-disk ledger (`trader.risk.ledger`).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Policy:
    """Hard limits. All USD figures are caps on *attempted* spend (notional + gas)."""

    allowlist: frozenset[str]        # tradeable asset symbols (upper-case; chain pinned below)
    per_trade_usd: float             # max USD notional of one trade
    daily_usd: float                 # max summed attempted USD per UTC day
    max_slippage_pct: float          # max slippage tolerance / quote-implied slippage, %
    drawdown_stop_pct: float         # halt all trading at this % below the equity high-water
    lifetime_usd_ceiling: float      # max summed attempted USD, ever (the spike ceiling)
    chain: str = "bsc"               # the ONLY chain trades may touch


# The Phase-2 spike policy (runbook Step 4) — a guardrail, not a suggestion. The allowlist
# holds bare symbols because that is what the twak CLI and its quotes speak; the "-BSC"
# qualifier of "USDT-BSC" is carried by the pinned `chain="bsc"` (every check refuses any
# other chain, so a non-BSC USDT can never slip through on symbol alone).
SPIKE_POLICY = Policy(
    allowlist=frozenset({"BNB", "USDT"}),
    per_trade_usd=2.0,
    daily_usd=6.0,
    max_slippage_pct=1.0,
    drawdown_stop_pct=30.0,
    lifetime_usd_ceiling=10.0,
    chain="bsc",
)
