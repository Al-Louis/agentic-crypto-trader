"""The decision interface — `DecisionCore`: observations in, trade intents out.

The strategy *brain* lives behind this protocol so the execution loop stays
strategy-agnostic ([[Project Overview]], [[Trading Strategies]]). The loop never
imports a strategy; it holds a `DecisionCore` and calls `.decide(obs)`. The RL
champion ([[AI Training]]) plugs in later by implementing the same two methods —
nothing in `loop.py` changes.

Contract:
  * `decide(obs) -> list[Intent]` — pure given the observation: NO I/O, NO signing,
    NO clock. It proposes *target* trades; the loop applies guardrails and fills
    them (paper or live). Returning `[]` is a valid hold.
  * Intents are *desires*, not orders. The loop re-checks every intent against the
    hard `risk/` guardrails before any fill — a core that proposes an out-of-policy
    trade is refused, never obeyed. The interface deliberately cannot reach the
    signing path.

`HoldCore` is the trivial shipped stub (always holds). It exists so the whole loop
runs end-to-end *now*, before any strategy is committed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable


@dataclass(frozen=True)
class Observation:
    """One tick's market view handed to the decision core.

    `prices` is `{SYMBOL: usd_price}` for the eligible universe this tick (a missing
    symbol = no observation, not a zero). `equity_usd` and `positions` describe the
    current paper/live portfolio. `ts` is the tick's UTC ISO timestamp. Extra signal
    fields (indicators, on-chain features) attach via `extra` without changing this
    frozen shape or the loop.
    """

    ts: str
    prices: dict[str, float]
    equity_usd: float
    positions: dict[str, float] = field(default_factory=dict)  # SYMBOL -> token units held
    cash_usd: float = 0.0
    extra: dict = field(default_factory=dict)


@dataclass(frozen=True)
class Intent:
    """A proposed swap the core wants this tick: spend `usd` of `from_asset` into `to_asset`.

    Strategy-side only — it carries no signing power. The loop maps it onto a
    `risk.TradeIntent`, runs the guardrails, then fills it (paper) or routes it through
    `execute_trade` (live). `reason` is a short audit tag (e.g. "rebalance", "stub-hold").
    """

    from_asset: str
    to_asset: str
    usd: float
    slippage_pct: float = 1.0
    reason: str = ""


@runtime_checkable
class DecisionCore(Protocol):
    """The swappable strategy brain. Implement these two methods; the loop owns the rest."""

    name: str

    def decide(self, obs: Observation) -> list[Intent]:
        """Propose this tick's trade intents (possibly empty). Pure — no I/O, no clock."""
        ...


class HoldCore:
    """Trivial stub: never trades. Proves the loop end-to-end before a strategy exists.

    NOT a strategy — a placeholder. Swapping in the RL champion (or any candidate)
    means constructing the loop with a different `DecisionCore`; this file and the loop
    are untouched. Deliberately makes the no-op behaviour explicit so a forward-run on
    `HoldCore` is unambiguously "no decisions yet", never a silent failure.
    """

    name = "hold-stub"

    def decide(self, obs: Observation) -> list[Intent]:  # noqa: ARG002 — stub holds always
        return []
