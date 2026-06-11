"""Agent layer — the autonomous orchestration loop.

Event-driven read -> decide -> sign -> confirm, hands-off, inside the risk guardrails.
Coordinates the data, strategy, risk, and execution layers.

Public surface:
  Loop, LoopConfig          the engine (agent.loop)
  DecisionCore, HoldCore,   the swappable strategy interface + trivial stub (agent.decide)
    Observation, Intent
  PriceFeed, CmcPriceFeed,  the read step (agent.feed)
    FakeFeed
  PaperFill, fill           the paper broker (agent.paper)
Run with `python -m trader.agent` (agent.__main__).
"""

from trader.agent.decide import DecisionCore, HoldCore, Intent, Observation
from trader.agent.feed import CmcPriceFeed, FakeFeed, PriceFeed
from trader.agent.loop import Loop, LoopConfig

__all__ = [
    "CmcPriceFeed", "DecisionCore", "FakeFeed", "HoldCore", "Intent", "Loop",
    "LoopConfig", "Observation", "PriceFeed",
]
