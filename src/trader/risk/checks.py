"""Pure guardrail checks — `check_trade(policy, intent, state) -> Verdict`.

No I/O, no network, no clock: state comes in as a `RiskState` (derived from the ledger by
`trader.risk.ledger.state_from_ledger`), so the same function judges both the *intent*
(the wish) and the *quote-derived* intent (the truth) in `execute_trade`'s two-phase check.

Fail-closed semantics: an unavailable/None state, or an intent whose numbers cannot be
trusted (non-finite / non-positive USD), refuses with STATE_UNAVAILABLE — never a pass.
Caps are inclusive ("spend up to": exactly $2.00 passes a $2 cap); the drawdown stop is a
halt-at threshold (exactly 30% down stops). A tiny EPS absorbs float noise at boundaries.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from trader.risk.policy import Policy

EPS = 1e-9

# The eight refusal codes (runbook §guardrail skeleton spec). Coded, never free-form.
CHAIN_NOT_BSC = "CHAIN_NOT_BSC"
NOT_ALLOWLISTED = "NOT_ALLOWLISTED"
PER_TRADE_CAP = "PER_TRADE_CAP"
DAILY_CAP = "DAILY_CAP"
LIFETIME_CEILING = "LIFETIME_CEILING"
SLIPPAGE_BOUND = "SLIPPAGE_BOUND"
DRAWDOWN_STOP = "DRAWDOWN_STOP"
STATE_UNAVAILABLE = "STATE_UNAVAILABLE"

REFUSAL_CODES = (CHAIN_NOT_BSC, NOT_ALLOWLISTED, PER_TRADE_CAP, DAILY_CAP, LIFETIME_CEILING,
                 SLIPPAGE_BOUND, DRAWDOWN_STOP, STATE_UNAVAILABLE)


@dataclass(frozen=True)
class TradeIntent:
    """One desired swap. `usd` is the notional; `slippage_pct` the tolerance to request."""

    from_asset: str
    to_asset: str
    usd: float
    chain: str = "bsc"
    slippage_pct: float = 1.0


@dataclass(frozen=True)
class RiskState:
    """Spend/equity state the caps read — derived from the persisted ledger, never memory."""

    spent_today_usd: float = 0.0
    spent_lifetime_usd: float = 0.0
    equity_usd: float | None = None        # latest recorded equity (None until first record)
    high_water_usd: float | None = None    # max recorded equity (drawdown anchor)
    available: bool = True                 # False = ledger unreadable -> fail closed
    detail: str = ""


@dataclass(frozen=True)
class Verdict:
    """`allowed` + every applicable refusal as `{"code": ..., "detail": ...}`."""

    allowed: bool
    refusals: tuple[dict, ...] = ()

    @property
    def codes(self) -> list[str]:
        return [r["code"] for r in self.refusals]


def _finite_positive(x) -> bool:
    return isinstance(x, (int, float)) and math.isfinite(x) and x > 0


def check_trade(policy: Policy, intent: TradeIntent, state: RiskState | None) -> Verdict:
    """Judge one intent against the policy + persisted state. Collects ALL refusals."""
    # Fail closed first: no trustworthy state / no trustworthy numbers => no trade.
    if state is None or not state.available:
        why = getattr(state, "detail", "") or "no risk state"
        return Verdict(False, ({"code": STATE_UNAVAILABLE, "detail": why},))
    if not _finite_positive(intent.usd) or not (isinstance(intent.slippage_pct, (int, float))
                                                and math.isfinite(intent.slippage_pct)
                                                and intent.slippage_pct >= 0):
        return Verdict(False, ({"code": STATE_UNAVAILABLE,
                                "detail": f"intent numbers invalid (usd={intent.usd!r}, "
                                          f"slippage_pct={intent.slippage_pct!r})"},))

    refusals: list[dict] = []
    if intent.chain != policy.chain:
        refusals.append({"code": CHAIN_NOT_BSC,
                         "detail": f"chain {intent.chain!r} != pinned {policy.chain!r}"})
    for asset in (intent.from_asset, intent.to_asset):
        if str(asset).upper() not in policy.allowlist:
            refusals.append({"code": NOT_ALLOWLISTED, "detail": f"asset {asset!r}"})
    if intent.usd > policy.per_trade_usd + EPS:
        refusals.append({"code": PER_TRADE_CAP,
                         "detail": f"${intent.usd:.2f} > ${policy.per_trade_usd:.2f}"})
    if state.spent_today_usd + intent.usd > policy.daily_usd + EPS:
        refusals.append({"code": DAILY_CAP,
                         "detail": f"${state.spent_today_usd:.2f} spent today + "
                                   f"${intent.usd:.2f} > ${policy.daily_usd:.2f}"})
    if state.spent_lifetime_usd + intent.usd > policy.lifetime_usd_ceiling + EPS:
        refusals.append({"code": LIFETIME_CEILING,
                         "detail": f"${state.spent_lifetime_usd:.2f} lifetime + "
                                   f"${intent.usd:.2f} > ${policy.lifetime_usd_ceiling:.2f}"})
    if intent.slippage_pct > policy.max_slippage_pct + EPS:
        refusals.append({"code": SLIPPAGE_BOUND,
                         "detail": f"{intent.slippage_pct:.4f}% > "
                                   f"{policy.max_slippage_pct:.4f}%"})
    # Drawdown stop: only measurable once equity has ever been recorded (pre-funding there is
    # no equity and nothing at risk yet — the spend caps still bind).
    if (state.high_water_usd is not None and state.equity_usd is not None
            and state.high_water_usd > 0):
        dd_pct = (1.0 - state.equity_usd / state.high_water_usd) * 100.0
        if dd_pct >= policy.drawdown_stop_pct - EPS:
            refusals.append({"code": DRAWDOWN_STOP,
                             "detail": f"drawdown {dd_pct:.2f}% >= stop "
                                       f"{policy.drawdown_stop_pct:.2f}% "
                                       f"(equity ${state.equity_usd:.2f} / "
                                       f"high-water ${state.high_water_usd:.2f})"})
    return Verdict(allowed=not refusals, refusals=tuple(refusals))
