"""Deterministic run gates — the honest "did this run actually improve?" checks.

The loop must not chase training reward (which overfits / reward-hacks). Before any tuning
decision, a run is gated on the things that actually matter for the competition and for
trusting a result: survive the drawdown DQ, positive risk-adjusted return, not bleeding to
fees, beat the baseline, and meet the >=1-trade/day activity rule. A run that games the reward
but fails these is a FAILURE, not progress — the TradeSim post-mortem's discipline, encoded.

Pure: a metrics dict in, a structured verdict out (vault "AI Training"). Metrics fields match
`trader.sim.metrics.MetricsReport` (non-finite values arrive as None and are skipped).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

DD_GATE = 0.30          # weekly max-drawdown DQ
FEE_DRAG_GATE = 0.50    # fees as a fraction of |PnL|
MIN_TRADES_PER_DAY = 1.0


@dataclass
class Gate:
    name: str
    passed: bool
    value: float
    threshold: float
    note: str


def _num(metrics: dict[str, Any], key: str) -> float | None:
    v = metrics.get(key)
    try:
        return float(v) if v is not None else None
    except (TypeError, ValueError):
        return None


def diagnose(metrics: dict[str, Any], *, baseline_return: float | None = None,
             days: float | None = None, dd_gate: float = DD_GATE,
             fee_drag_gate: float = FEE_DRAG_GATE,
             min_trades_per_day: float = MIN_TRADES_PER_DAY) -> dict[str, Any]:
    """Apply the gates to a run's metrics. Returns ``{verdict, failed, gates}``.

    `baseline_return` (e.g. buy&hold of the token) enables the beats-baseline gate; `days` (the
    run's span) enables the activity gate. Gates whose metric is missing/None are skipped.
    """
    gates: list[Gate] = []

    dd = _num(metrics, "max_drawdown_pct")
    if dd is not None:
        gates.append(Gate("drawdown_ok", dd < dd_gate, dd, dd_gate,
                          "max drawdown under the ~30% DQ gate"))

    sharpe = _num(metrics, "sharpe_ratio")
    if sharpe is not None:
        gates.append(Gate("positive_sharpe", sharpe > 0.0, sharpe, 0.0,
                          "risk-adjusted return is positive"))

    fee = _num(metrics, "fees_as_pct_of_pnl")
    if fee is not None:
        gates.append(Gate("fee_drag_ok", fee < fee_drag_gate, fee, fee_drag_gate,
                          "fees are not eating the PnL"))

    ret = _num(metrics, "total_return_pct")
    if baseline_return is not None and ret is not None:
        gates.append(Gate("beats_baseline", ret > baseline_return, ret, float(baseline_return),
                          "out-returns the buy&hold / baseline"))

    trades = _num(metrics, "total_trades")
    if days and trades is not None:
        tpd = trades / days
        gates.append(Gate("activity_ok", tpd >= min_trades_per_day, tpd, min_trades_per_day,
                          ">=1 trade/day competition activity rule"))

    failed = [g.name for g in gates if not g.passed]
    return {"verdict": "pass" if not failed else "fail",
            "failed": failed,
            "gates": [asdict(g) for g in gates]}
