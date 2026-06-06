"""Trading performance metrics — ported from TradeSim (`evaluation/metrics.py`).

The full risk panel (return, Sharpe/Sortino/Calmar, max-drawdown + duration, 95% VaR/CVaR,
FIFO win-rate / profit-factor, fee drag) from an equity curve + trade list. Decoupled from
the env: a minimal `Trade` dataclass is inlined. These metrics mirror the competition's
scoring so the sim's "good" matches the live "good" (vault "Simulated Market").
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Trade:
    """Minimal executed-trade record (enough for round-trip PnL + fee accounting)."""

    side: str            # "buy" | "sell"
    quantity: float
    price: float
    fee: float = 0.0
    notional: float = 0.0
    step: int = 0


@dataclass
class MetricsReport:
    """Complete performance metrics report."""

    total_return_pct: float
    annualized_return_pct: float
    step_returns_mean: float
    step_returns_std: float
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    max_drawdown_pct: float
    max_drawdown_duration: int
    var_95: float
    cvar_95: float
    total_trades: int
    win_rate: float
    profit_factor: float
    avg_win_pct: float
    avg_loss_pct: float
    max_consecutive_losses: int
    total_fees_paid: float
    fees_as_pct_of_pnl: float

    def to_dict(self) -> dict[str, float | int]:
        return dict(self.__dict__)

    def summary(self) -> str:
        return "\n".join([
            f"Total Return:     {self.total_return_pct:+.2%}",
            f"Ann. Return:      {self.annualized_return_pct:+.2%}",
            f"Sharpe Ratio:     {self.sharpe_ratio:.3f}",
            f"Sortino Ratio:    {self.sortino_ratio:.3f}",
            f"Calmar Ratio:     {self.calmar_ratio:.3f}",
            f"Max Drawdown:     {self.max_drawdown_pct:.2%}",
            f"Max DD Duration:  {self.max_drawdown_duration} steps",
            f"VaR 95%:          {self.var_95:.4f}",
            f"CVaR 95%:         {self.cvar_95:.4f}",
            f"Total Trades:     {self.total_trades}",
            f"Win Rate:         {self.win_rate:.1%}",
            f"Profit Factor:    {self.profit_factor:.2f}",
            f"Avg Win:          {self.avg_win_pct:+.4f}",
            f"Avg Loss:         {self.avg_loss_pct:+.4f}",
            f"Max Consec Loss:  {self.max_consecutive_losses}",
            f"Total Fees:       ${self.total_fees_paid:.2f}",
            f"Fees/PnL:         {self.fees_as_pct_of_pnl:.1%}",
        ])


class PerformanceMetrics:
    """Computes all trading performance metrics from an equity curve and trade list."""

    @staticmethod
    def compute_all(
        equity_curve: np.ndarray,
        trades: list[Trade] | None = None,
        risk_free_rate: float = 0.0,
        steps_per_year: float = 525_600.0,  # 1-minute candles per year
    ) -> MetricsReport:
        if trades is None:
            trades = []

        equity = np.asarray(equity_curve, dtype=np.float64)
        n = len(equity)

        total_return = (equity[-1] - equity[0]) / equity[0] if n and equity[0] > 0 else 0.0
        step_returns = np.diff(equity) / equity[:-1] if n > 1 else np.array([0.0])
        step_returns = np.nan_to_num(step_returns, nan=0.0, posinf=0.0, neginf=0.0)
        mean_ret = float(np.mean(step_returns))
        std_ret = float(np.std(step_returns))
        with np.errstate(over="ignore"):  # degenerate short curves annualize to inf
            annualized_return = (1 + total_return) ** (steps_per_year / max(n, 1)) - 1
        if not np.isfinite(annualized_return):
            annualized_return = float("inf") if total_return > 0 else -1.0

        rf_per_step = (1 + risk_free_rate) ** (1 / steps_per_year) - 1
        excess = step_returns - rf_per_step
        sharpe = (float(np.mean(excess) / (np.std(excess) + 1e-10) * np.sqrt(steps_per_year))
                  if n > 1 else 0.0)
        downside = excess[excess < 0]
        downside_std = float(np.std(downside)) if len(downside) > 0 else 1e-10
        sortino = float(np.mean(excess) / (downside_std + 1e-10) * np.sqrt(steps_per_year))

        max_dd = PerformanceMetrics._max_drawdown(equity)
        calmar = annualized_return / (max_dd + 1e-10)
        max_dd_duration = PerformanceMetrics._max_drawdown_duration(equity)

        var_95 = float(np.percentile(step_returns, 5)) if n > 1 else 0.0
        cvar_mask = step_returns <= var_95
        cvar_95 = float(np.mean(step_returns[cvar_mask])) if cvar_mask.any() else var_95

        trade_pnls = PerformanceMetrics._compute_trade_pnls(trades)
        wins = [p for p in trade_pnls if p > 0]
        losses = [p for p in trade_pnls if p <= 0]
        win_rate = len(wins) / max(len(trade_pnls), 1)
        gross_profit = sum(wins) if wins else 0.0
        gross_loss = abs(sum(losses)) if losses else 0.0
        profit_factor = gross_profit / (gross_loss + 1e-10)
        avg_win = float(np.mean(wins)) if wins else 0.0
        avg_loss = float(np.mean(losses)) if losses else 0.0
        max_consec_losses = PerformanceMetrics._max_consecutive_losses(trade_pnls)

        total_fees = sum(t.fee for t in trades)
        total_pnl = equity[-1] - equity[0] if n > 0 else 0.0
        fees_pct = total_fees / (abs(total_pnl) + 1e-10)

        return MetricsReport(
            total_return_pct=total_return,
            annualized_return_pct=annualized_return,
            step_returns_mean=mean_ret,
            step_returns_std=std_ret,
            sharpe_ratio=sharpe,
            sortino_ratio=sortino,
            calmar_ratio=calmar,
            max_drawdown_pct=max_dd,
            max_drawdown_duration=max_dd_duration,
            var_95=var_95,
            cvar_95=cvar_95,
            total_trades=len(trades),
            win_rate=win_rate,
            profit_factor=profit_factor,
            avg_win_pct=avg_win,
            avg_loss_pct=avg_loss,
            max_consecutive_losses=max_consec_losses,
            total_fees_paid=total_fees,
            fees_as_pct_of_pnl=fees_pct,
        )

    @staticmethod
    def _max_drawdown(equity: np.ndarray) -> float:
        if len(equity) < 2:
            return 0.0
        peak = np.maximum.accumulate(equity)
        return float(np.max((peak - equity) / (peak + 1e-10)))

    @staticmethod
    def _max_drawdown_duration(equity: np.ndarray) -> int:
        if len(equity) < 2:
            return 0
        peak = np.maximum.accumulate(equity)
        in_dd = equity < peak
        max_dur = current = 0
        for dd in in_dd:
            current = current + 1 if dd else 0
            max_dur = max(max_dur, current)
        return max_dur

    @staticmethod
    def _compute_trade_pnls(trades: list[Trade]) -> list[float]:
        """FIFO round-trip PnL: pair each sell with the oldest open buy."""
        pnls = []
        open_trades: list[Trade] = []
        for t in trades:
            if t.side == "buy":
                open_trades.append(t)
            elif t.side == "sell" and open_trades:
                entry = open_trades.pop(0)
                pnl = (t.price - entry.price) * min(t.quantity, entry.quantity) - t.fee - entry.fee
                pnls.append(pnl)
        return pnls

    @staticmethod
    def _max_consecutive_losses(trade_pnls: list[float]) -> int:
        max_consec = current = 0
        for pnl in trade_pnls:
            current = current + 1 if pnl <= 0 else 0
            max_consec = max(max_consec, current)
        return max_consec
