"""Produce one Apentic dashboard bundle from a real single-asset backtest.

A pipeline-first proof: it exercises the *entire* dashboard contract (candles + entry/exit
markers + trade table + metrics + equity curve) on real token candles, using a transparent
trend heuristic as a stand-in for the eventual trained policy. NOT the portfolio strategy of
record — `model_name` says so. This is the job the orchestrator dispatches; it writes its
bundle into ``--out`` (the run's artifact dir) and emits ``progress.json``.

Run:  python scripts/export_demo_run.py --out <dir> [--token HUMA] [--run-id huma-trend-ema50]
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from datetime import datetime, timezone

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from remote_train.progress import write_progress  # noqa: E402
from trader.report import apentic as ap  # noqa: E402
from trader.sim.metrics import PerformanceMetrics  # noqa: E402

HOURS_PER_YEAR = 24 * 365


def _load_token_ohlcv(token: str) -> pd.DataFrame:
    dirs = glob.glob(os.path.join("data", "ohlcv", "hour_1", f"{token}_*"))
    if not dirs:
        raise SystemExit(f"no hourly OHLCV for {token} under data/ohlcv/hour_1/")
    files = sorted(glob.glob(os.path.join(dirs[0], "*.parquet")))
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    return df.drop_duplicates("timestamp").sort_values("timestamp").reset_index(drop=True)


def _liquidity(token: str) -> float:
    rows = [s for s in json.load(open("data/selection.json", encoding="utf-8"))
            if s["symbol"] == token]
    return float(rows[0].get("liq_usd") or 0.0) if rows else 0.0


def main() -> None:
    ap_ = argparse.ArgumentParser()
    ap_.add_argument("--out", required=True, help="artifact dir to write the bundle into")
    ap_.add_argument("--token", default="HUMA", help="token symbol (default: top-vol HUMA)")
    ap_.add_argument("--run-id", default=None, help="dashboard run id (default: <token>-trend-ema50)")
    ap_.add_argument("--ema", type=int, default=168)   # ~1 week on hourly bars → low churn
    ap_.add_argument("--band", type=float, default=0.04, help="hysteresis band around the EMA")
    ap_.add_argument("--capital", type=float, default=10_000.0)
    args = ap_.parse_args()
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:  # noqa: BLE001
        pass

    token = args.token
    run_id = args.run_id or f"{token.lower()}-trend-ema{args.ema}"
    df = _load_token_ohlcv(token)
    close = df.set_index("timestamp")["close"].astype(float)

    # Transparent causal policy: a hysteresis trend filter. Go long when close pushes above
    # EMA·(1+band), exit when it falls below EMA·(1−band); the dead-band kills crossover
    # whipsaw. Signal is decided one bar ahead of execution (no look-ahead).
    ema = close.ewm(span=args.ema, adjust=False).mean()
    up, down = close > ema * (1 + args.band), close < ema * (1 - args.band)
    state, holding = [], 0
    for u, d in zip(up.to_numpy(), down.to_numpy()):
        holding = 1 if u else (0 if d else holding)
        state.append(holding)
    position = pd.Series(state, index=close.index, dtype=float).shift(1).fillna(0.0)

    trips, equity, trade_objs = ap.roundtrips_from_position(
        close, position, capital=args.capital, liquidity_usd=_liquidity(token))
    report = PerformanceMetrics.compute_all(equity.to_numpy(), trades=trade_objs,
                                            steps_per_year=HOURS_PER_YEAR)
    metrics = ap.metrics_to_frontend(report)
    candles = ap.candles_from_ohlcv(df)

    entry = ap.export_run(
        args.out, run_id,
        equity=equity, metrics=metrics, trades=trips, candles=candles,
        symbol=token, model_name=f"{token} trend-ema{args.ema} (demo heuristic)",
        regime="full-sample", n_episodes=1, simulation=False,
        indicators_used=[f"ema{args.ema}"],
        timestamp=datetime.now(timezone.utc).isoformat(),
    )

    write_progress(args.out, run_id=run_id, token=token, bars=len(df),
                   trades=len(trips), final_equity=float(equity.iloc[-1]),
                   total_return=report.total_return_pct, sharpe=report.sharpe_ratio,
                   max_drawdown=report.max_drawdown_pct, state="complete")

    print(f"[export_demo_run] {run_id}: {len(df)} bars, {len(trips)} round-trips, "
          f"return {report.total_return_pct:+.1%}, Sharpe {report.sharpe_ratio:.2f}, "
          f"maxDD {report.max_drawdown_pct:.1%}")
    print(f"  bundle → {os.path.join(args.out, run_id)}  (manifest entry id={entry['id']})")


if __name__ == "__main__":
    main()
