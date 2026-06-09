"""Publish a rule-strategy backtest to the Apentic frontend so it's *visible* — per-token candles
with the strategy's actual buy/sell markers + the allocation-over-time, exactly like the RL bundles.

Aggregate metrics can't tell a discretionary trader whether the rules are too rigid (missed a runup)
or churning — only the candles + markers can. This runs a weights-fn through an instrumented
backtest (recording every trade), then reuses the portfolio-bundle pipeline to publish it.

    python scripts/publish_strategy.py            # publishes rung0 + vol-top8 + trend50 on TEST
"""
from __future__ import annotations

import os
import sys
from datetime import datetime, timezone

sys.path.insert(0, "scripts")
sys.path.insert(0, "src")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from train_rl import build_portfolio_artifacts, build_volume_panel, trade_stats  # noqa: E402
from trader import config  # noqa: E402
from trader.report import apentic as ap  # noqa: E402
from trader.sim.broker import DEFAULT_GAS_USD, DEFAULT_LP_FEE_BPS, amm_cost_usd  # noqa: E402
from trader.sim.metrics import PerformanceMetrics  # noqa: E402
from trader.strategy.candidate import build_candidate, select_vol_tokens  # noqa: E402
from trader.strategy.rung0 import build_rung0, run_rung0  # noqa: E402

DEFAULT_TARGET = "s3://alexlouis-apentic-data"
DEFAULT_CF = "E14F268NIY6WLZ"
WARMUP, REBAL = 168, 24


def run_instrumented(returns, weights_fn, liq, capital=10_000.0):
    """Backtest that records per-rebalance weights + per-token trades (markers) — like the RL env."""
    syms = list(returns.columns)
    pos = pd.Series(0.0, index=syms)
    cash = capital
    eq = np.empty(len(returns))
    records, fees = [], 0.0
    nxt = WARMUP
    for i in range(len(returns)):
        r = returns.iloc[i].reindex(syms).fillna(0.0).to_numpy()
        pos = pd.Series(pos.to_numpy() * (1.0 + r), index=syms)
        equity = float(pos.sum() + cash)
        if i >= WARMUP and i >= nxt and equity > 1.0:
            w = weights_fn(returns.iloc[: i + 1]).reindex(syms).fillna(0.0).clip(lower=0.0)
            if w.sum() > 1.0:
                w = w / w.sum()
            target = w * equity
            tu, tf = {}, {}
            for s in syms:
                trade = float(target[s] - pos[s])
                if abs(trade) < 5.0:
                    continue
                c = amm_cost_usd(trade, liq.get(s, 0.0), DEFAULT_LP_FEE_BPS, DEFAULT_GAS_USD)
                cash -= trade + c
                pos[s] += trade
                fees += c
                tu[s], tf[s] = trade, c
            records.append({"time": int(returns.index[i]),
                            "weights": {s: float(w[s]) for s in syms if w[s] > 0},
                            "trades_usd": tu, "trade_fees": tf})
            nxt = i + REBAL
        eq[i] = float(pos.sum() + cash)
    return pd.Series(eq, index=returns.index), records, fees


def publish_bundle(run_id, model_name, eq, records, fees, target, dist, d0, d1):
    """`eq`/`records` cover a pre-window warmup; [d0, d1] is the window we display/score."""
    eq = eq.loc[d0:d1]                                  # display only the window (drop the warmup region)
    records = [r for r in records if d0 <= r["time"] <= d1]
    fees = sum(sum(r["trade_fees"].values()) for r in records)
    universe = sorted({t for rec in records for t in rec["weights"]}
                      | {t for rec in records for t in rec["trades_usd"]})
    rep = PerformanceMetrics.compute_all(eq.iloc[::REBAL].to_numpy(), steps_per_year=365)
    metrics = ap.metrics_to_frontend(rep)
    metrics["total_fees_paid"] = fees
    weights, candles, trades = build_portfolio_artifacts(records, universe, d0, d1)
    metrics.update(trade_stats(trades))
    out = os.path.join("runs-rl", run_id)
    os.makedirs(out, exist_ok=True)
    entry = ap.export_portfolio_run(out, run_id, equity=eq.iloc[::REBAL], metrics=metrics,
                                    weights=weights, token_candles=candles, token_trades=trades,
                                    universe=universe, model_name=model_name, action_mode="rule",
                                    regime="test", timestamp=datetime.now(timezone.utc).isoformat())
    ap.publish_run(os.path.join(out, run_id), run_id, entry, target, cloudfront_dist_id=dist)
    print(f"  published {run_id}: return {rep.total_return_pct:+.1%}, maxDD {rep.max_drawdown_pct:.1%}, "
          f"{metrics['total_trades']} trades over {len(universe)} tokens")


def main():
    config.load_dotenv()
    target = config.get("APENTIC_PUBLISH_TARGET") or DEFAULT_TARGET
    dist = config.get("APENTIC_CLOUDFRONT_DIST_ID") or DEFAULT_CF
    from train_rl import load_data, time_split
    returns, btc, anchor, liq = load_data()
    _, _, test_r = time_split(returns)
    ts = returns.index.get_loc(test_r.index[0])
    warmed = returns.iloc[ts - WARMUP:]                 # warm up on PRE-test data -> trade from day 1
    d0, d1 = int(test_r.index[0]), int(test_r.index[-1])
    uni = select_vol_tokens(test_r, 8)
    vol = build_volume_panel(uni, returns.index)
    print(f"publishing TEST strategies (rung0 = intra-day event-driven; universe {uni}) -> {target}")
    # rung-0: event-driven, evaluated EVERY hour (no daily clock); holds run untrimmed
    eq, rec, fee = run_rung0(warmed, build_rung0(warmed, tokens=uni, volume=vol), liq)
    publish_bundle("rung0-widestop-v3", "Rung-0 volume-ignition, wide stop 25% (TEST)", eq, rec, fee,
                   target, dist, d0, d1)
    # baselines: daily-rebalanced by design
    for rid, name, fn in [
            ("voltop8-test", "vol-top8 plain hold (TEST)", build_candidate(warmed, tokens=uni, overlay="none")),
            ("trend50-test", "vol-top8 trend50 (TEST)", build_candidate(warmed, btc, tokens=uni, overlay="trend50"))]:
        eqb, recb, feeb = run_instrumented(warmed, fn, liq)
        publish_bundle(rid, name, eqb, recb, feeb, target, dist, d0, d1)


if __name__ == "__main__":
    main()
