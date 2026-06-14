"""Measure the DEPLOYMENT-honest bar (the 2026-06-14 fork): grade the rung-0 RULE and Buy&Hold
over COLD weekly sessions (Mon-00:00-UTC, fresh $10k, no cross-week holds) of the held data, the
way the competition actually scores — instead of one flattering continuous episode. Torch-free
(no policy), so it runs on the laptop.

  python scripts/eval_weekly_baselines.py [--k 8] [--vol-target 0.005] [--all-weeks]

Prints a per-week table (return / within-week maxDD / trade-days / regime / split) and the
aggregate read: rung-0 vs Buy&Hold weekly, the worst-week drawdown (the cold DQ axis), the
>=1-trade/day activity-floor reality, and the BULL-GAP (B&H - rung-0 over bull weeks) — the number
the substrate decision waits on.
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, "scripts")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from trader import config  # noqa: E402
from trader.train import weekly_eval as we  # noqa: E402

WARMUP = we.WARMUP


def grade_week(ws, win, btc, liq, vol, split, k, vol_target, cap_floor):
    """Grade rung-0 + Buy&Hold over one cold week (the shared `weekly_eval` baseline grader), tagging
    the split so the aggregate can restrict to OOS."""
    r = we.grade_week_baselines(ws, win, liq, vol, k=k, vol_target=vol_target, cap_floor=cap_floor)
    r.split = split
    return r


def main() -> None:
    import datetime as dt

    def _date(t):
        return dt.datetime.fromtimestamp(int(t), dt.timezone.utc).date()

    p = argparse.ArgumentParser()
    p.add_argument("--k", type=int, default=8)
    p.add_argument("--vol-target", type=float, default=0.005)
    p.add_argument("--cap-floor", type=float, default=0.02)
    p.add_argument("--all-weeks", action="store_true", help="aggregate over ALL weeks, not just OOS")
    args = p.parse_args()
    config.load_dotenv()

    from train_rl import build_volume_panel, load_data, time_split

    returns, btc, _anchor, liq = load_data()
    train_r, val_r, test_r = time_split(returns)
    val_start, test_start = int(val_r.index[0]), int(test_r.index[0])
    vol = build_volume_panel(list(returns.columns), returns.index)

    span0, span1 = _date(returns.index[0]), _date(returns.index[-1])
    print(f"data: {len(returns)} hourly bars {span0}..{span1}, {len(returns.columns)} tokens | "
          f"splits: train<{_date(val_start)} "
          f"val<{_date(test_start)} test..")
    print(f"config: rung-0 rule, k={args.k} voltopk, risk-parity vol_target={args.vol_target}\n")

    weeks = []
    print(f"{'week':>11} {'split':>5} {'regime':>6} {'rung0':>8} {'maxDD':>6} {'days':>4} "
          f"{'B&H':>8} {'gap':>8}")
    for ws, win in we.cold_week_windows(returns):
        split = we.split_label(ws, val_start, test_start)
        r = grade_week(ws, win, btc, liq, vol, split, args.k, args.vol_target, args.cap_floor)
        weeks.append(r)
        d = _date(ws)
        flag = "" if r.rung0_active_ok else "  <Rule-1"
        dqf = " DQ" if r.rung0_dd > we.DD_GATE else ""
        print(f"{str(d):>11} {split:>5} {r.regime:>6} {r.rung0_ret:>+7.1%} {r.rung0_dd:>5.0%}{dqf} "
              f"{r.rung0_trade_days:>2}/7 {r.buyhold_ret:>+7.1%} {r.buyhold_ret - r.rung0_ret:>+7.1%}{flag}")

    agg = we.aggregate(weeks, oos_only=not args.all_weeks)
    scope = "ALL weeks" if args.all_weeks else "OOS weeks (val+test)"
    print(f"\n=== AGGREGATE — {scope} (n={agg.n_weeks}, splits {agg.splits}) ===")
    print(f"rung-0  : mean {agg.rung0_mean:+.1%}  median {agg.rung0_median:+.1%}  "
          f"win-rate {agg.rung0_winrate:.0%}  best {agg.rung0_best:+.1%}  worst {agg.rung0_worst:+.1%}")
    print(f"Buy&Hold: mean {agg.buyhold_mean:+.1%}  win-rate {agg.buyhold_winrate:.0%}")
    print(f"worst-week maxDD: {agg.rung0_worst_dd:.0%}  ({agg.rung0_dq_weeks} week(s) breach the "
          f"{we.DD_GATE:.0%} DQ gate)")
    print(f"activity floor (>=1 trade/day): {agg.activity_fail_weeks}/{agg.n_weeks} weeks MISS a day "
          f"-> Rule-1 DQ risk")
    print(f"BULL-GAP (B&H - rung-0 over {agg.n_bull} bull weeks): {agg.bull_gap_mean:+.1%}  "
          f"<- the substrate-decision number")


if __name__ == "__main__":
    main()
