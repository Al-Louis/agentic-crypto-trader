"""Unit tests for the cold-weekly-session grader's slicing + aggregation (the error-prone parts;
the per-strategy grading is exercised by scripts/eval_weekly_baselines.py against real data)."""
import numpy as np
import pandas as pd

from trader.train import weekly_eval as we


def _hourly_index(n, start=we.MONDAY_PHASE):
    """n consecutive hourly unix timestamps starting at a 00:00-UTC Monday."""
    return [start + i * we.HOUR for i in range(n)]


def test_monday_phase_lands_on_monday():
    # MONDAY_PHASE must be 00:00 UTC Monday: weekday() == 0, hour/min/sec == 0.
    import datetime as dt
    d = dt.datetime.fromtimestamp(we.MONDAY_PHASE, dt.timezone.utc)
    assert d.weekday() == 0 and (d.hour, d.minute, d.second) == (0, 0, 0)


def test_cold_week_windows_warmup_and_alignment():
    # 3 full weeks of hourly bars; only Mondays with a full warmup prepad behind them qualify.
    n = we.WARMUP + 3 * we.WEEK_BARS
    idx = _hourly_index(n)
    df = pd.DataFrame({"A": np.zeros(n), "B": np.zeros(n)}, index=idx)
    wins = list(we.cold_week_windows(df))
    # week 0 starts at row 0 (no warmup behind it) -> skipped; weeks at row WARMUP and WARMUP+168 qualify.
    starts = [ws for ws, _ in wins]
    assert all(ws % we.WEEK_SECS == we.MONDAY_PHASE for ws in starts)        # all Monday-aligned
    assert all(win.index[0] == ws - we.WARMUP * we.HOUR for ws, win in wins)  # warmup prepad present
    assert len(wins) >= 1
    # the window carries warmup + the week's bars, never more than needed.
    for _ws, win in wins:
        assert we.WARMUP < len(win) <= we.WARMUP + we.WEEK_BARS


def test_split_label_boundaries():
    val, test = 1000, 2000
    assert we.split_label(999, val, test) == "train"
    assert we.split_label(1000, val, test) == "val"      # boundary is inclusive on the later split
    assert we.split_label(1999, val, test) == "val"
    assert we.split_label(2000, val, test) == "test"


def _wr(split, regime, r0, dd, days, bh):
    return we.WeekResult(ws=0, split=split, regime=regime, rung0_ret=r0, rung0_dd=dd,
                         rung0_trade_days=days, rung0_active_ok=(days >= 7),
                         buyhold_ret=bh, universe_ew_ret=r0)


def test_aggregate_oos_only_and_bull_gap():
    weeks = [
        _wr("train", "bull", 0.50, 0.05, 7, 0.10),   # in-sample, excluded from OOS aggregate
        _wr("val", "bull", 0.05, 0.10, 5, 0.25),     # OOS bull: gap +0.20
        _wr("val", "flat", -0.02, 0.40, 6, -0.05),   # OOS, DQ breach (dd 0.40 > 0.30)
        _wr("test", "bull", 0.10, 0.08, 7, 0.20),    # OOS bull: gap +0.10, activity OK
    ]
    agg = we.aggregate(weeks, oos_only=True)
    assert agg.n_weeks == 3                                   # train week excluded
    assert agg.splits == {"train": 1, "val": 2, "test": 1}
    assert agg.n_bull == 2
    assert abs(agg.bull_gap_mean - 0.15) < 1e-9              # mean of +0.20 and +0.10
    assert agg.rung0_dq_weeks == 1
    assert abs(agg.rung0_worst_dd - 0.40) < 1e-9
    assert agg.activity_fail_weeks == 2                      # the two with days < 7
    assert abs(agg.rung0_winrate - (2 / 3)) < 1e-9          # +0.05, +0.10 positive; -0.02 not


def test_bootstrap_mean_ci_brackets_mean_and_is_deterministic():
    vals = [0.10, -0.05, 0.20, -0.02, 0.08, 0.15, -0.10, 0.03]
    lo, hi, mean = we.bootstrap_mean_ci(vals, seed=0)
    assert lo < mean < hi
    assert abs(mean - float(np.mean(vals))) < 1e-12
    lo2, hi2, _ = we.bootstrap_mean_ci(vals, seed=0)
    assert (lo, hi) == (lo2, hi2)                    # deterministic on a fixed seed
    # one giant week must NOT drag the CI lower bound up enough to look robust.
    spiky = [0.92] + [-0.03] * 10                    # the s0 +92%-week pathology
    lo_s, _, mean_s = we.bootstrap_mean_ci(spiky, seed=0)
    assert mean_s > 0 and lo_s < mean_s              # mean positive, but the CI floor is far below it


def test_weekly_gate_pass_and_binding():
    # A clearly-winning config: beats B&H + rung-0 with a positive CI floor, survives, fully active.
    pol = [0.10, 0.08, 0.12, 0.09, 0.11]
    ok = we.weekly_gate(pol, [0.05] * 5, bh_rets=[0.02] * 5, rung0_rets=[0.01] * 5,
                        activity_ok=[True] * 5)
    assert ok["pass"] and ok["binding"] is None

    # A DQ breach binds first regardless of return.
    dq = we.weekly_gate(pol, [0.05, 0.40, 0.05, 0.05, 0.05], bh_rets=[0.0] * 5,
                        rung0_rets=[0.0] * 5, activity_ok=[True] * 5)
    assert not dq["pass"] and dq["binding"] == "survives_dq"

    # Loses to holding -> binds on beats_buyhold.
    lose = we.weekly_gate([0.01] * 5, [0.05] * 5, bh_rets=[0.15] * 5, rung0_rets=[0.0] * 5,
                          activity_ok=[True] * 5)
    assert not lose["pass"] and lose["binding"] == "beats_buyhold"

    # Activity is informational by default (a deploy-time rebalance guardrail, not a strategy gate):
    # a policy that wins returns but misses a day still PASSES unless require_activity is set.
    miss = [True, True, False, True, True]
    soft = we.weekly_gate(pol, [0.05] * 5, bh_rets=[0.0] * 5, rung0_rets=[0.0] * 5, activity_ok=miss)
    assert soft["pass"] and soft["activity_fail_weeks"] == 1
    hard = we.weekly_gate(pol, [0.05] * 5, bh_rets=[0.0] * 5, rung0_rets=[0.0] * 5,
                          activity_ok=miss, require_activity=True)
    assert not hard["pass"] and hard["binding"] == "activity_floor"


def test_trade_days_counts_distinct_traded_days_only():
    ws = we.MONDAY_PHASE
    records = [
        {"time": ws + 2 * we.HOUR, "trades_usd": {"A": 100.0}},        # day 0
        {"time": ws + 26 * we.HOUR, "trades_usd": {}},                 # day 1, no trade -> ignored
        {"time": ws + 30 * we.HOUR, "trades_usd": {"B": 50.0}},        # day 1, traded
        {"time": ws + 30 * we.HOUR, "trades_usd": {"B": 50.0}},        # same day 1 again -> dedup
        {"time": ws + 7 * 24 * we.HOUR, "trades_usd": {"C": 5.0}},     # day 7 -> out of 0..6 range
    ]
    assert we._trade_days(records, ws) == {0, 1}
