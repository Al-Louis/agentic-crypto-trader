"""Unit tests for simulate_weekly's PURE, torch-free helpers — the split labeling and the
3-window aggregation that feed the dashboard leaderboard's overview (Dashboard Leaderboard.md
Phase 1). The full end-to-end run is DESKTOP-ONLY (torch + the policy); these pin the maths that
ARE testable on a torch-free laptop. Importing the module must itself stay torch-free (the torch
import lives inside main()), so this test imports the pure functions at module load."""

from __future__ import annotations

import os
import sys

sys.path.insert(0, "scripts")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import simulate_weekly as sw  # noqa: E402  (must import torch-free)


# --- split labeling -------------------------------------------------------------------------------

def test_label_week_split_boundaries():
    # train_end / val_end are the INCLUSIVE last bar of each split (train_r.index[-1], val_r.index[-1]).
    train_end, val_end = 1000, 2000
    assert sw.label_week_split(999, train_end, val_end) == "train"
    assert sw.label_week_split(1000, train_end, val_end) == "train"   # train_end belongs to train
    assert sw.label_week_split(1001, train_end, val_end) == "val"
    assert sw.label_week_split(2000, train_end, val_end) == "val"     # val_end belongs to val
    assert sw.label_week_split(2001, train_end, val_end) == "test"
    assert sw.label_week_split(9999, train_end, val_end) == "test"


def test_label_matches_time_split_partition():
    """Labels derived from time_split boundaries must partition a synthetic week-start grid the SAME
    way time_split slices it (the gate's split) — train+val+test = all weeks, no overlap, in order."""
    import numpy as np
    import pandas as pd
    from train_rl import time_split

    n = 280
    idx = [sw.MONDAY_PHASE + i * sw.WEEK_SECS for i in range(n)]   # 280 Monday-aligned week starts
    returns = pd.DataFrame({"A": np.zeros(n)}, index=idx)
    train_r, val_r, test_r = time_split(returns)                  # 60 / 20 / 20 by row position
    train_end, val_end = int(train_r.index[-1]), int(val_r.index[-1])

    labels = [sw.label_week_split(t, train_end, val_end) for t in idx]
    # the label counts must equal the time_split slice lengths exactly
    assert labels.count("train") == len(train_r)
    assert labels.count("val") == len(val_r)
    assert labels.count("test") == len(test_r)
    # and the assignment must be set-identical to the actual slices (not just counts)
    assert {t for t, lab in zip(idx, labels) if lab == "train"} == set(int(t) for t in train_r.index)
    assert {t for t, lab in zip(idx, labels) if lab == "val"} == set(int(t) for t in val_r.index)
    assert {t for t, lab in zip(idx, labels) if lab == "test"} == set(int(t) for t in test_r.index)
    # monotone: train weeks all precede val weeks all precede test weeks (it's a time split)
    assert max(t for t, l in zip(idx, labels) if l == "train") < min(t for t, l in zip(idx, labels) if l == "val")
    assert max(t for t, l in zip(idx, labels) if l == "val") < min(t for t, l in zip(idx, labels) if l == "test")


# --- window aggregation ---------------------------------------------------------------------------

def _wk(start, split, ret, dd):
    return {"start": start, "split": split, "return": ret, "dd": dd}


def test_summarize_windows_arithmetic():
    weeks = [
        _wk(0, "train", 0.10, 0.05),
        _wk(1, "train", -0.04, 0.12),
        _wk(2, "val", 0.20, 0.08),
        _wk(3, "val", 0.00, 0.03),     # return == 0 is NOT a win (strictly > 0)
        _wk(4, "test", -0.10, 0.25),
    ]
    w = sw.summarize_windows(weeks)

    # train: 2 weeks, returns 0.10 + -0.04
    assert w["train"]["n_weeks"] == 2
    assert abs(w["train"]["ret_sum"] - 0.06) < 1e-12
    assert abs(w["train"]["ret_mean"] - 0.03) < 1e-12
    assert w["train"]["worst_week_dd"] == 0.12
    assert w["train"]["win_rate"] == 0.5

    # val: 2 weeks, one win (0.20), one zero (not a win)
    assert w["val"]["n_weeks"] == 2
    assert abs(w["val"]["ret_sum"] - 0.20) < 1e-12
    assert w["val"]["win_rate"] == 0.5
    assert w["val"]["worst_week_dd"] == 0.08

    # test: 1 week, a loss
    assert w["test"]["n_weeks"] == 1
    assert abs(w["test"]["ret_sum"] - (-0.10)) < 1e-12
    assert w["test"]["win_rate"] == 0.0
    assert w["test"]["worst_week_dd"] == 0.25


def test_summarize_windows_overall_is_all_weeks():
    weeks = [
        _wk(0, "train", 0.10, 0.05),
        _wk(1, "train", -0.04, 0.12),
        _wk(2, "val", 0.20, 0.08),
        _wk(3, "val", 0.00, 0.03),
        _wk(4, "test", -0.10, 0.25),
    ]
    w = sw.summarize_windows(weeks)
    # overall = every week, regardless of split
    assert w["overall"]["n_weeks"] == len(weeks)
    assert abs(w["overall"]["ret_sum"] - sum(x["return"] for x in weeks)) < 1e-12
    assert w["overall"]["worst_week_dd"] == max(x["dd"] for x in weeks)
    # the 3 windows PARTITION overall: train + val + test counts == overall count
    assert w["train"]["n_weeks"] + w["val"]["n_weeks"] + w["test"]["n_weeks"] == w["overall"]["n_weeks"]
    # ret_sum is additive across the partition
    assert abs(w["train"]["ret_sum"] + w["val"]["ret_sum"] + w["test"]["ret_sum"]
               - w["overall"]["ret_sum"]) < 1e-12


def test_summarize_windows_empty_split_is_zeroed_not_crash():
    # a window with no weeks (e.g. all weeks in train) must yield zeros, not a div-by-zero.
    weeks = [_wk(0, "train", 0.10, 0.05), _wk(1, "train", 0.05, 0.02)]
    w = sw.summarize_windows(weeks)
    for empty in ("val", "test"):
        assert w[empty]["n_weeks"] == 0
        assert w[empty]["ret_sum"] == 0.0
        assert w[empty]["ret_mean"] == 0.0
        assert w[empty]["worst_week_dd"] == 0.0
        assert w[empty]["win_rate"] == 0.0
    assert w["overall"]["n_weeks"] == 2


def test_summarize_windows_no_weeks():
    w = sw.summarize_windows([])
    for split in ("train", "val", "test", "overall"):
        assert w[split] == {"ret_sum": 0.0, "ret_mean": 0.0, "worst_week_dd": 0.0,
                            "win_rate": 0.0, "n_weeks": 0}


def test_window_keys_match_contract():
    # the emitted schema (Dashboard Leaderboard.md): each window has exactly these 5 fields.
    w = sw.summarize_windows([_wk(0, "val", 0.1, 0.05)])
    assert set(w.keys()) == {"train", "val", "test", "overall"}
    for split in w:
        assert set(w[split].keys()) == {"ret_sum", "ret_mean", "worst_week_dd", "win_rate", "n_weeks"}


# --- per-week return + drawdown (the eq-based mark) -----------------------------------------------

def test_week_return_dd_uses_full_eq_not_a_dropped_slice():
    """Regression for the worst_dd=0 bug: evaluate_event_policy's eq is already week-only (seeded at
    reset(start=WARMUP)), so week_return_dd must mark the FULL series, not eq.iloc[WARMUP:]. A series
    that rises then dips must report a NONZERO drawdown."""
    import pandas as pd
    cap = sw.START_CAPITAL
    eq = pd.Series([cap, cap * 1.05, cap * 0.98, cap * 1.02])     # peak +5%, dip to -2%, end +2%
    ret, dd = sw.week_return_dd(eq)
    assert abs(ret - 0.02) < 1e-12                                # week return = last/cap - 1
    assert abs(dd - (1.0 - 0.98 / 1.05)) < 1e-12                  # worst DD from the +5% peak (~6.67%)
    assert dd > 0.0                                               # the bug returned 0.0


def test_week_return_dd_empty_is_zeroed():
    import pandas as pd
    assert sw.week_return_dd(pd.Series([], dtype=float)) == (0.0, 0.0)


# --- fold_positions: dust crumb / negative-exit_price regression --------------------------------
def _mk(side, t, price, usd, fee=0.0):
    return {"side": side, "time": t, "price": price, "usd": usd, "fee": fee}


def test_fold_positions_drops_dust_no_corrupt_exit():
    """Regression for the SIREN exit_price=-0.124 / -230% bug: a FIFO-unwind float crumb (qty ~1e-5)
    must be DROPPED, not emitted and then snapped (which divided the ledger residual by the dust qty
    and blew exit_price negative). The real round-trip survives with a sane, positive exit."""
    buy_qty = 1000.0 / 0.10                                       # 10000.0
    markers = [_mk("buy", 100, 0.10, 1000.0),
               _mk("sell", 200, 0.09, 0.09 * (buy_qty - 1e-5))]  # consumes all but a ~1e-5 crumb lot
    out = sw.fold_positions(markers, last_t=300, ledger_pnl=-50.0)
    assert out, "the real round-trip must be emitted"
    assert all(p["qty"] * p["entry_price"] > sw.POSITION_DUST_USD for p in out)   # dust dropped
    assert all(p["exit_price"] > 0 for p in out)                                  # never negative
    assert all(abs(p["exit_price"] / p["entry_price"] - 1.0) < 1.5 for p in out)  # never absurd
    recon = sum(p["qty"] * (p["exit_price"] - p["entry_price"]) for p in out)
    assert abs(recon - (-50.0)) < 1e-6                           # snap still hits the exact ledger PnL


def test_fold_positions_snaps_onto_largest_notional():
    """The ledger snap goes onto the LARGEST-notional position (so the per-unit nudge is tiny), not
    the last one. The small round-trip keeps its natural exit; the large one absorbs the residual."""
    markers = [_mk("buy", 100, 0.10, 500.0), _mk("sell", 150, 0.11, 0.11 * 5000.0),     # small qty 5000
               _mk("buy", 200, 0.10, 2000.0), _mk("sell", 250, 0.12, 0.12 * 20000.0)]   # large qty 20000
    out = sw.fold_positions(markers, last_t=300, ledger_pnl=1234.5)
    assert len(out) == 2
    small = min(out, key=lambda p: p["qty"]); large = max(out, key=lambda p: p["qty"])
    assert abs(small["exit_price"] - 0.11) < 1e-9                # small untouched (natural sell price)
    assert large["exit_price"] > 0.12                           # large absorbed the +residual
    recon = sum(p["qty"] * (p["exit_price"] - p["entry_price"]) for p in out)
    assert abs(recon - 1234.5) < 1e-6                           # exact ledger recon preserved
