"""Unit tests for publish_leaderboard's PURE, torch- and I/O-free logic — the top-3 insert/sort/
truncate/evict ranking, the two-score resolver (`resolve_scores`), and the config-aggregate guard
that feed the rolling `simulated_leaderboard.json` ([[Dashboard Leaderboard]] Phase 2). The PUBLISH
step (PUT + CloudFront) is desktop-only (creds) and is NOT exercised here; these pin the maths that
ARE laptop-testable. Importing the module must stay torch-free (the torch imports live in main())."""

from __future__ import annotations

import os
import sys

sys.path.insert(0, "scripts")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import publish_leaderboard as pl  # noqa: E402  (must import torch-free)


# --- helpers --------------------------------------------------------------------------------------

def _entry(run_id: str, score: float | None) -> dict:
    """A minimal leaderboard entry; `score` drives the ranking via `cumulative_score` (the 6-mo-return
    ranker). weekly_score is set equal so the tests stay agnostic to which field is the rank key."""
    return {"run_id": run_id, "weekly_score": score, "cumulative_score": score,
            "config_seed_mean": 0.0, "dq_pass": True, "windows": {},
            "trades_path": f"{run_id}/simulated_trades.json", "generated": "t"}


def _ranks(board: list[dict]) -> list[int]:
    return [e["rank"] for e in board]


# --- update_leaderboard: insertion ----------------------------------------------------------------

def test_insert_into_empty_board():
    board, evicted = pl.update_leaderboard([], _entry("a", 0.10), k=3)
    assert [e["run_id"] for e in board] == ["a"]
    assert board[0]["rank"] == 1
    assert evicted == []


def test_insert_into_under_k_board():
    cur = [dict(_entry("a", 0.20), rank=1)]
    board, evicted = pl.update_leaderboard(cur, _entry("b", 0.10), k=3)
    assert [e["run_id"] for e in board] == ["a", "b"]      # a (0.20) > b (0.10)
    assert _ranks(board) == [1, 2]
    assert evicted == []


def test_insert_that_makes_the_top_k_no_eviction():
    # board has room (2 < k=3); the new entry slots in and nothing is evicted.
    cur = [dict(_entry("a", 0.30), rank=1), dict(_entry("b", 0.10), rank=2)]
    board, evicted = pl.update_leaderboard(cur, _entry("c", 0.20), k=3)
    assert [e["run_id"] for e in board] == ["a", "c", "b"]   # 0.30, 0.20, 0.10
    assert _ranks(board) == [1, 2, 3]
    assert evicted == []


# --- update_leaderboard: eviction -----------------------------------------------------------------

def test_insert_evicts_old_kth():
    # full board (a:0.30, b:0.20, c:0.10); new d:0.25 bumps c off the bottom.
    cur = [dict(_entry("a", 0.30), rank=1), dict(_entry("b", 0.20), rank=2),
           dict(_entry("c", 0.10), rank=3)]
    board, evicted = pl.update_leaderboard(cur, _entry("d", 0.25), k=3)
    assert [e["run_id"] for e in board] == ["a", "d", "b"]   # 0.30, 0.25, 0.20
    assert _ranks(board) == [1, 2, 3]
    assert evicted == ["c"]                                  # the old #3, de-listed
    assert all(e["run_id"] != "c" for e in board)            # and genuinely dropped


def test_new_entry_that_misses_the_cut_is_itself_evicted():
    # full board, new entry's score is below every member -> it never lists; nothing existing drops.
    cur = [dict(_entry("a", 0.30), rank=1), dict(_entry("b", 0.20), rank=2),
           dict(_entry("c", 0.10), rank=3)]
    board, evicted = pl.update_leaderboard(cur, _entry("z", 0.05), k=3)
    assert [e["run_id"] for e in board] == ["a", "b", "c"]   # unchanged top-3
    assert _ranks(board) == [1, 2, 3]
    assert evicted == []                                     # no PRIOR member was de-listed
    assert all(e["run_id"] != "z" for e in board)            # the newcomer is not listed


# --- update_leaderboard: upsert -------------------------------------------------------------------

def test_upsert_replaces_existing_run_id_no_dup():
    # re-publishing 'b' with a better score must REPLACE (not duplicate) and re-sort.
    cur = [dict(_entry("a", 0.30), rank=1), dict(_entry("b", 0.10), rank=2),
           dict(_entry("c", 0.05), rank=3)]
    board, evicted = pl.update_leaderboard(cur, _entry("b", 0.40), k=3)
    ids = [e["run_id"] for e in board]
    assert ids == ["b", "a", "c"]                            # b now 0.40 -> #1
    assert ids.count("b") == 1                               # no duplicate
    assert _ranks(board) == [1, 2, 3]
    assert evicted == []                                     # b was listed and still is


def test_upsert_carries_new_payload_not_stale():
    cur = [dict(_entry("a", 0.10), rank=1)]
    fresh = dict(_entry("a", 0.10), config_seed_mean=0.99, generated="NEW")
    board, _ = pl.update_leaderboard(cur, fresh, k=3)
    assert board[0]["config_seed_mean"] == 0.99
    assert board[0]["generated"] == "NEW"


# --- update_leaderboard: ties + determinism -------------------------------------------------------

def test_tie_on_weekly_score_is_deterministic_by_run_id():
    # equal scores -> stable order by run_id ascending, independent of input order.
    a, b, c = _entry("zeta", 0.10), _entry("alpha", 0.10), _entry("mid", 0.10)
    board1, _ = pl.update_leaderboard([a, b], c, k=3)
    board2, _ = pl.update_leaderboard([c, a], b, k=3)        # different input order, same set
    assert [e["run_id"] for e in board1] == ["alpha", "mid", "zeta"]
    assert [e["run_id"] for e in board1] == [e["run_id"] for e in board2]
    assert _ranks(board1) == [1, 2, 3]


def test_none_weekly_score_sorts_last():
    cur = [dict(_entry("a", 0.10), rank=1)]
    board, evicted = pl.update_leaderboard(cur, _entry("b", None), k=3)
    assert [e["run_id"] for e in board] == ["a", "b"]        # None sorts below 0.10
    assert evicted == []


def test_none_weekly_score_never_evicts_a_scored_entry():
    cur = [dict(_entry("a", 0.30), rank=1), dict(_entry("b", 0.20), rank=2),
           dict(_entry("c", 0.10), rank=3)]
    board, evicted = pl.update_leaderboard(cur, _entry("z", None), k=3)
    assert [e["run_id"] for e in board] == ["a", "b", "c"]
    assert evicted == []


# --- update_leaderboard: rank reassignment + k -----------------------------------------------------

def test_ranks_are_1_to_k_contiguous():
    cur = []
    for rid, s in [("a", 0.5), ("b", 0.4), ("c", 0.3), ("d", 0.2)]:
        cur, _ = pl.update_leaderboard(cur, _entry(rid, s), k=3)
    assert _ranks(cur) == [1, 2, 3]
    assert len(cur) == 3
    assert [e["run_id"] for e in cur] == ["a", "b", "c"]     # d (0.2) never made it


def test_k_one_keeps_single_best_and_evicts_prior():
    cur = [dict(_entry("a", 0.10), rank=1)]
    board, evicted = pl.update_leaderboard(cur, _entry("b", 0.20), k=1)
    assert [e["run_id"] for e in board] == ["b"]
    assert evicted == ["a"]


def test_invalid_k_and_missing_run_id_raise():
    import pytest
    with pytest.raises(ValueError):
        pl.update_leaderboard([], _entry("a", 0.1), k=0)
    with pytest.raises(ValueError):
        pl.update_leaderboard([], {"weekly_score": 0.1}, k=3)


def test_does_not_mutate_inputs():
    cur = [dict(_entry("a", 0.30), rank=1), dict(_entry("b", 0.20), rank=2),
           dict(_entry("c", 0.10), rank=3)]
    snapshot = [dict(e) for e in cur]
    new_entry = _entry("d", 0.25)
    new_snapshot = dict(new_entry)
    pl.update_leaderboard(cur, new_entry, k=3)
    assert cur == snapshot          # the caller's current list is untouched
    assert new_entry == new_snapshot


# --- resolve_scores: the two model-own scores from meta.windows -----------------------------------

def test_resolve_scores_oos_weekly_mean_and_cumulative():
    # weekly_score = (val.ret_sum + test.ret_sum) / (val.n_weeks + test.n_weeks); cumulative = overall.ret_sum.
    windows = {
        "train": {"ret_sum": 0.57, "n_weeks": 17},
        "val": {"ret_sum": 0.101, "n_weeks": 6},
        "test": {"ret_sum": 0.256, "n_weeks": 5},
        "overall": {"ret_sum": 0.93, "n_weeks": 28},
    }
    weekly, cum, src = pl.resolve_scores(windows)
    assert abs(weekly - (0.101 + 0.256) / (6 + 5)) < 1e-12   # OOS per-week mean (~+3.25%/wk)
    assert abs(cum - 0.93) < 1e-12
    assert "OOS" in src and "overall" in src


def test_resolve_scores_excludes_train_from_weekly():
    # train weeks must NOT enter weekly_score (only val + test). A huge train number can't move it.
    windows = {
        "train": {"ret_sum": 99.0, "n_weeks": 17},
        "val": {"ret_sum": 0.12, "n_weeks": 6},
        "test": {"ret_sum": 0.18, "n_weeks": 6},
        "overall": {"ret_sum": 99.30, "n_weeks": 29},
    }
    weekly, cum, _ = pl.resolve_scores(windows)
    assert abs(weekly - (0.12 + 0.18) / 12) < 1e-12          # train's 99.0 is excluded
    assert abs(cum - 99.30) < 1e-12                          # cumulative IS the full overall


def test_resolve_scores_handles_empty_test_window():
    # n_weighting means an empty test window (n=0) reduces weekly_score to val-only, no div-by-zero.
    windows = {"val": {"ret_sum": 0.12, "n_weeks": 6}, "test": {"ret_sum": 0.0, "n_weeks": 0},
               "overall": {"ret_sum": 0.12, "n_weeks": 6}}
    weekly, cum, _ = pl.resolve_scores(windows)
    assert abs(weekly - 0.12 / 6) < 1e-12
    assert abs(cum - 0.12) < 1e-12


def test_resolve_scores_unavailable_when_no_oos_weeks():
    weekly, cum, src = pl.resolve_scores({"train": {"ret_sum": 0.5, "n_weeks": 10}})
    assert weekly is None
    assert cum is None
    assert src == "unavailable"


def test_resolve_scores_not_a_dict_is_graceful():
    weekly, cum, src = pl.resolve_scores(None)
    assert weekly is None and cum is None and src == "unavailable"


# --- config_aggregate_from_ledger -----------------------------------------------------------------

def _row(run_id, cfg, ret, legal_dd, split="val"):
    return {"run_id": run_id, "config_label": cfg, "return": ret, "legal_dd": legal_dd, "split": split}


def test_config_aggregate_means_seeds_and_dq_pass():
    cfg = "ppo-event-rdLe4-wkw-ef0af8f"
    rows = [_row(f"{cfg}-s{i}", cfg, r, True) for i, r in
            enumerate([-0.0286, 0.0061, 0.0388, 0.1648])]
    mean, dq, label = pl.config_aggregate_from_ledger(f"{cfg}-s3", rows)
    assert label == cfg
    assert abs(mean - (sum([-0.0286, 0.0061, 0.0388, 0.1648]) / 4)) < 1e-9
    assert dq is True


def test_config_aggregate_dq_fails_if_any_seed_illegal():
    cfg = "cfgX"
    rows = [_row(f"{cfg}-s0", cfg, 0.1, True), _row(f"{cfg}-s1", cfg, 0.2, False)]
    mean, dq, _ = pl.config_aggregate_from_ledger(f"{cfg}-s0", rows)
    assert dq is False


def test_config_aggregate_missing_config_is_graceful():
    mean, dq, label = pl.config_aggregate_from_ledger("ppo-unknown-cfg-s2", [])
    assert mean is None
    assert dq is False
    assert label == "ppo-unknown-cfg"     # the derived label, still useful for logging


def test_config_aggregate_matches_via_run_id_prefix_when_label_field_absent():
    # ledger rows without a config_label still match by stripping the seed suffix off run_id.
    cfg = "cfgY"
    rows = [{"run_id": f"{cfg}-s0", "return": 0.1, "legal_dd": True, "split": "val"},
            {"run_id": f"{cfg}-s1", "return": 0.3, "legal_dd": True, "split": "val"}]
    mean, dq, label = pl.config_aggregate_from_ledger(f"{cfg}-s1", rows)
    assert label == cfg
    assert abs(mean - 0.2) < 1e-12
    assert dq is True


def test_resolve_config_guard_override_skips_ledger():
    # override_mean given -> use the passed values, ignore the ledger entirely (laptop-computed guard).
    mean, dq, label = pl.resolve_config_guard("ppo-x-cfg-s3", [], override_mean=0.045, override_dq=True)
    assert mean == 0.045
    assert dq is True
    assert label == "ppo-x-cfg"


def test_resolve_config_guard_override_dq_false_default():
    mean, dq, _ = pl.resolve_config_guard("ppo-x-cfg-s3", [], override_mean=-0.0275)
    assert mean == -0.0275
    assert dq is False                       # override_dq defaults False (omit --dq-pass => fail)


def test_resolve_config_guard_falls_back_to_ledger_without_override():
    cfg = "cfgZ"
    rows = [_row(f"{cfg}-s0", cfg, 0.1, True), _row(f"{cfg}-s1", cfg, 0.3, True)]
    mean, dq, label = pl.resolve_config_guard(f"{cfg}-s1", rows)   # no override -> ledger
    assert abs(mean - 0.2) < 1e-12
    assert dq is True
    assert label == cfg


def test_strip_seed_suffix():
    assert pl.strip_seed_suffix("ppo-event-rdLe4-wkw-ef0af8f-s3") == "ppo-event-rdLe4-wkw-ef0af8f"
    assert pl.strip_seed_suffix("ppo-event-rdLe4r-68b268f-s0") == "ppo-event-rdLe4r-68b268f"
    assert pl.strip_seed_suffix("no-seed-here") == "no-seed-here"
    assert pl.strip_seed_suffix("ends-s12") == "ends"


# --- build_entry: the published schema ------------------------------------------------------------

def test_build_entry_schema():
    windows = {"train": {}, "val": {}, "test": {}, "overall": {}}
    e = pl.build_entry("rid-s0", 0.0325, 0.93, "windows: OOS(val+test) per-week mean + overall.ret_sum",
                       0.045, True, windows, generated="G")
    assert set(e.keys()) == {"run_id", "weekly_score", "cumulative_score", "score_source",
                             "config_seed_mean", "dq_pass", "windows", "trades_path", "generated"}
    assert e["run_id"] == "rid-s0"
    assert e["weekly_score"] == 0.0325
    assert e["cumulative_score"] == 0.93
    assert e["trades_path"] == "rid-s0/simulated_trades.json"
    assert e["generated"] == "G"
    assert "rank" not in e          # rank is assigned by update_leaderboard, not build_entry
