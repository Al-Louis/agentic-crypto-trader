"""The event-agent entry point's pure logic: tick scheduling, the paper-only mode gate, and
the provenance/selection loaders. The full tick loop needs the checkpoint (torch) and is
covered by the EventRunner tests + the on-box dry-run; here we cover everything else."""

import json
import os

import pytest

from trader.agent import event_agent as ea


# --- scheduling math ---------------------------------------------------------

def test_seconds_until_next_tick_lands_on_boundary_plus_offset():
    interval, offset = 3600, 180
    for now in (1_000_000, 1_000_181, 999_999, 1_762_322_400):
        wait = ea.seconds_until_next_tick(now, interval, offset)
        target = now + wait
        assert wait > 0
        assert wait <= interval                       # never more than one period away
        assert int(target) % interval == offset       # exactly on a boundary + offset


def test_seconds_until_next_tick_skips_the_current_offset_if_passed():
    # exactly at a boundary+offset -> must target the NEXT period (strictly future)
    now = (1_000_000 // 3600) * 3600 + 180
    assert ea.seconds_until_next_tick(now, 3600, 180) == 3600.0


# --- mode gate ---------------------------------------------------------------

def test_mode_gate_defaults_to_paper(monkeypatch):
    monkeypatch.delenv("TRADER_MODE", raising=False)
    assert ea._resolve_mode() == "paper"


def test_mode_gate_refuses_live_and_garbage(monkeypatch):
    monkeypatch.setenv("TRADER_MODE", "live")
    with pytest.raises(SystemExit) as e:
        ea._resolve_mode()
    assert e.value.code == 2
    monkeypatch.setenv("TRADER_MODE", "wibble")
    with pytest.raises(SystemExit):
        ea._resolve_mode()


# --- loaders -----------------------------------------------------------------

def test_load_provenance_extracts_nested_block(tmp_path):
    run = tmp_path / "rid"
    run.mkdir()
    (run / "metrics.json").write_text(json.dumps(
        {"provenance": {"k": 8, "recurrent": True, "reward_mode": "relative"}}), encoding="utf-8")
    prov = ea.load_provenance(str(tmp_path), "rid")
    assert prov["k"] == 8 and prov["recurrent"] is True


def test_load_selection_reads_symbol_and_pool():
    sel_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            "data", "selection.json")
    if not os.path.isfile(sel_path):
        pytest.skip("recorded selection.json absent")
    sel = ea.load_selection(sel_path)
    assert sel and all("symbol" in s and "pair_address" in s for s in sel)


def test_parser_requires_run_dir_and_defaults():
    args = ea.build_parser().parse_args(["--run-dir", "runs-rl/foo"])
    assert args.run_dir == "runs-rl/foo" and args.run_id is None
    assert args.interval_secs == ea.HOUR and not args.once
    with pytest.raises(SystemExit):
        ea.build_parser().parse_args([])               # --run-dir is required
