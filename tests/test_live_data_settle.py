"""The settle-wait fix for the GeckoTerminal candle-lag (live_data.update_live).

The live B trade ate ~5% slip because the HH:03 tick fetched before Gecko had finalized the
just-closed bar, so the env missed it for a FULL hour. `update_live(..., settle_max_wait>0)`
re-polls the active pools within the same tick until the just-closed bar lands, so the agent
decides on it THIS tick. These tests seed their own tmp pool (no recorded cache, no network),
so they always run — `fetch_fn`/`sleep` are injected to drive the timing deterministically.
"""

import os

import pandas as pd

from trader.agent import live_data as ld
from trader.data.downloader import OHLCV_COLS, _store_dir

H = 3600


def _seed_pool(root: str, sym: str, pool: str, bars: list[list]) -> None:
    d = _store_dir(root, sym, pool, "hour_1")
    os.makedirs(d, exist_ok=True)
    df = pd.DataFrame(bars, columns=OHLCV_COLS)
    df["timestamp"] = df["timestamp"].astype("int64")
    df.to_parquet(os.path.join(d, f"p_{int(df['timestamp'].min())}.parquet"), index=False)


def _no_network(monkeypatch) -> None:
    """Neutralize the anchor refresh + factor regen so update_live exercises only the OHLCV path."""
    monkeypatch.setattr(ld, "refresh_anchors", lambda **k: {})
    monkeypatch.setattr(ld, "refresh_factor_features", lambda *a, **k: [])


def test_just_closed_open_boundaries():
    assert ld.just_closed_open(100 * H + 200) == 99 * H     # 3m20s after the close
    assert ld.just_closed_open(100 * H) == 99 * H           # exactly on the boundary
    assert ld.just_closed_open(100 * H + H - 1) == 99 * H   # just before the next close


def test_settle_disabled_is_single_pass(tmp_path, monkeypatch):
    """settle_max_wait=0 -> the pre-existing single fetch pass, byte-identical (no re-polls)."""
    _no_network(monkeypatch)
    root = str(tmp_path / "ohlcv")
    _seed_pool(root, "FOO", "0xfoo", [[98 * H, 1, 1, 1, 1, 1]])
    calls = {"n": 0}

    def fetch_fn(_pool):
        calls["n"] += 1
        return [[98 * H, 1, 1, 1, 1, 1]]                    # never provides the just-closed bar

    out = ld.update_live([{"symbol": "FOO", "pair_address": "0xfoo"}], 100 * H + 200,
                         ohlcv_root=root, settle_max_wait=0.0, fetch_fn=fetch_fn,
                         sleep=lambda *_: None, min_interval=0.0)
    assert calls["n"] == 1                                  # exactly one fetch; no settle re-polls
    assert out["settle"]["enabled"] is False


def test_settle_catches_late_just_closed_bar(tmp_path, monkeypatch):
    """The decisive case: Gecko publishes the just-closed bar LATE; the wait catches it this tick."""
    _no_network(monkeypatch)
    root = str(tmp_path / "ohlcv")
    now, target = 100 * H + 200, 99 * H
    _seed_pool(root, "FOO", "0xfoo", [[97 * H, 1, 1, 1, 1, 1], [98 * H, 1, 1, 1, 1, 1]])  # active
    calls = {"n": 0}

    def fetch_fn(_pool):
        calls["n"] += 1
        if calls["n"] < 2:                                 # initial pass: Gecko hasn't published it
            return [[98 * H, 1, 1, 1, 1, 1]]
        return [[98 * H, 1, 1, 1, 1, 1], [target, 2, 2, 2, 2, 2]]   # later poll: now it's there

    out = ld.update_live([{"symbol": "FOO", "pair_address": "0xfoo"}], now, ohlcv_root=root,
                         settle_max_wait=300.0, settle_poll=45.0, fetch_fn=fetch_fn,
                         sleep=lambda *_: None, min_interval=0.0)
    assert ld.cached_newest_ts("FOO", "0xfoo", root=root) == target   # the late bar landed THIS tick
    assert calls["n"] >= 2 and out["settle"]["polls"] >= 1
    assert out["settle"]["still_missing"] == []


def test_settle_skips_perma_stale_pool(tmp_path, monkeypatch):
    """A pool whose newest bar is older than the active window is NOT waited on (would never come)."""
    _no_network(monkeypatch)
    root = str(tmp_path / "ohlcv")
    _seed_pool(root, "STALE", "0xstale", [[79 * H, 1, 1, 1, 1, 1], [80 * H, 1, 1, 1, 1, 1]])  # 20h old
    calls = {"n": 0}

    def fetch_fn(_pool):
        calls["n"] += 1
        return [[80 * H, 1, 1, 1, 1, 1]]                    # still stale, never the just-closed bar

    out = ld.update_live([{"symbol": "STALE", "pair_address": "0xstale"}], 100 * H + 200,
                         ohlcv_root=root, settle_max_wait=300.0, settle_poll=45.0, fetch_fn=fetch_fn,
                         sleep=lambda *_: None, min_interval=0.0)
    assert out["settle"]["active"] == 0                    # outside the 6h window -> not waited on
    assert calls["n"] == 1                                 # only the initial pass; loop body skipped


def test_settle_deadline_backstop(tmp_path, monkeypatch):
    """An active pool whose bar never arrives must stop at the deadline, not hang."""
    _no_network(monkeypatch)
    root = str(tmp_path / "ohlcv")
    _seed_pool(root, "FOO", "0xfoo", [[98 * H, 1, 1, 1, 1, 1]])     # active but the bar never comes
    out = ld.update_live([{"symbol": "FOO", "pair_address": "0xfoo"}], 100 * H + 200,
                         ohlcv_root=root, settle_max_wait=90.0, settle_poll=45.0,
                         fetch_fn=lambda _p: [[98 * H, 1, 1, 1, 1, 1]],
                         sleep=lambda *_: None, min_interval=0.0)
    assert out["settle"]["still_missing"] == ["FOO"]       # gave up at the deadline, didn't hang
    assert out["settle"]["polls"] == 2 and out["settle"]["waited"] == 90.0


def test_settle_partial_universe_some_late_some_stale(tmp_path, monkeypatch):
    """Mixed universe: one active pool settles late (caught), one stale pool ignored, one already-current."""
    _no_network(monkeypatch)
    root = str(tmp_path / "ohlcv")
    now, target = 100 * H + 200, 99 * H
    _seed_pool(root, "LATE", "0xlate", [[98 * H, 1, 1, 1, 1, 1]])      # active, will settle on poll 2
    _seed_pool(root, "CUR", "0xcur", [[98 * H, 1, 1, 1, 1, 1]])        # active, has target on pass 1
    _seed_pool(root, "STALE", "0xstale", [[80 * H, 1, 1, 1, 1, 1]])    # stale, ignored
    n = {"LATE": 0}

    def fetch_fn(pool):
        if pool == "0xcur":
            return [[target, 5, 5, 5, 5, 5]]                # current immediately
        if pool == "0xstale":
            return [[80 * H, 1, 1, 1, 1, 1]]
        n["LATE"] += 1                                      # LATE: only after the first fetch
        return [[target, 2, 2, 2, 2, 2]] if n["LATE"] >= 2 else [[98 * H, 1, 1, 1, 1, 1]]

    sel = [{"symbol": "LATE", "pair_address": "0xlate"},
           {"symbol": "CUR", "pair_address": "0xcur"},
           {"symbol": "STALE", "pair_address": "0xstale"}]
    out = ld.update_live(sel, now, ohlcv_root=root, settle_max_wait=300.0, settle_poll=45.0,
                         fetch_fn=fetch_fn, sleep=lambda *_: None, min_interval=0.0)
    assert out["settle"]["active"] == 2                    # LATE + CUR (STALE excluded)
    assert out["settle"]["still_missing"] == []            # LATE caught, CUR already had it
    assert ld.cached_newest_ts("LATE", "0xlate", root=root) == target
    assert ld.cached_newest_ts("CUR", "0xcur", root=root) == target
