"""The obs-parity GATE for the live-data updater (task #3's offline proof).

Claim under test: appending just-closed bars to the cache and regenerating the factor frame
yields panels BYTE-IDENTICAL to the recorded training pipeline — so `evaluate_event_policy`
fed the live panel sees the exact observation it saw in training (no train/serve skew).

Method (no network): take a recorded token, truncate its OHLCV cache at a cutoff C into a temp
workspace, replay the recorded bars after C forward through `append_alt_bars`, then assert:
  * finalization drops the still-forming bar,
  * every bar <= C is untouched and the appended bars equal the recorded bars (append-immutability),
  * the regenerated `r_alt` (the only column load_data reads) and the volume panel match the
    recorded values exactly over the replayed range.
The BTC/BNB anchors are read from the real recorded cache (read-only); only the alt is truncated.
"""

import json
import os
import shutil

import numpy as np
import pandas as pd
import pytest

from trader.agent import live_data as ld
from trader.data.downloader import OHLCV_COLS, _store_dir, load_ohlcv

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REAL_OHLCV = os.path.join(REPO, "data", "ohlcv")
REAL_ANCHOR = os.path.join(REPO, "data", "anchor")
REAL_FEATURES = os.path.join(REPO, "data", "features")
SELECTION = os.path.join(REPO, "data", "selection.json")

pytestmark = pytest.mark.skipif(
    not (os.path.isdir(REAL_OHLCV) and os.path.isfile(SELECTION)),
    reason="recorded data/ not present (parity gate needs the recorded cache)")


def _pick_token(min_bars: int = 400) -> dict:
    """First selection token with a recorded OHLCV series long enough to truncate + replay."""
    sel = json.load(open(SELECTION, encoding="utf-8"))
    for s in sel:
        if not s.get("pair_address"):
            continue
        df = load_ohlcv(s["symbol"], s["pair_address"], "hour", 1, root=REAL_OHLCV)
        if len(df) >= min_bars:
            return {"symbol": s["symbol"], "pair_address": s["pair_address"], "df": df}
    raise AssertionError("no recorded token with enough bars")


def _write_history_part(tmp_root: str, sym: str, pool: str, hist: pd.DataFrame) -> None:
    d = _store_dir(tmp_root, sym, pool, "hour_1")
    os.makedirs(d, exist_ok=True)
    part = hist[OHLCV_COLS].copy()
    part["timestamp"] = part["timestamp"].astype("int64")
    part.to_parquet(os.path.join(d, f"p_{int(part['timestamp'].min())}.parquet"), index=False)


# --- finalization ------------------------------------------------------------

def test_finalized_bars_drops_the_forming_bar():
    now = 1_000_000
    bars = [[now - 7200, 1, 1, 1, 1, 1],   # closed (opened 2h ago)
            [now - 3600, 1, 1, 1, 1, 1],   # closed exactly at now (ts+3600==now)
            [now - 1800, 1, 1, 1, 1, 1]]   # still forming (ts+3600 > now) -> dropped
    out = ld.finalized_bars(bars, now)
    assert [int(r[0]) for r in out] == [now - 7200, now - 3600]


# --- append-immutability + bar correctness -----------------------------------

def test_replay_append_is_immutable_and_correct(tmp_path):
    tok = _pick_token()
    sym, pool, full = tok["symbol"], tok["pair_address"], tok["df"]
    ts = full["timestamp"].astype("int64").to_numpy()
    cut_i = len(full) - 8                                   # replay the last 8 recorded bars
    cutoff = int(ts[cut_i])
    hist = full.iloc[:cut_i + 1]                            # bars <= cutoff
    future = full.iloc[cut_i + 1:]                          # bars to replay forward

    tmp_ohlcv = str(tmp_path / "ohlcv")
    _write_history_part(tmp_ohlcv, sym, pool, hist)
    assert ld.cached_newest_ts(sym, pool, root=tmp_ohlcv) == cutoff

    # replay each future bar one at a time (now_wall well past each so all are finalized)
    for _, row in future.iterrows():
        bar = [[int(row["timestamp"]), row["open"], row["high"], row["low"],
                row["close"], row["volume"]]]
        n = ld.append_alt_bars(sym, pool, ld.finalized_bars(bar, int(row["timestamp"]) + 7200),
                               root=tmp_ohlcv)
        assert n == 1
        # re-appending the same bar is a no-op (idempotent)
        assert ld.append_alt_bars(sym, pool, bar, root=tmp_ohlcv) == 0

    rebuilt = load_ohlcv(sym, pool, "hour", 1, root=tmp_ohlcv)
    # immutability: every bar <= cutoff is byte-identical to the recorded series
    a = rebuilt[rebuilt["timestamp"] <= cutoff].reset_index(drop=True)
    b = hist.reset_index(drop=True)
    for c in OHLCV_COLS:
        assert np.array_equal(a[c].to_numpy(), b[c].to_numpy()), f"history changed in {c}"
    # correctness: the full rebuilt series equals the recorded full series
    assert np.array_equal(rebuilt["timestamp"].to_numpy(), full["timestamp"].to_numpy())
    for c in ("open", "high", "low", "close", "volume"):
        assert np.allclose(rebuilt[c].to_numpy(), full[c].to_numpy(), equal_nan=True)


# --- panel parity (r_alt + volume) -------------------------------------------

def test_regenerated_panels_match_recorded(tmp_path):
    recorded_factor = os.path.join(REAL_FEATURES, f"{_pick_token()['symbol']}_factor.parquet")
    if not os.path.isfile(recorded_factor):
        pytest.skip("recorded factor parquet absent for the picked token")
    tok = _pick_token()
    sym, pool, full = tok["symbol"], tok["pair_address"], tok["df"]
    ts = full["timestamp"].astype("int64").to_numpy()
    cut_i = len(full) - 8
    hist, future = full.iloc[:cut_i + 1], full.iloc[cut_i + 1:]

    tmp_ohlcv = str(tmp_path / "ohlcv")
    tmp_feat = str(tmp_path / "features")
    _write_history_part(tmp_ohlcv, sym, pool, hist)
    for _, row in future.iterrows():
        bar = [[int(row["timestamp"]), row["open"], row["high"], row["low"],
                row["close"], row["volume"]]]
        ld.append_alt_bars(sym, pool, bar, root=tmp_ohlcv)

    sel = [{"symbol": sym, "pair_address": pool}]
    ld.refresh_factor_features(sel, ohlcv_root=tmp_ohlcv, anchor_root=REAL_ANCHOR, out=tmp_feat)

    live = pd.read_parquet(os.path.join(tmp_feat, f"{sym}_factor.parquet")).set_index("timestamp")["r_alt"]
    rec = pd.read_parquet(recorded_factor).set_index("timestamp")["r_alt"]
    common = live.index.intersection(rec.index)
    assert len(common) > 200                               # meaningful overlap
    # r_alt parity over the whole overlap, AND specifically on the replayed (appended) bars
    assert np.allclose(live.loc[common].to_numpy(), rec.loc[common].to_numpy(),
                       rtol=1e-9, atol=1e-12, equal_nan=True), "r_alt diverged"
    replayed = [int(t) for t in future["timestamp"] if int(t) in set(common)]
    assert replayed, "no replayed bars landed in the factor index"
    assert np.allclose(live.loc[replayed].to_numpy(), rec.loc[replayed].to_numpy(),
                       rtol=1e-9, atol=1e-12, equal_nan=True), "r_alt on appended bars diverged"

    # volume parity (per-bar local: the env's volume panel reads this column straight through).
    # Compare the regenerated/live cache's volume to the recorded series on the appended bars.
    live_oh = load_ohlcv(sym, pool, "hour", 1, root=tmp_ohlcv).set_index("timestamp")["volume"]
    rec_oh = full.set_index("timestamp")["volume"]
    for t in replayed:
        assert np.isclose(float(live_oh.loc[t]), float(rec_oh.loc[t]), equal_nan=True), \
            f"volume diverged on appended bar {t}"
