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


# --- 429 retry/backoff (the bug that degenerated the live universe) ----------

def test_fetch_alt_latest_retries_on_429_then_succeeds(monkeypatch):
    import time
    import urllib.error
    n = {"c": 0}

    def fake(pool, **kw):
        n["c"] += 1
        if n["c"] < 3:
            raise urllib.error.HTTPError("u", 429, "Too Many Requests", None, None)
        return [[1700000000, 1, 1, 1, 1, 1]]

    monkeypatch.setattr(ld.gt, "fetch_ohlcv", fake)
    monkeypatch.setattr(time, "sleep", lambda *_: None)        # no real backoff in the test
    out = ld.fetch_alt_latest("0xpool")
    assert n["c"] == 3 and out == [[1700000000, 1, 1, 1, 1, 1]]


def test_fetch_alt_latest_reraises_after_max_retries(monkeypatch):
    import time
    import urllib.error

    def always_429(pool, **kw):
        raise urllib.error.HTTPError("u", 429, "rate", None, None)

    monkeypatch.setattr(ld.gt, "fetch_ohlcv", always_429)
    monkeypatch.setattr(time, "sleep", lambda *_: None)
    with pytest.raises(urllib.error.HTTPError):
        ld.fetch_alt_latest("0xpool", max_429_retries=2)


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


# --- CMC feed (the cutover: fast just-closed-bar finalization, keyed by token) ----

def test_fetch_alt_latest_cmc_parses_kline(monkeypatch):
    """CMC k-line candle [o,h,l,c,v,time_ms,count] -> our [ts_sec,o,h,l,c,v] rows, by token."""
    import urllib.request
    payload = {"data": [[1.0, 2.0, 0.5, 1.5, 100.0, 1_700_000_000_000, 7],
                        [1.5, 3.0, 1.0, 2.0, 200.0, 1_700_003_600_000, 9]]}

    class _Resp:
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def read(self): return json.dumps(payload).encode()

    cap = {}

    def fake_urlopen(req, **kw):
        cap["url"] = req.full_url
        return _Resp()

    monkeypatch.setattr("trader.config.get", lambda k, *a, **kw: "TESTKEY")
    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    out = ld.fetch_alt_latest_cmc("0xToKeN")
    assert out == [[1_700_000_000, 1.0, 2.0, 0.5, 1.5, 100.0],
                   [1_700_003_600, 1.5, 3.0, 1.0, 2.0, 200.0]]
    assert "address=0xToKeN" in cap["url"] and "platform=bsc" in cap["url"] and "interval=1h" in cap["url"]


def test_update_live_feed_selector_routes_cmc_vs_gecko(monkeypatch):
    """feed='cmc' fetches by token_address via the CMC path; default/unset -> Gecko by pool."""
    import time
    sel = [{"symbol": "AAA", "pair_address": "0xpoolA", "token_address": "0xtokA"},
           {"symbol": "BBB", "pair_address": "0xpoolB", "token_address": "0xtokB"}]
    calls = {"cmc": [], "gecko": []}
    monkeypatch.setattr(ld, "refresh_anchors", lambda **kw: {"BTC": 1})
    monkeypatch.setattr(ld, "refresh_factor_features", lambda *a, **kw: [])
    monkeypatch.setattr(ld, "finalized_bars", lambda page, now, *a, **kw: page)
    monkeypatch.setattr(ld, "append_alt_bars", lambda sym, pool, bars, **kw: len(bars))
    monkeypatch.setattr(ld, "fetch_alt_latest_cmc",
                        lambda tok, **kw: (calls["cmc"].append(tok) or [[1, 1, 1, 1, 1, 1]]))
    monkeypatch.setattr(ld, "fetch_alt_latest",
                        lambda pool, **kw: (calls["gecko"].append(pool) or [[1, 1, 1, 1, 1, 1]]))
    monkeypatch.setattr(time, "sleep", lambda *_: None)

    out = ld.update_live(sel, 1_700_000_000, feed="cmc")          # explicit cmc -> token-keyed
    assert calls["cmc"] == ["0xtokA", "0xtokB"] and calls["gecko"] == []
    assert out["appended"] == {"AAA": 1, "BBB": 1}

    calls["cmc"].clear(); calls["gecko"].clear()
    monkeypatch.setattr("trader.config.get", lambda k, *a, **kw: None)  # CANDLE_FEED unset -> gecko
    ld.update_live(sel, 1_700_000_000)                            # default -> pool-keyed Gecko
    assert calls["gecko"] == ["0xpoolA", "0xpoolB"] and calls["cmc"] == []
