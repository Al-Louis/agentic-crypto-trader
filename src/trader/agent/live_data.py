"""Hourly live-data updater — keep the `data/` panels fresh so the validated loaders + env
run UNCHANGED on live data (the parity guarantee behind `trader.agent.event_live`).

The training pipeline reads three things, all under `data/`:
  * `data/ohlcv/hour_1/<slug>_<pool[:10]>/p_*.parquet`  — per-token 1h OHLCV (GeckoTerminal, sec ts)
  * `data/anchor/{BTC_USDT,BNB_USDT}/1h.parquet`         — BTC/BNB 1h anchor (ccxt, ms ts)
  * `data/features/<sym>_factor.parquet`                 — the factor frame; `load_data` reads `r_alt`

This module advances all three FORWARD by the just-closed bar(s), reusing the existing producers
(`trader.data.anchor.download_anchor`, `trader.data.downloader`, `trader.features.factor`) so the
live files are byte-compatible with the recorded ones. After an update, `load_data()` /
`build_volume_panel()` / `build_ohlc_frac_panels()` / `evaluate_event_policy()` are called UNCHANGED.

Two invariants make the weekly-replay safe (covered by the parity gate, `tests/test_live_data.py`):
  * **Finalization** — only CLOSED bars are appended (`ts + 3600 <= now`); the still-forming bar is
    never written, so a decision is never taken on a partial bar.
  * **Append-immutability** — appends are new part files for new timestamps only; every previously
    written bar stays byte-identical, so re-running the week's replay reproduces past decisions.

Network calls (GeckoTerminal / ccxt) live in the `fetch_*` / `refresh_anchors` functions; the
append + factor-regen core is pure (filesystem only) so the parity gate runs offline on recorded data.
"""

from __future__ import annotations

import os

import pandas as pd

from trader.data import geckoterminal as gt
from trader.data.anchor import _slug as _anchor_slug  # noqa: PLC2701 — reuse the exact slug/layout
from trader.data.downloader import OHLCV_COLS, _store_dir, load_ohlcv  # noqa: PLC2701
from trader.features.factor import compute_factor_features

BAR_SECS = 3600
FACTOR_WINDOW = 168     # matches scripts/build_factor_features.WINDOW
FACTOR_MOM_SPAN = 24    # matches scripts/build_factor_features.MOM_SPAN
OHLCV_ROOT = os.path.join("data", "ohlcv")
ANCHOR_ROOT = os.path.join("data", "anchor")
FEATURES_OUT = os.path.join("data", "features")


# --- finalization ------------------------------------------------------------

def finalized_bars(bars: list[list], now_wall: int, bar_secs: int = BAR_SECS) -> list[list]:
    """Keep only CLOSED bars: a bar opening at `ts` is closed once `ts + bar_secs <= now_wall`.
    `bars` are `[ts_seconds, o, h, l, c, v]` rows (any order). Returns them sorted ascending.
    The currently-forming bar is dropped — a decision is never taken on a partial bar."""
    closed = [r for r in bars if int(r[0]) + bar_secs <= int(now_wall)]
    return sorted(closed, key=lambda r: int(r[0]))


# --- append (pure, filesystem only) -----------------------------------------

def cached_newest_ts(symbol: str, pool: str, root: str = OHLCV_ROOT) -> int | None:
    """Newest cached 1h bar timestamp (seconds) for a token, or None if nothing cached."""
    df = load_ohlcv(symbol, pool, "hour", 1, root=root)
    return int(df["timestamp"].max()) if len(df) else None


def append_alt_bars(symbol: str, pool: str, bars: list[list], root: str = OHLCV_ROOT) -> int:
    """Append finalized 1h bars NEWER than what's cached, as a new idempotent part file (same
    layout as the backfill downloader). Returns the count written. Earlier bars are never touched
    (append-immutability). `bars` must already be finalized (`finalized_bars`)."""
    newest = cached_newest_ts(symbol, pool, root=root)
    fresh = [r for r in bars if newest is None or int(r[0]) > newest]
    if not fresh:
        return 0
    fresh.sort(key=lambda r: int(r[0]))
    d = _store_dir(root, symbol, pool, "hour_1")
    os.makedirs(d, exist_ok=True)
    df = pd.DataFrame(fresh, columns=OHLCV_COLS)
    df["timestamp"] = df["timestamp"].astype("int64")
    oldest_new = int(df["timestamp"].min())
    path = os.path.join(d, f"p_{oldest_new}.parquet")
    if not os.path.exists(path):           # idempotent — never overwrite an existing part
        df.to_parquet(path, index=False)
    return len(fresh)


# --- factor regen (reuses the exact training feature recipe) -----------------

def _anchor_seconds(symbol: str, root: str = ANCHOR_ROOT) -> pd.DataFrame:
    """Anchor OHLCV with ms timestamps normalized to seconds (mirrors build_factor_features)."""
    from trader.data.anchor import load_anchor  # noqa: PLC0415
    df = load_anchor(symbol, "1h", root=root)
    df = df[["timestamp", "open", "high", "low", "close", "volume"]].copy()
    df["timestamp"] = df["timestamp"] // 1000
    return df


def refresh_factor_features(selection: list[dict], *, ohlcv_root: str = OHLCV_ROOT,
                            anchor_root: str = ANCHOR_ROOT, out: str = FEATURES_OUT) -> list[str]:
    """Regenerate `data/features/<sym>_factor.parquet` for each selected token from the current
    cache, using the EXACT training recipe (`compute_factor_features`, window=168/mom=24). `r_alt`
    — the only column `load_data` consumes — is thus identical to what the trainer would compute.
    Returns the symbols refreshed. Network-free (reads the local cache)."""
    os.makedirs(out, exist_ok=True)
    btc, bnb = _anchor_seconds("BTC/USDT", anchor_root), _anchor_seconds("BNB/USDT", anchor_root)
    done = []
    for s in selection:
        sym, pool = s["symbol"], s["pair_address"]
        alt = load_ohlcv(sym, pool, "hour", 1, root=ohlcv_root)
        if alt.empty:
            continue
        fac = compute_factor_features(alt, btc, bnb, window=FACTOR_WINDOW, mom_span=FACTOR_MOM_SPAN)
        fac.to_parquet(os.path.join(out, f"{sym}_factor.parquet"), index=False)
        done.append(sym)
    return done


# --- network fetch (live only; not exercised by the offline parity gate) -----

def fetch_alt_latest(pool: str, *, network: str = "bsc", limit: int = 300,
                     max_429_retries: int = 5) -> list[list]:
    """Most-recent 1h OHLCV page for a pool (GeckoTerminal). `[ts_sec, o, h, l, c, v]` rows.

    Retries on HTTP 429 with exponential backoff (GeckoTerminal rate-limits hard — without this,
    a single 429 silently drops a token for the whole tick, which degenerated the live vol-top-8
    to only the tokens that happened not to 429). Mirrors the backfill downloader's 429 policy."""
    import time  # noqa: PLC0415
    import urllib.error  # noqa: PLC0415

    attempt = 0
    while True:
        try:
            return gt.fetch_ohlcv(pool, timeframe="hour", aggregate=1, limit=limit, network=network)
        except urllib.error.HTTPError as e:
            if e.code == 429 and attempt < max_429_retries:
                time.sleep(min(8 * (2 ** attempt), 60))
                attempt += 1
                continue
            raise


def refresh_anchors(days: int = 10, root: str = ANCHOR_ROOT) -> dict:
    """Forward-incremental BTC/BNB 1h anchor refresh (ccxt/binanceus; appends only newer bars)."""
    from trader.data.anchor import download_anchor  # noqa: PLC0415
    return download_anchor(["BTC/USDT", "BNB/USDT"], ["1h"], days=days, root=root)


def _stderr(msg: str) -> None:
    """Default logger -> stderr (systemd journal). stdout is block-buffered under systemd, so a
    stdout WARN can sit invisibly in the buffer for a long time — exactly how the 429 skips hid."""
    import sys  # noqa: PLC0415
    print(msg, file=sys.stderr, flush=True)


def update_live(selection: list[dict], now_wall: int, *, ohlcv_root: str = OHLCV_ROOT,
                anchor_root: str = ANCHOR_ROOT, features_out: str = FEATURES_OUT,
                anchor_days: int = 10, min_interval: float = 3.0, logger=_stderr) -> dict:
    """One hourly refresh of all three surfaces, then the factor regen. `now_wall` (unix seconds,
    injectable for tests) gates bar finalization. Returns a per-token appended-bar count + the
    anchor totals. The caller (the loop) then runs the validated loaders/driver UNCHANGED.

    Token fetches are PACED by `min_interval` seconds: GeckoTerminal rate-limits hard (HTTP 429
    after a handful of rapid calls), so 20 back-to-back fetches would 429 most of them. ~2.5s ×
    20 tokens ≈ 50s/refresh — well inside the hourly cadence."""
    import time  # noqa: PLC0415

    anchors = refresh_anchors(days=anchor_days, root=anchor_root)
    appended: dict[str, int] = {}
    for i, s in enumerate(selection):
        sym, pool = s["symbol"], s["pair_address"]
        if i and min_interval > 0:
            time.sleep(min_interval)               # pace GeckoTerminal (skip before the first)
        try:
            page = fetch_alt_latest(pool)
            n = append_alt_bars(sym, pool, finalized_bars(page, now_wall), root=ohlcv_root)
            appended[sym] = n
        except Exception as e:  # noqa: BLE001 — one bad token must not abort the hourly refresh
            logger(f"  live-data WARN {sym}: {e!r}")
            appended[sym] = 0
    refreshed = refresh_factor_features(selection, ohlcv_root=ohlcv_root,
                                        anchor_root=anchor_root, out=features_out)
    return {"appended": appended, "anchors": anchors, "factors_refreshed": len(refreshed)}
