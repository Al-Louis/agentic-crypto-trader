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


def just_closed_open(now_wall: int, bar_secs: int = BAR_SECS) -> int:
    """Open ts of the most-recently CLOSED bar at `now_wall`: the bar whose close is the last hour
    boundary <= now. This is the bar a tick firing just after the hour should be deciding on — and
    the one GeckoTerminal is slowest to finalize, so the `settle` wait in `update_live` re-polls
    until the active pools carry it (else the agent misses it for a WHOLE hour: the HH:03 fetch
    races Gecko's candle finalization, the bar is absent, and the next attempt is HH+1:03)."""
    return (int(now_wall) // bar_secs) * bar_secs - bar_secs


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


def build_close_panel(selection: list[dict], index, root: str = OHLCV_ROOT):
    """Per-token REAL USD close aligned to `index` — used to translate the env's internal
    return-index fill prices (which start at 1.0 at the window's warmup start) into real market
    prices for the telemetry. Mirrors `build_volume_panel`'s alignment (reindex + ffill)."""
    cols = {}
    for s in selection:
        df = load_ohlcv(s["symbol"], s["pair_address"], "hour", 1, root=root)
        if df.empty:
            continue
        ts = df["timestamp"].to_numpy()
        ts = (ts // 1000) if len(ts) and ts.max() > 1e12 else ts
        cols[s["symbol"]] = pd.Series(df["close"].to_numpy(), index=ts).reindex(index).ffill()
    return pd.DataFrame(cols, index=index)


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


def _refresh_pool(s: dict, now_wall: int, *, root: str, fetch_fn, logger) -> int:
    """Fetch one pool's latest page and append its just-closed bars. Returns the appended count
    (0 on ANY error — one bad pool must never abort the hourly refresh). Shared by the initial
    pass and the settle re-poll so both finalize against the SAME `now_wall`."""
    sym, pool = s["symbol"], s["pair_address"]
    try:
        page = fetch_fn(pool)
        return append_alt_bars(sym, pool, finalized_bars(page, now_wall), root=root)
    except Exception as e:  # noqa: BLE001 — one bad token must not abort the hourly refresh
        logger(f"  live-data WARN {sym}: {e!r}")
        return 0


def update_live(selection: list[dict], now_wall: int, *, ohlcv_root: str = OHLCV_ROOT,
                anchor_root: str = ANCHOR_ROOT, features_out: str = FEATURES_OUT,
                anchor_days: int = 10, min_interval: float = 3.0,
                settle_max_wait: float = 0.0, settle_poll: float = 45.0,
                settle_active_window: int = 6 * 3600, fetch_fn=fetch_alt_latest,
                sleep=None, logger=_stderr) -> dict:
    """One hourly refresh of all three surfaces, then the factor regen. `now_wall` (unix seconds,
    injectable for tests) gates bar finalization. Returns a per-token appended-bar count + the
    anchor totals + a `settle` diagnostics block. The caller runs the validated loaders UNCHANGED.

    Token fetches are PACED by `min_interval` seconds: GeckoTerminal rate-limits hard (HTTP 429
    after a handful of rapid calls), so 20 back-to-back fetches would 429 most of them. ~2.5s ×
    20 tokens ≈ 50s/refresh — well inside the hourly cadence.

    SETTLE WAIT (the candle-lag fix, OFF by default -> byte-identical single pass): the HH:03 tick
    often fetches before GeckoTerminal has finalized the just-closed bar for a pool, so the env
    misses that bar for a FULL hour (the +5% slip the live B trade ate). When `settle_max_wait>0`,
    after the initial pass re-poll ONLY the *active* pools still missing the just-closed bar, every
    `settle_poll`s, until they all carry it or the deadline — so the agent decides on the bar THIS
    tick. A pool is "active" if its newest cached bar (before this tick) is within
    `settle_active_window`s of now, so a perma-stale pool (e.g. inactive XAUt, never in the
    vol-top-8) does NOT hold up the tick. The bars are identical GeckoTerminal candles (append-
    immutable), so this changes only WHEN a decision lands, never WHICH (no train/serve skew).
    `fetch_fn`/`sleep` are injectable for offline tests."""
    if sleep is None:
        import time  # noqa: PLC0415
        sleep = time.sleep
    now_wall = int(now_wall)
    target = just_closed_open(now_wall)

    # classify the pools that SHOULD settle the just-closed bar (only when the wait is enabled —
    # the disabled path stays exactly the pre-existing single pass). Snapshot newest cached bars
    # BEFORE any appends: a pool producing bars (newest within the active window) is expected to
    # get the just-closed bar; a stale one is not waited on.
    active: list[dict] = []
    if settle_max_wait > 0.0 and settle_poll > 0.0:
        active = [s for s in selection
                  if (nt := cached_newest_ts(s["symbol"], s["pair_address"], root=ohlcv_root)) is not None
                  and nt >= now_wall - settle_active_window]

    anchors = refresh_anchors(days=anchor_days, root=anchor_root)
    appended: dict[str, int] = {}
    for i, s in enumerate(selection):
        if i and min_interval > 0:
            sleep(min_interval)                    # pace GeckoTerminal (skip before the first)
        appended[s["symbol"]] = _refresh_pool(s, now_wall, root=ohlcv_root,
                                              fetch_fn=fetch_fn, logger=logger)

    settle = {"enabled": settle_max_wait > 0.0, "target": target, "active": len(active),
              "waited": 0.0, "polls": 0, "still_missing": []}
    if settle_max_wait > 0.0 and settle_poll > 0.0 and active:
        def _missing() -> list[dict]:
            return [s for s in active
                    if (cached_newest_ts(s["symbol"], s["pair_address"], root=ohlcv_root) or -1) < target]
        waited, polls = 0.0, 0
        miss = _missing()
        while miss and waited < settle_max_wait:
            sleep(settle_poll)
            waited += settle_poll
            polls += 1
            for j, s in enumerate(miss):
                if j and min_interval > 0:
                    sleep(min_interval)
                appended[s["symbol"]] = appended.get(s["symbol"], 0) + _refresh_pool(
                    s, now_wall, root=ohlcv_root, fetch_fn=fetch_fn, logger=logger)
            miss = _missing()
        settle.update(waited=waited, polls=polls, still_missing=[s["symbol"] for s in miss])
        logger(f"  live-data settle: target={target} active={len(active)} waited={waited:.0f}s "
               f"polls={polls} still_missing={settle['still_missing']}")

    refreshed = refresh_factor_features(selection, ohlcv_root=ohlcv_root,
                                        anchor_root=anchor_root, out=features_out)
    return {"appended": appended, "anchors": anchors,
            "factors_refreshed": len(refreshed), "settle": settle}
