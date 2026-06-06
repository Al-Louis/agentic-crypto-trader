"""Resumable, cached OHLCV downloader -> Parquet.

Pulls GeckoTerminal OHLCV (keyed by pool address) for a set of tokens across
timeframes into an append-only Parquet cache under ``data/ohlcv/``, with a JSON
manifest tracking per-(pool, timeframe) progress so an interrupted run resumes at
the last fetched page.

Design:
  - Each fetched page is written as its own part file ``p_<oldest_ts>.parquet`` —
    append-only, no rewrites, idempotent (re-fetching a page is a no-op).
  - The manifest is checkpointed **after every page**, recording the oldest
    timestamp reached. On resume we continue paginating backward from there.
  - GeckoTerminal rate-limits hard (HTTP 429 after a handful of rapid calls), so
    requests are paced by ``min_interval`` and retried with exponential backoff.

Layout::

    data/ohlcv/
      _manifest.json
      <timeframe>_<aggregate>/<symbol-slug>_<pool[:10]>/p_<oldest_ts>.parquet
"""

from __future__ import annotations

import json
import os
import re
import time
import urllib.error
from datetime import datetime, timezone

import pandas as pd

from trader.data import geckoterminal as gt

DEFAULT_ROOT = os.path.join("data", "ohlcv")
OHLCV_COLS = ["timestamp", "open", "high", "low", "close", "volume"]


def slug(s: str) -> str:
    """Filesystem-safe token slug (non-ASCII tickers exist in the universe)."""
    return re.sub(r"[^A-Za-z0-9._-]", "_", s) or "tok"


def tf_key(timeframe: str, aggregate: int) -> str:
    return f"{timeframe}_{aggregate}"


def _store_dir(root: str, symbol: str, pool: str, tfk: str) -> str:
    return os.path.join(root, tfk, f"{slug(symbol)}_{pool[:10]}")


def load_ohlcv(symbol: str, pool: str, timeframe: str = "hour",
               aggregate: int = 1, root: str = DEFAULT_ROOT) -> pd.DataFrame:
    """Read all part files for a series, deduped and sorted oldest->newest."""
    d = _store_dir(root, symbol, pool, tf_key(timeframe, aggregate))
    if not os.path.isdir(d):
        return pd.DataFrame(columns=OHLCV_COLS)
    parts = [pd.read_parquet(os.path.join(d, f))
             for f in os.listdir(d) if f.endswith(".parquet")]
    if not parts:
        return pd.DataFrame(columns=OHLCV_COLS)
    df = pd.concat(parts, ignore_index=True)
    df = (df.drop_duplicates("timestamp")
            .sort_values("timestamp")
            .reset_index(drop=True))
    df["datetime"] = pd.to_datetime(df["timestamp"], unit="s", utc=True)
    return df


class OHLCVDownloader:
    """Paced, resumable GeckoTerminal -> Parquet backfill."""

    def __init__(self, root: str = DEFAULT_ROOT, network: str = "bsc",
                 min_interval: float = 3.0, max_days: int = 190,
                 page_limit: int = 1000, max_429_retries: int = 6, logger=print):
        self.root = root
        self.network = network
        self.min_interval = min_interval
        self.max_days = max_days
        self.page_limit = page_limit
        self.max_429_retries = max_429_retries
        self.log = logger
        self._last_req = 0.0
        os.makedirs(root, exist_ok=True)
        self.manifest_path = os.path.join(root, "_manifest.json")
        self.manifest = self._load_manifest()

    # --- manifest (atomic write) ------------------------------------------
    def _load_manifest(self) -> dict:
        if os.path.exists(self.manifest_path):
            with open(self.manifest_path, encoding="utf-8") as f:
                return json.load(f)
        return {}

    def _save_manifest(self) -> None:
        tmp = self.manifest_path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(self.manifest, f, indent=2, ensure_ascii=False)
        os.replace(tmp, self.manifest_path)

    # --- paced fetch with 429 backoff -------------------------------------
    def _throttle(self) -> None:
        dt = time.time() - self._last_req
        if dt < self.min_interval:
            time.sleep(self.min_interval - dt)
        self._last_req = time.time()

    def _fetch_page(self, pool, timeframe, aggregate, before_ts):
        attempt = 0
        while True:
            self._throttle()
            try:
                return gt.fetch_ohlcv(
                    pool, timeframe=timeframe, aggregate=aggregate,
                    limit=self.page_limit, network=self.network,
                    before_timestamp=before_ts,
                )
            except urllib.error.HTTPError as e:
                if e.code == 429 and attempt < self.max_429_retries:
                    wait = min(10 * (2 ** attempt), 120)
                    self.log(f"    429 -> backoff {wait}s "
                             f"(retry {attempt + 1}/{self.max_429_retries})")
                    time.sleep(wait)
                    attempt += 1
                    continue
                raise

    def _write_part(self, d: str, page: list) -> int:
        os.makedirs(d, exist_ok=True)
        df = pd.DataFrame(page, columns=OHLCV_COLS)
        df["timestamp"] = df["timestamp"].astype("int64")
        oldest = int(df["timestamp"].min())
        path = os.path.join(d, f"p_{oldest}.parquet")
        if not os.path.exists(path):  # idempotent: a re-fetched page is a no-op
            df.to_parquet(path, index=False)
        return oldest

    # --- one series -------------------------------------------------------
    def download(self, symbol: str, pool: str, timeframe: str = "hour",
                 aggregate: int = 1, force: bool = False) -> dict:
        tfk = tf_key(timeframe, aggregate)
        key = f"{pool}|{tfk}"
        ent = self.manifest.get(key) or {
            "symbol": symbol, "pool": pool, "network": self.network,
            "timeframe": timeframe, "aggregate": aggregate,
            "oldest_ts": None, "newest_ts": None, "pages": 0, "complete": False,
        }
        if ent.get("complete") and not force:
            self.log(f"  [{symbol} {tfk}] complete ({ent['pages']} pages) -> skip")
            return ent

        d = _store_dir(self.root, symbol, pool, tfk)
        floor_ts = int(time.time()) - self.max_days * 86_400
        before = ent.get("oldest_ts")  # resume point (None on first run)
        self.log(f"  [{symbol} {tfk}] {'resume' if before else 'start'}"
                 f"{f' before={before}' if before else ''}")

        while True:
            page = self._fetch_page(pool, timeframe, aggregate, before)
            if not page:
                ent["complete"] = True
                break
            oldest = self._write_part(d, page)
            newest = max(int(r[0]) for r in page)
            ent["oldest_ts"] = oldest if ent["oldest_ts"] is None else min(ent["oldest_ts"], oldest)
            ent["newest_ts"] = newest if ent["newest_ts"] is None else max(ent["newest_ts"], newest)
            ent["pages"] += 1
            ent["updated_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
            self.manifest[key] = ent
            self._save_manifest()  # checkpoint EVERY page -> page-level resume
            self.log(f"    page {ent['pages']}: {len(page):4} rows, "
                     f"oldest={datetime.fromtimestamp(oldest, timezone.utc).date()}")
            if len(page) < self.page_limit or oldest <= floor_ts:
                ent["complete"] = True  # reached start-of-history or max_days
                break
            before = oldest

        self.manifest[key] = ent
        self._save_manifest()
        return ent

    # --- many series ------------------------------------------------------
    def download_many(self, selections: list[dict],
                      timeframes: list[tuple[str, int]]) -> list[dict]:
        """selections: [{symbol, pair_address}]; timeframes: [(timeframe, aggregate)]."""
        results = []
        n = len(selections) * len(timeframes)
        i = 0
        for sel in selections:
            for tf, agg in timeframes:
                i += 1
                self.log(f"[{i}/{n}] {sel['symbol']} {tf}/{agg}")
                try:
                    results.append(self.download(sel["symbol"], sel["pair_address"], tf, agg))
                except Exception as e:  # noqa: BLE001 — checkpointed; don't lose other series
                    self.log(f"  ERROR {sel['symbol']} {tf}: {e!r} (progress saved)")
                    results.append({"symbol": sel["symbol"], "timeframe": tf,
                                    "aggregate": agg, "error": repr(e)})
        return results
