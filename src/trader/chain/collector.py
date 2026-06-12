"""Chunked, resumable eth_getLogs collector -> Parquet under data/chain/.

One global forward scan over the block range, all (active) pools in a single
address-array filter, topic0-filtered to our event set. Mirrors the OHLCV
downloader's manifest pattern (``trader.data.downloader``): progress is
checkpointed so an interrupted run resumes at the cursor.

Span adaptation: public endpoints cap ``eth_getLogs`` by block span, result
count, and response size in undocumented ways. The scan starts at
``span_init`` blocks, halves on ``SpanTooWide`` (down to ``span_min``), and
creeps back up after consecutive easy chunks. A response of >= 9_500 logs is
treated as possible silent truncation -> the chunk is split and re-fetched.

Block->time: per-chunk boundary timestamps are sampled into
``data/chain/blockindex/samples.parquet`` (every ``ts_every`` chunks);
panels interpolate per-log timestamps from these (hourly bucketing needs
nowhere near per-block precision, and BSC block time moved 3s -> 0.45s
across our window, so a global linear fit would be wrong).

Layout::

    data/chain/
      _pools.json                      # registry (trader.chain.registry)
      _manifest.json                   # {"scan": {from_block, to_block, cursor}}
      blockindex/samples.parquet       # (block, ts) samples
      logs/<SYM>_<pool10>/p_<from>_<to>.parquet   # unified decoded rows
"""

from __future__ import annotations

import json
import os
import time
from datetime import datetime, timezone

import pandas as pd

from trader.chain import events
from trader.chain.registry import load_registry
from trader.chain.rpc import BscRpc, SpanTooWide

DEFAULT_ROOT = os.path.join("data", "chain")
# therpc.io demonstrably returns >12k logs in one response; providers that cap
# results raise errors (handled as SpanTooWide) rather than silently truncate.
TRUNCATION_SUSPECT = 30_000


class LogCollector:
    def __init__(self, root: str = DEFAULT_ROOT, rpc: BscRpc | None = None,
                 span_init: int = 10_000, span_min: int = 500,
                 span_max: int = 20_000, flush_rows: int = 250_000,
                 ts_every: int = 5, logger=print):
        self.root = root
        self.rpc = rpc or BscRpc()
        self.span_init, self.span_min, self.span_max = span_init, span_min, span_max
        self.flush_rows = flush_rows
        self.ts_every = ts_every
        self.log = logger
        self.pools = load_registry(os.path.join(root, "_pools.json"))
        self.by_addr = {p["pool"].lower(): p for p in self.pools}
        self.manifest_path = os.path.join(root, "_manifest.json")
        self.manifest = self._load_json(self.manifest_path) or {}
        self._buf: dict[str, list[dict]] = {}   # symbol -> rows
        self._buf_n = 0
        self._buf_from: int | None = None       # first block covered by buffer
        self._ts_samples: list[tuple[int, int]] = []

    # --- persistence --------------------------------------------------------

    @staticmethod
    def _load_json(path):
        if os.path.exists(path):
            with open(path, encoding="utf-8") as f:
                return json.load(f)
        return None

    def _save_manifest(self) -> None:
        tmp = self.manifest_path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(self.manifest, f, indent=1)
        os.replace(tmp, self.manifest_path)

    def _pool_dir(self, p: dict) -> str:
        return os.path.join(self.root, "logs", f"{p['symbol']}_{p['pool'][:10]}")

    def _flush(self, through_block: int) -> None:
        """Write buffered rows as per-pool part files and advance the cursor.
        Part files are keyed by block range -> idempotent on re-scan."""
        frm = self._buf_from
        for sym, rows in self._buf.items():
            if not rows:
                continue
            p = next(pp for pp in self.pools if pp["symbol"] == sym)
            d = self._pool_dir(p)
            os.makedirs(d, exist_ok=True)
            path = os.path.join(d, f"p_{frm}_{through_block}.parquet")
            if not os.path.exists(path):
                pd.DataFrame(rows, columns=events.ROW_COLUMNS).to_parquet(path, index=False)
        if self._ts_samples:
            d = os.path.join(self.root, "blockindex")
            os.makedirs(d, exist_ok=True)
            path = os.path.join(d, "samples.parquet")
            df = pd.DataFrame(self._ts_samples, columns=["block", "ts"])
            if os.path.exists(path):
                df = pd.concat([pd.read_parquet(path), df], ignore_index=True)
                df = df.drop_duplicates("block").sort_values("block")
            df.to_parquet(path, index=False)
            self._ts_samples = []
        self._buf, self._buf_n, self._buf_from = {}, 0, None
        self.manifest["scan"]["cursor"] = through_block
        self.manifest["scan"]["updated_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
        self._save_manifest()

    # --- scanning -----------------------------------------------------------

    def _fetch_chunk(self, addresses: list[str], frm: int, to: int) -> list[dict]:
        """One getLogs call; on truncation suspicion split the range in half
        recursively (re-fetch is idempotent — logs are keyed by range)."""
        logs = self.rpc.get_logs(addresses, frm, to, topics=[events.ALL_TOPICS])
        if len(logs) >= TRUNCATION_SUSPECT and to > frm:
            mid = (frm + to) // 2
            self.log(f"    {len(logs)} logs in [{frm},{to}] — truncation suspect, splitting")
            return (self._fetch_chunk(addresses, frm, mid)
                    + self._fetch_chunk(addresses, mid + 1, to))
        return logs

    def scan(self, from_block: int | None = None, to_block: int | None = None,
             max_chunks: int | None = None) -> dict:
        """Run (or resume) the scan. ``from_block``/``to_block`` are only used
        when starting fresh; a resumed scan keeps its recorded range."""
        sc = self.manifest.get("scan")
        if sc is None:
            assert from_block is not None and to_block is not None, \
                "first run needs an explicit block range"
            sc = {"from_block": from_block, "to_block": to_block,
                  "cursor": from_block - 1}
            self.manifest["scan"] = sc
            self._save_manifest()
        elif to_block is not None and to_block > sc["to_block"]:
            sc["to_block"] = to_block          # extend (live tail)

        addresses = [p["pool"] for p in self.pools]
        span = self.span_init
        easy = 0
        chunks = 0
        t0 = time.time()
        cursor = sc["cursor"]
        self._buf_from = cursor + 1

        while cursor < sc["to_block"]:
            if max_chunks is not None and chunks >= max_chunks:
                break
            frm = cursor + 1
            to = min(frm + span - 1, sc["to_block"])
            try:
                logs = self._fetch_chunk(addresses, frm, to)
            except SpanTooWide:
                if span <= self.span_min:
                    raise
                span = max(span // 2, self.span_min)
                self.log(f"    span too wide -> {span}")
                easy = 0
                continue
            for lg in logs:
                p = self.by_addr.get(lg["address"].lower())
                if p is None:
                    continue
                row = events.decode_log(lg, p["dec0"], p["dec1"])
                if row is not None:
                    self._buf.setdefault(p["symbol"], []).append(row)
                    self._buf_n += 1
            cursor = to
            chunks += 1
            if chunks % self.ts_every == 0 or cursor >= sc["to_block"]:
                self._ts_samples.append((to, self.rpc.block_timestamp(to)))
            if self._buf_n >= self.flush_rows:
                self._flush(cursor)
                self._buf_from = cursor + 1
            # grow toward responses of ~8-12k logs — therpc returns >12k fine,
            # and per-call fixed cost dominates when chunks are small
            if len(logs) < 8_000:
                easy += 1
                if easy >= 3 and span < self.span_max:
                    span = min(int(span * 1.5), self.span_max)
                    easy = 0
            else:
                easy = 0
            if chunks % 25 == 0:
                done = cursor - sc["from_block"] + 1
                total = sc["to_block"] - sc["from_block"] + 1
                rate = done / max(time.time() - t0, 1e-9)
                self.log(f"  [{done/total:6.1%}] block {cursor:,} span={span} "
                         f"buf={self._buf_n} ({rate:,.0f} blk/s, "
                         f"eta {(total-done)/max(rate,1e-9)/3600:.1f}h)")
        self._flush(cursor)
        return self.manifest["scan"]


def load_pool_logs(symbol: str, root: str = DEFAULT_ROOT) -> pd.DataFrame:
    """All decoded rows for one token's pool, deduped and time-ordered."""
    pools = load_registry(os.path.join(root, "_pools.json"))
    p = next(pp for pp in pools if pp["symbol"] == symbol)
    d = os.path.join(root, "logs", f"{p['symbol']}_{p['pool'][:10]}")
    if not os.path.isdir(d):
        return pd.DataFrame(columns=events.ROW_COLUMNS)
    parts = [pd.read_parquet(os.path.join(d, f))
             for f in sorted(os.listdir(d)) if f.endswith(".parquet")]
    if not parts:
        return pd.DataFrame(columns=events.ROW_COLUMNS)
    df = pd.concat(parts, ignore_index=True)
    return (df.drop_duplicates(["block", "log_index"])
              .sort_values(["block", "log_index"])
              .reset_index(drop=True))


def load_block_index(root: str = DEFAULT_ROOT) -> pd.DataFrame:
    path = os.path.join(root, "blockindex", "samples.parquet")
    return pd.read_parquet(path).sort_values("block").reset_index(drop=True)
