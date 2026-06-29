"""NodeReal MegaNode Enhanced-API client — per-wallet asset transfers on BSC (read-only).

`nr_getAssetTransfers` returns a wallet's transfers including **native BNB** (`category` external +
internal) and tokens (`20`) — the native legs `eth_getLogs` can't see, which is what makes a clean
deposit-vs-swap classification possible. Free tier; key from `NODEREAL_API_KEY` (.env), else the
docs' public shared key (lower rate limits — set your own for the full 123-wallet run).

Stdlib-only. Per-call block range caps at 2,000,000, so the scan chunks; results paginate via pageKey.
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request

from trader import config

PUBLIC_KEY = "64a9df0874fb4a93b9d0a3849de012d3"   # NodeReal docs' shared demo key (rate-limited)
MAX_RANGE = 2_000_000                              # per-call block-range cap
MAX_COUNT = "0x3e8"                                # 1000 transfers/page (max)
_HEADERS = {"Content-Type": "application/json", "User-Agent": "act-competition/0.1"}


def endpoint(key: str | None = None) -> str:
    key = key or config.get("NODEREAL_API_KEY") or PUBLIC_KEY
    return f"https://bsc-mainnet.nodereal.io/v1/{key}"


class NodeReal:
    def __init__(self, key: str | None = None, *, min_interval: float = 0.12,
                 timeout: float = 40.0, max_retries: int = 5, logger=print):
        self.url = endpoint(key)
        self.using_public = (key or config.get("NODEREAL_API_KEY") or PUBLIC_KEY) == PUBLIC_KEY
        self.min_interval = min_interval
        self.timeout = timeout
        self.max_retries = max_retries
        self.log = logger
        self._last = 0.0
        self.n_calls = 0          # underlying JSON-RPC requests (the real quota unit)

    def _call(self, method: str, params: list):
        self.n_calls += 1
        dt = time.time() - self._last
        if dt < self.min_interval:
            time.sleep(self.min_interval - dt)
        body = json.dumps({"jsonrpc": "2.0", "id": 1, "method": method, "params": params}).encode()
        for attempt in range(self.max_retries + 1):
            try:
                req = urllib.request.Request(self.url, data=body, headers=_HEADERS)
                with urllib.request.urlopen(req, timeout=self.timeout) as r:
                    res = json.loads(r.read())
                self._last = time.time()
                if "error" in res:
                    raise RuntimeError(f"nodereal {method}: {res['error']}")
                return res["result"]
            except urllib.error.HTTPError as e:
                if e.code in (429, 503) and attempt < self.max_retries:
                    time.sleep(min(1.5 * (2 ** attempt), 30))
                    continue
                raise

    def block_number(self) -> int:
        return int(self._call("eth_blockNumber", []), 16)

    def asset_transfers(self, *, from_block: int, to_block: int, from_address: str | None = None,
                        to_address: str | None = None,
                        category=("external", "internal", "20")) -> list[dict]:
        """All transfers in `[from_block, to_block]` for the given address/direction, chunked to the
        2M-block cap and paginated. Returns normalized rows: `{category, asset, qty, from, to, hash,
        block, ts, contract}`. `qty` is decimal-scaled (value/10**decimal)."""
        out: list[dict] = []
        lo = from_block
        while lo <= to_block:
            hi = min(lo + MAX_RANGE - 1, to_block)
            page = None
            while True:
                p: dict = {"category": list(category), "fromBlock": hex(lo), "toBlock": hex(hi),
                           "maxCount": MAX_COUNT, "order": "asc"}
                if from_address:
                    p["fromAddress"] = from_address
                if to_address:
                    p["toAddress"] = to_address
                if page:
                    p["pageKey"] = page
                res = self._call("nr_getAssetTransfers", [p])
                for t in res.get("transfers", []):
                    out.append(_normalize(t))
                page = res.get("pageKey")
                if not page:
                    break
            lo = hi + 1
        return out


def _normalize(t: dict) -> dict:
    dec = int(t.get("decimal") or 18)
    raw = t.get("value") or "0x0"
    try:
        qty = int(raw, 16) / (10 ** dec)
    except (ValueError, TypeError):
        qty = 0.0
    return {
        "id": t.get("id"),                         # NodeReal's unique transfer id — cache dedup key
        "category": t.get("category"),
        "asset": t.get("asset"),
        "qty": qty,
        "from": (t.get("from") or "").lower(),
        "to": (t.get("to") or "").lower(),
        "hash": t.get("hash"),
        "block": int(t.get("blockNum"), 16) if t.get("blockNum") else None,
        "ts": int(t.get("blockTimeStamp") or 0) or None,
        "contract": (t.get("contractAddress") or "").lower(),
    }


class CachedNodeReal:
    """Incremental cache over NodeReal `asset_transfers`: persists each wallet's transfers + the last
    block scanned, so each hourly run fetches only NEW blocks (the window→latest range shrinks from the
    whole window to ~1 hour) instead of rescanning per wallet. Drop-in for `NodeReal` in
    `flows`/`build_leaderboard` (implements `asset_transfers` + `block_number`). Call `save()` after.
    """

    cached = True   # marker: build_leaderboard skips the suspect re-fetch (the cache is reliable)
    LOOKBACK = 3000  # re-scan ~last 1-2h of blocks each run so a transient partial read self-heals
                     # next hour (dedup by id makes the overlap free); without it a missed transfer
                     # would be lost forever once `scanned` advances past its block.

    def __init__(self, nr, cache_path: str, window_start_block: int):
        self.nr = nr
        self.cache_path = cache_path
        self.window_start_block = window_start_block
        data: dict = {}
        try:
            with open(cache_path, encoding="utf-8") as f:
                data = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            pass
        self.scanned = int(data.get("scanned_to_block") or 0)
        self.wallets = data.get("wallets", {})       # addr -> {"in": [t,...], "out": [t,...]}
        self._refreshed: set = set()
        self._max_to = self.scanned
        self._block_no = None
        self.n_fetches = 0
        self.blocks_scanned = 0

    def block_number(self) -> int:
        if self._block_no is None:
            self._block_no = self.nr.block_number()
        return self._block_no

    def asset_transfers(self, *, from_block, to_block, from_address=None, to_address=None,
                        category=("external", "internal", "20")):
        addr = (from_address or to_address or "").lower()
        direction = "out" if from_address else "in"
        entry = self.wallets.setdefault(addr, {"in": [], "out": []})
        arr = entry.setdefault(direction, [])
        if (addr, direction) not in self._refreshed:
            start = max(self.window_start_block, self.scanned - self.LOOKBACK)   # lookback; dedup by id
            if start <= to_block:
                new = self.nr.asset_transfers(from_block=start, to_block=to_block,
                                              from_address=from_address, to_address=to_address,
                                              category=category)
                self.n_fetches += 1
                self.blocks_scanned += (to_block - start + 1)   # CU is charged per block-range scanned
                seen = {t.get("id") for t in arr}
                for t in new:
                    if t.get("id") not in seen:
                        arr.append(t)
                        seen.add(t.get("id"))
            self._refreshed.add((addr, direction))
            self._max_to = max(self._max_to, to_block)
        return [t for t in arr if from_block <= (t.get("block") or 0) <= to_block]

    def save(self) -> None:
        import os  # noqa: PLC0415
        self.scanned = self._max_to
        os.makedirs(os.path.dirname(self.cache_path) or ".", exist_ok=True)
        tmp = self.cache_path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump({"scanned_to_block": self.scanned, "wallets": self.wallets}, f, separators=(",", ":"))
        os.replace(tmp, self.cache_path)
