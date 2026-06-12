"""Minimal JSON-RPC client for BSC with endpoint failover and pacing.

Free public endpoints differ wildly (probed 2026-06-12, see [[Pool-Event Data
Layer]]): ``bsc.therpc.io`` serves *deep historical* ``eth_getLogs`` (the only
free endpoint found that does) but 502s above ~10-20k-block spans;
``bsc-rpc.publicnode.com`` is fast and generous on spans but prunes log
history to roughly the last day; the dataseed nodes reject ``eth_getLogs``
outright ("limit exceeded" at any span). So: ordered endpoint list, classify
errors, fail over on prune/permission errors, back off on rate limits — and
the *caller* owns span adaptation (a ``SpanTooWide`` signal, not a retry).
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request

from trader import config

# Ordered by deep-history capability; BSC_RPC_URL (.env) is appended as a
# final fallback (the dataseed default still serves eth_call/getBlock fine).
DEFAULT_ENDPOINTS = [
    "https://bsc.therpc.io",
    "https://bsc-rpc.publicnode.com",
]

_HEADERS = {
    "Content-Type": "application/json",
    # publicnode/drpc 403 the default Python-urllib agent
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) trader-chain/0.1",
}


class RpcError(RuntimeError):
    def __init__(self, code: int | None, message: str):
        super().__init__(f"RPC error {code}: {message}")
        self.code = code
        self.message = message


class SpanTooWide(RpcError):
    """The node refused the request for being too large — narrow and retry."""


class HistoryPruned(RpcError):
    """This endpoint does not have logs that far back — try another."""


def _classify(code: int | None, message: str) -> RpcError:
    msg = (message or "").lower()
    if code == -32701 or "pruned" in msg or "header not found" in msg:
        return HistoryPruned(code, message)
    if (code in (-32005, -32602, -32000) and
            ("limit" in msg or "range" in msg or "too many" in msg or
             "response size" in msg or "results" in msg)):
        return SpanTooWide(code, message)
    return RpcError(code, message)


class BscRpc:
    """Paced multi-endpoint JSON-RPC. Endpoints are tried in order; an
    endpoint that raises ``HistoryPruned`` is skipped for that call only
    (it may still serve recent blocks and plain calls)."""

    def __init__(self, endpoints: list[str] | None = None,
                 min_interval: float = 0.25, timeout: float = 60.0,
                 max_429_retries: int = 6, logger=print):
        env_url = config.get("BSC_RPC_URL")
        self.endpoints = list(endpoints or DEFAULT_ENDPOINTS)
        if env_url and env_url not in self.endpoints:
            self.endpoints.append(env_url)
        self.min_interval = min_interval
        self.timeout = timeout
        self.max_429_retries = max_429_retries
        self.log = logger
        self._last_req = 0.0

    def _throttle(self) -> None:
        dt = time.time() - self._last_req
        if dt < self.min_interval:
            time.sleep(self.min_interval - dt)
        self._last_req = time.time()

    def _post(self, url: str, method: str, params: list):
        body = json.dumps({"jsonrpc": "2.0", "id": 1,
                           "method": method, "params": params}).encode()
        req = urllib.request.Request(url, data=body, headers=_HEADERS)
        with urllib.request.urlopen(req, timeout=self.timeout) as r:
            res = json.loads(r.read())
        if "error" in res:
            err = res["error"] or {}
            raise _classify(err.get("code"), str(err.get("message", err)))
        return res["result"]

    def call(self, method: str, params: list):
        """Try each endpoint in order. ``SpanTooWide`` propagates immediately
        (every endpoint would balk the same way and the caller must narrow);
        429s back off and retry the same endpoint; other failures fall
        through to the next endpoint."""
        last: Exception | None = None
        for url in self.endpoints:
            attempt = 0
            while True:
                self._throttle()
                try:
                    return self._post(url, method, params)
                except SpanTooWide:
                    raise
                except urllib.error.HTTPError as e:
                    if e.code in (429, 503) and attempt < self.max_429_retries:
                        wait = min(2.0 * (2 ** attempt), 60)
                        time.sleep(wait)
                        attempt += 1
                        continue
                    if e.code in (502, 504):  # therpc 502s on oversized spans
                        last = SpanTooWide(e.code, f"HTTP {e.code}")
                        break
                    last = e
                    break
                except (HistoryPruned, RpcError, OSError) as e:
                    last = e
                    break
        if isinstance(last, SpanTooWide):
            raise last
        raise last if last else RpcError(None, "no endpoints configured")

    # --- conveniences -------------------------------------------------------

    def block_number(self) -> int:
        return int(self.call("eth_blockNumber", []), 16)

    def block_timestamp(self, block: int) -> int:
        res = self.call("eth_getBlockByNumber", [hex(block), False])
        if res is None:
            raise RpcError(None, f"block {block} not found")
        return int(res["timestamp"], 16)

    def get_logs(self, addresses: list[str], from_block: int, to_block: int,
                 topics: list | None = None) -> list[dict]:
        flt: dict = {"address": addresses if len(addresses) > 1 else addresses[0],
                     "fromBlock": hex(from_block), "toBlock": hex(to_block)}
        if topics:
            flt["topics"] = topics
        return self.call("eth_getLogs", [flt])

    def eth_call(self, to: str, data: str) -> str:
        return self.call("eth_call", [{"to": to, "data": data}, "latest"])

    def block_at_timestamp(self, ts: int, lo: int = 1, hi: int | None = None) -> int:
        """Binary-search the first block with timestamp >= ts."""
        hi = hi if hi is not None else self.block_number()
        while lo < hi:
            mid = (lo + hi) // 2
            if self.block_timestamp(mid) < ts:
                lo = mid + 1
            else:
                hi = mid
        return lo
