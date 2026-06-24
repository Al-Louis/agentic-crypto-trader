"""Compare candle FRESHNESS across three feeds, for 1h AND 15m bars:

  1. GeckoTerminal   (current feed)              hour/agg1  + minute/agg15
  2. CMC k-line      (/v1/k-line/candles)        interval=1h + interval=15min, platform=bsc, unit=usd
  3. On-chain swaps  (NodeReal eth_getLogs)      interval-agnostic; bar readable at ~block lag

Modes:
  * snapshot (default): per token, each feed's newest 1h bar + same-bar close parity, and the on-chain
    just-closed-bar swap count / volume / close. Instant.
  * --watch: measure FINALIZATION LAG at each boundary over a run window. Fires at every 15-min mark
    (15:15/30/45/16:00...) for the just-closed 15m bar, AND additionally at the top of the hour for the
    just-closed 1h bar. Polls Gecko + CMC (PACED) until each serves the bar; on-chain = block lag.
    Aggregates median/P90/max per (interval, feed) -> the data for backlog #2 (faster feed) and #3
    (does 15m data arrive fast enough to bother with a faster cadence?).

Read-only. Keys from .env. Run:
    PYTHONPATH=src python scripts/compare_candle_feeds.py
    PYTHONPATH=src python scripts/compare_candle_feeds.py --watch --hours 2     # 8 pools (credit-bounded)
    PYTHONPATH=src python scripts/compare_candle_feeds.py --watch --limit 0     # all 20 pools
"""
from __future__ import annotations

import argparse
import json
import time
import urllib.parse
import urllib.request
import datetime as dt

from trader import config
from trader.chain import events
from trader.chain.rpc import BscRpc
from trader.data import geckoterminal as gt

POOLS_JSON = "data/chain/_pools.json"
CMC_BASE = "https://pro-api.coinmarketcap.com/v1/k-line/candles"

# interval -> (seconds, GeckoTerminal timeframe/aggregate, CMC k-line interval)
INTERVALS = {
    "1h":  {"secs": 3600, "gt_tf": "hour",   "gt_agg": 1,  "cmc": "1h"},
    "15m": {"secs": 900,  "gt_tf": "minute", "gt_agg": 15, "cmc": "15min"},
}


def _utc(s: int) -> str:
    return dt.datetime.fromtimestamp(int(s), dt.timezone.utc).strftime("%H:%M")


def load_universe() -> list[dict]:
    with open(POOLS_JSON, encoding="utf-8") as f:
        pools = json.load(f)
    for p in pools:
        p["token"] = p["token0"] if p["token_side"] == 0 else p["token1"]
    return pools


# --- feeds (return {bar_open_ts: close_usd}) -------------------------------

def gecko_bars(pool: str, tf: str = "hour", agg: int = 1) -> dict[int, float]:
    try:
        rows = gt.fetch_ohlcv(pool, timeframe=tf, aggregate=agg, limit=8, network="bsc")
        return {int(r[0]): float(r[4]) for r in rows}
    except Exception:  # noqa: BLE001
        return {}


def cmc_bars(token: str, key: str, interval: str = "1h") -> dict[int, float]:
    """CMC k-line candle = [o,h,l,c,v,time_ms,count] (oldest-first)."""
    q = {"platform": "bsc", "address": token, "interval": interval, "unit": "usd", "limit": "8"}
    req = urllib.request.Request(f"{CMC_BASE}?{urllib.parse.urlencode(q)}",
                                 headers={"X-CMC_PRO_API_KEY": key, "Accept": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=40) as r:
            data = json.loads(r.read()).get("data") or []
        return {int(c[5]) // 1000: float(c[3]) for c in data}
    except Exception:  # noqa: BLE001
        return {}


def onchain_candle(rpc: BscRpc, p: dict, from_block: int, to_block: int) -> dict:
    """Swap-built bar: count + volume (quote side) + close (quote/token)."""
    try:
        logs = rpc.get_logs([p["pool"]], from_block, to_block, topics=[[events.V2_SWAP, events.V3_SWAP]])
    except Exception as e:  # noqa: BLE001
        return {"err": type(e).__name__}
    rows = [events.decode_log(lg, p["dec0"], p["dec1"]) for lg in logs]
    swaps = [r for r in rows if r and r["event"] == "swap"]
    qa, ta = ("a1", "a0") if p["token_side"] == 0 else ("a0", "a1")
    vol = sum(abs(r[qa]) for r in swaps if r.get(qa) is not None)
    close = next((abs(r[qa] / r[ta]) for r in reversed(swaps) if r.get(qa) and r.get(ta)), None)
    return {"n_swaps": len(swaps), "vol_quote": vol, "close_quote": close, "quote": p.get("quote")}


# --- snapshot --------------------------------------------------------------

def snapshot(uni: list[dict], key: str, rpc: BscRpc, limit: int) -> None:
    now = int(time.time())
    cur_open = (now // 3600) * 3600
    blk_open = rpc.block_at_timestamp(cur_open, lo=rpc.block_number() - 8000)
    head = rpc.block_number()
    print(f"now {_utc(now)}  current 1h-bar {_utc(cur_open)} (forming)\n")
    print(f"{'token':<8} {'Gecko 1h':<14} {'CMC 1h':<14} {'parity':<7} {'on-chain cur-bar (sw/vol/close)':<38}")
    print("-" * 90)
    for p in (uni[:limit] if limit else uni):
        g, c = gecko_bars(p["pool"]), cmc_bars(p["token"], key)
        gn, cn = (max(g) if g else None), (max(c) if c else None)
        par = "?"
        common = (set(g) & set(c)) - {cur_open}
        if common:
            b = max(common); par = f"{abs(c[b]-g[b])/g[b]*100:.2f}%" if g.get(b) else "?"
        oc = onchain_candle(rpc, p, blk_open, head)
        ocs = (f"{oc['n_swaps']}sw/{oc['vol_quote']:.2f} {oc.get('quote')}/"
               f"{('%.5g' % oc['close_quote']) if oc.get('close_quote') else '?'}") if "err" not in oc else oc["err"]
        print(f"{p['symbol']:<8} {(_utc(gn)+'+'+str((now-gn)//60)+'m') if gn else 'MISS':<14} "
              f"{(_utc(cn)+'+'+str((now-cn)//60)+'m') if cn else 'MISS':<14} {par:<7} {ocs:<38}")
        time.sleep(0.4)
    print("\nSnapshot ties on the forming bar; run --watch for finalization lag of the just-closed bar.")


# --- watch (multi-interval finalization lag) -------------------------------

def _stats(vals: list[float]) -> dict:
    v = sorted(x for x in vals if x is not None)
    pc = lambda q: v[min(len(v) - 1, int(q * (len(v) - 1)))] if v else None  # noqa: E731
    return {"n": len(v), "median": pc(0.5), "p90": pc(0.9), "max": (v[-1] if v else None)}


def measure_boundary(uni, key, rpc, boundary: int, due: list[str], pace: float, cap_s: float) -> list[dict]:
    # on-chain: the bar is readable once the boundary block is mined
    t_oc = time.time()
    while time.time() - boundary < 90:
        try:
            if rpc.block_timestamp(rpc.block_number()) >= boundary:
                break
        except Exception:  # noqa: BLE001
            pass
        time.sleep(2)
    oc_lag = round(time.time() - boundary, 1)

    lag: dict[tuple, float] = {}
    pend = {(p["symbol"], iv, fd) for p in uni for iv in due for fd in ("gecko", "cmc")}
    while pend and (time.time() - boundary) < cap_s:
        for p in uni:
            for iv in due:
                tgt = boundary - INTERVALS[iv]["secs"]
                kg = (p["symbol"], iv, "gecko")
                if kg in pend:
                    if tgt in gecko_bars(p["pool"], INTERVALS[iv]["gt_tf"], INTERVALS[iv]["gt_agg"]):
                        lag[kg] = round(time.time() - boundary, 1); pend.discard(kg)
                    time.sleep(pace)
                kc = (p["symbol"], iv, "cmc")
                if kc in pend:
                    if tgt in cmc_bars(p["token"], key, INTERVALS[iv]["cmc"]):
                        lag[kc] = round(time.time() - boundary, 1); pend.discard(kc)
                    time.sleep(pace)
        done = len(uni) * len(due) * 2 - len(pend)
        print(f"  [{_utc(boundary)} {'+'.join(due)}] +{int(time.time()-boundary)}s settled "
              f"{done}/{len(uni)*len(due)*2}  onchain=+{oc_lag}s", flush=True)
        if pend:
            time.sleep(20)

    recs = []
    for iv in due:
        for p in uni:
            recs.append({"boundary": boundary, "interval": iv, "feed": "onchain", "lag": oc_lag})
            for fd in ("gecko", "cmc"):
                recs.append({"boundary": boundary, "interval": iv, "feed": fd,
                             "lag": lag.get((p["symbol"], iv, fd))})
    return recs


def watch(uni, key, rpc, limit: int, hours: float, pace: float) -> None:
    uni = uni[:limit] if limit else uni
    now = int(time.time())
    first = ((now // 900) + 1) * 900
    if first - now < 20:
        first += 900
    end = now + int(hours * 3600)
    boundaries = list(range(first, end + 1, 900))
    print(f"[watch] {len(uni)} pools, pace {pace}s; boundaries {_utc(boundaries[0])}..{_utc(boundaries[-1])} "
          f"(15m every mark, +1h at the hour). Sleeping {first-now}s.", flush=True)
    all_recs = []
    for b in boundaries:
        due = ["15m"] + (["1h"] if b % 3600 == 0 else [])
        time.sleep(max(0, b - int(time.time())))
        all_recs += measure_boundary(uni, key, rpc, b, due, pace, cap_s=480)
        # incremental dump so a long run is never lost
        with open("data/candle_feed_latency.json", "w", encoding="utf-8") as f:
            json.dump(all_recs, f)

    print("\n=== FINALIZATION LAG (seconds after bar close) — median / P90 / max (n) ===")
    for iv in ("15m", "1h"):
        print(f"  [{iv}]")
        for fd in ("onchain", "cmc", "gecko"):
            s = _stats([r["lag"] for r in all_recs if r["interval"] == iv and r["feed"] == fd])
            print(f"    {fd:<8} median {s['median']}s  P90 {s['p90']}s  max {s['max']}s  (n={s['n']})")
    print("  wrote data/candle_feed_latency.json")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--watch", action="store_true")
    ap.add_argument("--hours", type=float, default=2.0, help="watch run length (hours)")
    ap.add_argument("--limit", type=int, default=8, help="pools to use (0 = all 20)")
    ap.add_argument("--pace", type=float, default=1.5, help="per-call pacing seconds (watch)")
    ap.add_argument("--endpoint", default=None)
    args = ap.parse_args()
    config.load_dotenv()
    key = config.get("CMC_API_KEY")
    nr = args.endpoint or f"https://bsc-mainnet.nodereal.io/v1/{config.get('NODEREAL_API_KEY')}"
    rpc = BscRpc(endpoints=[nr])
    uni = load_universe()
    if args.watch:
        watch(uni, key, rpc, args.limit, args.hours, args.pace)
    else:
        snapshot(uni, key, rpc, args.limit)


if __name__ == "__main__":
    main()
