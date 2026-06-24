"""Probe: does the CMC key's tier expose DEX OHLCV, and does it match GeckoTerminal?

Tests the hypothesis (user, 2026-06-22) that we can switch the live decision feed from
GeckoTerminal to CMC's DEX API. Two questions decide it:
  1. ACCESS  - does CMC_API_KEY's plan reach /v4/dex/... at all? (structured error -> tier)
  2. PARITY  - do CMC's hourly candles match Gecko's for the same pool, well enough that the
               agent's ignition gate (surge/cushion/rising/ema_up) would fire on the SAME bars?
  3. FRESH   - which feed has the just-closed candle sooner (the root-cause lever)?

Run: PYTHONPATH=src python scripts/probe_cmc_dex.py
"""
from __future__ import annotations

import datetime as dt
import json
import time
import urllib.error
import urllib.parse
import urllib.request

from trader import config
from trader.data import geckoterminal as gt

BASE = "https://pro-api.coinmarketcap.com"
B_TOKEN = "0x6bdcce4a559076e37755a78ce0c06214e59e4444"   # B token contract (BSC)
B_WBNB_POOL = "0x203d66ecb7263EfE424FCbA0898761fc9FC9a8c0"  # the pool we trade on (Pancake v2, WBNB)


def utc(ts) -> str:
    return dt.datetime.fromtimestamp(int(ts), dt.timezone.utc).strftime("%Y-%m-%d %H:%M")


def cmc_get(endpoint: str, params: dict, key: str) -> tuple[int, dict]:
    url = f"{BASE}{endpoint}?{urllib.parse.urlencode(params)}"
    req = urllib.request.Request(url, headers={"X-CMC_PRO_API_KEY": key, "Accept": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=40) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        try:
            return e.code, json.loads(e.read())
        except Exception:  # noqa: BLE001
            return e.code, {"raw": str(e)}


def show_status(label: str, code: int, body: dict) -> None:
    st = body.get("status") or {}
    print(f"[{label}] HTTP {code}  err={st.get('error_code')} msg={st.get('error_message')!r} "
          f"credits={st.get('credit_count')}")


def main() -> None:
    config.load_dotenv()
    key = config.get("CMC_API_KEY")
    if not key:
        print("no CMC_API_KEY"); return
    now = int(time.time())
    print(f"now UTC {utc(now)}\n")

    # 1) ACCESS — can the key reach the DEX surface at all?
    code, body = cmc_get("/v4/dex/networks/list", {}, key)
    show_status("dex/networks/list", code, body)
    if code == 200:
        nets = body.get("data") or []
        bsc = [n for n in nets if str(n.get("network_slug") or n.get("slug")).lower() in ("bsc", "bnb", "binance-smart-chain")]
        print("  BSC network rows:", json.dumps(bsc[:3])[:300])
    print()

    # 2) OHLCV latest for the WBNB pool — try a few param spellings, keep the first that works.
    variants = [
        {"network_slug": "bsc", "contract_address": B_WBNB_POOL, "convert": "USD"},
        {"network_id": "14", "contract_address": B_WBNB_POOL, "convert": "USD"},
        {"network_slug": "bsc", "contract_address": B_WBNB_POOL, "time_period": "hourly",
         "interval": "1h", "count": "14", "convert": "USD"},
    ]
    for ep in ("/v4/dex/pairs/ohlcv/latest", "/v4/dex/pairs/ohlcv/historical"):
        for v in variants:
            code, body = cmc_get(ep, v, key)
            show_status(f"{ep}  {list(v)}", code, body)
            if code == 200 and body.get("data"):
                print("  DATA sample:", json.dumps(body["data"])[:700])
                break
        print()

    # 3) discover B's pools + volumes via spot-pairs (confirm the 7% WBNB / 37% USD1 split).
    for v in ({"network_slug": "bsc", "base_asset_contract_address": B_TOKEN, "convert": "USD"},
              {"network_slug": "bsc", "token_address": B_TOKEN, "convert": "USD"}):
        code, body = cmc_get("/v4/dex/spot-pairs/latest", v, key)
        show_status(f"spot-pairs/latest {list(v)}", code, body)
        if code == 200 and body.get("data"):
            for p in (body["data"] or [])[:8]:
                print("  pool:", p.get("contract_address"), p.get("dex_slug") or p.get("dex_name"),
                      "quote=", (p.get("quote_asset") or {}).get("symbol"),
                      "vol24=", (p.get("quote") or [{}])[0].get("volume_24h") if isinstance(p.get("quote"), list)
                      else (p.get("quote") or {}).get("USD", {}).get("volume_24h"))
            break
    print()

    # 4) GeckoTerminal freshness for the same pool, side by side.
    grows = gt.fetch_ohlcv(B_WBNB_POOL, timeframe="hour", aggregate=1, limit=6, network="bsc")
    print("GeckoTerminal newest 6 (WBNB pool):")
    for r in grows[:6]:
        print(f"  {utc(r[0])}  close={float(r[4]):.5f} vol={float(r[5]):.0f}  age={(now-int(r[0]))/3600:.2f}h")


if __name__ == "__main__":
    main()
