"""Prep for the Gecko-vs-CMC signal parity: build a CMC-sourced anchor (BTC/BNB), a fresh Gecko alt
OHLCV store, and BOTH feature sets via the real `refresh_factor_features`. Both feature builds share
the same CMC anchor so the comparison isolates the ALT price feed. Laptop-runnable (no torch)."""
from __future__ import annotations

import json
import os
import time
import urllib.parse
import urllib.request

import pandas as pd

from trader import config
from trader.agent.live_data import refresh_factor_features
from trader.data import geckoterminal as gt
from trader.data.downloader import OHLCV_COLS, _store_dir, tf_key

config.load_dotenv()
KEY = config.get("CMC_API_KEY")
TFK = tf_key("hour", 1)
SEL = json.load(open("data/selection.json", encoding="utf-8"))
BTCB = "0x7130D2A12B9BCbFAe4f2634d864A1Ee1Ce3Ead9c"
WBNB = "0xBB4CdB9CBd36B01bD1cBaEBF2De08d9173bc095c"


def cmc_kline(token: str, limit: int = 1000) -> pd.DataFrame:
    q = {"platform": "bsc", "address": token, "interval": "1h", "unit": "usd", "limit": str(limit)}
    req = urllib.request.Request(f"https://pro-api.coinmarketcap.com/v1/k-line/candles?{urllib.parse.urlencode(q)}",
                                 headers={"X-CMC_PRO_API_KEY": KEY, "Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=60) as r:
        data = json.loads(r.read()).get("data") or []
    rows = [(int(c[5]), float(c[0]), float(c[1]), float(c[2]), float(c[3]), float(c[4])) for c in data]  # ms ts
    return pd.DataFrame(rows, columns=OHLCV_COLS)


# 1) CMC anchor (BTC/BNB) — ms timestamps, the format load_anchor expects
print("[1] CMC anchor BTC/BNB...")
for sym, tok in (("BTC_USDT", BTCB), ("BNB_USDT", WBNB)):
    df = cmc_kline(tok)
    d = os.path.join("data", "anchor_cmc", sym)
    os.makedirs(d, exist_ok=True)
    df.to_parquet(os.path.join(d, "1h.parquet"), index=False)
    print(f"    {sym}: {len(df)} bars")
    time.sleep(0.4)

# 2) fresh Gecko alt store (the gecko side of the parity)
print("[2] fresh Gecko alt store (20 pools)...")
for s in SEL:
    sym, pool = s["symbol"], s["pair_address"]
    try:
        rows = gt.fetch_ohlcv(pool, timeframe="hour", aggregate=1, limit=1000, network="bsc")  # newest-first
    except Exception as e:  # noqa: BLE001
        print(f"    {sym:<10} gecko FAIL {type(e).__name__}"); continue
    if not rows:
        print(f"    {sym:<10} 0"); continue
    df = pd.DataFrame([(int(r[0]), float(r[1]), float(r[2]), float(r[3]), float(r[4]), float(r[5]))
                       for r in rows], columns=OHLCV_COLS).sort_values("timestamp").reset_index(drop=True)
    d = _store_dir(os.path.join("data", "ohlcv_gecko"), sym, pool, TFK)
    os.makedirs(d, exist_ok=True)
    df.to_parquet(os.path.join(d, f"p_{int(df['timestamp'].iloc[0])}.parquet"), index=False)
    print(f"    {sym:<10} {len(df)} bars")
    time.sleep(2.2)   # GeckoTerminal courtesy pacing

# 3) features for both, SAME CMC anchor
print("[3] building features (gecko + cmc) on the shared CMC anchor...")
anc = os.path.join("data", "anchor_cmc")
g = refresh_factor_features(SEL, ohlcv_root=os.path.join("data", "ohlcv_gecko"), anchor_root=anc,
                            out=os.path.join("data", "features_gecko"))
c = refresh_factor_features(SEL, ohlcv_root=os.path.join("data", "ohlcv_cmc"), anchor_root=anc,
                            out=os.path.join("data", "features_cmc"))
print(f"    gecko features: {len(g)} tokens | cmc features: {len(c)} tokens")
# coverage check
import datetime as dt
for label, out in (("gecko", "features_gecko"), ("cmc", "features_cmc")):
    f = os.path.join("data", out, "TAC_factor.parquet")
    if os.path.exists(f):
        d = pd.read_parquet(f)
        ts = int(d["timestamp"].iloc[-1])
        print(f"    {label} TAC latest bar: {dt.datetime.fromtimestamp(ts, dt.timezone.utc):%m-%d %H:%M}  ({len(d)} rows)")
