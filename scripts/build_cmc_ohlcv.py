"""Build a parallel OHLCV store sourced from CMC k-line (by-token aggregate), in the EXACT format
the env reads (`data/ohlcv/hour_1/<slug>_<pool10>/p_*.parquet`), so `refresh_factor_features` and the
volume/shape panels can be pointed at it for an APPLES-TO-APPLES Gecko-vs-CMC policy replay.

Keyed by each selection token's `pair_address[:10]` (so the real feature build finds it) but the
candles come from CMC's `address=<token>` aggregate. BTC/BNB anchors are left as-is — this isolates
the ALT price feed. Read-only w.r.t. the live agent. Run:
    PYTHONPATH=src python scripts/build_cmc_ohlcv.py
"""
from __future__ import annotations

import json
import os
import time
import urllib.parse
import urllib.request

import pandas as pd

from trader import config
from trader.data.downloader import OHLCV_COLS, _store_dir, tf_key

CMC = "https://pro-api.coinmarketcap.com/v1/k-line/candles"
ROOT = os.path.join("data", "ohlcv_cmc")


def fetch(token: str, key: str, limit: int = 1000) -> pd.DataFrame:
    q = {"platform": "bsc", "address": token, "interval": "1h", "unit": "usd", "limit": str(limit)}
    req = urllib.request.Request(f"{CMC}?{urllib.parse.urlencode(q)}",
                                 headers={"X-CMC_PRO_API_KEY": key, "Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=60) as r:
        data = json.loads(r.read()).get("data") or []      # candle = [o,h,l,c,v,time_ms,count]
    rows = [(int(c[5]) // 1000, float(c[0]), float(c[1]), float(c[2]), float(c[3]), float(c[4]))
            for c in data]
    return pd.DataFrame(rows, columns=OHLCV_COLS)


def main() -> None:
    config.load_dotenv()
    key = config.get("CMC_API_KEY")
    sel = json.load(open("data/selection.json", encoding="utf-8"))
    tfk = tf_key("hour", 1)
    for s in sel:
        sym, pool, tok = s["symbol"], s["pair_address"], s.get("token_address")
        if not tok:
            print(f"  {sym:<10} no token_address — skip"); continue
        try:
            df = fetch(tok, key)
        except Exception as e:  # noqa: BLE001
            print(f"  {sym:<10} FETCH FAIL {type(e).__name__} {str(e)[:50]}"); continue
        if df.empty:
            print(f"  {sym:<10} 0 candles"); continue
        d = _store_dir(ROOT, sym, pool, tfk)
        os.makedirs(d, exist_ok=True)
        df.to_parquet(os.path.join(d, f"p_{int(df['timestamp'].iloc[0])}.parquet"), index=False)
        import datetime as dt
        span = (f"{dt.datetime.fromtimestamp(int(df['timestamp'].iloc[0]), dt.timezone.utc):%m-%d}.."
                f"{dt.datetime.fromtimestamp(int(df['timestamp'].iloc[-1]), dt.timezone.utc):%m-%d %H:%M}")
        print(f"  {sym:<10} {len(df):>4} bars  {span}  -> {d}")
        time.sleep(0.35)
    print(f"\nCMC OHLCV store built at {ROOT}/")


if __name__ == "__main__":
    main()
