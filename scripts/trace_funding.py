"""Validate the capital-model rewrite: per-token markers from the REAL run_rung0.

Before the rewrite, ZEC's May-1 ignition fired but was unfunded (cash=-$5) and the state machine
phantom-held it through a +28% runup. After the rewrite (loser-funded rotation + funded-only held),
ZEC should get a real, funded buy near May 1. This dumps every token's buy/sell marker times from
the actual executor and highlights ZEC. ASCII-only.

    python scripts/trace_funding.py
"""
from __future__ import annotations

import datetime as dt
import sys

sys.path.insert(0, "scripts")
sys.path.insert(0, "src")

import pandas as pd  # noqa: E402

from train_rl import build_volume_panel, load_data, time_split  # noqa: E402
from trader.strategy.candidate import select_vol_tokens  # noqa: E402
from trader.strategy.rung0 import build_rung0, run_rung0  # noqa: E402

WARMUP = 168


def when(ts):
    return dt.datetime.fromtimestamp(int(ts), dt.timezone.utc).strftime("%b%d %H:%M")


def main():
    returns, btc, anchor, liq = load_data()
    _, _, test_r = time_split(returns)
    tloc = returns.index.get_loc(test_r.index[0])
    warmed = returns.iloc[tloc - WARMUP:]
    uni = select_vol_tokens(test_r, 8)
    vol = build_volume_panel(uni, returns.index)

    eq, records, fees = run_rung0(warmed, build_rung0(warmed, tokens=uni, volume=vol), liq)
    nbuy = sum(1 for r in records for v in r["trades_usd"].values() if v > 0)
    nsell = sum(1 for r in records for v in r["trades_usd"].values() if v < 0)
    print(f"return {eq.iloc[-1] / eq.iloc[0] - 1:+.1%}  fees ${fees:,.0f}  "
          f"funded buys {nbuy}  sells {nsell}\n")
    print(f"  {'token':>7}  buys / sells (UTC)")
    for t in uni:
        buys = [when(r["time"]) for r in records if r["trades_usd"].get(t, 0.0) > 0]
        sells = [when(r["time"]) for r in records if r["trades_usd"].get(t, 0.0) < 0]
        mark = "  <-- was missed (phantom-held)" if t == "ZEC" else ""
        print(f"  {t:>7}  BUY {buys}")
        print(f"  {'':>7}  SELL {sells}{mark}")


if __name__ == "__main__":
    main()
