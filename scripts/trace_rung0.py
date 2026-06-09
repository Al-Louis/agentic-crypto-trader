"""Trace the rung-0 decision per rebalance for one token on the test split — what fired, what
blocked, at each daily 00:00 check. Answers 'why no buy in the first week / why this trade'.

    python scripts/trace_rung0.py SIREN B SKYAI
"""
from __future__ import annotations

import datetime as dt
import glob
import sys

sys.path.insert(0, "scripts")
sys.path.insert(0, "src")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from train_rl import load_data, time_split  # noqa: E402
from trader.strategy.candidate import select_vol_tokens  # noqa: E402

WARMUP, REBAL = 168, 24
EMA, BRK, STOP, COOL = 72, 72, 0.11, 2


def candle_closes(tok):
    ds = glob.glob(f"data/ohlcv/hour_1/{tok}_*/*.parquet")
    if not ds:
        return None
    oh = (pd.concat([pd.read_parquet(f) for f in ds]).drop_duplicates("timestamp")
          .sort_values("timestamp").set_index("timestamp"))
    oh.index = (oh.index // 1000) if oh.index.max() > 1e12 else oh.index
    return oh["close"]


def trace(test_r, tok):
    closes = candle_closes(tok)
    print(f"\n=== {tok} ===  (warmup ends, first possible trade: "
          f"{dt.datetime.fromtimestamp(int(test_r.index[WARMUP]), dt.timezone.utc):%b %d %H:%M})")
    print(f"  {'date':12}{'candle$':>10}{'up?':>5}{'newHi?':>7}{'cool?':>6}{'reclaim?':>9}{'state':>7}  action")
    held, origin, peak, exit_reb, prior = False, None, None, -10 ** 9, None
    reb = 0
    for i in range(WARMUP, len(test_r), REBAL):
        px = (1.0 + test_r[tok].iloc[: i + 1].fillna(0.0)).cumprod()
        price, ema = float(px.iloc[-1]), float(px.ewm(span=EMA, adjust=False).mean().iloc[-1])
        rhigh = float(px.iloc[-BRK:].max())
        up, newhi = price > ema, price >= rhigh - 1e-12
        cooled = (reb - exit_reb) >= COOL
        reclaim = prior is None or price > prior
        t = int(test_r.index[i])
        cprice = (float(closes.iloc[int(np.abs(closes.index.to_numpy() - t).argmin())])
                  if closes is not None else float("nan"))
        act = "-"
        if held:
            peak = max(peak, price)
            if price < peak * (1.0 - STOP) or price < ema:
                held, prior, exit_reb, act = False, origin, reb, "** EXIT (sell) **"
            else:
                act = "hold"
        else:
            if up and newhi and cooled and reclaim:
                held, origin, peak, act = True, price, price, "** ENTER (buy) **"
            else:
                blk = [n for n, c in [("not-up", up), ("not-newHigh", newhi),
                                      ("cooldown", cooled), ("below-origin", reclaim)] if not c]
                act = "flat  (blocked: " + ", ".join(blk) + ")"
        d = dt.datetime.fromtimestamp(t, dt.timezone.utc).strftime("%b %d %H:%M")
        st = "HELD" if held else "flat"
        print(f"  {d:12}{cprice:>10.4g}{('Y' if up else 'n'):>5}{('Y' if newhi else 'n'):>7}"
              f"{('Y' if cooled else 'n'):>6}{('Y' if reclaim else 'n'):>9}{st:>7}  {act}")
        reb += 1


def main():
    returns, btc, anchor, liq = load_data()
    _, _, test_r = time_split(returns)
    uni = select_vol_tokens(test_r, 8)
    for tok in (sys.argv[1:] or ["SIREN"]):
        if tok in uni:
            trace(test_r, tok)
        else:
            print(f"\n{tok} not in test universe {uni}")


if __name__ == "__main__":
    main()
