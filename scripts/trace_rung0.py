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

from train_rl import build_volume_panel, load_data, time_split  # noqa: E402
from trader.strategy.candidate import select_vol_tokens  # noqa: E402

WARMUP, REBAL = 168, 24
EMA, STOP, COOL = 72, 0.11, 2
VMULT, VSPIKE, VBASE = 2.5, 24, 168


def candle_closes(tok):
    ds = glob.glob(f"data/ohlcv/hour_1/{tok}_*/*.parquet")
    if not ds:
        return None
    oh = (pd.concat([pd.read_parquet(f) for f in ds]).drop_duplicates("timestamp")
          .sort_values("timestamp").set_index("timestamp"))
    oh.index = (oh.index // 1000) if oh.index.max() > 1e12 else oh.index
    return oh["close"]


def trace(warmed, tok, vol):
    closes = candle_closes(tok)
    vser = vol[tok]
    print(f"\n=== {tok} ===  (window start / first possible trade: "
          f"{dt.datetime.fromtimestamp(int(warmed.index[WARMUP]), dt.timezone.utc):%b %d %H:%M})")
    print(f"  {'date':12}{'candle$':>10}{'volX':>6}{'spike?':>7}{'rising?':>8}{'cool?':>6}"
          f"{'reclaim?':>9}{'state':>7}  action")
    held, origin, peak, exit_reb, prior = False, None, None, -10 ** 9, None
    reb = 0
    for i in range(WARMUP, len(warmed), REBAL):
        t = int(warmed.index[i])
        px = (1.0 + warmed[tok].iloc[: i + 1].fillna(0.0)).cumprod()
        price, ema = float(px.iloc[-1]), float(px.ewm(span=EMA, adjust=False).mean().iloc[-1])
        v = vser.loc[:t].to_numpy()
        recent = v[-VSPIKE:].mean()
        base = v[-VBASE:-VSPIKE].mean() if len(v) > VBASE else 0.0
        volx = recent / base if base > 0 else 0.0
        spike = base > 0 and recent >= VMULT * base
        rising = len(px) > VSPIKE and price > float(px.iloc[-VSPIKE - 1])
        cooled = (reb - exit_reb) >= COOL
        reclaim = prior is None or price > prior
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
            if spike and rising and cooled and reclaim:
                held, origin, peak, act = True, price, price, "** ENTER (buy) **"
            else:
                blk = [n for n, c in [("no-spike", spike), ("not-rising", rising),
                                      ("cooldown", cooled), ("below-origin", reclaim)] if not c]
                act = "flat  (" + ", ".join(blk) + ")"
        d = dt.datetime.fromtimestamp(t, dt.timezone.utc).strftime("%b %d %H:%M")
        st = "HELD" if held else "flat"
        print(f"  {d:12}{cprice:>10.4g}{volx:>6.1f}{('Y' if spike else 'n'):>7}"
              f"{('Y' if rising else 'n'):>8}{('Y' if cooled else 'n'):>6}"
              f"{('Y' if reclaim else 'n'):>9}{st:>7}  {act}")
        reb += 1


def main():
    returns, btc, anchor, liq = load_data()
    _, _, test_r = time_split(returns)
    ts = returns.index.get_loc(test_r.index[0])
    warmed = returns.iloc[ts - WARMUP:]                  # warm on pre-window data -> trade from day 1
    uni = select_vol_tokens(test_r, 8)
    vol = build_volume_panel(uni, returns.index)
    for tok in (sys.argv[1:] or ["SIREN"]):
        if tok in uni:
            trace(warmed, tok, vol)
        else:
            print(f"\n{tok} not in test universe {uni}")


if __name__ == "__main__":
    main()
