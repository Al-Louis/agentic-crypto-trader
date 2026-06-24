"""Ad-hoc: measure how far GeckoTerminal's latest hourly candle for a pool lags wall-clock.

Direct evidence for the 'Gecko is slow to publish candles' hypothesis behind the live
execution lag. Run: `PYTHONPATH=src python scripts/probe_gecko_lag.py [pool] [symbol]`.
"""
from __future__ import annotations

import datetime as dt
import sys
import time

from trader.data import geckoterminal as gt

POOL = sys.argv[1] if len(sys.argv) > 1 else "0x203d66ecb7263EfE424FCbA0898761fc9FC9a8c0"
SYM = sys.argv[2] if len(sys.argv) > 2 else "B"


def utc(ts: int) -> str:
    return dt.datetime.utcfromtimestamp(int(ts)).strftime("%Y-%m-%d %H:%M")


def main() -> None:
    now = int(time.time())
    rows = gt.fetch_ohlcv(POOL, timeframe="hour", aggregate=1, limit=14, network="bsc")
    print(f"symbol={SYM} pool={POOL}")
    print(f"wall-clock now UTC: {utc(now)}  ({now})")
    print(f"rows returned: {len(rows)}\n")
    print(f"{'bar_open_UTC':<18}{'ts':>12}{'close':>14}{'volume':>16}  age_now")
    for r in rows[:14]:
        tsr = int(r[0])
        age = (now - tsr) / 3600.0
        print(f"{utc(tsr):<18}{tsr:>12}{float(r[4]):>14.5f}{float(r[5]):>16.2f}  {age:6.2f}h")
    if not rows:
        return
    closed = [int(r[0]) for r in rows if int(r[0]) + 3600 <= now]
    newest_open = int(rows[0][0])
    newest_closed = max(closed) if closed else None
    print()
    print(f"newest candle bar_open : {utc(newest_open)} UTC (closes {utc(newest_open + 3600)})")
    if newest_closed is not None:
        # what the agent would act on this tick = newest CLOSED bar Gecko actually returns
        behind_h = (now - 3600 - newest_closed) / 3600.0
        print(f"newest CLOSED bar offered: {utc(newest_closed)} UTC")
        print(f"=> lag of newest-closed bar behind (now-1h): {behind_h:.2f} h "
              f"(0 = on time; >0 = Gecko is behind, the agent can't see the just-closed bar yet)")


if __name__ == "__main__":
    main()
