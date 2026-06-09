"""Per-bar gate tracer for rung-0 — WHY did a trade fire / not fire / exit when it did?

Replays the rung-0 state machine for one token over the warmed window and prints, per bar,
the four entry gates (spike / rising / cooled / reclaimed), the two exit conditions
(stop / ema-break), the state, and the action. Crucially it ALSO prints the real candle close
beside the strategy-space price (cumprod of r_alt returns) so we can see when the signal the
strategy trades on DISAGREES with the candle on the chart (a data-space divergence, not a logic
bug). ASCII-only for the Windows console.

    python scripts/trace_gates.py
"""
from __future__ import annotations

import datetime as dt
import json
import sys
import urllib.request

sys.path.insert(0, "scripts")
sys.path.insert(0, "src")

import pandas as pd  # noqa: E402

from train_rl import build_volume_panel, load_data, time_split  # noqa: E402
from trader.strategy.candidate import select_vol_tokens  # noqa: E402

WARMUP, EMA, STOP, COOL, VMULT, VSPK, VBASE = 168, 72, 0.25, 48, 2.5, 24, 168
RUN_ID = "rung0-widestop-v3"


def candle_closes(token):
    url = f"https://data.alexlouis.dev/{RUN_ID}/tk_{token}_candles.json"
    try:
        with urllib.request.urlopen(url, timeout=20) as r:
            return {int(c["time"]): float(c["close"]) for c in json.load(r)}
    except Exception as e:
        print(f"  (no candle file for {token}: {e})")
        return {}


def replay(warmed, t, vol):
    """Return {timestamp: row-dict} of gate/state for every traded bar of token t."""
    px = (1.0 + warmed[t].fillna(0.0)).cumprod()
    ema = px.ewm(span=EMA, adjust=False).mean()
    idx = warmed.index
    held, origin, peak, exit_reb, prior = False, None, None, -10 ** 9, None
    bar, rows = 0, {}
    for i in range(len(warmed)):
        if i < WARMUP:
            continue
        price, e, i_now = float(px.iloc[i]), float(ema.iloc[i]), int(idx[i])
        ratio, spike = float("nan"), False
        if t in vol.columns:
            v = vol[t].loc[:idx[i]].to_numpy()
            if len(v) > VBASE:
                recent, base = v[-VSPK:].mean(), v[-VBASE:-VSPK].mean()
                ratio = recent / base if base > 0 else float("nan")
                spike = base > 0 and recent >= VMULT * base
        rising = i >= VSPK and price > float(px.iloc[i - VSPK])
        cooled = (bar - exit_reb) >= COOL
        reclaimed = prior is None or price > prior
        act, stop_hit, ema_hit = "-", False, False
        if held:
            peak = max(peak, price)
            stop_hit, ema_hit = price < peak * (1.0 - STOP), price < e
            if stop_hit or ema_hit:
                held, prior, exit_reb = False, origin, bar
                act = "SELL(stop)" if stop_hit else "SELL(ema)"
            else:
                act = "hold"
        elif spike and rising and cooled and reclaimed:
            held, origin, peak, act = True, price, price, "BUY"
        rows[i_now] = dict(px=price, ema=e, ratio=ratio, spike=spike, rising=rising,
                           cooled=cooled, reclaimed=reclaimed, prior=prior, st=held, act=act)
        bar += 1
    return rows


def trace(warmed, vol, token, start, end, note=""):
    rows = replay(warmed, token, vol)
    cnd = candle_closes(token)
    t0, t1 = int(start.timestamp()), int(end.timestamp())
    win = [ts for ts in rows if t0 <= ts <= t1]
    if not win:
        print(f"\n== {token} {start:%b %d %H:%M}-{end:%b %d %H:%M}  (no bars) ==")
        return
    px0 = rows[win[0]]["px"]
    c0 = next((cnd[ts] for ts in win if ts in cnd), None)
    print(f"\n== {token}  {start:%b %d %H:%M} - {end:%b %d %H:%M} UTC  {note} ==")
    print(f"  {'time':14}{'cndl$':>9}{'cndD%':>7}{'pxD%':>7}{'volX':>6}"
          f"{'spk':>4}{'ris':>4}{'cool':>5}{'rec':>4}{'st':>4}  act")
    for ts in win:
        r = rows[ts]
        c = cnd.get(ts)
        cd = f"{(c / c0 - 1) * 100:+5.0f}" if c and c0 else "   - "
        pd_ = f"{(r['px'] / px0 - 1) * 100:+5.0f}"
        cs = f"{c:8.4g}" if c else "     -  "
        when = dt.datetime.fromtimestamp(ts, dt.timezone.utc).strftime("%b%d %H:%M")
        print(f"  {when:14}{cs:>9}{cd:>7}{pd_:>7}{r['ratio']:>6.1f}"
              f"{'Y' if r['spike'] else '.':>4}{'Y' if r['rising'] else '.':>4}"
              f"{'Y' if r['cooled'] else '.':>5}{'Y' if r['reclaimed'] else '.':>4}"
              f"{'H' if r['st'] else '.':>4}  {r['act']}")


def main():
    returns, btc, anchor, liq = load_data()
    _, _, test_r = time_split(returns)
    ts = returns.index.get_loc(test_r.index[0])
    warmed = returns.iloc[ts - WARMUP:]
    uni = select_vol_tokens(test_r, 8)
    vol = build_volume_panel(uni, returns.index)
    print(f"universe = {uni}")

    def d(s):
        return dt.datetime.strptime(s, "%Y-%m-%d %H:%M").replace(tzinfo=dt.timezone.utc)

    # ZEC: user expected a BUY ~May 1 16:00 (huge runup) but only traded May 23 -> what blocked it?
    if "ZEC" in uni:
        trace(warmed, vol, "ZEC", d("2026-04-30 12:00"), d("2026-05-05 12:00"), "[expected BUY ~May01 16:00]")
        trace(warmed, vol, "ZEC", d("2026-05-23 06:00"), d("2026-05-24 00:00"), "[the trade that DID fire]")
    # B: entered May 11 17:00, user says volume said ~06:00 -> entry lag?
    if "B" in uni:
        trace(warmed, vol, "B", d("2026-05-11 00:00"), d("2026-05-11 22:00"), "[entered 17:00, expected ~06:00]")
    # SKYAI: BUY May 23 02:00 -> SELL 04:00, 2-hour whipsaw -> stop or ema?
    if "SKYAI" in uni:
        trace(warmed, vol, "SKYAI", d("2026-05-22 22:00"), d("2026-05-23 08:00"), "[2-hr whipsaw 02:00->04:00]")
    # UB: second entry bought a local top then sold -> entry below trend?
    if "UB" in uni:
        trace(warmed, vol, "UB", d("2026-05-20 00:00"), d("2026-05-25 00:00"), "[2nd entry buys top?]")


if __name__ == "__main__":
    main()
