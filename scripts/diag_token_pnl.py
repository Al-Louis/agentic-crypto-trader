"""Reconcile a portfolio bundle's per-token PnL against its headline return, and stress-test the
price series for data artifacts (single-bar spikes that fake a moonshot on illiquid microcaps).

    python scripts/diag_token_pnl.py [run_id]   # default ppo-sharpe-s1

For each universe token, computes honest realized+unrealized PnL from the *candle* prices the
frontend sees (FIFO not needed: cash_flow + final_qty*last_price), flags spikes, and checks
whether the per-token PnLs sum to the portfolio's actual profit. A big gap => the displayed PnL
is computed off a different price series than the equity curve (the candles-vs-returns split).
"""
from __future__ import annotations

import json
import sys
import urllib.request

HOST = "https://data.alexlouis.dev"


def g(run, name):
    with urllib.request.urlopen(f"{HOST}/{run}/{name}", timeout=30) as r:
        return json.load(r)


def ts(t):
    import datetime as dt
    return dt.datetime.fromtimestamp(t / 1000 if t > 1e12 else t, dt.timezone.utc).strftime("%m-%d")


def main():
    run = sys.argv[1] if len(sys.argv) > 1 else "ppo-sharpe-s1"
    info = g(run, "run_info.json")
    metrics = g(run, "metrics.json")
    uni = {u["symbol"]: u["slug"] for u in info.get("universe", [])}
    cap = 10_000.0
    tot_ret = metrics.get("total_return_pct", 0.0)
    print(f"\n=== {run} ===  headline return {tot_ret:+.1%}  -> profit ${tot_ret*cap:,.0f} on ${cap:,.0f}\n")

    rows, total_pnl = [], 0.0
    for sym, slug in uni.items():
        try:
            candles = g(run, f"tk_{slug}_candles.json")
            trades = g(run, f"tk_{slug}_trades.json")
        except Exception as e:  # noqa: BLE001
            print(f"  {sym}: fetch failed ({e})"); continue
        if not candles:
            continue
        closes = [c["close"] for c in candles]
        last = closes[-1]
        # biggest single-bar move (spike detector)
        spike, spike_i = 0.0, 0
        for i in range(1, len(closes)):
            if closes[i - 1] > 0:
                r = closes[i] / closes[i - 1] - 1.0
                if abs(r) > abs(spike):
                    spike, spike_i = r, i
        # honest PnL from candle prices: net cash flow + final position marked to last close
        qty = cashflow = fees = 0.0
        for m in trades:
            px = m.get("price") or 0.0
            if not px:
                continue
            q = m["usd"] / px
            fees += m.get("fee", 0.0)
            if m["side"] == "buy":
                qty += q; cashflow -= m["usd"]
            else:
                qty -= q; cashflow += m["usd"]
        pnl = cashflow + qty * last - fees
        total_pnl += pnl
        rows.append((sym, pnl, closes[0], last, last / closes[0] - 1 if closes[0] else 0,
                     min(closes), max(closes), spike, spike_i, len(trades), qty * last))

    rows.sort(key=lambda r: -r[1])
    print(f"  {'token':10}{'PnL$':>10}{'px0':>10}{'pxLast':>10}{'px chg':>9}{'max/min':>9}"
          f"{'maxbar':>9}{'trades':>7}{'endpos$':>10}")
    for sym, pnl, p0, pl, chg, lo, hi, sp, spi, nt, endv in rows:
        print(f"  {sym:10}{pnl:>+10,.0f}{p0:>10.4g}{pl:>10.4g}{chg:>+8.0%}"
              f"{hi/lo if lo else 0:>8.1f}x{sp:>+8.0%}{nt:>7}{endv:>+10,.0f}")
    print(f"\n  sum per-token PnL = ${total_pnl:,.0f}   vs   headline profit ${tot_ret*cap:,.0f}")
    gap = total_pnl - tot_ret * cap
    print(f"  reconciliation gap = ${gap:,.0f}  ({'OK' if abs(gap) < 0.15*abs(tot_ret*cap)+500 else 'MISMATCH — PnL computed off a different price series than equity'})")


if __name__ == "__main__":
    main()
