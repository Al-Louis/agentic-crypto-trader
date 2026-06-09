"""Confirm the cash-starvation / phantom-held bug in run_rung0.

Replays the real portfolio run and logs every time an entry SIGNAL fires but can't be funded
(no cash), and every time an exit fires on a token the portfolio doesn't actually hold. If ZEC's
May-1 ignition shows up here as UNFUNDED, that's why the frontend shows no ZEC trade until May 23.
ASCII-only.

    python scripts/trace_funding.py
"""
from __future__ import annotations

import datetime as dt
import sys

sys.path.insert(0, "scripts")
sys.path.insert(0, "src")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from train_rl import build_volume_panel, load_data, time_split  # noqa: E402
from trader.sim.broker import DEFAULT_GAS_USD, DEFAULT_LP_FEE_BPS, amm_cost_usd  # noqa: E402
from trader.strategy.candidate import select_vol_tokens  # noqa: E402
from trader.strategy.rung0 import build_rung0  # noqa: E402

WARMUP, ENTRY_FRAC = 168, 0.20


def when(ts):
    return dt.datetime.fromtimestamp(int(ts), dt.timezone.utc).strftime("%b%d %H:%M")


def main():
    returns, btc, anchor, liq = load_data()
    _, _, test_r = time_split(returns)
    tloc = returns.index.get_loc(test_r.index[0])
    warmed = returns.iloc[tloc - WARMUP:]
    uni = select_vol_tokens(test_r, 8)
    vol = build_volume_panel(uni, returns.index)
    step = build_rung0(warmed, tokens=uni, volume=vol)

    syms = list(warmed.columns)
    pos = pd.Series(0.0, index=syms)
    cash = 10_000.0
    unfunded, phantom, funded = [], [], 0
    for i in range(len(warmed)):
        r = warmed.iloc[i].reindex(syms).fillna(0.0).to_numpy()
        pos = pd.Series(pos.to_numpy() * (1.0 + r), index=syms)
        equity = float(pos.sum() + cash)
        if i >= WARMUP and equity > 1.0:
            entries, exits = step(warmed.iloc[: i + 1])
            for t in exits:
                v = float(pos[t])
                if abs(v) >= 1.0:
                    c = amm_cost_usd(-v, liq.get(t, 0.0), DEFAULT_LP_FEE_BPS, DEFAULT_GAS_USD)
                    cash += v - c
                    pos[t] = 0.0
                else:
                    phantom.append((warmed.index[i], t))           # exit signal, nothing held
            for t in entries:
                size = min(ENTRY_FRAC * equity, cash)
                if size >= 1.0:
                    c = amm_cost_usd(size, liq.get(t, 0.0), DEFAULT_LP_FEE_BPS, DEFAULT_GAS_USD)
                    cash -= size + c
                    pos[t] += size
                    funded += 1
                else:
                    unfunded.append((warmed.index[i], t, equity, cash,
                                     int((pos > 1e-6).sum()), sorted(pos[pos > 1e-6].index)))

    print(f"funded entries: {funded}   UNFUNDED (cash-starved) signals: {len(unfunded)}   "
          f"phantom exits: {len(phantom)}\n")
    print("UNFUNDED entry signals (signal fired, no cash to buy):")
    print(f"  {'time':14}{'token':>7}{'equity':>9}{'cash':>8}{'#held':>6}  held")
    for ts, t, eq, ca, n, names in unfunded:
        print(f"  {when(ts):14}{t:>7}{eq:>9.0f}{ca:>8.0f}{n:>6}  {names}")
    zec = [(ts, t) for ts, t in [(u[0], u[1]) for u in unfunded] if t == "ZEC"]
    print(f"\nZEC unfunded count: {len(zec)}; first: {when(zec[0][0]) if zec else '-'}")
    print(f"phantom exits (paper-exit with no position): {[(when(ts), t) for ts, t in phantom][:12]}")


if __name__ == "__main__":
    main()
