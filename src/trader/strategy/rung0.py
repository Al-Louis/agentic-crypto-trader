"""Committed candidate v1, rung 0 — volume-ignition trend-hold (vault "Trading Strategies").

A per-token state machine encoding the user's discretionary discipline, evaluated **every bar
(intra-day)** — NOT on a fixed daily clock. Entries/exits fire the hour the signal fires.

Per token, each bar:
  - **Enter** (flat → held) on **ignition** — a 2–3× volume spike (recent volume vs its trailing
    baseline) while price is rising — past a `cooldown`, and only above the prior cycle's **runup
    origin** (no FOMO re-entry / dead-zone churn). Record origin + peak.
  - **Hold** (let winners run): stay in while the trend is intact; track the rolling peak. The
    position is **never trimmed** — it runs freely until an exit.
  - **Exit** (held → flat) on the rollover: price `stop_k` off the peak, OR price < trend-EMA → cash.

`build_rung0` returns a stateful `step(hist) -> (entries, exits)`; `run_rung0` is the event-driven
backtester (buy `entry_frac` of equity on an entry, sell to cash on an exit, leave holds alone).
Build a fresh strategy + run per backtest (the state machine is stateful).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from trader.sim.broker import DEFAULT_GAS_USD, DEFAULT_LP_FEE_BPS, amm_cost_usd
from trader.strategy.candidate import select_vol_tokens


def build_rung0(returns: pd.DataFrame, k: int = 8, ema_span: int = 72, stop_k: float = 0.25,
                cooldown: int = 48, max_weight: float = 0.25, tokens: list[str] | None = None,
                volume: pd.DataFrame | None = None, vol_mult: float = 2.5, vol_spike: int = 24,
                vol_base: int = 168):
    """Return a stateful `step(hist) -> (entries, exits)` — the per-bar ignition state machine.

    Args:
        ema_span: trend-EMA span in bars (72 = ~3 days hourly). stop_k: trailing-stop off the peak.
        cooldown: BARS (hours) to wait after an exit before re-entry (48 = 2 days).
        volume: per-token volume panel aligned to `returns.index` (`train_rl.build_volume_panel`).
        vol_mult/vol_spike/vol_base: entry needs recent volume ≥ vol_mult × its baseline (the spike).
        max_weight: kept for API compat; sizing is `entry_frac` in `run_rung0`, holds aren't trimmed.
    """
    universe = list(tokens) if tokens is not None else select_vol_tokens(returns, k)
    vol = volume
    st = {t: {"held": False, "origin": None, "peak": None, "exit_reb": -10 ** 9,
              "prior_origin": None} for t in universe}
    counter = {"bar": 0}

    def step(hist: pd.DataFrame):
        bar = counter["bar"]
        counter["bar"] += 1
        i_now = hist.index[-1]
        entries, exits = [], []
        for t in universe:
            if t not in hist.columns:
                continue
            s = st[t]
            px = (1.0 + hist[t].fillna(0.0)).cumprod()
            price = float(px.iloc[-1])
            ema = float(px.ewm(span=ema_span, adjust=False).mean().iloc[-1])

            if s["held"]:
                s["peak"] = max(s["peak"], price)
                if price < s["peak"] * (1.0 - stop_k) or price < ema:   # rollover → exit to cash
                    s["held"], s["prior_origin"], s["exit_reb"] = False, s["origin"], bar
                    exits.append(t)
            else:
                spike = False                                          # 2–3× volume spike
                if vol is not None and t in vol.columns:
                    v = vol[t].loc[:i_now].to_numpy()
                    if len(v) > vol_base:
                        recent, base = v[-vol_spike:].mean(), v[-vol_base:-vol_spike].mean()
                        spike = base > 0 and recent >= vol_mult * base
                rising = len(px) > vol_spike and price > float(px.iloc[-vol_spike - 1])
                cooled = (bar - s["exit_reb"]) >= cooldown
                reclaimed = s["prior_origin"] is None or price > s["prior_origin"]   # above runup origin
                if spike and rising and cooled and reclaimed:
                    s["held"], s["origin"], s["peak"] = True, price, price
                    entries.append(t)
        return entries, exits

    return step


def run_rung0(returns: pd.DataFrame, step_fn, liquidity: dict, capital: float = 10_000.0,
              warmup: int = 168, entry_frac: float = 0.20, lp_fee_bps: float = DEFAULT_LP_FEE_BPS,
              gas_usd: float = DEFAULT_GAS_USD, record_every: int = 24) -> tuple:
    """Event-driven backtest: evaluate the strategy EVERY bar (intra-day). Buy `entry_frac` of equity
    from cash on an entry, sell to cash on an exit, and **leave held positions alone** — winners run,
    no daily/hourly trim. Returns `(equity Series, records, total_fees)`."""
    syms = list(returns.columns)
    pos = pd.Series(0.0, index=syms)
    cash = float(capital)
    eq = np.empty(len(returns))
    records, fees = [], 0.0
    for i in range(len(returns)):
        r = returns.iloc[i].reindex(syms).fillna(0.0).to_numpy()
        pos = pd.Series(pos.to_numpy() * (1.0 + r), index=syms)        # positions drift with the bar
        equity = float(pos.sum() + cash)
        tu, tf = {}, {}
        if i >= warmup and equity > 1.0:
            entries, exits = step_fn(returns.iloc[: i + 1])
            for t in exits:
                v = float(pos[t])
                if abs(v) >= 1.0:                                      # sell the whole position → cash
                    c = amm_cost_usd(-v, liquidity.get(t, 0.0), lp_fee_bps, gas_usd)
                    cash += v - c
                    fees += c
                    tu[t], tf[t], pos[t] = -v, c, 0.0
            for t in entries:
                size = min(entry_frac * equity, cash)                 # buy a slice of cash; holds untouched
                if size >= 1.0:
                    c = amm_cost_usd(size, liquidity.get(t, 0.0), lp_fee_bps, gas_usd)
                    cash -= size + c
                    fees += c
                    tu[t], tf[t] = size, c
                    pos[t] += size
        eq[i] = float(pos.sum() + cash)
        if tu or (i >= warmup and i % record_every == 0):             # markers + daily allocation snapshots
            e = eq[i] if eq[i] > 0 else 1.0
            records.append({"time": int(returns.index[i]),
                            "weights": {s: float(pos[s] / e) for s in syms if pos[s] > 1e-6},
                            "trades_usd": tu, "trade_fees": tf})
    return pd.Series(eq, index=returns.index), records, fees
