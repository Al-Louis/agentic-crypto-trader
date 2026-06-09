"""Committed candidate v1, rung 0 — volume-ignition trend-hold (vault "Trading Strategies").

A per-token discipline evaluated **every bar (intra-day)** — NOT on a fixed daily clock. The
strategy splits cleanly: `build_rung0` is a **stateless per-bar signal** (price / EMA / volume
surge / trend), and `run_rung0` is the **event-driven executor** that owns all held-state, cash,
sizing, and capital rotation. Keeping funding in one place fixes the phantom-held bug (a signal
that can't be funded must NOT flip a token to "held").

Per token, each bar:
  - **Enter** (flat -> held) on **ignition** — a sharp `vol_fast`-bar volume surge >= `vol_mult` x
    its trailing baseline, while price is rising AND **clearly above a rising trend-EMA** (the
    quality gate that rejects brief micro-spikes that whipsaw). Gated by `cooldown` and the prior
    cycle's **runup origin** (no FOMO re-entry / dead-zone churn below where a runup began).
  - **Hold** (let winners run): never trimmed by time. The position runs until an exit OR until it
    becomes the **weakest holding** and its capital is rotated into a stronger fresh ignition.
  - **Exit** (held -> flat) on the rollover: price `stop_k` off the peak, OR price < trend-EMA.

**Capital model — loser-funded rotation.** Sizing is `entry_frac` of equity per entry. When a
fresh ignition can't be funded from cash, `run_rung0` frees capital from the **weakest current
holding** (lowest price/EMA cushion) — but only if that holding is weaker than the incoming
candidate, so a winner stronger than the new opportunity is never trimmed. This stops first-mover
ignitions from permanently crowding out better later ones (the ZEC failure). Rotation sells are
recorded as markers so per-token PnL reconciles.

Build a fresh signal + run per backtest (the executor is stateful).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from trader.sim.broker import DEFAULT_GAS_USD, DEFAULT_LP_FEE_BPS, amm_cost_usd
from trader.strategy.candidate import select_vol_tokens


def build_rung0(returns: pd.DataFrame, k: int = 8, ema_span: int = 72, tokens: list[str] | None = None,
                volume: pd.DataFrame | None = None, vol_mult: float = 2.5, vol_spike: int = 24,
                vol_base: int = 168, vol_fast: int = 4, trend_buf: float = 0.0,
                trend_gate: bool = True, max_weight: float = 0.25):
    """Return a stateless `signal(hist) -> {token: dict}` of per-bar primitives.

    Each token dict has: price, ema, spike, rising, ignite, cushion (= price/ema - 1, used by the
    executor to rank holding strength for rotation). State (held / cooldown / origin / dead-zone)
    and all funding live in `run_rung0`, not here.

    Args:
        ema_span: trend-EMA span in bars (72 = ~3 days hourly).
        vol_fast: surge window — recent `vol_fast` bars' volume vs the baseline (sharp, not a 24-bar
            mean, so it fires near the ignition rather than ~11h late).
        vol_spike: price-rise lookback (price must exceed its level this many bars back).
        vol_base: trailing baseline window for the surge ratio.
        vol_mult: surge multiple — recent volume must be >= vol_mult x baseline.
        trend_gate/trend_buf: ignite only when price > ema*(1+trend_buf) and the EMA is rising
            (rejects micro-spikes near a flat EMA that whipsaw out immediately).
        max_weight: kept for API compat; sizing is `entry_frac` in `run_rung0`.
    """
    universe = list(tokens) if tokens is not None else select_vol_tokens(returns, k)
    vol = volume

    def signal(hist: pd.DataFrame):
        i_now = hist.index[-1]
        out = {}
        for t in universe:
            if t not in hist.columns:
                continue
            px = (1.0 + hist[t].fillna(0.0)).cumprod()
            ema_s = px.ewm(span=ema_span, adjust=False).mean()
            price, ema = float(px.iloc[-1]), float(ema_s.iloc[-1])
            spike = False                                          # sharp volume surge
            if vol is not None and t in vol.columns:
                v = vol[t].loc[:i_now].to_numpy()
                if len(v) > vol_base:
                    recent, base = v[-vol_fast:].mean(), v[-vol_base:-vol_fast].mean()
                    spike = base > 0 and recent >= vol_mult * base
            rising = len(px) > vol_spike and price > float(px.iloc[-vol_spike - 1])
            ema_prev = float(ema_s.iloc[-vol_fast - 1]) if len(ema_s) > vol_fast else ema
            trend_ok = (not trend_gate) or (price > ema * (1.0 + trend_buf) and ema >= ema_prev)
            out[t] = {"price": price, "ema": ema, "spike": spike, "rising": rising,
                      "ignite": bool(spike and rising and trend_ok), "cushion": price / ema - 1.0}
        return out

    return signal


def run_rung0(returns: pd.DataFrame, signal_fn, liquidity: dict, capital: float = 10_000.0,
              warmup: int = 168, entry_frac: float = 0.20, stop_k: float = 0.25, cooldown: int = 48,
              lp_fee_bps: float = DEFAULT_LP_FEE_BPS, gas_usd: float = DEFAULT_GAS_USD,
              record_every: int = 24, rotate: bool = True) -> tuple:
    """Event-driven backtest — evaluate EVERY bar (intra-day). Owns held-state, cash, sizing, and
    loser-funded rotation. Buy `entry_frac` of equity on a funded ignition, sell to cash on an exit,
    leave winners untrimmed; when cash is short, rotate out the weakest holding to fund a stronger
    fresh ignition. Returns `(equity Series, records, total_fees)`."""
    syms = list(returns.columns)
    pos = pd.Series(0.0, index=syms)
    cash, fees, bar = float(capital), 0.0, 0
    st = {s: {"held": False, "origin": None, "peak": None, "exit_reb": -10 ** 9,
              "prior_origin": None} for s in syms}
    eq = np.empty(len(returns))
    records: list = []

    for i in range(len(returns)):
        r = returns.iloc[i].reindex(syms).fillna(0.0).to_numpy()
        pos = pd.Series(pos.to_numpy() * (1.0 + r), index=syms)    # positions drift with the bar
        equity = float(pos.sum() + cash)
        tu, tf = {}, {}

        def trade(t, delta):                                       # signed: >0 buy, <0 sell
            nonlocal cash, fees
            c = amm_cost_usd(delta, liquidity.get(t, 0.0), lp_fee_bps, gas_usd)
            cash -= delta + c
            pos[t] += delta
            fees += c
            tu[t], tf[t] = tu.get(t, 0.0) + delta, tf.get(t, 0.0) + c

        if i >= warmup and equity > 1.0:
            sig = signal_fn(returns.iloc[: i + 1])
            for t in [s for s in sig if st[s]["held"]]:            # 1) exits on held positions
                s, d = st[t], sig[t]
                s["peak"] = max(s["peak"], d["price"])
                if d["price"] < s["peak"] * (1.0 - stop_k) or d["price"] < d["ema"]:
                    v = float(pos[t])
                    if abs(v) >= 1.0:
                        trade(t, -v)
                    s.update(held=False, prior_origin=s["origin"], exit_reb=bar)
            cands = [t for t in sig if sig[t]["ignite"] and not st[t]["held"]   # 2) fresh ignitions
                     and (bar - st[t]["exit_reb"]) >= cooldown
                     and (st[t]["prior_origin"] is None or sig[t]["price"] > st[t]["prior_origin"])]
            cands.sort(key=lambda t: sig[t]["cushion"], reverse=True)           # fund strongest first
            for t in cands:
                if min(entry_frac * equity, cash) < 1.0 and rotate:            # loser-funded rotation
                    held = [h for h in sig if st[h]["held"]]
                    if held:
                        weak = min(held, key=lambda h: sig[h]["cushion"])
                        if sig[weak]["cushion"] < sig[t]["cushion"]:           # swap weak for strong only
                            v = float(pos[weak])
                            if abs(v) >= 1.0:
                                trade(weak, -v)
                            st[weak].update(held=False, prior_origin=st[weak]["origin"], exit_reb=bar)
                            equity = float(pos.sum() + cash)
                size = min(entry_frac * equity, cash)
                if size >= 1.0:                                                # held=True ONLY when funded
                    trade(t, size)
                    st[t].update(held=True, origin=sig[t]["price"], peak=sig[t]["price"])
            bar += 1

        eq[i] = float(pos.sum() + cash)
        if tu or (i >= warmup and i % record_every == 0):         # markers + daily allocation snapshots
            e = eq[i] if eq[i] > 0 else 1.0
            records.append({"time": int(returns.index[i]),
                            "weights": {s: float(pos[s] / e) for s in syms if pos[s] > 1e-6},
                            "trades_usd": tu, "trade_fees": tf})
    return pd.Series(eq, index=returns.index), records, fees
