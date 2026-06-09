"""Committed candidate v1, rung 0 — disciplined trend-hold on vol-top8 (vault "Trading Strategies").

A per-token state machine encoding the user's discretionary discipline, on the validated vol-top8
universe. **Honest framing:** this is *not* a momentum-selection bet (that failed the IC gate here —
negative/mean-reverting; "entry alpha is dead, only low turnover survives"). The edge it targets is
**clean exits + low turnover** — the documented lane. Entry is a *state gate* (when to deploy), not
an alpha claim.

Per token, each rebalance:
  - **Enter** (flat → held) on **ignition** — a 2–3× volume spike (recent volume vs its trailing
    baseline) while price is rising — past a `cooldown`, and only above the prior cycle's **runup
    origin** (no FOMO re-entry / dead-zone churn). Record origin + peak.
  - **Hold** (let winners run): stay in while the trend is intact; track the rolling peak.
  - **Exit** (held → flat) on the rollover: price `stop_k` off the peak, OR close < trend-EMA → cash.
  - **Stand aside**: chop/bleed has no volume spike, so it never re-arms — no FOMO, no dead-zone churn.

Weights = equal share across currently-held tokens (per-token cap `max_weight`), rest in cash.

**Stateful:** relies on being called once per rebalance in chronological order, exactly how
`trader.sim.backtest.run_xs_backtest` drives a weights-fn. Build a fresh instance per backtest.
"""

from __future__ import annotations

import pandas as pd

from trader.strategy.candidate import select_vol_tokens


def build_rung0(returns: pd.DataFrame, k: int = 8, ema_span: int = 72,
                stop_k: float = 0.11, cooldown: int = 2, max_weight: float = 0.25,
                tokens: list[str] | None = None, volume: pd.DataFrame | None = None,
                vol_mult: float = 2.5, vol_spike: int = 24, vol_base: int = 168):
    """Return a stateful backtester weights-fn for the rung-0 disciplined trend-hold strategy.

    Args:
        returns: alt returns panel (vol-top-k universe is picked from it unless `tokens` is given).
        k: universe size (vol-top-k).
        ema_span: trend-EMA span in bars (72 = ~3 days hourly — catches multi-day moves, not noise).
        stop_k: trailing-stop fraction off the rolling peak (0.11 = exit 11% below the high).
        cooldown: rebalances to wait after an exit before re-entry is allowed.
        max_weight: per-token weight cap.
        tokens: explicit universe (overrides vol-selection) — pass a *fixed* set when rebuilding fresh
            state per window in a walk-forward sweep, so the universe doesn't drift between windows.
        volume: per-token volume panel aligned to `returns.index` (from `train_rl.build_volume_panel`).
        vol_mult: entry needs recent volume >= this × its baseline (the 2–3× spike).
        vol_spike: recent window (bars) for the spike; vol_base: the trailing baseline window.
    """
    universe = list(tokens) if tokens is not None else select_vol_tokens(returns, k)
    vol = volume
    st = {t: {"held": False, "origin": None, "peak": None, "exit_reb": -10 ** 9,
              "prior_origin": None} for t in universe}
    counter = {"reb": 0}

    def weights(hist: pd.DataFrame) -> pd.Series:
        reb = counter["reb"]
        counter["reb"] += 1
        i_now = hist.index[-1]
        active = []
        for t in universe:
            if t not in hist.columns:
                continue
            s = st[t]
            px = (1.0 + hist[t].fillna(0.0)).cumprod()        # synthetic price (consistent base)
            price = float(px.iloc[-1])
            ema = float(px.ewm(span=ema_span, adjust=False).mean().iloc[-1])

            if s["held"]:
                s["peak"] = max(s["peak"], price)
                if price < s["peak"] * (1.0 - stop_k) or price < ema:   # rollover → exit to cash
                    s["held"] = False
                    s["prior_origin"] = s["origin"]
                    s["exit_reb"] = reb
                else:
                    active.append(t)                          # let it run
            else:
                # entry = ignition: a 2–3× volume spike while price is rising
                spike = False
                if vol is not None and t in vol.columns:
                    v = vol[t].loc[:i_now].to_numpy()
                    if len(v) > vol_base:
                        recent = v[-vol_spike:].mean()
                        base = v[-vol_base:-vol_spike].mean()
                        spike = base > 0 and recent >= vol_mult * base
                rising = len(px) > vol_spike and price > float(px.iloc[-vol_spike - 1])
                cooled = (reb - s["exit_reb"]) >= cooldown
                reclaimed = s["prior_origin"] is None or price > s["prior_origin"]   # above runup origin
                if spike and rising and cooled and reclaimed:
                    s["held"] = True
                    s["origin"] = price
                    s["peak"] = price
                    active.append(t)

        if not active:
            return pd.Series(dtype=float)                     # all cash
        w = min(1.0 / len(active), max_weight)
        return pd.Series({t: w for t in active})

    return weights
