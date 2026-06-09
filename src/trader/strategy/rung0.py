"""Committed candidate v1, rung 0 — disciplined trend-hold on vol-top8 (vault "Trading Strategies").

A per-token state machine encoding the user's discretionary discipline, on the validated vol-top8
universe. **Honest framing:** this is *not* a momentum-selection bet (that failed the IC gate here —
negative/mean-reverting; "entry alpha is dead, only low turnover survives"). The edge it targets is
**clean exits + low turnover** — the documented lane. Entry is a *state gate* (when to deploy), not
an alpha claim.

Per token, each rebalance:
  - **Enter** (flat → held) on a confirmed breakout: price > trend-EMA AND a new `breakout_n`-bar
    high, past a `cooldown`, and only if price has reclaimed the prior cycle's **runup origin**
    (so it never re-enters the sideways bleed below where a runup started). Record origin + peak.
  - **Hold** (let winners run): stay in while the trend is intact; track the rolling peak.
  - **Exit** (held → flat) on the rollover: price `stop_k` off the peak, OR close < trend-EMA → cash.
  - **Stand aside**: a token bleeding/chopping below its origin won't make a new high above the EMA,
    so it never re-arms — no FOMO re-entry, no dead-zone churn.

Weights = equal share across currently-held tokens (per-token cap `max_weight`), rest in cash.

**Stateful:** relies on being called once per rebalance in chronological order, exactly how
`trader.sim.backtest.run_xs_backtest` drives a weights-fn. Build a fresh instance per backtest.
"""

from __future__ import annotations

import pandas as pd

from trader.strategy.candidate import select_vol_tokens


def build_rung0(returns: pd.DataFrame, k: int = 8, ema_span: int = 72, breakout_n: int = 72,
                stop_k: float = 0.11, cooldown: int = 2, max_weight: float = 0.25):
    """Return a stateful backtester weights-fn for the rung-0 disciplined trend-hold strategy.

    Args:
        returns: alt returns panel (vol-top-k universe is picked from it, matching the baseline).
        k: universe size (vol-top-k).
        ema_span: trend-EMA span in bars (72 = ~3 days hourly — catches multi-day moves, not noise).
        breakout_n: a new high over this many bars confirms entry.
        stop_k: trailing-stop fraction off the rolling peak (0.11 = exit 11% below the high).
        cooldown: rebalances to wait after an exit before re-entry is allowed.
        max_weight: per-token weight cap.
    """
    universe = select_vol_tokens(returns, k)
    st = {t: {"held": False, "origin": None, "peak": None, "exit_reb": -10 ** 9,
              "prior_origin": None} for t in universe}
    counter = {"reb": 0}

    def weights(hist: pd.DataFrame) -> pd.Series:
        reb = counter["reb"]
        counter["reb"] += 1
        active = []
        for t in universe:
            if t not in hist.columns:
                continue
            s = st[t]
            px = (1.0 + hist[t].fillna(0.0)).cumprod()        # synthetic price (consistent base)
            price = float(px.iloc[-1])
            ema = float(px.ewm(span=ema_span, adjust=False).mean().iloc[-1])
            recent_high = float(px.iloc[-breakout_n:].max())
            uptrend = price > ema
            new_high = price >= recent_high - 1e-12           # current is a fresh breakout_n-bar high

            if s["held"]:
                s["peak"] = max(s["peak"], price)
                if price < s["peak"] * (1.0 - stop_k) or price < ema:   # rollover → exit to cash
                    s["held"] = False
                    s["prior_origin"] = s["origin"]
                    s["exit_reb"] = reb
                else:
                    active.append(t)                          # let it run
            else:
                cooled = (reb - s["exit_reb"]) >= cooldown
                reclaimed = s["prior_origin"] is None or price > s["prior_origin"]   # above runup origin
                if uptrend and new_high and cooled and reclaimed:
                    s["held"] = True
                    s["origin"] = price
                    s["peak"] = price
                    active.append(t)

        if not active:
            return pd.Series(dtype=float)                     # all cash
        w = min(1.0 / len(active), max_weight)
        return pd.Series({t: w for t in active})

    return weights
