"""Live driver for the event-driven RL champion (sbq-s1) — the weekly-replay harness.

The champion (`ppo-event-rdLe4-sbq-3c84b4a-s1`, RecurrentPPO/LSTM on `EventRungEnv`) is only
valid fed the EXACT observation its env produced in training. Rather than reimplement the obs
or the exit machinery live (train/serve skew = out-of-distribution = silent breakage), this
driver **re-runs the validated inference loop verbatim** and swaps only the data layer:

  recorded panel  ->  a live rolling panel (the same `data/` files, kept fresh each hour)

The key property exploited: `EventRungEnv` precomputes its signals over a panel and is fully
deterministic given that panel, and the deployment cadence is already defined by
`simulate_weekly` — each **calendar week (00:00 UTC Monday)** is an independent COLD session
(fresh $10k, vol-top-8 reselected at the week open, LSTM reset, no cross-week compounding).

So each hour at bar-close we:
  1. (caller) refresh the `data/` panels with the just-closed bar,
  2. slice the CURRENT cold-week window (warmup prepad -> now),
  3. run `evaluate_event_policy` over it with a cold-at-week-start predictor, and
  4. DIFF the fills against the prior hour -> any fill on the newest bar is THIS hour's decision.

Because closed bars never change, the replay for bars < now is stable across hours; only the
newest bar can introduce a new event. The strategy machinery (universe pick, ignition, trailing
stop / loss floor / tp rungs / intrabar floor / rule overlay) is reused unchanged from
`trader.train.event_env` via `scripts/train_event.evaluate_event_policy`; this module owns ONLY
the live orchestration (window math + fill diff + weekly reset), which is what these tests cover.

The heavy, torch-touching pieces (model load, `model.predict`) are injected, so the window/diff
logic is exercised offline with a fake predictor over the recorded panel.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

# The validated inference core lives in scripts/ (owned by the training side — imported, never
# reimplemented). Add the repo's scripts/ + src/ to the path once. This file is at
# src/trader/agent/event_live.py, so the repo root is four dirnames up.
_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
for _p in (os.path.join(_REPO, "scripts"), os.path.join(_REPO, "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

WEEK_SECS = 7 * 24 * 3600
MONDAY_PHASE = 345600          # t % WEEK_SECS for 00:00 UTC Monday (the unix epoch was a Thursday)
WARMUP = 168                   # signal warmup bars (== train_event.WARMUP); kept local to avoid a torch import
WEEK_BARS = 168                # a full calendar week of hourly bars
START_CAPITAL = 10_000.0


def week_start_for(ts: int) -> int:
    """The 00:00-UTC-Monday timestamp of the calendar week containing `ts` (seconds)."""
    ts = int(ts)
    return ts - ((ts - MONDAY_PHASE) % WEEK_SECS)


def cold_week_window(returns: pd.DataFrame, now_ts: int) -> tuple[pd.DataFrame, int, int]:
    """Slice the CURRENT cold-week window from the panel: a `WARMUP`-bar prepad ending just
    before the week open, then the week's bars up to and including the latest bar <= `now_ts`.

    Mirrors `simulate_weekly`'s `win = returns.iloc[i0-WARMUP : i0-WARMUP+WARMUP+WEEK_BARS]`,
    but TRUNCATED at `now` (live has no future bars). Returns `(win, week_start_ts, i0)` where
    `i0` is the integer position of the week open in `returns.index`. Raises if there isn't a
    full warmup before the week open, or the week open isn't in the panel yet.
    """
    idx = [int(t) for t in returns.index]
    pos_of = {t: i for i, t in enumerate(idx)}
    ws = week_start_for(now_ts)
    i0 = pos_of.get(ws)
    if i0 is None:
        raise ValueError(f"week open {ws} not present in the panel (data not caught up?)")
    if i0 < WARMUP:
        raise ValueError(f"only {i0} bars before week open {ws}; need {WARMUP} for warmup")
    # latest bar at or before now (the just-closed bar); never index past the week's 168 bars
    cur = i0
    for i in range(i0, min(i0 + WEEK_BARS, len(idx))):
        if idx[i] <= int(now_ts):
            cur = i
        else:
            break
    win = returns.iloc[i0 - WARMUP: cur + 1]
    return win, ws, i0


@dataclass
class Fill:
    """One execution the env produced this hour. `usd` > 0 is a buy, < 0 a sell. `price` is the
    env's price-index fill (the same scale the paper book and equity are computed in)."""

    token: str
    usd: float
    fee: float
    time: int
    price: float
    reason: str
    obs: dict = field(default_factory=dict)

    @property
    def side(self) -> str:
        return "buy" if self.usd >= 0 else "sell"


def fills_from_records(records: list[dict]) -> list[Fill]:
    """Flatten `evaluate_event_policy` records (each `{time, fills:[...], ...}`) into `Fill`s,
    in chronological (record, then within-record) order."""
    out: list[Fill] = []
    for rec in records:
        for f in rec.get("fills", []):
            out.append(Fill(token=f["token"], usd=float(f["usd"]), fee=float(f.get("fee", 0.0)),
                            time=int(f["time"]), price=float(f.get("px", 0.0)),
                            reason=str(f.get("reason", "")), obs=f.get("obs", {})))
    return out


def new_fills(records: list[dict], after_ts: int) -> list[Fill]:
    """Fills on bars STRICTLY AFTER `after_ts` — this hour's freshly-decided executions. The
    replay is deterministic and past bars are immutable, so fills at times <= `after_ts` were
    already emitted on a prior hour; only later-bar fills are new. `after_ts < week_start` on a
    fresh week surfaces the week's whole fill stream (incl. a basket_default open at the open)."""
    return [f for f in fills_from_records(records) if f.time > int(after_ts)]


class LiveEventTrader:
    """Holds the checkpoint + provenance and runs one hourly evaluation of the current cold week.

    Construct with either loaded `(model, vn)` (tests / preloaded) or `(policy_path, vecnorm_path)`
    to lazily load on first use (torch + sb3-contrib — desktop/box only). `prov` is the published
    `metrics.json` provenance dict (the exact training config). `env_kwargs` is built ONCE over the
    full returns panel (matching `simulate_weekly`); the env reindexes btc/vol/frac panels to each
    week's window internally.
    """

    def __init__(self, prov: dict, *, policy_path: str | None = None, vecnorm_path: str | None = None,
                 model=None, vn=None, act_last_bar: bool = False):
        self.prov = prov
        self.act_last_bar = bool(act_last_bar)   # LIVE: act on the just-closed bar at this tick (no +1-bar lag)
        self.recurrent = bool(prov.get("recurrent"))
        self._policy_path, self._vecnorm_path = policy_path, vecnorm_path
        self._model, self._vn = model, vn

    # -- checkpoint -----------------------------------------------------------
    def _ensure_loaded(self) -> None:
        if self._model is not None and self._vn is not None:
            return
        if not (self._policy_path and self._vecnorm_path):
            raise RuntimeError("LiveEventTrader needs a loaded (model, vn) or both checkpoint paths")
        import pickle  # noqa: PLC0415
        if self.recurrent:
            from sb3_contrib import RecurrentPPO  # noqa: PLC0415
            self._model = RecurrentPPO.load(self._policy_path, device="cpu")
        else:
            from stable_baselines3 import PPO  # noqa: PLC0415
            self._model = PPO.load(self._policy_path, device="cpu")
        with open(self._vecnorm_path, "rb") as fh:
            self._vn = pickle.load(fh)

    def _predict_fn(self):
        """A FRESH cold-LSTM predictor (state reset) — one per week, matching the validated
        cold-weekly eval where every session starts cold (in-distribution)."""
        self._ensure_loaded()
        from simulate import make_predict  # noqa: PLC0415 — validated predictor glue
        return make_predict(self._model, self._vn, self.recurrent)

    def env_kwargs(self, returns: pd.DataFrame) -> dict:
        """The EXACT EventRungEnv kwargs the checkpoint trained with, rebuilt from provenance
        over the full panel (frac panels get reindexed to each week's window inside the env)."""
        from simulate import env_kwargs_from_provenance  # noqa: PLC0415
        from train_rl import build_ohlc_frac_panels  # noqa: PLC0415
        return env_kwargs_from_provenance(self.prov, returns, build_ohlc_frac_panels)

    # -- one hourly evaluation ------------------------------------------------
    def evaluate_week(self, returns: pd.DataFrame, btc: pd.Series, liq: dict, vol: pd.DataFrame,
                      now_ts: int, env_kwargs: dict, *, predict_fn=None) -> dict:
        """Run `evaluate_event_policy` over the current cold-week window up to `now_ts`. Returns
        `{week_start, equity, records, universe, token_pnl, fills, win_index}`. `predict_fn` is
        injectable (tests pass a deterministic stub); production passes None -> a cold LSTM."""
        from train_event import evaluate_event_policy  # noqa: PLC0415 — the validated loop
        win, ws, _i0 = cold_week_window(returns, now_ts)
        pf = predict_fn if predict_fn is not None else self._predict_fn()
        eq, records, universe, fees, raw, token_pnl = evaluate_event_policy(
            pf, win, btc, liq, vol, env_kwargs, act_last_bar=self.act_last_bar)
        return {"week_start": ws, "equity": eq, "records": records, "universe": universe,
                "token_pnl": token_pnl, "fills": fills_from_records(records),
                "win_index": [int(t) for t in win.index], "raw_actions": raw, "fees": fees}
