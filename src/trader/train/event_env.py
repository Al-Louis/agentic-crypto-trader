"""Event-driven RL env wrapping rung-0's machinery — rung 1 (RL learns the discretion).

Unlike `PortfolioEnv` (daily rebalance-to-target on a fixed `step_bars` clock), this steps at
rung-0's **events**: the agent acts only when a volume-ignition fires (size it / skip it) or a held
position trips its trailing-stop or EMA-break (cut it / hold through). Between events the env
advances bar-by-bar — positions drift, no trades — so execution is **intra-day and event-timed,
never a 00:00 clock**. It is a **semi-MDP**: `step()` applies the decision, then runs forward to the
next event and returns that interval's equity change as the reward.

Division of labor (the point of rung 1): **rung-0 supplies the edge** — *when* to consider acting
(its ignition timing, exit triggers, and dead-zone / cooldown anti-churn discipline). **The RL
learns the discretion rung-0 hard-codes** — entry sizing (conviction), and whether to **override**
an exit (hold a winner through its stop, or cut early). Funding a new ignition still uses rung-0's
loser-funded rotation rule for v1.

Pure numpy/pandas so it runs and tests on the laptop; a thin gymnasium adapter wraps it for SB3 on
the desktop. Signals are precomputed once over the panel (causal: cumprod/rolling/shift use only
past bars; scale-invariant ratios), and positions are valued by price-index ratios, so the per-bar
advance is cheap.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from trader.sim.broker import DEFAULT_GAS_USD, DEFAULT_LP_FEE_BPS, amm_cost_usd

OBS_DIM = 12   # ... + the rung-0 rule's current exposure (0 in absolute mode)
SURGE_CLIP, CUSHION_CLIP, RET_CLIP = 10.0, 1.0, 5.0
SKIP_EPS, HOLD_EPS = 0.05, 0.95   # action < SKIP_EPS on entry = skip; >= HOLD_EPS on exit = full hold


class EventRungEnv:
    """Event-driven rung-0 env. `reset()`/`step(action)` are plain numpy in/out (one scalar action
    in [0,1], interpreted by event type). Obs is a fixed `OBS_DIM` vector + an event-type flag."""

    def __init__(self, returns: pd.DataFrame, btc_close: pd.Series, liquidity: dict, *,
                 volume: pd.DataFrame, k: int = 8, ema_span: int = 72, warmup: int = 168,
                 episode_bars: int = 168, capital: float = 10_000.0,
                 lp_fee_bps: float = DEFAULT_LP_FEE_BPS, gas_usd: float = DEFAULT_GAS_USD,
                 vol_mult: float = 2.5, vol_spk: int = 24, vol_base: int = 168, vol_fast: int = 4,
                 stop_k: float = 0.25, cooldown: int = 48, max_entry_frac: float = 0.34,
                 dd_soft: float = 0.15, dd_gate: float = 0.30, dd_lambda: float = 2.0,
                 reward_mode: str = "absolute", record_trace: bool = False, seed: int | None = None):
        self.returns = returns.sort_index()
        self.btc = btc_close.reindex(self.returns.index).ffill().bfill()
        self.btc_ema = self.btc.ewm(span=ema_span, adjust=False).mean()
        self.liquidity = liquidity
        self.k, self.warmup, self.episode_bars = k, warmup, episode_bars
        self.capital = float(capital)
        self.lp_fee_bps, self.gas_usd = lp_fee_bps, gas_usd
        self.stop_k, self.cooldown, self.max_entry_frac = stop_k, cooldown, max_entry_frac
        self.dd_soft, self.dd_gate, self.dd_lambda = dd_soft, dd_gate, dd_lambda
        self.reward_mode = reward_mode                      # "absolute" | "relative" (vs the rung-0 rule)
        self.rule_entry_frac = 0.20                         # the rung-0 RULE's fixed sizing (the benchmark)
        self.record_trace = bool(record_trace)              # eval-only: per-bar equity curve + markers
        self.obs_dim, self.action_dim = OBS_DIM, 1
        self.n_bars = len(self.returns)
        self.cols = list(self.returns.columns)
        self.col_ix = {t: j for j, t in enumerate(self.cols)}

        # --- precompute global, causal, scale-invariant signal arrays [bar x token] ---
        vol = volume.reindex(self.returns.index).fillna(0.0)
        px = (1.0 + self.returns.fillna(0.0)).cumprod()
        ema = px.ewm(span=ema_span, adjust=False).mean()
        vrec = vol.rolling(vol_fast, min_periods=1).mean()
        vbase = vol.shift(vol_fast).rolling(max(vol_base - vol_fast, 1), min_periods=1).mean()
        surge = (vrec / vbase.replace(0.0, np.nan)).fillna(0.0)
        cushion = px / ema - 1.0
        rising = px / px.shift(vol_spk) - 1.0
        ema_up = ema >= ema.shift(vol_fast)
        ignite = ((surge >= vol_mult) & (rising > 0) & (cushion > 0) & ema_up)
        self._px = px.to_numpy()
        self._cush = cushion.to_numpy()
        self._surge = surge.clip(0.0, SURGE_CLIP).to_numpy()
        self._ignite = ignite.to_numpy()
        self._std = self.returns.rolling(warmup, min_periods=8).std().to_numpy()  # for causal universe

        self._min_start = warmup
        self._max_start = self.n_bars - episode_bars - 1
        if self._max_start < self._min_start:           # == is fine (a single full-window episode, eval)
            raise ValueError("series too short for episode_bars/warmup")
        self.rng = np.random.default_rng(seed)

    # -- lifecycle ----------------------------------------------------------
    def reset(self, *, start: int | None = None, seed: int | None = None) -> np.ndarray:
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.start = int(start) if start is not None else int(
            self.rng.integers(self._min_start, max(self._max_start, self._min_start + 1)))
        self.end = self.start + self.episode_bars
        self.bar = self.start
        self.universe = self._pick_universe(self.start)          # causal vol-top-k, fixed for episode
        self.cash = self.capital
        self.peak_eq = self.capital
        self.pos = {}                                            # tok -> dict(usd, entry_bar, peak_px, ref_px)
        self.cool = {t: -10 ** 9 for t in self.universe}        # last exit bar (cooldown)
        self.prior_origin = {t: None for t in self.universe}    # dead-zone: runup origin of last cycle
        self.ignite_armed = {t: True for t in self.universe}    # entry edge: only prompt on a fresh ignite
        self._queue = []                                        # pending (event_type, tok) on the current bar
        self._eq_mark = self.capital
        self._trades = []                                       # trades made in the current step (markers)
        self._eq_trace = [(int(self.returns.index[self.start]), self.capital)]  # per-bar equity (eval)
        # relative/residual reward: precompute the rung-0 RULE's per-bar equity AND per-token weights
        # over the episode - the benchmark the agent must BEAT. "relative" = beat the rule's portfolio
        # return; "residual" = beat it per-decision (only the agent's weight DEVIATIONS earn/lose).
        if self.reward_mode in ("relative", "residual"):
            self._rule_eq, self._rule_w = self._rule_equity_curve(self.start, self.end)
        else:
            self._rule_eq = self._rule_w = None
        self._rule_eq_mark = float(self._rule_eq[0]) if self._rule_eq is not None else self.capital
        self._prev_bar = self.start                             # residual: interval + agent-weight snapshot
        self._agent_w_snap = {}
        self._advance_to_event(first=True)                      # roll to the first decision point
        return self._obs()

    def step(self, action) -> tuple:
        # action in [-1,1]: neutral a=0 -> m=0.5 lands in the INTERIOR (the policy trades from init),
        # so a Gaussian head can't dead-gradient collapse to "never trade" at a [0,1] skip boundary
        a = float(np.clip(np.asarray(action).reshape(-1)[0], -1.0, 1.0))
        m = (a + 1.0) / 2.0                                # -> [0,1] sizing / keep-fraction
        if self._done:
            return self._obs(), 0.0, True, self._info()
        eq_pre = self._equity()
        # reward = interval return since the previous decision's post-action equity, minus dd brake.
        # In "relative" mode subtract the rung-0 RULE's interval return -> only BEATING the rule scores.
        ret = eq_pre / self._eq_mark - 1.0 if self._eq_mark > 0 else 0.0
        if self.reward_mode == "relative":
            ri = min(self.bar - self.start, len(self._rule_eq) - 1)
            rule_ret = self._rule_eq[ri] / self._rule_eq_mark - 1.0 if self._rule_eq_mark > 0 else 0.0
            ret = ret - rule_ret
        elif self.reward_mode == "residual":                   # per-decision: only DEVIATIONS earn/lose
            ret = self._residual_return()
        dd = (self.peak_eq - eq_pre) / self.peak_eq if self.peak_eq > 0 else 0.0
        reward = float(np.clip(ret, -RET_CLIP, RET_CLIP)) - self.dd_lambda * self._dd_penalty(dd)

        decision_time = int(self.returns.index[self.bar])
        self._trades = []
        etype, tok = self._pending
        if etype == "entry":
            self._do_entry(tok, m)
        else:
            self._do_exit(tok, m)
        eq_post = self._equity()
        traded = list(self._trades)
        weights = {t: self._pos_value(t) / max(eq_post, 1.0) for t in self.pos}
        self._eq_mark = eq_post
        if self.reward_mode == "relative":
            self._rule_eq_mark = float(self._rule_eq[min(self.bar - self.start, len(self._rule_eq) - 1)])
        elif self.reward_mode == "residual":                   # snapshot agent weights for the next interval
            self._agent_w_snap = {t: self._pos_value(t) / max(eq_post, 1.0) for t in self.pos}
            self._prev_bar = self.bar
        self._advance_to_event()
        info = self._info()
        info.update({"trades": traded, "trade_time": decision_time, "weights": weights})
        return self._obs(), float(reward), bool(self._done), info

    # -- event engine -------------------------------------------------------
    def _advance_to_event(self, first: bool = False):
        """Roll the bar cursor forward (positions drift) until the next decision point, draining any
        queued events on the current bar first. Sets self._pending / self._done."""
        self._done = False
        while True:
            if self._queue:
                self._pending = self._queue.pop(0)
                return
            self.bar += 0 if first else 1
            first = False
            eqb = self._equity()
            if self.bar >= self.end or eqb <= 1.0:
                self._done, self._pending = True, ("none", None)
                return
            self.peak_eq = max(self.peak_eq, eqb)
            if self.record_trace:
                self._eq_trace.append((int(self.returns.index[self.bar]), eqb))
            self._queue = self._scan_bar(self.bar)
            # re-arm entry edges where ignite has dropped (so a future ignite prompts again)
            for t in self.universe:
                if not self._ignite[self.bar, self.col_ix[t]]:
                    self.ignite_armed[t] = True

    def _scan_bar(self, bar: int) -> list:
        """Decision points on `bar`: exits on held positions first, then fresh fundable ignitions."""
        ev = []
        for t, p in self.pos.items():                            # update peaks, test exit triggers
            j = self.col_ix[t]
            p["peak_px"] = max(p["peak_px"], self._px[bar, j])
            stop_hit = self._px[bar, j] < p["ref_px"] * (1.0 - self.stop_k)
            ema_hit = self._cush[bar, j] < 0.0
            if stop_hit or ema_hit:
                ev.append(("exit", t))
        for t in self.universe:                                  # fresh ignitions on flat, cooled, reclaimed
            if t in self.pos or not self.ignite_armed[t]:
                continue
            j = self.col_ix[t]
            if not self._ignite[bar, j]:
                continue
            cooled = (bar - self.cool[t]) >= self.cooldown
            po = self.prior_origin[t]
            reclaimed = po is None or self._px[bar, j] > po
            if cooled and reclaimed:
                ev.append(("entry", t))
        return ev

    # -- decisions ----------------------------------------------------------
    def _do_entry(self, tok: str, a: float):
        self.ignite_armed[tok] = False                           # consume this ignition edge
        if a < SKIP_EPS:
            return                                               # agent declined the ignition
        eq = self._equity()
        want = a * self.max_entry_frac * eq
        if want > self.cash:                                     # rung-0 loser-funded rotation
            self._rotate_for(tok, want)
        size = min(want, self.cash)
        if size < 1.0:
            return
        j = self.col_ix[tok]
        c = amm_cost_usd(size, self.liquidity.get(tok, 0.0), self.lp_fee_bps, self.gas_usd)
        self.cash -= size + c
        self.pos[tok] = {"usd": size, "entry_bar": self.bar, "peak_px": self._px[self.bar, j],
                         "ref_px": self._px[self.bar, j], "origin": self._px[self.bar, j]}
        self._trades.append((tok, size, c))                      # +buy marker

    def _do_exit(self, tok: str, a: float):
        """a = fraction of the position to KEEP. a>=HOLD_EPS overrides the exit (re-arm the stop)."""
        p = self.pos.get(tok)
        if p is None:
            return
        if a >= HOLD_EPS:                                        # override: hold through, re-reference the stop
            p["ref_px"] = self._px[self.bar, self.col_ix[tok]]
            return
        keep = max(a, 0.0)
        val = self._pos_value(tok)
        sell_val = (1.0 - keep) * val
        j = self.col_ix[tok]
        c = amm_cost_usd(-sell_val, self.liquidity.get(tok, 0.0), self.lp_fee_bps, self.gas_usd)
        self.cash += sell_val - c
        if sell_val >= 1.0:
            self._trades.append((tok, -sell_val, c))             # -sell marker (incl. rotation sells)
        if keep <= 1e-6:                                         # full exit -> cooldown + dead-zone
            self.cool[tok] = self.bar
            self.prior_origin[tok] = p["origin"]
            del self.pos[tok]
        else:                                                    # partial trim, keep the rest running
            p["usd"] *= keep
            p["ref_px"] = self._px[self.bar, j]

    def _rotate_for(self, tok: str, want: float):
        """Free cash for `tok` by closing the WEAKEST holding (lowest cushion) — but only if it's
        weaker than the incoming candidate (rung-0's swap-weak-for-strong guard)."""
        while self.cash < want and self.pos:
            cur_cush = self._cush[self.bar, self.col_ix[tok]]
            weak = min(self.pos, key=lambda h: self._cush[self.bar, self.col_ix[h]])
            if self._cush[self.bar, self.col_ix[weak]] >= cur_cush:
                break
            self._do_exit(weak, 0.0)                             # full close of the laggard

    # -- valuation / obs / reward -------------------------------------------
    def _pos_value(self, tok: str) -> float:
        p = self.pos[tok]
        j = self.col_ix[tok]
        return p["usd"] * self._px[self.bar, j] / self._px[p["entry_bar"], j]

    def _equity(self) -> float:
        return self.cash + sum(self._pos_value(t) for t in self.pos)

    def _dd_penalty(self, dd: float) -> float:
        ramp = float(np.clip((dd - self.dd_soft) / (self.dd_gate - self.dd_soft), 0.0, 1.0))
        return ramp * ramp

    def _pick_universe(self, at: int) -> list:
        row = self._std[at - 1]
        order = np.argsort(np.nan_to_num(row, nan=-1.0))[::-1][:self.k]
        return [self.cols[j] for j in order]

    def _rule_equity_curve(self, start: int, end: int):
        """The rung-0 RULE's per-bar equity AND per-token weights over [start, end] on this episode's
        universe — a faithful mirror of `run_rung0` (exits on stop/EMA, ignition entries, loser-funded
        rotation) using the precomputed signals. The benchmark the agent must beat. Returns
        `(eq[n], w[n, k])` with `w[bar, i]` = the rule's weight in `universe[i]`. O(bars x k)."""
        px, cush, ig, cix = self._px, self._cush, self._ignite, self.col_ix
        sk, cd, ef = self.stop_k, self.cooldown, self.rule_entry_frac
        fee, gas, liq = self.lp_fee_bps, self.gas_usd, self.liquidity
        cash, pos = self.capital, {}                            # t -> dict(usd, entry_bar, peak_px, origin)
        cool = {t: -10 ** 9 for t in self.universe}
        prior = {t: None for t in self.universe}
        upos = {t: i for i, t in enumerate(self.universe)}     # universe-position index for the weight matrix
        eq = np.empty(end - start + 1)
        w = np.zeros((end - start + 1, len(self.universe)))

        def value(t, bar):
            p = pos[t]
            j = cix[t]
            return p["usd"] * px[bar, j] / px[p["entry_bar"], j]

        for kbar, bar in enumerate(range(start, end + 1)):
            equity = cash + sum(value(t, bar) for t in pos)
            if equity > 1.0:
                for t in list(pos):                            # 1) exits (stop off the peak, or below EMA)
                    j = cix[t]
                    p = pos[t]
                    p["peak_px"] = max(p["peak_px"], px[bar, j])
                    if px[bar, j] < p["peak_px"] * (1.0 - sk) or cush[bar, j] < 0.0:
                        v = value(t, bar)
                        cash += v - amm_cost_usd(-v, liq.get(t, 0.0), fee, gas)
                        cool[t], prior[t] = bar, p["origin"]
                        del pos[t]
                cands = [t for t in self.universe if ig[bar, cix[t]] and t not in pos
                         and (bar - cool[t]) >= cd
                         and (prior[t] is None or px[bar, cix[t]] > prior[t])]
                cands.sort(key=lambda t: cush[bar, cix[t]], reverse=True)
                for t in cands:                                # 2) fund strongest ignitions; rotate losers
                    if min(ef * equity, cash) < 1.0 and pos:
                        weak = min(pos, key=lambda h: cush[bar, cix[h]])
                        if cush[bar, cix[weak]] < cush[bar, cix[t]]:
                            v = value(weak, bar)
                            cash += v - amm_cost_usd(-v, liq.get(weak, 0.0), fee, gas)
                            cool[weak], prior[weak] = bar, pos[weak]["origin"]
                            del pos[weak]
                            equity = cash + sum(value(tt, bar) for tt in pos)
                    size = min(ef * equity, cash)
                    if size >= 1.0:
                        j = cix[t]
                        cash -= size + amm_cost_usd(size, liq.get(t, 0.0), fee, gas)
                        pos[t] = {"usd": size, "entry_bar": bar, "peak_px": px[bar, j], "origin": px[bar, j]}
            e = cash + sum(value(t, bar) for t in pos)
            eq[kbar] = e
            if e > 0:
                for t in pos:
                    w[kbar, upos[t]] = value(t, bar) / e
        return eq, w

    def _residual_return(self) -> float:
        """Per-decision (residual) signal: the agent's weight DEVIATIONS from the rule, dotted with
        token returns over the interval since the last decision. Shared positions (agent_w==rule_w)
        cancel, so only the agent's *active bets vs the rule* earn/lose - the gradient the
        whole-portfolio relative reward smeared into base-divergence noise."""
        pb, cb = self._prev_bar, self.bar
        if cb <= pb:
            return 0.0
        ro = pb - self.start
        s = 0.0
        for i, t in enumerate(self.universe):
            j = self.col_ix[t]
            p0 = self._px[pb, j]
            if p0 > 0:
                tok_ret = self._px[cb, j] / p0 - 1.0
                s += (self._agent_w_snap.get(t, 0.0) - self._rule_w[ro, i]) * tok_ret
        return float(s)

    def _obs(self) -> np.ndarray:
        eq = self._equity()
        eq = eq if eq > 0 else 1.0
        etype, tok = self._pending
        is_exit = 1.0 if etype == "exit" else 0.0
        cush = surge = unreal = held_frac = giveback = 0.0
        if tok is not None:
            j = self.col_ix[tok]
            cush = float(np.clip(self._cush[self.bar, j], -CUSHION_CLIP, CUSHION_CLIP))
            surge = float(self._surge[self.bar, j])
            if tok in self.pos:
                p = self.pos[tok]
                unreal = float(np.clip(self._px[self.bar, j] / self._px[p["entry_bar"], j] - 1.0,
                                       -RET_CLIP, RET_CLIP))
                held_frac = (self.bar - p["entry_bar"]) / max(self.episode_bars, 1)
                giveback = float(np.clip(self._px[self.bar, j] / p["peak_px"] - 1.0, -CUSHION_CLIP, 0.0))
        exposure = sum(self._pos_value(t) for t in self.pos) / eq
        dd = (self.peak_eq - eq) / self.peak_eq if self.peak_eq > 0 else 0.0
        ema = float(self.btc_ema.iloc[self.bar])
        btc_trend = float(self.btc.iloc[self.bar]) / ema - 1.0 if ema else 0.0
        rule_expo = 0.0                                        # the rung-0 rule's current invested fraction
        if self._rule_w is not None:
            rule_expo = float(self._rule_w[min(self.bar - self.start, len(self._rule_w) - 1)].sum())
        obs = [is_exit, cush, surge, unreal, held_frac, giveback,
               self.cash / eq, exposure, len(self.pos) / self.k, dd, btc_trend, rule_expo]
        return np.nan_to_num(np.array(obs, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)

    def _info(self) -> dict:
        etype, tok = self._pending
        return {"equity": self._equity(), "bar": self.bar,
                "time": int(self.returns.index[min(self.bar, self.n_bars - 1)]),
                "event": etype, "token": tok, "n_pos": len(self.pos), "cash": self.cash,
                "drawdown": (self.peak_eq - self._equity()) / self.peak_eq if self.peak_eq > 0 else 0.0}
