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
from trader.train.event_reward import entry_forward_reward

OBS_DIM = 13   # ... rung-0 exposure (0 in absolute mode) + universe breadth (alts' own regime)
SURGE_CLIP, CUSHION_CLIP, RET_CLIP = 10.0, 1.0, 5.0
SKIP_EPS, HOLD_EPS = 0.05, 0.95   # action < SKIP_EPS on entry = skip; >= HOLD_EPS on exit = full hold

# rung-1b `rule_default` action tables (discrete, 4 levels): index 0 EXECUTES rung-0's decision —
# entry at the rule's sizing, exit cut in full — so deviating from the rule is a learned act, never
# the path of least resistance (the g2b forensics: the policy skipped every strong ignition and
# partially vetoed every exit for free). Entries are multiples of `rule_entry_frac`, still clipped
# by the risk-parity cap; exits are keep-fractions (1.0 = hold-through/override).
RULE_DEFAULT_ENTRY_MULT = (1.0, 0.5, 0.0, 2.0)   # idx -> multiple of the RULE's sizing (0 = skip)
RULE_DEFAULT_EXIT_KEEP = (0.0, 1.0 / 3.0, 2.0 / 3.0, 1.0)   # idx -> fraction kept (0 = the rule's cut)
RULE_DEFAULT_TP_KEEP = (1.0, 2.0 / 3.0, 1.0 / 3.0, 0.0)   # profit prompts: idx0 = let it run (the rule)

# `basket_default` (the long-default OVERLAY, 2026-06-14): the env starts fully long the risk-parity
# basket (= Buy&Hold) and events TILT names off it, so the default action HOLDS. The exit/profit
# tables INVERT rule-default — idx 0 keeps the basket weight (hold-through), deviations trim — so
# doing nothing ≈ B&H, the floor the event-only skeleton lacked (it trailed B&H by a +13% bull-gap on
# cold weekly sessions, [[Experiment Log]] §2026-06-14). Entries (re-buying a name trimmed to flat)
# reuse the rule-default entry table. With `rule_prior` on idx 0, the untrained policy ≈ Buy&Hold.
BASKET_EXIT_KEEP = (1.0, 2.0 / 3.0, 1.0 / 3.0, 0.0)   # idx0 = HOLD the basket (default); idx3 = full cut
BASKET_TP_KEEP = (1.0, 2.0 / 3.0, 1.0 / 3.0, 0.0)     # idx0 = let it run (default); higher = trim into strength


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
                 reward_mode: str = "absolute", r4_beta: float = 0.0, res_gamma: float = 0.0,
                 fwd_horizon: int = 24, ungate: bool = False,
                 action_mode: str = "continuous", n_action_levels: int = 4,
                 universe_mode: str = "voltopk", vol_target: float = 0.0, cap_floor: float = 0.02,
                 harvest_obs: bool = False,
                 rule_default: bool = False, basket_default: bool = False,
                 exit_commit: int = 0, dust_usd: float = 0.0,
                 tp_rungs: tuple | list = (), loss_floor: float = 0.0,
                 det_blacklist: int = 0, det_surge: float = 8.0, det_drop: float = -0.15,
                 low_frac: pd.DataFrame | None = None, intrabar_floor: bool = False,
                 high_frac: pd.DataFrame | None = None, wick_reject: float = 0.0,
                 cycle_obs: bool = False, universe_lookback: int = 0, no_btc_obs: bool = False,
                 record_trace: bool = False, seed: int | None = None):
        self.returns = returns.sort_index()
        self.btc = btc_close.reindex(self.returns.index).ffill().bfill()
        self.btc_ema = self.btc.ewm(span=ema_span, adjust=False).mean()
        self.liquidity = liquidity
        self.k, self.warmup, self.episode_bars = k, warmup, episode_bars
        self.capital = float(capital)
        self.lp_fee_bps, self.gas_usd = lp_fee_bps, gas_usd
        self.stop_k, self.cooldown, self.max_entry_frac = stop_k, cooldown, max_entry_frac
        self.dd_soft, self.dd_gate, self.dd_lambda = dd_soft, dd_gate, dd_lambda
        self.reward_mode = reward_mode                      # absolute|relative|residual|residual_ranked
        self.r4_beta = r4_beta                              # residual: foregone-opportunity penalty weight
        self.res_gamma = res_gamma                          # residual_ranked: quadratic deviation-budget weight
        self.fwd_horizon = int(fwd_horizon)                 # entry_forward: forward-return window (bars)
        self.ungate = bool(ungate)                          # exp5 selector: fire on every in-universe ignition
                                                            # (drop rung-0's cooled&reclaimed gate -> ~960 vs 39)
        self.action_mode = action_mode                      # "continuous" (Box[-1,1]) | "discrete" (Discrete)
        self.n_action_levels = int(n_action_levels)         # discrete: # of size/keep levels spanning [0,1]
        self.universe_mode = universe_mode                  # voltopk | broad (vol-stratified) | lowvol (calm)
        self.vol_target = float(vol_target)                 # >0: per-token weight cap proportional to vol_target/vol
        self.cap_floor = float(cap_floor)                   # risk-parity: min per-token weight cap (keep upside)
        self.harvest_obs = bool(harvest_obs)                # lever-2: append r24/r3d/r7d momentum slots (13->16)
        self.rule_default = bool(rule_default)              # rung-1b: action idx 0 EXECUTES rung-0's decision
        self.exit_commit = int(exit_commit)                 # bars an exit decision (non-cut) commits for
        self.dust_usd = float(dust_usd)                     # partial keeps below this force a full close
        self.intrabar_floor = bool(intrabar_floor)          # the floor is a RESTING STOP: filled
        #   intra-bar where the price PATH crossed it (bar low from `low_frac`), not 60 minutes
        #   later at the next close — closes the Q hole (a −53% bar blowing through a −20% floor).
        self.wick_reject = float(wick_reject)               # kill ignitions on EXTREME-rejection
        #   trigger bars (close < (1-wick_reject)*high): the dump is mid-flight at the fill price.
        #   Probe (`probe_wick.py`): rare (~3 events/5mo) and catastrophic every observed time;
        #   the mild version (0.10) is REFUTED — bars closing AT their highs are the worst bucket.
        if self.intrabar_floor and (low_frac is None or loss_floor <= 0.0):
            raise ValueError("intrabar_floor needs low_frac data AND loss_floor > 0")
        if self.wick_reject > 0.0 and high_frac is None:
            raise ValueError("wick_reject needs high_frac data")
        self._lowf = (low_frac.reindex(self.returns.index).fillna(1.0).clip(0.01, 1.0)
                      .reindex(columns=self.returns.columns).fillna(1.0).to_numpy()
                      if low_frac is not None else None)
        self._highf = (high_frac.reindex(self.returns.index).fillna(1.0).clip(0.0, 1.0)
                       .reindex(columns=self.returns.columns).fillna(1.0).to_numpy()
                       if high_frac is not None else None)
        self.loss_floor = float(loss_floor)                 # disaster floor: a position below
        #   entry*(1-loss_floor) CANNOT be overridden — forced full cut, and the floor punctures the
        #   exit-commit window. Closes the one unbounded loss path (the Q ride: override down -45%).
        #   Winners (above entry) keep the full override; only deep losers lose the right to be ridden.
        self.tp_rungs = tuple(sorted(float(x) for x in tp_rungs))   # profit-take prompts at these
        #   unrealized-gain levels (e.g. 0.25,0.5,1,2): the only way to SELL INTO STRENGTH — exit
        #   prompts fire on weakness only. Each rung prompts once per position; default = let it run.
        if self.rule_default and (action_mode != "discrete" or self.n_action_levels != 4):
            raise ValueError("rule_default needs action_mode='discrete' with n_action_levels=4")
        self.basket_default = bool(basket_default)          # long-default OVERLAY: start fully long the
        #   risk-parity basket; events tilt off it; the default action HOLDS (exit/profit tables invert).
        if self.basket_default and not self.rule_default:
            raise ValueError("basket_default builds on rule_default's discrete 4-level head; set rule_default=True")
        self.rule_entry_frac = 0.20                         # the rung-0 RULE's fixed sizing (the benchmark)
        self.record_trace = bool(record_trace)              # eval-only: per-bar equity curve + markers
        self.no_btc_obs = bool(no_btc_obs)                  # neutralize the btc_trend obs slot to a
        #   constant 0: the universe was selected for LOW BTC correlation, so a BTC-anchored regime
        #   signal is near-noise (the alts decouple). Slot kept (obs_dim unchanged) so the policy
        #   just learns to ignore a dead input; the regime signal that earns its keep is breadth.
        self.cycle_obs = bool(cycle_obs)                    # SPENT-MOVE knowledge (probe_knowledge:
        #   an ignition whose token's PRIOR ignition already paid >10% returns −6..−7% fwd-24h vs
        #   −1..−2% fresh, on BOTH train and val) — 2 slots: ret-since / bars-since prior ignition.
        self.obs_dim = (OBS_DIM + (3 if self.harvest_obs else 0)   # +r24/r3d/r7d when harvest_obs
                        + (2 if self.cycle_obs else 0))
        self.action_dim = 1
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
        if self.wick_reject > 0.0:                          # extreme-rejection wick guard (user
            ignite = ignite.to_numpy() & (self._highf >= (1.0 - self.wick_reject))   # idea, probed)
        self.det_blacklist = int(det_blacklist)
        if self.det_blacklist > 0:
            # DETONATION blacklist (the Q pattern, probe-gated by scripts/probe_detonation.py):
            # a massive surge WHILE price collapses marks the token untradeable — its later
            # ignitions are poison (fwd48 −8%/−24% train/val, win 8–21%) until ~4wk out, where
            # they revert to baseline. Zero the ignite signal for `det_blacklist` bars after.
            det = ((surge >= det_surge) & (rising <= det_drop)).to_numpy()
            ig_np = (ignite if isinstance(ignite, np.ndarray) else ignite.to_numpy()).copy()
            for j in range(det.shape[1]):
                for b in np.where(det[:, j])[0]:
                    ig_np[b: b + self.det_blacklist, j] = False
            ignite = ig_np
        self._px = px.to_numpy()
        self._cush = cushion.to_numpy()
        self._surge = surge.clip(0.0, SURGE_CLIP).to_numpy()
        self._ignite = ignite if isinstance(ignite, np.ndarray) else ignite.to_numpy()
        # causal universe-selection volatility: trailing `universe_lookback` bars (0 = the historical
        # default, warmup=168h/7d). The lookback is an UNTESTED axis (user simulator design,
        # 2026-06-12): 24=1d, 168=1wk, 720=1mo, 2160=3mo, 4320=6mo (data permitting).
        ulb = int(universe_lookback) if universe_lookback else warmup
        self.universe_lookback = ulb
        self._std = self.returns.rolling(ulb, min_periods=8).std().to_numpy()
        if self.cycle_obs:                                  # last-ignition bar per (bar, token):
            ig_arr = ignite if isinstance(ignite, np.ndarray) else ignite.to_numpy()
            last = np.full(ig_arr.shape, -1, dtype=np.int32)   # causal cumulative pass — the prior
            run = np.full(ig_arr.shape[1], -1, dtype=np.int32)  # ignition STRICTLY BEFORE this bar
            for b in range(ig_arr.shape[0]):
                last[b] = run
                run = np.where(ig_arr[b], b, run)
            self._last_ig = last
        # entry_forward: the TYPICAL-ignition forward return (the demean null). A single scalar over
        # this panel's ignitions -> the preflight computes it identically, so the reward landscapes match.
        self._mu_base = self._ignition_base_rate() if reward_mode == "entry_forward" else 0.0

        self._min_start = warmup
        self._max_start = self.n_bars - episode_bars - 1
        if self._max_start < self._min_start:           # == is fine (a single full-window episode, eval)
            raise ValueError("series too short for episode_bars/warmup")
        self.rng = np.random.default_rng(seed)

    # -- curriculum ---------------------------------------------------------
    def set_episode_bars(self, n: int) -> None:
        """Horizon-curriculum hook: change the episode length BETWEEN episodes (takes effect on the
        next `reset()`, never mid-episode). SHRINKING is always safe — a shorter episode fits any
        window the longer one did, so `_max_start` only widens. The env must be CONSTRUCTED at the
        LARGEST horizon the schedule uses (its `__init__` `_max_start` is the tightest bound);
        growing past that would index past the panel. See `trader.train.curriculum`."""
        n = int(n)
        new_max = self.n_bars - n - 1
        if new_max < self._min_start:
            raise ValueError(f"episode_bars {n} too large for series (n_bars={self.n_bars}, "
                             f"warmup={self.warmup})")
        self.episode_bars = n
        self._max_start = new_max

    # -- lifecycle ----------------------------------------------------------
    def reset(self, *, start: int | None = None, seed: int | None = None) -> np.ndarray:
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.start = int(start) if start is not None else int(
            self.rng.integers(self._min_start, max(self._max_start, self._min_start + 1)))
        self.end = self.start + self.episode_bars
        self.bar = self.start
        self.universe = self._pick_universe(self.start)          # causal vol-ranked, fixed for episode
        self._uni_ix = np.array([self.col_ix[t] for t in self.universe])  # for the breadth regime feature
        self._tok_cap = self._token_caps(self.start)             # per-token weight cap (risk-parity if vol_target>0)
        self.cash = self.capital
        self.peak_eq = self.capital
        self.pos = {}                                            # tok -> dict(usd, entry_bar, peak_px, origin)
        self._realized = {}                                      # tok -> cumulative realized cash flow (PnL ledger)
        self.cool = {t: -10 ** 9 for t in self.universe}        # last exit bar (cooldown)
        self._exit_decided = {}                                 # tok -> bar of last committed non-cut exit
        self.prior_origin = {t: None for t in self.universe}    # dead-zone: runup origin of last cycle
        self.ignite_armed = {t: True for t in self.universe}    # entry edge: only prompt on a fresh ignite
        self._queue = []                                        # pending (event_type, tok) on the current bar
        self._eq_mark = self.capital
        self._trades = []                                       # trades made in the current step (markers)
        self._eq_trace = [(int(self.returns.index[self.start]), self.capital)]  # per-bar equity (eval)
        if self.basket_default:                                 # long-default OVERLAY: start fully long the
            self._buy_basket()                                  # risk-parity basket (= B&H); events tilt off it
            self._eq_mark = self._equity()                      # post-buy equity (entry cost paid) is the mark
            self._eq_trace = [(int(self.returns.index[self.start]), self._eq_mark)]
        # relative/residual reward: precompute the BENCHMARK per-bar equity AND per-token weights over the
        # episode - the rung-0 RULE normally, or (basket_default) the held-basket B&H. "relative" = beat the
        # benchmark's portfolio return; "residual" = beat it per-decision (only weight DEVIATIONS earn/lose).
        if self.reward_mode in ("relative", "residual", "residual_ranked"):
            self._rule_eq, self._rule_w = (self._basket_equity_curve(self.start, self.end)
                                           if self.basket_default
                                           else self._rule_equity_curve(self.start, self.end))
        else:
            self._rule_eq = self._rule_w = None
        self._rule_eq_mark = float(self._rule_eq[0]) if self._rule_eq is not None else self.capital
        self._prev_bar = self.start                             # residual: interval + agent-weight snapshot
        self._agent_w_snap = {}
        self._pending_entries = []                              # entry_forward: (entry_bar, dev, tok) awaiting outcome
        self._matured_reward = 0.0                             # entry_forward: matured entry rewards since last step
        self._advance_to_event(first=True)                      # roll to the first decision point
        return self._obs()

    def step(self, action) -> tuple:
        # action in [-1,1]: neutral a=0 -> m=0.5 lands in the INTERIOR (the policy trades from init),
        # so a Gaussian head can't dead-gradient collapse to "never trade" at a [0,1] skip boundary
        if self.action_mode == "discrete":                 # Discrete level idx -> m in {0, .., 1}
            idx = int(round(float(np.asarray(action).reshape(-1)[0])))
            idx = int(np.clip(idx, 0, self.n_action_levels - 1))
            if self.rule_default:                          # idx 0 = EXECUTE the default; deviations earned
                etype0 = self._pending[0]
                if etype0 == "entry":
                    m = RULE_DEFAULT_ENTRY_MULT[idx]       # re-buy a trimmed-flat name at rule sizing
                elif etype0 == "profit":                   # basket_default inverts: idx0 = let it run / hold
                    m = (BASKET_TP_KEEP if self.basket_default else RULE_DEFAULT_TP_KEEP)[idx]
                else:                                      # exit: idx0 = HOLD the basket (basket_default)
                    m = (BASKET_EXIT_KEEP if self.basket_default else RULE_DEFAULT_EXIT_KEEP)[idx]
            else:
                m = idx / max(self.n_action_levels - 1, 1)  # 4 levels -> {0, 1/3, 2/3, 1}
        else:
            a = float(np.clip(np.asarray(action).reshape(-1)[0], -1.0, 1.0))
            m = (a + 1.0) / 2.0                            # -> [0,1] sizing / keep-fraction
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
        elif self.reward_mode in ("residual", "residual_ranked"):  # per-decision: only DEVIATIONS earn/lose
            ret = self._residual_return()
        elif self.reward_mode == "entry_forward":              # delayed entry-forward residual (semi-MDP)
            ret = self._matured_reward                          # entry rewards matured since the last step
            self._matured_reward = 0.0
        dd = (self.peak_eq - eq_pre) / self.peak_eq if self.peak_eq > 0 else 0.0
        reward = float(np.clip(ret, -RET_CLIP, RET_CLIP)) - self.dd_lambda * self._dd_penalty(dd)

        decision_time = int(self.returns.index[self.bar])
        self._trades = []
        etype, tok = self._pending
        if etype == "entry":
            self._do_entry(tok, m)
        elif etype == "profit":
            self._do_profit(tok, m)
        else:
            self._do_exit(tok, m)
        eq_post = self._equity()
        traded = list(self._trades)
        weights = {t: self._pos_value(t) / max(eq_post, 1.0) for t in self.pos}
        self._eq_mark = eq_post
        if self.reward_mode == "relative":
            self._rule_eq_mark = float(self._rule_eq[min(self.bar - self.start, len(self._rule_eq) - 1)])
        elif self.reward_mode in ("residual", "residual_ranked"):  # snapshot weights for the next interval
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
            if self.intrabar_floor and self.pos:            # the RESTING-STOP floor: fill where the
                for t in list(self.pos):                    # bar's LOW crossed entry*(1-floor) —
                    p = self.pos[t]                         # not at the next close (the Q hole)
                    j = self.col_ix[t]
                    floor_px = self._px[p["entry_bar"], j] * (1.0 - self.loss_floor)
                    if self._px[self.bar, j] * self._lowf[self.bar, j] <= floor_px:
                        self._stop_fill(t, floor_px)
            eqb = self._equity()
            if self.record_trace:                              # record EVERY advanced bar incl. the FINAL
                self._eq_trace.append((int(self.returns.index[self.bar]), eqb))   # one — the done-check
            if self.bar >= self.end or eqb <= 1.0:             # used to return FIRST, leaving eq one bar
                self._done, self._pending = True, ("none", None)   # stale vs the open-position marks
                return
            self.peak_eq = max(self.peak_eq, eqb)
            if self.reward_mode == "entry_forward":            # mature entries whose forward window elapsed
                self._mature_entries(self.bar)
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
            p["peak_px"] = max(p["peak_px"], self._px[bar, j])   # peak tracks highs even while committed
            floored = (self.loss_floor > 0.0
                       and self._px[bar, j] < self._px[p["entry_bar"], j] * (1.0 - self.loss_floor))
            if (not floored and self.exit_commit > 0
                    and (bar - self._exit_decided.get(t, -10 ** 9)) < self.exit_commit):
                continue                                         # committed exit decision: not re-prompted
            #                                  (the disaster floor PUNCTURES the commit window)
            if floored:
                ev.append(("exit", t))
                continue
            stop_hit = self._px[bar, j] < p["peak_px"] * (1.0 - self.stop_k)   # TRAILING stop off the
            #                              peak (matches canonical rung0.py:121 + _rule_equity_curve:400)
            ema_hit = self._cush[bar, j] < 0.0
            if stop_hit or ema_hit:
                ev.append(("exit", t))
        if self.tp_rungs:                                        # profit prompts: a position crossed its
            for t, p in self.pos.items():                        # next unrealized-gain rung (sell-into-
                if p.get("tp_i", 0) >= len(self.tp_rungs):       # strength is now expressible)
                    continue
                j = self.col_ix[t]
                unreal = self._px[bar, j] / self._px[p["entry_bar"], j] - 1.0
                if unreal >= self.tp_rungs[p["tp_i"]]:
                    ev.append(("profit", t))
        for t in self.universe:                                  # fresh ignitions on flat, cooled, reclaimed
            if t in self.pos or not self.ignite_armed[t]:
                continue
            j = self.col_ix[t]
            if not self._ignite[bar, j]:
                continue
            cooled = (bar - self.cool[t]) >= self.cooldown
            po = self.prior_origin[t]
            reclaimed = po is None or self._px[bar, j] > po
            if self.ungate or (cooled and reclaimed):          # exp5: let the agent gate, not the rule
                ev.append(("entry", t))
        return ev

    # -- decisions ----------------------------------------------------------
    def _do_entry(self, tok: str, a: float):
        self.ignite_armed[tok] = False                           # consume this ignition edge
        eq = self._equity()
        size = 0.0
        if a >= SKIP_EPS:
            if self.rule_default:                            # a = multiple of the RULE's sizing,
                frac = min(a * self.rule_entry_frac,         # still clipped by the risk-parity cap
                           self._tok_cap.get(tok, self.max_entry_frac))
                want = frac * eq
            else:
                want = a * self._tok_cap.get(tok, self.max_entry_frac) * eq
            if want > self.cash:                                 # rung-0 loser-funded rotation
                self._rotate_for(tok, want)
            size = min(want, self.cash)
            if size >= 1.0:
                j = self.col_ix[tok]
                c = amm_cost_usd(size, self.liquidity.get(tok, 0.0), self.lp_fee_bps, self.gas_usd)
                self.cash -= size + c
                self._realized[tok] = self._realized.get(tok, 0.0) - (size + c)   # ledger: cash out
                self.pos[tok] = {"usd": size, "entry_bar": self.bar, "peak_px": self._px[self.bar, j],
                                 "origin": self._px[self.bar, j], "tp_i": 0}
                self._trades.append((tok, size, c, int(self.returns.index[self.bar]),
                                     self._px[self.bar, j]))      # +buy @ the bar's price index
            else:
                size = 0.0                                       # couldn't fund -> a skip (dev = -0.20)
        if self.reward_mode == "entry_forward":                  # record for delayed forward crediting
            self._pending_entries.append((self.bar, size / max(eq, 1.0) - self.rule_entry_frac, tok))

    def _do_exit(self, tok: str, a: float):
        """a = fraction of the position to KEEP. a>=HOLD_EPS overrides the exit (re-arm the stop)."""
        p = self.pos.get(tok)
        if p is None:
            return
        if (self.loss_floor > 0.0 and self._px[self.bar, self.col_ix[tok]]
                < self._px[p["entry_bar"], self.col_ix[tok]] * (1.0 - self.loss_floor)):
            self._sell_down(tok, 0.0)                            # disaster floor: deep losers cannot be
            return                                               # overridden or trimmed — forced cut
        if a >= HOLD_EPS:                                        # override: hold through
            if self.rule_default:                                # rung-1b: COMMIT the hold (no re-prompt
                self._exit_decided[tok] = self.bar               # for exit_commit bars) and do NOT
            else:                                                # re-anchor — the old re-anchor let
                p["peak_px"] = self._px[self.bar, self.col_ix[tok]]  # repeated overrides ratchet the
            return                                               # stop down a crash (g2b-s2, 63.7% DD)
        self._sell_down(tok, max(a, 0.0))

    def _do_profit(self, tok: str, a: float):
        """Take-profit prompt (the position crossed its next unrealized-gain rung): `a` = fraction
        to KEEP. Default (rule_default idx 0 -> keep 1.0) = rung-0's let-winners-run; selling here
        is the 'sell into strength' deviation that exit prompts (weakness-triggered) cannot express."""
        p = self.pos.get(tok)
        if p is None:
            return
        j = self.col_ix[tok]
        unreal = self._px[self.bar, j] / self._px[p["entry_bar"], j] - 1.0
        while p["tp_i"] < len(self.tp_rungs) and unreal >= self.tp_rungs[p["tp_i"]]:
            p["tp_i"] += 1                                       # consume every rung crossed
        if a >= HOLD_EPS:
            return                                               # let it run (the rule's behavior)
        self._sell_down(tok, max(a, 0.0))

    def _stop_fill(self, tok: str, fill_px: float):
        """Force-fill a full close at `fill_px` (the resting-stop price the intra-bar path
        crossed) instead of the bar close — AMM cost applies; cooldown + dead-zone arm as on any
        full exit. The position's value at the stop = usd * fill_px/entry_px = usd*(1-floor)."""
        p = self.pos[tok]
        j = self.col_ix[tok]
        val = p["usd"] * fill_px / self._px[p["entry_bar"], j]
        c = amm_cost_usd(-val, self.liquidity.get(tok, 0.0), self.lp_fee_bps, self.gas_usd)
        self.cash += val - c
        self._realized[tok] = self._realized.get(tok, 0.0) + (val - c)   # ledger: cash in (even at $0)
        if val > 0.0:                                        # record EVERY close (even sub-$1 crash
            self._trades.append((tok, -val, c, int(self.returns.index[self.bar]),   # closes) so the
                                 fill_px))                          # marker stream nets to flat -sell @ stop price index
        self.cool[tok] = self.bar
        self.prior_origin[tok] = p["origin"]
        del self.pos[tok]

    def _sell_down(self, tok: str, keep: float):
        """Sell a position down to `keep` of its value (shared by exit cuts/trims and profit-takes):
        dust floor, AMM cost, markers, cooldown + dead-zone on a full close, commit on a trim."""
        p = self.pos[tok]
        val = self._pos_value(tok)
        if self.dust_usd > 0.0 and keep * val < self.dust_usd:  # dust floor: a remainder below $dust
            keep = 0.0                                          # is a full close, not a gas-bleeding tail
        sell_val = (1.0 - keep) * val
        j = self.col_ix[tok]
        c = amm_cost_usd(-sell_val, self.liquidity.get(tok, 0.0), self.lp_fee_bps, self.gas_usd)
        self.cash += sell_val - c
        self._realized[tok] = self._realized.get(tok, 0.0) + (sell_val - c)   # ledger: cash in
        if sell_val > 0.0:                                   # record EVERY sell (even sub-$1) so the
            self._trades.append((tok, -sell_val, c, int(self.returns.index[self.bar]),   # marker stream
                                 self._px[self.bar, j]))            # nets to flat -sell @ bar index (incl. rotation)
        if keep <= 1e-6:                                         # full exit -> cooldown + dead-zone
            self.cool[tok] = self.bar
            self.prior_origin[tok] = p["origin"]
            del self.pos[tok]
        else:                                                    # partial trim, keep the rest running
            p["usd"] *= keep
            if self.rule_default:                                # rung-1b: a trim is ONE committed decision
                self._exit_decided[tok] = self.bar               # (no per-bar liquidation drip), stop NOT
            else:                                                # re-anchored (no ratchet)
                p["peak_px"] = self._px[self.bar, j]             # legacy: re-anchor on the trim

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

    def token_pnls(self) -> dict:
        """Per-token EXACT PnL = realized cash flow + current open-position value. Sums to
        (equity - capital); the ledger the dashboard export reconciles each asset's positions to."""
        toks = set(self._realized) | set(self.pos)
        return {t: self._realized.get(t, 0.0) + (self._pos_value(t) if t in self.pos else 0.0)
                for t in toks}

    def _equity(self) -> float:
        return self.cash + sum(self._pos_value(t) for t in self.pos)

    def _dd_penalty(self, dd: float) -> float:
        ramp = float(np.clip((dd - self.dd_soft) / (self.dd_gate - self.dd_soft), 0.0, 1.0))
        return ramp * ramp

    def _pick_universe(self, at: int) -> list:
        """Causal vol-ranked universe. `universe_mode` is the curriculum's VOLATILITY axis:
        `voltopk` = the k most volatile (current — maximum chaos); `lowvol` = the k calmest
        (S0: learn basics on tractable dynamics); `broad` = a vol-stratified spread across the
        distribution (calm + volatile together, for risk-parity allocation)."""
        row = np.nan_to_num(self._std[at - 1], nan=-1.0)
        order = np.argsort(row)[::-1]                        # all tokens, high -> low vol
        if self.universe_mode == "lowvol":                   # calmest k (curriculum S0: learn basics)
            valid = [j for j in order if row[j] > 0] or list(order)
            pick = valid[::-1][:self.k]
        elif self.universe_mode == "broad":                  # vol-stratified spread (calm + volatile)
            valid = [j for j in order if row[j] > 0] or list(order)
            if len(valid) > self.k:
                idx = np.linspace(0, len(valid) - 1, self.k).round().astype(int)
                pick = [valid[i] for i in idx]
            else:
                pick = valid[:self.k]
        else:                                                # voltopk (default): EXACT prior behavior
            pick = list(order[:self.k])
        return [self.cols[j] for j in pick]

    def _token_caps(self, at: int) -> dict:
        """Per-token max weight cap. With `vol_target>0`: RISK-PARITY — cap proportional to
        `vol_target/trailing_vol`, clipped to `[cap_floor, max_entry_frac]`, so high-vol tokens get
        small caps (bounded drawdown contribution) while staying present for convex upside; calm
        tokens anchor at the ceiling. With `vol_target<=0`: flat `max_entry_frac` for every token."""
        if self.vol_target <= 0.0:
            return {t: self.max_entry_frac for t in self.universe}
        caps = {}
        for t in self.universe:
            v = self._std[at - 1, self.col_ix[t]]
            v = float(v) if v and v > 0 else self.max_entry_frac
            caps[t] = float(np.clip(self.vol_target / v, self.cap_floor, self.max_entry_frac))
        return caps

    def _ignition_base_rate(self) -> float:
        """Mean forward-`fwd_horizon` return over every ignition in the panel — the typical-ignition
        return the entry-forward reward demeans against. Shared verbatim with the preflight."""
        H, rs = self.fwd_horizon, []
        for j in range(len(self.cols)):
            for bar in range(self.warmup, self.n_bars - H):
                if self._ignite[bar, j] and self._px[bar, j] > 0:
                    rs.append(self._px[bar + H, j] / self._px[bar, j] - 1.0)
        return float(np.mean(rs)) if rs else 0.0

    def _mature_entries(self, bar: int):
        """Credit entries whose forward horizon has just elapsed (semi-MDP delayed reward) via the
        shared `entry_forward_reward`. Entries whose window runs past the episode end stay pending and
        are dropped at done — no look-ahead, no truncated-window bias."""
        keep = []
        for eb, dev, tok in self._pending_entries:
            mb = eb + self.fwd_horizon
            if mb > bar:                                       # outcome not realized yet
                keep.append((eb, dev, tok))
            elif mb < self.end:                                # realized within the episode -> credit
                j = self.col_ix[tok]
                p0 = self._px[eb, j]
                if p0 > 0:
                    fwd = self._px[mb, j] / p0 - 1.0
                    self._matured_reward += entry_forward_reward(dev, fwd, self._mu_base, self.res_gamma)
            # else (mb >= end): dropped — window not realized within the episode
        self._pending_entries = keep

    def _basket_weights(self) -> dict:
        """Risk-parity basket weights (cap-normalized, fully invested) — the long-default target."""
        wsum = sum(self._tok_cap[t] for t in self.universe) or 1.0
        return {t: self._tok_cap[t] / wsum for t in self.universe}

    def _buy_basket(self):
        """Long-default OVERLAY (`basket_default`): buy the full risk-parity basket at reset so the env
        starts like Buy&Hold. Cost is baked into the position basis (value = alloc - cost, matching
        `buy_and_hold_return` and `_basket_equity_curve`), cash drops to ~0. Held names then only get
        exit/profit (trim) prompts; doing nothing holds the basket — the floor the skeleton lacked."""
        w = self._basket_weights()
        for t in self.universe:
            j = self.col_ix[t]
            alloc = w[t] * self.capital
            c = amm_cost_usd(alloc, self.liquidity.get(t, 0.0), self.lp_fee_bps, self.gas_usd)
            usd = alloc - c                                      # cost baked into the basis (B&H convention)
            if usd < 1.0:
                continue
            self.cash -= alloc
            self._realized[t] = self._realized.get(t, 0.0) - alloc
            self.pos[t] = {"usd": usd, "entry_bar": self.start, "peak_px": self._px[self.start, j],
                           "origin": self._px[self.start, j], "tp_i": 0}
            self.ignite_armed[t] = False                         # already held -> no immediate entry re-prompt
            self._trades.append((t, usd, c, int(self.returns.index[self.start]), self._px[self.start, j]))

    def _basket_equity_curve(self, start: int, end: int):
        """Buy&Hold of the risk-parity basket over [start, end] — the benchmark under `basket_default`
        (a do-nothing agent matches it, so only correct tilts score). Buy at `start` (cost baked in),
        hold, drift. Returns `(eq[n], w[n, k])` with `w[bar, i]` = the basket's weight in `universe[i]`."""
        w0 = self._basket_weights()
        invested = {}
        for t in self.universe:
            alloc = w0[t] * self.capital
            c = amm_cost_usd(alloc, self.liquidity.get(t, 0.0), self.lp_fee_bps, self.gas_usd)
            invested[t] = alloc - c
        px, cix = self._px, self.col_ix
        eq = np.empty(end - start + 1)
        w = np.zeros((end - start + 1, len(self.universe)))
        for kbar, bar in enumerate(range(start, end + 1)):
            vals = []
            for t in self.universe:
                j = cix[t]
                p0 = px[start, j]
                vals.append(invested[t] * px[bar, j] / p0 if p0 > 0 else invested[t])
            e = sum(vals)
            eq[kbar] = e
            if e > 0:
                for i, v in enumerate(vals):
                    w[kbar, i] = v / e
        return eq, w

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
        whole-portfolio relative reward smeared into base-divergence noise.

        Two shapes:
        - `residual` (+ optional R4 `r4_beta>0`): `Σ dev·ret`; R4 charges `beta`x surrendered upside
          when under-sizing a winner. A reward LINEAR in dev → the per-decision gradient is a constant
          direction → the policy corners (all-big or all-small by the lambdas). The corner is a
          functional-form limit, not a tuning one.
        - `residual_ranked` (the fix): **demean** the return by the interval's cross-sectional mean and
          add a **quadratic deviation budget**: `Σ dev·(ret − ret_bar) − res_gamma·Σ dev²`. Centering
          removes the constant drift gradient (E[ret−ret_bar]=0 for a skill-less agent), so the only way
          to score is to make `dev` track the obs-PREDICTABLE part of the return (cush) → conditional
          sizing. The quadratic budget makes the optimum interior (`dev* ∝ (ret−ret_bar)/2γ` =
          rank-correct), so neither corner is optimal. Maximized by conditional sizing and only by it."""
        pb, cb = self._prev_bar, self.bar
        if cb <= pb:
            return 0.0
        ro = pb - self.start
        rets, devs = [], []
        for i, t in enumerate(self.universe):
            j = self.col_ix[t]
            p0 = self._px[pb, j]
            rets.append(self._px[cb, j] / p0 - 1.0 if p0 > 0 else 0.0)
            devs.append(self._agent_w_snap.get(t, 0.0) - self._rule_w[ro, i])
        if self.reward_mode == "residual_ranked":
            rbar = sum(rets) / len(rets) if rets else 0.0      # interval cross-sectional mean (causal)
            return float(sum(d * (r - rbar) - self.res_gamma * d * d for d, r in zip(devs, rets)))
        s = 0.0                                                # plain residual (+ optional R4)
        for d, r in zip(devs, rets):
            s += d * r
            if self.r4_beta > 0.0 and d < 0.0 and r > 0.0:
                s += self.r4_beta * d * r
        return float(s)

    def _obs(self) -> np.ndarray:
        eq = self._equity()
        eq = eq if eq > 0 else 1.0
        etype, tok = self._pending
        is_exit = 1.0 if etype == "exit" else 0.5 if etype == "profit" else 0.0
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
        btc_trend = 0.0 if self.no_btc_obs else (float(self.btc.iloc[self.bar]) / ema - 1.0 if ema else 0.0)
        rule_expo = 0.0                                        # the rung-0 rule's current invested fraction
        if self._rule_w is not None:
            rule_expo = float(self._rule_w[min(self.bar - self.start, len(self._rule_w) - 1)].sum())
        # universe breadth: fraction of the TRADED basket above its EMA right now — the alts' OWN regime
        # signal (a pump lifts most names; a crash breaks most). Decoupled from btc_trend, which misleads
        # because the alts trade idiosyncratically from BTC. Low breadth = de-risk; high = harvest.
        breadth = float(np.mean(self._cush[self.bar, self._uni_ix] > 0.0)) if len(self._uni_ix) else 0.0
        obs = [is_exit, cush, surge, unreal, held_frac, giveback,
               self.cash / eq, exposure, len(self.pos) / self.k, dd, btc_trend, rule_expo, breadth]
        if self.harvest_obs:                                   # lever-2 HARVEST: the event token's short-horizon
            harv = [0.0, 0.0, 0.0]                             # momentum (r24/r3d/r7d), causal (past px only),
            if tok is not None:                                # tanh-squashed so fat alt tails land in [-1,1]
                j = self.col_ix[tok]
                for i, n in enumerate((24, 72, 168)):
                    p0 = self._px[self.bar - n, j] if self.bar - n >= 0 else 0.0
                    r = (self._px[self.bar, j] / p0 - 1.0) if p0 > 0 else 0.0
                    harv[i] = float(np.tanh(3.0 * np.clip(r, -RET_CLIP, RET_CLIP)))
            obs += harv
        if self.cycle_obs:                                 # SPENT-MOVE: the event token's prior-
            ret_since, bars_since = 0.0, 1.0               # ignition payoff + staleness (fresh = 0/1)
            if tok is not None:
                j = self.col_ix[tok]
                pb = int(self._last_ig[self.bar, j])
                if pb >= 0 and self._px[pb, j] > 0:
                    ret_since = float(np.tanh(2.0 * (self._px[self.bar, j] / self._px[pb, j] - 1.0)))
                    bars_since = float(min(self.bar - pb, 672) / 672.0)
            obs += [ret_since, bars_since]
        return np.nan_to_num(np.array(obs, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)

    def _info(self) -> dict:
        etype, tok = self._pending
        return {"equity": self._equity(), "bar": self.bar,
                "time": int(self.returns.index[min(self.bar, self.n_bars - 1)]),
                "event": etype, "token": tok, "n_pos": len(self.pos), "cash": self.cash,
                "drawdown": (self.peak_eq - self._equity()) / self.peak_eq if self.peak_eq > 0 else 0.0}
