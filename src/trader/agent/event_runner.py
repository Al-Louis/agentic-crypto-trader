"""The paper-trading runner around the event-driven champion — one hourly tick.

Glues the three pieces together for the forward-run:
  * `live_data.update_live` — refresh the `data/` panels with the just-closed bar (skippable /
    injectable for tests),
  * `event_live.LiveEventTrader` — replay the current cold week and surface this hour's NEW fills,
  * the hard `trader.risk` guardrails + the `trader.agent.store` ledger + the `trading/` publisher.

The EventRungEnv is a self-contained simulator: it picks the vol-top-8, sizes/exits via its own
risk machinery, and prices fills on its internal index — so in PAPER mode **the env's equity IS
the paper book** (a fresh $10k cold session each calendar week, exactly as the model was validated
and the competition scores). This runner does NOT re-simulate; it records the env's fills, runs the
EXTERNAL guardrails over each one as the live-signing safety wrapper would (allowlist, per-trade /
daily caps, slippage, the ~30% drawdown stop), logs every decision, tracks the >=1-trade/day floor,
and publishes telemetry. In LIVE mode the same fills would route through TWAK and a guardrail
refusal would BLOCK the signing; in paper a blocked fill is still recorded but flagged.

Caps in `forward_run_policy` are PAPER placeholders sized to the env's $10k scale so faithful env
fills are not spuriously refused; the live dollar caps are quant's Phase-G call against the real
bankroll. The allowlist and the drawdown stop are real and binding.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from trader.agent import store
from trader.agent.event_live import LiveEventTrader, new_fills, week_start_for
from trader.risk import Policy, RiskState, TradeIntent, check_trade

# the env's USD "cash" leg maps to a stable for the allowlist / live routing (token<->USDT swaps)
CASH_LEG = "USDT"
DD_STOP_PCT = 30.0


def _exec_summary(res: dict | None) -> dict:
    """Compact a live-executor result (`execute_trade`'s return) into fill-row fields. Returns
    `{}` when there was no live signing (paper / no executor) so the paper fill row stays
    BYTE-IDENTICAL to before this sleeve existed."""
    if res is None:
        return {}
    status = ("dry_run" if res.get("dry_run") else "refused" if res.get("refused")
              else "error" if res.get("error") else res.get("status") or "unknown")
    out = {"exec_status": status, "tx_hash": res.get("tx_hash")}
    if res.get("refused"):
        out["exec_refused"] = res["refused"]
    if res.get("error"):
        out["exec_error"] = res["error"]
    return out


def forward_run_policy(universe: list[str], capital: float = 10_000.0, *,
                       cash_leg: str = CASH_LEG, dd_stop_pct: float = DD_STOP_PCT,
                       max_slippage_pct: float = 1.0) -> Policy:
    """Guardrail policy for the paper forward-run: allowlist = the traded universe + the stable
    cash leg; caps sized so faithful env fills pass (paper placeholders — Phase G sets the live
    dollar caps); the allowlist + 30% drawdown stop are the binding limits."""
    from trader.agent.compliance import COMPLIANCE_TOKEN  # the daily >=1-trade/day round-trip leg
    allow = frozenset({str(t).upper() for t in universe} | {cash_leg.upper(), COMPLIANCE_TOKEN.upper()})
    return Policy(allowlist=allow, per_trade_usd=capital, daily_usd=capital * 10.0,
                  max_slippage_pct=max_slippage_pct, drawdown_stop_pct=dd_stop_pct,
                  lifetime_usd_ceiling=capital * 1_000.0, chain="bsc")


@dataclass
class TickResult:
    """What one hourly tick did — returned for logging/telemetry and asserted in tests."""

    now_ts: int
    week_start: int
    new_week: bool
    equity_usd: float
    drawdown_pct: float
    fills_recorded: int
    fills_blocked: int
    trades_today: int
    universe: list[str] = field(default_factory=list)
    compliance_trades: int = 0          # >=1-trade/day overlay fills recorded this tick (0/1)


class EventRunner:
    """Drives the event champion in paper mode, one `tick(now_ts)` per closed 1h bar."""

    def __init__(self, trader: LiveEventTrader, *, selection: list[dict],
                 agent_ledger_path: Path, capital: float = 10_000.0,
                 policy: Policy | None = None, publisher=None, mode: str = "paper",
                 compliance_frac: float = 0.03, bnb_price_fn=None,
                 execute_fn=None, live_policy: Policy | None = None,
                 live_compliance_usd: float | None = None, live_dry_run: bool = False):
        self.trader = trader
        self.selection = selection
        self.agent_ledger_path = Path(agent_ledger_path)
        self.capital = float(capital)
        self.policy = policy                      # built lazily from the env universe if None
        self._publisher = publisher
        self.mode = mode
        self.compliance_frac = float(compliance_frac)   # >=1-trade/day overlay size (0 disables it)
        # --- live execution (the TWAK signing path) — OFF unless ALL of: mode=="live" AND an
        # `execute_fn` is wired. Default (execute_fn=None) keeps paper byte-identical, so the EC2
        # paper service is untouched. `live_policy` (the tight real-money caps) governs the actual
        # signing; the env-parity guardrail check still uses self.policy. `live_compliance_usd`
        # overrides the compliance round-trip notional for live (the env equity is the $10k book,
        # NOT the real bankroll, so frac*equity would over-size the real swap). `live_dry_run`
        # routes through execute_trade's quote-only pre-flight (no signing).
        self.execute_fn = execute_fn
        self.live_policy = live_policy
        self.live_compliance_usd = (None if live_compliance_usd is None
                                    else float(live_compliance_usd))
        self.live_dry_run = bool(live_dry_run)
        self._bnb_price_fn = bnb_price_fn          # injectable BNB USD price(now_ts) for tests
        self._bnb_anchor = None                    # cached BNB close series (False = unavailable)
        self._compliance_pnl = 0.0                 # cumulative realized PnL of the compliance sleeve
        self._env_kwargs: dict | None = None
        self._close_panel = None                  # per-token real USD closes (set each tick)
        self._week_start: int | None = None       # the cold week currently being traded
        self._acted_ts: int | None = None         # newest bar whose fills we've already recorded
        self._week_peak_eq = capital              # within-week equity high-water (drawdown anchor)

    # -- helpers --------------------------------------------------------------
    def _ensure_env_kwargs(self, returns) -> dict:
        if self._env_kwargs is None:
            self._env_kwargs = self.trader.env_kwargs(returns)
        return self._env_kwargs

    def _risk_state(self, equity: float, spent_today: float) -> RiskState:
        """Guardrail state for THIS fill: env equity + within-week high-water (the drawdown
        anchor) + the day's recorded spend. Always available in paper (we hold the numbers)."""
        return RiskState(spent_today_usd=spent_today, spent_lifetime_usd=spent_today,
                         equity_usd=equity, high_water_usd=self._week_peak_eq, available=True)

    def _trades_today(self, day_utc: str) -> int:
        rows = store.read_rows(self.agent_ledger_path)
        return sum(1 for r in rows if r.get("kind") == "fill"
                   and str(r.get("ts") or "").startswith(day_utc))

    def _ledger_cursor(self, ws: int) -> int:
        """The fill-diff resume point: the latest fill bar_ts already recorded for the current
        week (>= ws) in the ledger, else ws-1. Read from disk so a restart resumes idempotently
        (the store's crash-recovery rule) instead of re-recording the week and duplicating fills."""
        bars = [int(r["bar_ts"]) for r in store.read_rows(self.agent_ledger_path)
                if r.get("kind") == "fill" and isinstance(r.get("bar_ts"), int)
                and int(r["bar_ts"]) >= ws]
        return max(bars) if bars else ws - 1

    def _real_price(self, token: str, bar_ts: int) -> float | None:
        """The token's real USD close at the fill's bar (for display) — None if unavailable, in
        which case the caller keeps the env's internal index price."""
        cp = self._close_panel
        if cp is None or token not in cp.columns:
            return None
        try:
            v = float(cp.at[bar_ts, token])
        except (KeyError, ValueError, TypeError):
            return None
        return v if v == v and v > 0 else None      # reject NaN / non-positive

    # -- live signing (the TWAK execution path; off in paper) -----------------
    def _sign_live(self, frm: str, to: str, usd: float, slippage_pct: float) -> dict | None:
        """Route a guardrail-PASSED fill through the real signing path (`execute_fn`, e.g.
        `trader.execution.execute.execute_trade`). Returns the executor's result dict, or None
        when live execution is not wired (paper / no executor) — then the fill is paper-only,
        exactly as before. NEVER raises: a signing exception is captured into an error dict so
        one bad swap cannot abort the hourly tick. Real spend is capped by `live_policy` inside
        the executor (its own two-phase check + risk ledger), independent of the env-parity check."""
        if self.execute_fn is None or self.mode != "live":
            return None
        intent = TradeIntent(from_asset=frm, to_asset=to, usd=float(usd), chain="bsc",
                             slippage_pct=slippage_pct)
        pol = self.live_policy if self.live_policy is not None else self.policy
        try:
            return self.execute_fn(intent, pol, dry_run=self.live_dry_run)
        except Exception as e:  # noqa: BLE001 — a swap failure must not crash the loop
            return {"error": f"{type(e).__name__}: {e}"[:300], "tx_hash": None}

    # -- daily >=1-trade/day compliance overlay -------------------------------
    def _bnb_price(self, now_ts: int) -> float | None:
        """BNB USD close at/just-before `now_ts`, from the BNB anchor parquet (same source the harness
        keeps fresh) — or an injected fn for tests. None if unavailable (then compliance is skipped,
        never fabricated)."""
        if self._bnb_price_fn is not None:
            return self._bnb_price_fn(int(now_ts))
        if self._bnb_anchor is None:
            try:
                import os  # noqa: PLC0415
                import pandas as pd  # noqa: PLC0415
                a = pd.read_parquet(os.path.join("data", "anchor", "BNB_USDT", "1h.parquet"))
                a = a.set_index("timestamp").sort_index()
                if a.index.max() > 1e12:                 # ms -> s (match load_data's BTC anchor)
                    a.index = (a.index // 1000).astype("int64")
                self._bnb_anchor = a["close"]
            except Exception:  # noqa: BLE001 — no BNB anchor on this box -> skip compliance, don't crash
                self._bnb_anchor = False
        if self._bnb_anchor is False:
            return None
        s = self._bnb_anchor
        prior = s.index[s.index <= int(now_ts)]
        if len(prior) == 0:
            return None
        v = float(s.loc[prior[-1]])
        return v if v == v and v > 0 else None

    def _run_compliance(self, now_ts: int, equity: float, spent_today: float) -> int:
        """The Rule-1 guardrail: BUY 3% BNB at 01:00 UTC, SELL it back at 23:00 UTC, each day. Records
        the leg as a `fill` (so it counts toward the >=1-trade/day floor), routed through the same risk
        guardrails. Idempotent off the ledger (a restart / re-tick never double-trades). Kept off the env
        book — a separate sleeve whose realized PnL is tracked in `self._compliance_pnl`. Returns 0/1."""
        from trader.agent.compliance import (BUY_REASON, CASH_LEG, COMPLIANCE_TOKEN, SELL_REASON,
                                             compliance_action, compliance_cost)
        if self.compliance_frac <= 0.0:
            return 0
        action = compliance_action(now_ts)
        if action is None:
            return 0
        # group "today" by the BAR time (bar_ts == now_ts for our fills), NOT the wall-clock `ts` the
        # store stamps — so the once-per-day idempotency holds under simulated-time replay AND live.
        day_start = (int(now_ts) // 86400) * 86400
        today = [r for r in store.read_rows(self.agent_ledger_path)
                 if r.get("compliance") and isinstance(r.get("bar_ts"), int)
                 and day_start <= int(r["bar_ts"]) < day_start + 86400]
        bought = next((r for r in today if r.get("reason") == BUY_REASON), None)
        sold = any(r.get("reason") == SELL_REASON for r in today)
        px = self._bnb_price(now_ts)
        if px is None:
            return 0                                     # no BNB price -> skip, never fabricate a fill

        if action == "buy" and bought is None:
            usd = self.compliance_frac * equity
            frm, to, reason = CASH_LEG, COMPLIANCE_TOKEN, BUY_REASON
        elif action == "sell" and bought is not None and not sold:
            usd_buy, px_buy = float(bought.get("usd_in") or 0.0), float(bought.get("price") or px)
            usd = (usd_buy / px_buy) * px if px_buy > 0 else usd_buy   # BNB-leg value at the sell price
            frm, to, reason = COMPLIANCE_TOKEN, CASH_LEG, SELL_REASON
        else:
            return 0                                     # already done today (idempotent) / nothing to sell

        cost = compliance_cost(usd)
        intent = TradeIntent(from_asset=frm, to_asset=to, usd=usd, chain="bsc",
                             slippage_pct=self.policy.max_slippage_pct)
        verdict = check_trade(self.policy, intent, self._risk_state(equity, spent_today))
        # live signing: the REAL swap uses live_compliance_usd (the env equity is the $10k book,
        # not the real bankroll, so frac*equity would over-size the on-chain swap); falls back to
        # `usd` if unset. The paper row keeps `usd` for $10k-book sleeve-PnL parity.
        exec_res = None
        if verdict.allowed:
            real_usd = self.live_compliance_usd if self.live_compliance_usd is not None else usd
            exec_res = self._sign_live(frm, to, real_usd, self.policy.max_slippage_pct)
        if reason == SELL_REASON:
            self._compliance_pnl += usd - float(bought.get("usd_in") or 0.0) - cost
        row = {"kind": "fill", "mode": self.mode, "compliance": True, "from": frm, "to": to,
               "usd_in": usd, "usd_out": usd, "cost_usd": cost, "price": px, "price_index": px,
               "reason": reason, "trigger": reason, "token": COMPLIANCE_TOKEN, "obs": None,
               "bar_ts": int(now_ts), "guardrail_ok": bool(verdict.allowed),
               "guardrail_codes": verdict.codes}
        row.update(_exec_summary(exec_res))
        store.append(row, self.agent_ledger_path, now=None)
        if not verdict.allowed:
            store.append({"kind": "refusal", "mode": self.mode, "compliance": True,
                          "intent": {"from": frm, "to": to, "usd": usd}, "refusals": verdict.codes},
                         self.agent_ledger_path, now=None)
        return 1

    # -- one hourly tick ------------------------------------------------------
    def tick(self, now_ts: int, *, panels=None, predict_fn=None,
             refresh_data: bool = True) -> TickResult:
        """Refresh data (unless injected), replay the current cold week, record this hour's new
        fills through the guardrails, mark equity/heartbeat, and publish. `panels` (returns, btc,
        liq, vol) and `predict_fn` are injectable for offline tests; `refresh_data=False` skips
        the network update."""
        now_ts = int(now_ts)
        if refresh_data and panels is None:
            from trader.agent.live_data import update_live  # noqa: PLC0415 (network path)
            update_live(self.selection, now_ts)
        if panels is None:
            from train_rl import build_volume_panel, load_data  # noqa: PLC0415
            returns, btc, _anchor, liq = load_data()
            vol = build_volume_panel(list(returns.columns), returns.index)
        else:
            returns, btc, liq, vol = panels

        # real USD closes to translate the env's internal return-index fill prices to market
        # prices (the env's _px starts at 1.0 at the window warmup start, not USD).
        self._close_panel = None
        if self.selection:
            from trader.agent.live_data import build_close_panel  # noqa: PLC0415
            self._close_panel = build_close_panel(self.selection, returns.index)

        env_kwargs = self._ensure_env_kwargs(returns)
        res = self.trader.evaluate_week(returns, btc, liq, vol, now_ts, env_kwargs,
                                        predict_fn=predict_fn)
        ws = res["week_start"]

        # week rollover OR a process restart (in-memory cursor lost): resume the fill-diff cursor
        # from the LEDGER — the latest fill bar already recorded for this week — so a restart never
        # re-records the week's fills (the duplicate-on-restart bug). A genuine new week has no
        # fills >= ws yet, so the cursor is ws-1 and the week records from its open.
        new_week = ws != self._week_start
        if new_week:
            self._week_start = ws
            self._acted_ts = self._ledger_cursor(ws)
        after = self._acted_ts if self._acted_ts is not None else self._ledger_cursor(ws)

        eq_series = res["equity"]
        equity = float(eq_series.iloc[-1]) if len(eq_series) else self.capital
        # week high-water from the full replay curve (restart-safe), not in-memory tracking
        self._week_peak_eq = max(self.capital, float(eq_series.max()) if len(eq_series) else self.capital)
        dd_pct = ((self._week_peak_eq - equity) / self._week_peak_eq * 100.0
                  if self._week_peak_eq > 0 else 0.0)

        if self.policy is None:
            self.policy = forward_run_policy(res["universe"], self.capital)

        day_utc = datetime.fromtimestamp(now_ts, timezone.utc).date().isoformat()
        spent_today = 0.0
        recorded = blocked = 0
        for f in new_fills(res["records"], after):
            usd = abs(float(f.usd))
            frm, to = (CASH_LEG, f.token) if f.side == "buy" else (f.token, CASH_LEG)
            intent = TradeIntent(from_asset=frm, to_asset=to, usd=usd, chain="bsc",
                                 slippage_pct=self.policy.max_slippage_pct)
            verdict = check_trade(self.policy, intent, self._risk_state(equity, spent_today))
            spent_today += usd
            exec_res = None
            if verdict.allowed:
                recorded += 1
                exec_res = self._sign_live(frm, to, usd, intent.slippage_pct)  # live only; None in paper
            else:
                blocked += 1
            # paper: record the env's fill either way, tagged with the guardrail verdict; in live
            # an !allowed verdict BLOCKS signing (no _sign_live call -> never mirrored on-chain).
            # `price` is the REAL USD market close at the bar; `price_index` is the env's internal
            # return-index the PnL/equity is computed in. `_exec_summary` adds tx_hash/exec_status
            # in live (else {} -> the paper row is byte-identical to before this sleeve).
            real_px = self._real_price(f.token, int(f.time))
            row = {"kind": "fill", "mode": self.mode, "from": frm, "to": to,
                   "usd_in": usd, "usd_out": usd, "cost_usd": float(f.fee),
                   "price": real_px if real_px is not None else float(f.price),
                   "price_index": float(f.price),
                   "reason": f.reason, "token": f.token,
                   "trigger": f.reason, "obs": f.obs, "bar_ts": int(f.time),
                   "guardrail_ok": bool(verdict.allowed),
                   "guardrail_codes": verdict.codes}
            row.update(_exec_summary(exec_res))
            store.append(row, self.agent_ledger_path, now=None)
            if not verdict.allowed:
                store.append({"kind": "refusal", "mode": self.mode,
                              "intent": {"from": frm, "to": to, "usd": usd},
                              "refusals": verdict.codes}, self.agent_ledger_path, now=None)

        # >=1-trade/day compliance overlay (Rule-1): a small BNB<->USDT round-trip, recorded as a fill
        # so it counts toward the daily floor. Off the env book (separate sleeve); idempotent off ledger.
        compliance_n = self._run_compliance(now_ts, equity, spent_today)

        store.append({"kind": "equity", "mode": self.mode, "tick": now_ts,
                      "equity_usd": equity, "peak_usd": self._week_peak_eq,
                      "drawdown_pct": dd_pct, "week_start": ws,
                      "compliance_pnl_usd": self._compliance_pnl,
                      "below_dust": equity <= 1.0}, self.agent_ledger_path, now=None)
        store.append({"kind": "heartbeat", "mode": self.mode, "tick": now_ts,
                      "equity_usd": equity, "week_start": ws}, self.agent_ledger_path, now=None)

        # advance the cursor by the latest BAR timestamp processed, NOT wall-clock now_ts: fills
        # are keyed by bar-time, which lags wall-time by ~the bar duration, so a wall-clock cursor
        # silently drops every fill after the first tick (the missed-exit bug).
        self._acted_ts = int(res["win_index"][-1]) if res.get("win_index") else now_ts
        if self._publisher is not None:
            try:
                self._publisher()
            except Exception as e:  # noqa: BLE001 — telemetry must never stop a tick
                print(f"publish warning (tick {now_ts}): {e}")

        return TickResult(now_ts=now_ts, week_start=ws, new_week=new_week, equity_usd=equity,
                          drawdown_pct=dd_pct, fills_recorded=recorded, fills_blocked=blocked,
                          trades_today=self._trades_today(day_utc), universe=list(res["universe"]),
                          compliance_trades=compliance_n)
