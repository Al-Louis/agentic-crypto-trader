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
from trader.agent.event_live import (LiveEventTrader, fills_from_records, new_fills,
                                     week_start_for)
from trader.risk import Policy, RiskState, TradeIntent, check_trade

# the env's USD "cash" leg maps to a stable for the allowlist / live routing (token<->USDT swaps)
CASH_LEG = "USDT"
DD_STOP_PCT = 30.0
# TWAK can't resolve microcap tickers ("Unknown token: UB on bsc") — live swaps key off the BEP-20
# CONTRACT via the assetId `c20000714_t<contract>` (c20000714 = the BSC coinId). BNB/USDT resolve by
# symbol, so only the universe tokens need this.
BSC_ASSET_PREFIX = "c20000714_t"


def _exec_summary(res: dict | None) -> dict:
    """Compact a live-executor result (`execute_trade`'s return) into fill-row fields. Returns
    `{}` when there was no live signing (paper / no executor) so the paper fill row stays
    BYTE-IDENTICAL to before this sleeve existed."""
    if res is None:
        return {}
    status = ("dry_run" if res.get("dry_run") else "skipped" if res.get("skipped")
              else "refused" if res.get("refused") else "error" if res.get("error")
              else res.get("status") or "unknown")
    out = {"exec_status": status, "tx_hash": res.get("tx_hash")}
    if res.get("skipped"):
        out["exec_skipped"] = res["skipped"]
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


def live_forward_policy(universe: list[str], bankroll_usd: float, *, asset_ids=None,
                        cash_leg: str = CASH_LEG, dd_stop_pct: float = DD_STOP_PCT,
                        max_slippage_pct: float = 1.0, max_entry_frac: float = 0.34) -> Policy:
    """The REAL-money guardrail policy for live signing — caps sized to the actual bankroll (NOT the
    $10k env book). A single scaled entry maxes at ~`max_entry_frac`*bankroll, so per-trade is set a
    little above that; daily allows several round-trips; the lifetime ceiling is a generous backstop
    (the 30% drawdown stop + allowlist are the binding limits). `asset_ids` adds the token CONTRACTS
    to the allowlist: BOTH the intent-phase and quote-phase checks allowlist on the INTENT's assets
    (the assetId the swap is keyed off — a contract can't route to a different token), while the
    realized USD + slippage are re-derived from the quote (`execute.py`). The `universe` SYMBOLS cover
    the cash legs. The env-parity check still uses the $10k-scale `forward_run_policy`."""
    from trader.agent.compliance import COMPLIANCE_TOKEN
    allow = {str(t).upper() for t in universe} | {cash_leg.upper(), COMPLIANCE_TOKEN.upper()}
    if asset_ids:
        allow |= {str(a).upper() for a in asset_ids if a}
    return Policy(allowlist=frozenset(allow),
                  per_trade_usd=round(bankroll_usd * max_entry_frac * 1.25, 4),
                  daily_usd=round(bankroll_usd * 4.0, 4),
                  max_slippage_pct=max_slippage_pct, drawdown_stop_pct=dd_stop_pct,
                  lifetime_usd_ceiling=round(bankroll_usd * 100.0, 4), chain="bsc")


def read_live_bankroll_usdt(*, chain: str = "bsc") -> float:
    """Read the wallet's current USDT balance (the bankroll anchor) for the launcher to pass as
    `live_bankroll_usd`. Read ONCE at startup, when the wallet is freshly funded and flat (all USDT);
    a mid-week restart holding token positions would under-read, so pass an explicit bankroll then."""
    from trader.execution import twak_cli
    b = twak_cli.wallet_balance(chain=chain)
    for t in b.get("tokens", []) or []:
        if str(t.get("symbol")).upper() == CASH_LEG.upper():
            return float(t.get("balance") or 0.0)
    return 0.0


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
                 execute_fn=None, execute_amount_fn=None, live_policy: Policy | None = None,
                 live_compliance_usd: float | None = None, live_dry_run: bool = False,
                 live_bankroll_usd: float | None = None, min_notional_usd: float = 0.0):
        self.trader = trader
        self.selection = selection
        # symbol -> TWAK assetId (contract), so live swaps resolve microcap tokens (BNB/USDT pass
        # through as symbols). Empty when the selection has no token_address (e.g. unit tests).
        self._asset_map = {str(s["symbol"]).upper(): BSC_ASSET_PREFIX + str(s["token_address"])
                           for s in (selection or []) if s.get("token_address")}
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
        self.execute_amount_fn = execute_amount_fn   # amount-in executor for the compliance qty-unwind
        self.live_policy = live_policy
        self._live_policy_explicit = live_policy is not None   # don't overwrite a caller-supplied one
        self.live_compliance_usd = (None if live_compliance_usd is None
                                    else float(live_compliance_usd))
        self.live_dry_run = bool(live_dry_run)
        # The decision env ALWAYS runs at `capital` ($10k, mandatory for the frozen model). A live
        # fill is the env's $10k-denominated `usd` re-based to the real bankroll by ONE fixed scale =
        # bankroll/capital (a 10% weight => $1k env => $1k*scale real). This is a FIXED scale on
        # purpose: the env compounds its equity WITHIN a cold week, and a fixed scale mirrors that
        # trajectory at 1:scale — so when the env book grows to $11k the real wallet has also grown to
        # ~bankroll*1.1 and a 10% weight is still 10% of the (mirrored) real equity. (Scaling by
        # current env_equity instead would strip the within-week equity-proportional sizing the model
        # learned — do NOT.) `live_bankroll_usd=None` => scale 1.0 (no re-basing; the dev-wallet tests
        # + the explicit live_compliance_usd override path rely on this). `is not None` + `capital>0`:
        # an explicit 0.0 bankroll (depleted wallet) => scale 0.0 => $0 intents the executor refuses
        # (fail-safe), NOT a silent fall-through to full $10k size.
        if self.capital <= 0:
            raise ValueError(f"capital must be > 0 (frozen-model book), got {self.capital}")
        self.live_bankroll_usd = (None if live_bankroll_usd is None else float(live_bankroll_usd))
        self._live_scale = (self.live_bankroll_usd / self.capital
                            if self.live_bankroll_usd is not None else 1.0)
        self.min_notional_usd = float(min_notional_usd)
        # Live signing with REAL money needs real-money caps: refuse to arm if neither the bankroll
        # (which auto-builds live_forward_policy in tick) NOR an explicit live_policy is given —
        # otherwise _sign_live would fall back to the $10k-scale env-parity policy on a real swap.
        if (self.mode == "live" and self.execute_fn is not None
                and self.live_bankroll_usd is None and self.live_policy is None):
            raise ValueError("live signing requires live_bankroll_usd (scales fills + sizes the "
                             "live caps) or an explicit live_policy — refusing to sign on the "
                             "$10k-scale env-parity policy")
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

    def _recorded_fill_keys(self, ws: int) -> set:
        """The (bar_ts, token, side) of every STRATEGY fill already recorded for the current week
        (>= ws) — the dedup set for the tick loop. Excludes the compliance sleeve (its bar_ts is the
        wall-clock tick, not a strategy bar). Read from disk so a restart dedups idempotently and a
        fill seeded as `missed` (to skip a pre-fix entry we won't chase) is never re-signed."""
        keys = set()
        for r in store.read_rows(self.agent_ledger_path):
            if r.get("kind") != "fill" or r.get("compliance"):
                continue
            bt = r.get("bar_ts")
            if not isinstance(bt, int) or int(bt) < ws:
                continue
            side = "buy" if r.get("to") == r.get("token") else "sell"
            keys.add((int(bt), str(r.get("token")), side))
        return keys

    def _onchain_held(self, token: str, ws: int) -> bool:
        """Did a strategy BUY of `token` actually LAND on-chain this week — a `confirmed` leg, net of
        confirmed sells? Gates a strategy SELL so the runner never signs an UNBACKED sell: the env
        book can be long a token the wallet never bought (a `missed` entry, or one the guardrails
        BLOCKED), and signing that exit would swap funds the wallet doesn't hold. Only
        `exec_status=="confirmed"` legs count — a missed/skipped/blocked buy does not."""
        net = 0
        for r in store.read_rows(self.agent_ledger_path):
            if (r.get("kind") != "fill" or r.get("compliance")
                    or str(r.get("token")) != token or r.get("exec_status") != "confirmed"):
                continue
            bt = r.get("bar_ts")
            if not isinstance(bt, int) or int(bt) < ws:
                continue
            net += 1 if r.get("to") == token else -1
        return net > 0

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
    def _asset_for(self, sym: str) -> str:
        """The TWAK swap leg for `sym`: the BEP-20 CONTRACT (assetId) for a universe token (TWAK
        can't resolve microcap tickers), else the symbol itself (BNB/USDT resolve by symbol)."""
        return self._asset_map.get(str(sym).upper(), sym)

    def _sign_live(self, frm: str, to: str, usd: float, slippage_pct: float,
                   *, prescaled: bool = False) -> dict | None:
        """Route a guardrail-PASSED fill through the real signing path (`execute_fn`, e.g.
        `trader.execution.execute.execute_trade`). The env's `usd` is re-based to the real bankroll
        by `self._live_scale` (= bankroll/$10k) UNLESS `prescaled=True` (the explicit
        `live_compliance_usd` override already gives a real figure). A scaled fill below
        `min_notional_usd` is SKIPPED on-chain (dust guard) — `{"skipped": ...}`. Returns None when
        live execution is not wired (paper / no executor) — then the fill is paper-only, exactly as
        before. NEVER raises: a signing exception is captured into an error dict so one bad swap
        cannot abort the hourly tick. Real spend is capped by `live_policy` inside the executor (its
        own two-phase check + risk ledger), independent of the env-parity check."""
        if self.execute_fn is None or self.mode != "live":
            return None
        real_usd = float(usd) if prescaled else float(usd) * self._live_scale
        if self.min_notional_usd > 0.0 and real_usd < self.min_notional_usd:
            # Dust guard: no on-chain mirror. This is CORRECT for the risk ledger — a skipped fill did
            # not trade, so it must NOT count toward real spent_today/lifetime (the executor's
            # state_from_ledger only sees actually-signed attempts). The env/agent book still records
            # the paper fill (tagged exec_status='skipped') for $10k-parity; the two ledgers are
            # separate scopes by design (real spend vs paper book), not a divergence.
            return {"skipped": "below_min_notional", "real_usd": real_usd, "tx_hash": None}
        intent = TradeIntent(from_asset=self._asset_for(frm), to_asset=self._asset_for(to),
                             usd=real_usd, chain="bsc", slippage_pct=slippage_pct)
        pol = self.live_policy if self.live_policy is not None else self.policy
        try:
            return self.execute_fn(intent, pol, dry_run=self.live_dry_run)
        except Exception as e:  # noqa: BLE001 — a swap failure must not crash the loop
            return {"error": f"{type(e).__name__}: {e}"[:300], "tx_hash": None}

    def _sign_live_amount(self, frm: str, to: str, amount: float, slippage_pct: float) -> dict | None:
        """Sign a swap of an EXACT token AMOUNT (the M4 unwind: the compliance SELL sells the precise
        BNB the BUY acquired, never a USD notional that could over/under-shoot the held balance).
        Routes through `execute_amount_fn` (e.g. `execute_swap_amount`) under `live_policy`. Returns
        None when amount-in live execution isn't wired (then the caller falls back to the USD path).
        NEVER raises."""
        if self.execute_amount_fn is None or self.mode != "live" or not (amount and amount > 0):
            return None
        pol = self.live_policy if self.live_policy is not None else self.policy
        try:
            return self.execute_amount_fn(self._asset_for(frm), self._asset_for(to), float(amount),
                                          pol, dry_run=self.live_dry_run)
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
        # live signing. SELL: unwind the EXACT BNB the BUY landed (amount-in via `_sign_live_amount`,
        # M4) so an intraday BNB move can't over/under-shoot the held balance or the wallet's gas
        # buffer; it falls back to the USD path only if the amount-in executor isn't wired or no qty
        # was recorded. BUY / no-qty: an explicit `live_compliance_usd` real figure (prescaled — dev
        # override) OR the env's `usd` re-based to the bankroll by `_live_scale`. Paper row keeps
        # `usd` for $10k-book sleeve-PnL parity.
        exec_res = None
        if verdict.allowed:
            sell_qty = float((bought or {}).get("bnb_qty") or 0.0) if reason == SELL_REASON else 0.0
            if self.live_compliance_usd is not None:                 # dev override: fixed USD both legs
                exec_res = self._sign_live(frm, to, self.live_compliance_usd,
                                           self.policy.max_slippage_pct, prescaled=True)
            elif reason == SELL_REASON:
                # Unwind the EXACT BNB the BUY landed (amount-in). Sign ONLY when a real qty was
                # captured (proof the BUY confirmed). NEVER sign a USD-sized SELL on a position that
                # may not exist (a refused/failed BUY) — that would sell the wallet's gas buffer or
                # revert. No captured qty in live => record a skip, do not sign. (NOTE: the live
                # wallet MUST hold a BNB gas buffer beyond the compliance position — the SELL's tx
                # fee is paid in BNB; selling the exact bought qty leaves the buffer for gas.)
                if sell_qty > 0.0:
                    exec_res = self._sign_live_amount(frm, to, sell_qty, self.policy.max_slippage_pct)
                if exec_res is None and self.mode == "live" and self.execute_amount_fn is not None:
                    exec_res = {"skipped": "no_bought_qty", "tx_hash": None}
            else:                                                    # BUY leg (env usd re-based by scale)
                exec_res = self._sign_live(frm, to, usd, self.policy.max_slippage_pct)
        if reason == SELL_REASON:
            self._compliance_pnl += usd - float(bought.get("usd_in") or 0.0) - cost
        row = {"kind": "fill", "mode": self.mode, "compliance": True, "from": frm, "to": to,
               "usd_in": usd, "usd_out": usd, "cost_usd": cost, "price": px, "price_index": px,
               "reason": reason, "trigger": reason, "token": COMPLIANCE_TOKEN, "obs": None,
               "bar_ts": int(now_ts), "guardrail_ok": bool(verdict.allowed),
               "guardrail_codes": verdict.codes}
        # persist the realized BNB qty the BUY acquired so the SELL leg can unwind it exactly (M4).
        # Gated on a CONFIRMED BUY with a real positive out_amount — a refused/pending/failed BUY or a
        # malformed swap output leaves NO bnb_qty, so the SELL skips signing (never over/under-shoots).
        if (reason == BUY_REASON and isinstance(exec_res, dict)
                and exec_res.get("status") == "confirmed"
                and str(exec_res.get("out_symbol")).upper() == COMPLIANCE_TOKEN
                and isinstance(exec_res.get("out_amount"), (int, float))
                and exec_res["out_amount"] > 0.0):
            row["bnb_qty"] = float(exec_res["out_amount"])
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

        # The week's fills are recorded by IDENTITY-DEDUP against the ledger (the loop below), so a
        # restart never re-records and the env's back-dated fills are never dropped. `new_week` only
        # drives the within-week drawdown-anchor reset.
        new_week = ws != self._week_start
        if new_week:
            self._week_start = ws

        eq_series = res["equity"]
        equity = float(eq_series.iloc[-1]) if len(eq_series) else self.capital
        # week high-water from the full replay curve (restart-safe), not in-memory tracking
        self._week_peak_eq = max(self.capital, float(eq_series.max()) if len(eq_series) else self.capital)
        dd_pct = ((self._week_peak_eq - equity) / self._week_peak_eq * 100.0
                  if self._week_peak_eq > 0 else 0.0)

        if self.policy is None:
            self.policy = forward_run_policy(res["universe"], self.capital)
        # Real-money caps for live signing: derive from the bankroll (NOT the $10k env book) once the
        # universe is known, unless an explicit live_policy was passed. Without this, _sign_live would
        # cap real swaps at the $10k-scale `self.policy` (the M1 hole). Rebuilt each tick so the
        # allowlist tracks the weekly universe; same max_slippage as the env-parity check.
        if self.live_bankroll_usd is not None and not self._live_policy_explicit:
            self.live_policy = live_forward_policy(
                res["universe"], self.live_bankroll_usd,
                asset_ids=[self._asset_map.get(str(t).upper()) for t in res["universe"]],
                max_slippage_pct=self.policy.max_slippage_pct)

        day_utc = datetime.fromtimestamp(now_ts, timezone.utc).date().isoformat()
        spent_today = 0.0
        recorded = blocked = 0
        # IDENTITY-DEDUP: record every env strategy fill in THIS week (bar >= ws) not already in the
        # ledger, keyed by (bar_ts, token, side). REPLACES the forward-only cursor that silently
        # dropped fills the env back-dates to a bar already behind the cursor (sbq-s1's lagged
        # week-open ignition). A late-surfaced fill is still caught; the key check makes double-signing
        # impossible — a recorded fill, INCLUDING one seeded as `missed` to skip a pre-fix entry we
        # won't chase, is never re-signed.
        recorded_keys = self._recorded_fill_keys(ws)
        for f in fills_from_records(res["records"]):
            ft = int(f.time)
            if ft < ws:                                       # cold-week fills only (>= the open)
                continue
            fkey = (ft, str(f.token), f.side)
            if fkey in recorded_keys:                         # already recorded -> never re-sign
                continue
            recorded_keys.add(fkey)
            usd = abs(float(f.usd))
            frm, to = (CASH_LEG, f.token) if f.side == "buy" else (f.token, CASH_LEG)
            intent = TradeIntent(from_asset=frm, to_asset=to, usd=usd, chain="bsc",
                                 slippage_pct=self.policy.max_slippage_pct)
            verdict = check_trade(self.policy, intent, self._risk_state(equity, spent_today))
            spent_today += usd
            exec_res = None
            if verdict.allowed:
                recorded += 1
                # LIVE sell-side position guard: never sign a SELL of a token the wallet didn't
                # actually buy on-chain. The env book can be long a position the wallet is NOT (a
                # missed or a guardrail-BLOCKED entry), and signing its exit would route a real swap
                # of funds the wallet doesn't hold. Record it skipped (no on-chain mirror), don't sign.
                # Paper is unaffected — _sign_live is a no-op there. Mirrors the compliance SELL guard.
                if (self.mode == "live" and f.side == "sell"
                        and not self._onchain_held(str(f.token), ws)):
                    exec_res = {"skipped": "no_onchain_position", "tx_hash": None}
                else:
                    exec_res = self._sign_live(frm, to, usd, intent.slippage_pct)  # live; None in paper
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

        if self._publisher is not None:
            try:
                self._publisher()
            except Exception as e:  # noqa: BLE001 — telemetry must never stop a tick
                print(f"publish warning (tick {now_ts}): {e}")

        return TickResult(now_ts=now_ts, week_start=ws, new_week=new_week, equity_usd=equity,
                          drawdown_pct=dd_pct, fills_recorded=recorded, fills_blocked=blocked,
                          trades_today=self._trades_today(day_utc), universe=list(res["universe"]),
                          compliance_trades=compliance_n)
