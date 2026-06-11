"""The autonomous execution loop — read -> decide -> execute -> confirm -> monitor.

A long-running, event-driven loop ([[Project Overview]], [[Real-time Monitoring]]):

  read     pull live prices for the eligible universe via a `PriceFeed`
  decide   hand the observation to the swappable `DecisionCore` -> trade intents
  execute  for each intent: re-check the HARD guardrails (`risk.check_trade`), then
             paper  -> simulate the fill against live price (`agent.paper.fill`)
             live   -> route through `execute_trade` and NOTHING else
  confirm  apply the fill to the portfolio, persist the row to the agent ledger
  monitor  mark equity + drawdown (scoring-mirror), write the heartbeat, re-arm

Two modes, one code path. **Paper is the default and a first-class mode** — real reads,
real decide(), realistic AMM-cost fills, persisted PnL. **Live is unreachable without an
explicit `mode="live"` config flag**, and even then every trade goes through the single
guarded signing path (`trader.execution.execute_trade`). Default-paper is enforced in the
`LoopConfig` constructor *and* re-asserted at the live branch — a misconfigured loop fails
closed to paper, never opens to live by accident.

Crash-safe: on construction the loop re-derives its portfolio from `agent.store` (the
[[Remote Capabilities]] idempotency rule). State is never held only in memory; every fill,
equity mark and heartbeat is on disk before the next tick.

Scoring-mirror ([[Real-time Monitoring]]): equity is marked each tick; drawdown =
(peak - equity)/peak feeds the same figure the `risk/` drawdown stop reads. An hour opening
at <= $1 equity scores 0% per the competition mechanic — surfaced, not silently averaged.
"""

from __future__ import annotations

import signal
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from trader.agent import paper, store
from trader.agent.decide import DecisionCore, HoldCore, Intent, Observation
from trader.agent.feed import PriceFeed
from trader.agent.store import CASH
from trader.risk import SPIKE_POLICY, Policy, TradeIntent, check_trade, ledger

DUST_EQUITY_USD = 1.0  # an hour opening <= this scores 0% (competition mechanic)


@dataclass(frozen=True)
class LoopConfig:
    """Loop wiring. `mode` defaults to 'paper'; 'live' is the ONLY value that reaches the
    signing path and must be set explicitly. Any other value fails closed to paper."""

    universe: list[str]
    mode: str = "paper"                         # "paper" (default) | "live"
    tick_seconds: float = 3600.0                # hourly cadence (scoring is hourly)
    max_ticks: int | None = None                # None = run until signalled (production)
    policy: Policy = SPIKE_POLICY
    agent_ledger_path: Path = store.AGENT_LEDGER_PATH
    risk_ledger_path: Path = ledger.LEDGER_PATH
    paper_liquidity_usd: float = paper.DEFAULT_LIQUIDITY_USD

    @property
    def is_live(self) -> bool:
        """True ONLY for the exact string 'live'. Anything else is paper (fail closed)."""
        return self.mode == "live"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class Loop:
    """The autonomous loop. Inject `feed` + `core` (+ optional `execute_fn`/`sleep`/`now_iso`
    for tests). Construct -> `run()` blocks until `max_ticks` or a stop signal."""

    def __init__(self, config: LoopConfig, feed: PriceFeed, core: DecisionCore | None = None,
                 *, execute_fn=None, sleep=time.sleep, now_iso=_now_iso):
        self.cfg = config
        self.feed = feed
        self.core = core or HoldCore()
        self._sleep = sleep
        self._now_iso = now_iso
        # Live signing is injectable for tests, but defaults to the ONE real path.
        if execute_fn is None:
            from trader.execution.execute import execute_trade
            execute_fn = execute_trade
        self._execute_trade = execute_fn
        self._stop = False
        # crash-safe: re-derive the portfolio from disk on construction
        self.state = store.derive_state(self.cfg.agent_ledger_path)

    # --- lifecycle ---------------------------------------------------------

    def request_stop(self, *_a) -> None:
        """Clean shutdown signal — the current tick finishes, then run() returns."""
        self._stop = True

    def install_signal_handlers(self) -> None:
        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                signal.signal(sig, self.request_stop)
            except (ValueError, OSError):  # not on the main thread (e.g. under a test runner)
                pass

    def run(self) -> int:
        """Run ticks until `max_ticks` reached or stop requested. Returns ticks executed."""
        executed = 0
        while not self._stop:
            if self.cfg.max_ticks is not None and executed >= self.cfg.max_ticks:
                break
            self.tick()
            executed += 1
            if self._stop or (self.cfg.max_ticks is not None and executed >= self.cfg.max_ticks):
                break
            self._sleep(self.cfg.tick_seconds)
        return executed

    # --- one cycle ---------------------------------------------------------

    def tick(self) -> dict:
        """One read -> decide -> execute -> confirm -> monitor cycle. Returns a tick summary."""
        ts = self._now_iso()
        tick_idx = self.state.tick

        # 1) READ — fail closed: an empty/failed read skips the decide step (no trade on no data)
        prices = self.feed.prices(self.cfg.universe)

        # 2) DECIDE — value the portfolio, build the observation, ask the core
        equity = self._equity(prices)
        obs = Observation(
            ts=ts, prices=prices, equity_usd=equity,
            positions={k: v for k, v in self.state.positions.items() if k != CASH},
            cash_usd=self.state.positions.get(CASH, 0.0),
        )
        intents = self.core.decide(obs) if prices else []

        # 3+4) EXECUTE + CONFIRM each intent (guardrails first, then mode-routed fill)
        results = []
        for intent in intents:
            results.append(self._handle_intent(intent, prices, ts))

        # 5) MONITOR — re-value (fills changed the book), mark equity/drawdown, heartbeat
        equity = self._equity(prices)
        dd_pct, peak = self._mark_equity(equity, ts, tick_idx)
        self._heartbeat(ts, tick_idx, equity)
        # advance the in-memory tick pointer (disk rows carry it for crash-recovery)
        self.state = store.derive_state(self.cfg.agent_ledger_path)
        return {"ts": ts, "tick": tick_idx, "equity_usd": equity, "drawdown_pct": dd_pct,
                "peak_usd": peak, "n_intents": len(intents), "results": results,
                "below_dust": equity <= DUST_EQUITY_USD, "mode": self.cfg.mode}

    # --- internals ---------------------------------------------------------

    def _equity(self, prices: dict[str, float]) -> float:
        """USD value of the book at this tick's prices. CASH is $1; a position with no live
        price this tick is valued at 0 *for this mark only* (conservative — it never inflates
        equity on a missing read, matching the scoring-honest posture)."""
        total = self.state.positions.get(CASH, 0.0)
        for sym, units in self.state.positions.items():
            if sym == CASH:
                continue
            px = prices.get(sym.upper())
            if px and px > 0:
                total += units * px
        return total

    def _handle_intent(self, intent: Intent, prices: dict[str, float], ts: str) -> dict:
        """Guardrail-check one intent, then fill it (paper) or sign it (live). Never bypasses
        the hard `risk/` checks; an out-of-policy intent is refused and audited, never obeyed."""
        risk_intent = TradeIntent(
            from_asset=intent.from_asset, to_asset=intent.to_asset, usd=float(intent.usd),
            chain=self.cfg.policy.chain, slippage_pct=intent.slippage_pct)
        state = ledger.state_from_ledger(self.cfg.risk_ledger_path)
        verdict = check_trade(self.cfg.policy, risk_intent, state)
        if not verdict.allowed:
            store.append({"kind": "refusal", "mode": self.cfg.mode,
                          "intent": {"from": intent.from_asset, "to": intent.to_asset,
                                     "usd": intent.usd}, "refusals": verdict.codes},
                         self.cfg.agent_ledger_path, now=None)
            return {"refused": verdict.codes, "from": intent.from_asset, "to": intent.to_asset}

        if self.cfg.is_live:
            return self._fill_live(risk_intent, intent, ts)
        return self._fill_paper(intent, prices, ts)

    def _fill_paper(self, intent: Intent, prices: dict[str, float], ts: str) -> dict:
        """Simulate the fill against live prices and persist it as a `fill` row."""
        try:
            f = paper.fill(intent.from_asset, intent.to_asset, intent.usd, prices,
                           liquidity_usd=self.cfg.paper_liquidity_usd)
        except ValueError as e:  # missing live price for a leg -> skip, don't fabricate a fill
            return {"skipped": str(e), "from": intent.from_asset, "to": intent.to_asset}
        # Spend must count against the caps even in paper mode, so a paper forward-run respects
        # the SAME daily/lifetime budget the live run will — write the attempt to the risk ledger.
        ledger.append({"kind": "attempt", "intent": {"from": intent.from_asset,
                       "to": intent.to_asset, "usd": f.usd_in}, "usd": f.usd_in, "mode": "paper"},
                      self.cfg.risk_ledger_path)
        store.append({"kind": "fill", "mode": "paper", "from": f.from_asset, "to": f.to_asset,
                      "usd_in": f.usd_in, "usd_out": f.usd_out, "cost_usd": f.cost_usd,
                      "units_from": f.units_from, "units_to": f.units_to,
                      "price_from": f.price_from, "price_to": f.price_to,
                      "reason": intent.reason}, self.cfg.agent_ledger_path)
        return {"filled": "paper", "from": f.from_asset, "to": f.to_asset,
                "usd_in": f.usd_in, "usd_out": f.usd_out, "cost_usd": f.cost_usd}

    def _fill_live(self, risk_intent: TradeIntent, intent: Intent, ts: str) -> dict:
        """Route a guardrail-passed intent through the ONE real signing path. The loop never
        signs anything itself — `execute_trade` re-runs the guardrails on the live quote, signs
        via TWAK, confirms, and writes the risk ledger. We mirror the outcome into the agent
        ledger as a `fill`/refusal row for the portfolio book."""
        out = self._execute_trade(risk_intent, self.cfg.policy,
                                   ledger_path=self.cfg.risk_ledger_path)
        if out.get("refused"):
            store.append({"kind": "refusal", "mode": "live",
                          "intent": {"from": intent.from_asset, "to": intent.to_asset,
                                     "usd": intent.usd}, "refusals": out["refused"]},
                         self.cfg.agent_ledger_path)
            return {"refused": out["refused"], "phase": out.get("phase")}
        tx_hash = out.get("tx_hash")
        if not tx_hash or out.get("status") != "confirmed":
            # broadcast/confirm uncertain — record without mutating the book (reconcile later)
            store.append({"kind": "result", "mode": "live", "from": intent.from_asset,
                          "to": intent.to_asset, "tx_hash": tx_hash,
                          "status": out.get("status") or out.get("error")},
                         self.cfg.agent_ledger_path)
            return {"unconfirmed": True, "tx_hash": tx_hash, "status": out.get("status")}
        # confirmed: the realized USD is the quote value; units reconcile against the wallet
        # read next tick (monitoring closes the loop — see [[Real-time Monitoring]]).
        usd = float(out.get("usd") or risk_intent.usd)
        store.append({"kind": "fill", "mode": "live", "from": intent.from_asset.upper(),
                      "to": intent.to_asset.upper(), "usd_in": usd, "usd_out": usd,
                      "cost_usd": 0.0, "units_from": 0.0, "units_to": 0.0,
                      "tx_hash": tx_hash, "reason": intent.reason},
                     self.cfg.agent_ledger_path)
        return {"filled": "live", "tx_hash": tx_hash, "usd": usd}

    def _mark_equity(self, equity: float, ts: str, tick_idx: int) -> tuple[float | None, float]:
        """Record the hourly equity mark + drawdown (scoring mirror). Also marks the risk
        ledger's equity so the `risk/` drawdown stop reads the same figure."""
        peak = self.state.peak_usd
        peak = equity if peak is None else max(peak, equity)
        dd_pct = ((peak - equity) / peak * 100.0) if peak and peak > 0 else None
        store.append({"kind": "equity", "mode": self.cfg.mode, "tick": tick_idx,
                      "equity_usd": equity, "peak_usd": peak,
                      "drawdown_pct": dd_pct, "below_dust": equity <= DUST_EQUITY_USD},
                     self.cfg.agent_ledger_path)
        # feed the live drawdown stop (only meaningful once equity has ever been funded)
        if equity > 0:
            ledger.append_equity(equity, self.cfg.risk_ledger_path)
        return dd_pct, peak

    def _heartbeat(self, ts: str, tick_idx: int, equity: float) -> None:
        """The dead-man input for /apentic/trading — a fresh timestamp every tick."""
        store.append({"kind": "heartbeat", "mode": self.cfg.mode, "tick": tick_idx,
                      "equity_usd": equity}, self.cfg.agent_ledger_path)
