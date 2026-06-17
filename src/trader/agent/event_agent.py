"""`python -m trader.agent.event_agent` — the live paper-trading entry point for the
event-driven RL champion (ef-s2).

Wires the checkpoint -> LiveEventTrader -> EventRunner -> the `trading/` publisher and drives
one tick per closed 1h bar, a few minutes after each hour so the bar's data has settled.

  python -m trader.agent.event_agent --run-dir runs-rl/ppo-event-rdLe4-ef-503b784-s2 --once
  python -m trader.agent.event_agent --run-dir runs-rl/<run-id>            # forever, hourly

Checkpoint layout (matches scripts/simulate_weekly): `<run-dir>/policy.zip`,
`<run-dir>/vecnormalize.pkl`, and `<run-dir>/<run-id>/metrics.json` (the provenance). `--run-id`
defaults to the run-dir basename.

PAPER ONLY for now: the event harness has no TWAK signing path yet (the runner records paper
fills + the guardrail audit; live routing is future work), so live mode REFUSES loudly rather
than silently doing nothing — the same fail-loud posture as `trader.agent.__main__`.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import sys
import threading
from datetime import datetime, timezone
from pathlib import Path

from trader import config

HOUR = 3600
MODE_ENV = "TRADER_MODE"            # paper | live (live refused here — no signing path yet)
LIVE_OPT_IN_ENV = "AGENT_ALLOW_LIVE"
DEFAULT_TICK_OFFSET = 180           # seconds after the hour to tick (let the bar's data settle)


def seconds_until_next_tick(now: int, interval: int = HOUR, offset: int = DEFAULT_TICK_OFFSET) -> float:
    """Seconds from `now` (unix sec) to the next `interval`-boundary + `offset`. Always > 0:
    if we're past this period's offset, target the next period. Pure — unit-tested."""
    base = (now // interval) * interval + offset
    while base <= now:
        base += interval
    return float(base - now)


def load_provenance(run_dir: str, run_id: str) -> dict:
    """Read the checkpoint's training provenance from `<run-dir>/<run-id>/metrics.json`."""
    path = os.path.join(run_dir, run_id, "metrics.json")
    with open(path, encoding="utf-8") as fh:
        meta = json.load(fh)
    return meta.get("provenance", meta)


def load_selection(path: str = os.path.join("data", "selection.json")) -> list[dict]:
    """The universe tokens + pool addresses the live-data updater needs (symbol + pair_address)."""
    sel = json.load(open(path, encoding="utf-8"))
    return [{"symbol": s["symbol"], "pair_address": s.get("pair_address")} for s in sel]


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="trader.agent.event_agent",
                                description="Live paper-trading loop for the event RL champion")
    p.add_argument("--run-dir", required=True, help="checkpoint dir (policy.zip + vecnormalize.pkl "
                   "+ <run-id>/metrics.json)")
    p.add_argument("--run-id", default=None, help="defaults to the run-dir basename")
    p.add_argument("--once", action="store_true", help="run a single tick then exit (the dry-run gate)")
    p.add_argument("--interval-secs", type=int, default=HOUR)
    p.add_argument("--tick-offset-secs", type=int, default=DEFAULT_TICK_OFFSET)
    p.add_argument("--no-refresh", action="store_true", help="skip the network data refresh "
                   "(assume data/ is already current — for the on-box dry-run against recorded data)")
    p.add_argument("--capital", type=float, default=10_000.0, help="cold-weekly session capital "
                   "(ef-s2 trained at 10000; changing it breaks AMM-cost/fill parity)")
    return p


def _resolve_mode() -> str:
    """Paper unless explicitly live; an invalid value fails loud (systemd restart-loops -> the
    dead-man fires). Live is refused outright — no signing path for the event harness yet."""
    mode = config.get(MODE_ENV) or "paper"
    if mode not in ("paper", "live"):
        print(f"refusing: {MODE_ENV}={mode!r} is not 'paper' or 'live'.", file=sys.stderr)
        raise SystemExit(2)
    if mode == "live":
        print("refusing live mode: the event harness has no TWAK signing path yet "
              "(paper only). Unset TRADER_MODE or set it to 'paper'.", file=sys.stderr)
        raise SystemExit(2)
    return mode


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config.load_dotenv()
    mode = _resolve_mode()
    run_id = args.run_id or os.path.basename(os.path.normpath(args.run_dir))

    from trader.agent.event_live import LiveEventTrader
    from trader.agent.event_runner import EventRunner
    from trader.agent.publish import build_publisher
    from trader.agent.store import AGENT_LEDGER_PATH

    prov = load_provenance(args.run_dir, run_id)
    trader = LiveEventTrader(prov,
                             policy_path=os.path.join(args.run_dir, "policy.zip"),
                             vecnorm_path=os.path.join(args.run_dir, "vecnormalize.pkl"))
    publish_target = config.get("APENTIC_PUBLISH_TARGET")
    publisher = build_publisher(AGENT_LEDGER_PATH, publish_target) if publish_target else None
    runner = EventRunner(trader, selection=load_selection(),
                         agent_ledger_path=AGENT_LEDGER_PATH, capital=args.capital,
                         publisher=publisher, mode=mode)

    print(f"event-agent start: run_id={run_id} mode={mode} recurrent={trader.recurrent} "
          f"refresh={not args.no_refresh} publish={publish_target or 'off'} "
          f"once={args.once}", file=sys.stderr)

    def _tick(now_ts: int) -> None:
        r = runner.tick(now_ts, refresh_data=not args.no_refresh)
        print(f"[tick {datetime.fromtimestamp(now_ts, timezone.utc).isoformat()}] "
              f"wk={datetime.fromtimestamp(r.week_start, timezone.utc).date()} "
              f"eq=${r.equity_usd:,.2f} dd={r.drawdown_pct:.1f}% "
              f"fills+{r.fills_recorded}/blocked{r.fills_blocked} "
              f"trades_today={r.trades_today} uni={len(r.universe)}", file=sys.stderr)

    if args.once:
        _tick(int(_now()))
        return 0

    stop = threading.Event()
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            signal.signal(sig, lambda *_: stop.set())
        except (ValueError, OSError):
            pass
    while not stop.is_set():
        wait = seconds_until_next_tick(int(_now()), args.interval_secs, args.tick_offset_secs)
        if stop.wait(wait):                         # interruptible sleep — SIGTERM returns at once
            break
        try:
            _tick(int(_now()))
        except Exception as e:  # noqa: BLE001 — one bad tick must not kill the loop (dead-man ages)
            print(f"tick error: {e!r}", file=sys.stderr)
    print("event-agent stop", file=sys.stderr)
    return 0


def _now() -> float:
    """Wall-clock seconds (indirection so tests can monkeypatch)."""
    import time  # noqa: PLC0415
    return time.time()


if __name__ == "__main__":
    raise SystemExit(main())
