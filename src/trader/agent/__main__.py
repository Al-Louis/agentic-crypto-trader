"""`python -m trader.agent` — run the autonomous loop (paper by default).

The entry point the future systemd unit launches ([[Remote Capabilities]]). It wires the
live CMC feed + the `HoldCore` stub (the RL champion swaps in here later via `--core`, not
yet), installs SIGINT/SIGTERM handlers for a clean shutdown, and runs.

Live mode is UNREACHABLE without `--mode live` *and* the matching env opt-in — paper is the
default and a misconfig fails closed to paper. This engagement runs paper only; the live
branch is wired but must not be exercised here.

  python -m trader.agent                       # paper, hourly, eligible universe, forever
  python -m trader.agent --ticks 3 --interval 0  # quick local smoke (3 ticks, no wait)
"""

from __future__ import annotations

import argparse
import sys

from trader import config
from trader.agent.decide import HoldCore
from trader.agent.feed import CmcPriceFeed
from trader.agent.loop import Loop, LoopConfig
from trader.data.eligible import eligible_symbols

# Live mode demands BOTH the mode selection and this env var — a second, independent gate
# so a stray `--mode live` (or env-file edit) can't open the signing path on its own.
LIVE_OPT_IN_ENV = "AGENT_ALLOW_LIVE"

# The systemd unit launches `python -m trader.agent` with no argv; mode then comes from the
# env-file (deploy/trader.env.template). An explicit --mode flag overrides the env.
MODE_ENV = "TRADER_MODE"


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="trader.agent", description="Autonomous trading loop")
    p.add_argument("--mode", choices=["paper", "live"], default=None,
                   help=f"paper (default, first-class sim) or live (requires "
                        f"{LIVE_OPT_IN_ENV}=1); unset falls back to ${MODE_ENV}, then paper")
    p.add_argument("--ticks", type=int, default=None,
                   help="run N ticks then stop (default: run until signalled)")
    p.add_argument("--interval", type=float, default=3600.0,
                   help="seconds between ticks (default 3600 = hourly, matching scoring)")
    p.add_argument("--no-stables", action="store_true",
                   help="exclude stablecoins from the traded universe")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.mode is None:
        env_mode = config.get(MODE_ENV)
        if env_mode not in (None, "", "paper", "live"):
            # A typo'd mode must be LOUD (systemd restart-loops -> dead-man fires), not a
            # silent week of paper trading during the scored window.
            print(f"refusing: {MODE_ENV}={env_mode!r} is not 'paper' or 'live'.",
                  file=sys.stderr)
            return 2
        args.mode = env_mode or "paper"

    if args.mode == "live" and config.get(LIVE_OPT_IN_ENV) != "1":
        print(f"refusing live mode: set {LIVE_OPT_IN_ENV}=1 to enable the signing path "
              f"(paper is the default).", file=sys.stderr)
        return 2

    api_key = config.require("CMC_API_KEY")
    feed = CmcPriceFeed(api_key)
    universe = eligible_symbols(include_stables=not args.no_stables)
    cfg = LoopConfig(universe=universe, mode=args.mode, tick_seconds=args.interval,
                     max_ticks=args.ticks)
    # Telemetry to the trading/ surface (put-only instance role on the host). Unset -> off;
    # the loop still records everything locally — publishing is read-side, never load-bearing.
    publish_target = config.get("APENTIC_PUBLISH_TARGET")
    publisher = None
    if publish_target:
        from trader.agent.publish import build_publisher
        publisher = build_publisher(cfg.agent_ledger_path, publish_target)
    loop = Loop(cfg, feed, HoldCore(), publisher=publisher)
    loop.install_signal_handlers()
    print(f"loop start: mode={cfg.mode} universe={len(universe)} core={loop.core.name} "
          f"interval={cfg.tick_seconds}s ticks={args.ticks or 'inf'} "
          f"publish={publish_target or 'off'}", file=sys.stderr)
    n = loop.run()
    print(f"loop stop: {n} ticks executed", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
