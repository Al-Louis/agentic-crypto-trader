"""`python -m trader.agent.live_event_agent` — the LIVE (real-money) entry point for the event
champion. Identical loop to `event_agent`, but it actually SIGNS via TWAK — behind a triple gate.

  TRADER_MODE=live AGENT_ALLOW_LIVE=1 AGENT_LIVE_CONFIRM=1 \
    python -m trader.agent.live_event_agent --run-dir <ckpt> [--bankroll-usd 100]

Gates (ALL required to sign real money): `TRADER_MODE=live`, `AGENT_ALLOW_LIVE=1`,
`AGENT_LIVE_CONFIRM=1`. `--dry-run` routes every fill through `execute_trade`'s quote-only
pre-flight (no signing) and needs only the first two gates — the safe validation pass.

Bankroll: read from the wallet's **USDT balance at startup** (flat = all-USDT) unless
`--bankroll-usd` is given. It re-bases the env's $10k fills to the real wallet
(`scale = bankroll/$10k`); the decision env stays at $10k (mandatory for the frozen model). The
compliance SELL unwinds the exact BNB the BUY acquired (amount-in), preserving the gas buffer.

FUNDING REQUIREMENTS for the live wallet:
  * **USDT** = the bankroll the strategy trades (the anchor read at startup).
  * **A small BNB gas buffer BEYOND the compliance position** — every tx fee is paid in BNB, and
    the compliance SELL sells the exact BNB it bought, so without a standing buffer the SELL has no
    BNB left to pay its own gas. The EC2 runbook funds ~$10 BNB; that is the buffer. Keep it topped
    up. (After a MID-WEEK restart while holding token positions, pass an explicit `--bankroll-usd`
    — the startup USDT read under-counts when capital is parked in tokens.)

This is a SEPARATE entry point from `event_agent` (which stays paper-only and refuses live), so the
EC2 paper service is never at risk of arming the signer.
"""
from __future__ import annotations

import argparse
import os
import signal
import sys
import threading
from datetime import datetime, timezone

from trader import config
from trader.agent.event_agent import (DEFAULT_CANDLE_WINDOW, DEFAULT_TICK_OFFSET, HOUR, _now,
                                       load_provenance, load_selection, publish_aux_feeds,
                                       seconds_until_next_tick)

MODE_ENV = "TRADER_MODE"
LIVE_OPT_IN_ENV = "AGENT_ALLOW_LIVE"
LIVE_CONFIRM_ENV = "AGENT_LIVE_CONFIRM"      # the final, signing-only gate


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="trader.agent.live_event_agent",
                                description="LIVE real-money loop for the event RL champion")
    p.add_argument("--run-dir", required=True, help="checkpoint dir (policy.zip + vecnormalize.pkl "
                   "+ <run-id>/metrics.json)")
    p.add_argument("--run-id", default=None, help="defaults to the run-dir basename")
    p.add_argument("--bankroll-usd", type=float, default=None, help="real bankroll anchor; default "
                   "= the wallet's USDT balance read at startup (re-bases $10k fills: scale=bankroll/$10k)")
    p.add_argument("--min-notional-usd", type=float, default=0.50, help="skip signing a scaled fill "
                   "below this size (dust guard); the env still records the paper fill")
    p.add_argument("--dry-run", action="store_true", help="quote-only pre-flight for every fill (no "
                   "signing) — needs only TRADER_MODE=live + AGENT_ALLOW_LIVE=1")
    p.add_argument("--once", action="store_true", help="run a single tick then exit")
    p.add_argument("--now", type=int, default=None, help="override wall-clock 'now' (unix sec) for --once")
    p.add_argument("--interval-secs", type=int, default=HOUR)
    p.add_argument("--tick-offset-secs", type=int, default=DEFAULT_TICK_OFFSET)
    p.add_argument("--no-refresh", action="store_true", help="skip the network data refresh")
    p.add_argument("--settle-max-wait", type=float, default=600.0, help="max seconds to re-poll "
                   "GeckoTerminal WITHIN a tick until the just-closed bar settles for the active "
                   "pools (the candle-lag fix: without it a bar Gecko publishes late is missed for "
                   "a FULL hour). 0 = single fetch pass (old behavior).")
    p.add_argument("--settle-poll", type=float, default=45.0,
                   help="seconds between settle re-polls of the pools still missing the just-closed bar")
    p.add_argument("--settle-active-window", type=int, default=6 * 3600, help="a pool is waited on "
                   "only if its newest cached bar is within this many seconds of now (excludes "
                   "perma-stale pools that would otherwise hold up every tick)")
    p.add_argument("--capital", type=float, default=10_000.0, help="cold-weekly env capital (the model "
                   "trained at 10000; do NOT change — only the SCALE to the real bankroll varies)")
    p.add_argument("--candle-window", type=int, default=DEFAULT_CANDLE_WINDOW, help="trailing 1h "
                   "candles to publish per token to trading/candles/ (default 168 = 7d)")
    p.add_argument("--publish-wallet", action="store_true", help="ALSO publish trading/wallet.json — the "
                   "ACTUAL on-chain wallet equity/PnL (not the $10k env book). OFF by default so the "
                   "proven loop + existing telemetry stay byte-identical until this is validated.")
    p.add_argument("--wallet-baseline-usd", type=float, default=None, help="funded cost basis for the "
                   "wallet PnL (e.g. 101.06); omitted -> equity shown, PnL null")
    p.add_argument("--ledger-path", default=None, help="agent-ledger override; a --dry-run defaults "
                   "to data/agent_ledger.dryrun.jsonl (NEVER the production ledger) and does not publish")
    return p


def _require_live_gates(dry_run: bool) -> None:
    """ALL gates must be set to SIGN real money. `--dry-run` skips the final CONFIRM gate (it never
    signs). Fail loud (exit 2) so a misconfigured unit restart-loops into the dead-man, never trades."""
    if config.get(MODE_ENV) != "live":
        print(f"refusing: the live launcher requires {MODE_ENV}=live", file=sys.stderr)
        raise SystemExit(2)
    if config.get(LIVE_OPT_IN_ENV) != "1":
        print(f"refusing: set {LIVE_OPT_IN_ENV}=1 to enable the signing path", file=sys.stderr)
        raise SystemExit(2)
    if not dry_run and config.get(LIVE_CONFIRM_ENV) != "1":
        print(f"refusing REAL signing: set {LIVE_CONFIRM_ENV}=1 (the final gate) or pass --dry-run",
              file=sys.stderr)
        raise SystemExit(2)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config.load_dotenv()
    _require_live_gates(args.dry_run)
    run_id = args.run_id or os.path.basename(os.path.normpath(args.run_dir))

    from trader.agent.event_live import LiveEventTrader
    from trader.agent.event_runner import EventRunner, read_live_bankroll_usdt
    from trader.agent.publish import build_publisher
    from trader.agent.store import AGENT_LEDGER_PATH
    from trader.execution.execute import execute_swap_amount, execute_trade

    selection = load_selection()
    wallet_addr = config.get("AGENT_WALLET_ADDRESS")
    # Bankroll anchor = the wallet's TOTAL on-chain USD equity (USDT + token positions + BNB), so the
    # $10k-book fills re-base to real capital that's PARKED IN TOKENS — not just the USDT balance — and a
    # mid-week restart self-corrects (no --bankroll-usd pinning). Falls back to the USDT-only read if the
    # address is unset or the on-chain/price read fails. An explicit --bankroll-usd always wins.
    if args.bankroll_usd is not None:
        bankroll = args.bankroll_usd
    elif wallet_addr:
        try:
            from trader.agent.wallet_recon import read_live_equity_usd  # noqa: PLC0415
            bankroll = read_live_equity_usd(wallet_addr, selection)
            print(f"bankroll = on-chain wallet equity ${bankroll:,.2f} (USDT + tokens + BNB)",
                  file=sys.stderr)
        except Exception as e:  # noqa: BLE001 — a transient read error falls back, never refuses
            bankroll = read_live_bankroll_usdt()
            print(f"wallet-equity read failed ({e!r}); bankroll = USDT-only ${bankroll:,.2f}",
                  file=sys.stderr)
    else:
        bankroll = read_live_bankroll_usdt()
    if not bankroll or bankroll <= 0:
        print(f"refusing: bankroll is {bankroll!r} — fund the wallet or pass --bankroll-usd",
              file=sys.stderr)
        raise SystemExit(2)

    prov = load_provenance(args.run_dir, run_id)
    trader = LiveEventTrader(prov, policy_path=os.path.join(args.run_dir, "policy.zip"),
                             vecnorm_path=os.path.join(args.run_dir, "vecnormalize.pkl"))
    # A dry-run is a VALIDATION — it must NEVER touch the production ledger or publish to the live
    # dashboard. Isolate to a separate ledger + disable publishing unless an explicit --ledger-path.
    from pathlib import Path  # noqa: PLC0415
    if args.ledger_path:
        ledger_path = Path(args.ledger_path)
    elif args.dry_run:
        ledger_path = Path(AGENT_LEDGER_PATH).with_name("agent_ledger.dryrun.jsonl")
    else:
        ledger_path = Path(AGENT_LEDGER_PATH)
    publish_target = None if args.dry_run else config.get("APENTIC_PUBLISH_TARGET")
    publisher = build_publisher(ledger_path, publish_target) if publish_target else None
    # settle-wait config -> live_data.update_live: re-poll Gecko within a tick until the just-closed
    # bar lands, so a late-published candle is traded THIS hour, not next (the candle-lag fix).
    live_data_kwargs = {"settle_max_wait": args.settle_max_wait, "settle_poll": args.settle_poll,
                        "settle_active_window": args.settle_active_window}
    runner = EventRunner(trader, selection=selection, agent_ledger_path=ledger_path,
                         capital=args.capital, publisher=publisher, mode="live",
                         execute_fn=execute_trade, execute_amount_fn=execute_swap_amount,
                         live_bankroll_usd=float(bankroll), min_notional_usd=args.min_notional_usd,
                         live_dry_run=args.dry_run, live_data_kwargs=live_data_kwargs)

    # on-chain wallet reconciliation (trading/wallet.json) — OFF unless --publish-wallet (additive).
    wallet_recon_on = bool(args.publish_wallet) and bool(publish_target) and bool(wallet_addr)
    if args.publish_wallet and not wallet_recon_on:
        print(f"wallet recon requested but disabled: publish_target={publish_target!r} "
              f"address={'set' if wallet_addr else 'MISSING (set AGENT_WALLET_ADDRESS)'}", file=sys.stderr)

    print(f"LIVE event-agent start: run_id={run_id} bankroll=${bankroll:,.2f} "
          f"scale={bankroll / args.capital:.4g} dry_run={args.dry_run} "
          f"min_notional=${args.min_notional_usd} once={args.once} ledger={ledger_path} "
          f"publish={publish_target or 'off'} wallet_recon={'on' if wallet_recon_on else 'off'} "
          f"settle={'off' if args.settle_max_wait <= 0 else f'<= {args.settle_max_wait:.0f}s/{args.settle_poll:.0f}s'}",
          file=sys.stderr)

    def _tick(now_ts: int) -> None:
        r = runner.tick(now_ts, refresh_data=not args.no_refresh)
        print(f"[{'DRY ' if args.dry_run else ''}LIVE tick "
              f"{datetime.fromtimestamp(now_ts, timezone.utc).isoformat()}] "
              f"eq=${r.equity_usd:,.2f} fills+{r.fills_recorded}/blk{r.fills_blocked} "
              f"compliance={r.compliance_trades} trades_today={r.trades_today} "
              f"uni={len(r.universe)}", file=sys.stderr)
        # resume the dashboard's candle + signals feeds (these froze at go-live until ported from
        # the paper launcher). publish_target is None on a --dry-run, so this no-ops there.
        publish_aux_feeds(publish_target, selection, trader, now_ts, candle_window=args.candle_window)
        # ACTUAL on-chain wallet equity/PnL (flag-gated, additive). Fail-safe — a recon error here
        # must never stop a trading tick.
        if wallet_recon_on:
            try:
                from trader.agent.wallet_recon import USDT_BSC, publish_wallet  # noqa: PLC0415
                assets = [{"symbol": s["symbol"], "contract": s.get("token_address")}
                          for s in selection if s.get("token_address")]
                assets.append({"symbol": "USDT", "contract": USDT_BSC})
                wp = publish_wallet(publish_target, address=wallet_addr, assets=assets,
                                    prices=runner.latest_token_prices(now_ts),
                                    baseline_usd=args.wallet_baseline_usd, ledger_path=ledger_path)
                print(f"[wallet] real equity=${wp['equity_usd']:,.2f} pnl_usd={wp['pnl_usd']} "
                      f"stale={wp['stale']} -> {publish_target}/wallet.json", file=sys.stderr)
            except Exception as e:  # noqa: BLE001
                print(f"wallet recon warning: {e!r}", file=sys.stderr)

    if args.once:
        _tick(int(args.now if args.now is not None else _now()))
        return 0

    stop = threading.Event()
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            signal.signal(sig, lambda *_: stop.set())
        except (ValueError, OSError):
            pass
    while not stop.is_set():
        try:
            _tick(int(_now()))
        except Exception as e:  # noqa: BLE001 — one bad tick must not kill the loop
            print(f"tick error: {e!r}", file=sys.stderr)
        if stop.wait(seconds_until_next_tick(int(_now()), args.interval_secs, args.tick_offset_secs)):
            break
    print("LIVE event-agent stop", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
