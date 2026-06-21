"""Local live-execution smoke — a tiny REAL BNB<->USDT round-trip on the DEV wallet through the
SAME `execute_trade` path the event harness now calls (`_sign_live`).

Safety:
  * Default is **DRY-RUN** (quote-only, no signing, nothing written). Pass `--execute` to sign.
  * Tight caps (`$0.50/trade, $1.50/day, $3 lifetime`) in an ISOLATED ledger
    (`data/risk_ledger_localtest.jsonl`) so the spike ledger is untouched and the local
    lifetime accounting is clean.
  * Uses whatever wallet the local `twak` keychain holds = the dev spike wallet
    `0x2C19…D32E`. It CANNOT reach the EC2 competition wallet (different box, different keystore).

Usage:
  python scripts/live_exec_smoke.py                      # dry-run both legs (quote-only)
  python scripts/live_exec_smoke.py --negative           # + show an over-cap refusal
  python scripts/live_exec_smoke.py --execute --leg sell # REAL BNB->USDT (one leg)
  python scripts/live_exec_smoke.py --execute --leg buy  # REAL USDT->BNB (one leg)
  python scripts/live_exec_smoke.py --execute            # REAL round-trip (buy then sell)
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from trader.execution import twak_cli                      # noqa: E402
from trader.execution.execute import execute_trade         # noqa: E402
from trader.risk import Policy, TradeIntent                # noqa: E402

TEST_LEDGER = Path("data/risk_ledger_localtest.jsonl")
TEST_POLICY = Policy(allowlist=frozenset({"BNB", "USDT"}), per_trade_usd=0.50, daily_usd=1.50,
                     max_slippage_pct=1.0, drawdown_stop_pct=30.0, lifetime_usd_ceiling=3.0,
                     chain="bsc")
LEGS = {                                                    # name -> (from, to)
    "buy":  ("USDT", "BNB"),                                # compliance BUY  (01:00): USDT -> BNB
    "sell": ("BNB", "USDT"),                                # compliance SELL (23:00): BNB  -> USDT
}


def _balance() -> str:
    try:
        b = twak_cli.wallet_balance(chain="bsc")
        toks = {t.get("symbol"): t.get("balance") for t in b.get("tokens", [])}
        return f"BNB={b.get('total')} (${b.get('totalUsd')})  tokens={toks}"
    except Exception as e:  # noqa: BLE001
        return f"(balance unavailable: {type(e).__name__}: {e})"


def _one_leg(name: str, usd: float, slippage: float, execute: bool) -> dict:
    frm, to = LEGS[name]
    print(f"\n--- {name.upper()}  {frm} -> {to}  ${usd:g}  (slippage <= {slippage}%) ---")
    res = execute_trade(TradeIntent(from_asset=frm, to_asset=to, usd=usd, chain="bsc",
                                    slippage_pct=slippage),
                        TEST_POLICY, ledger_path=TEST_LEDGER, dry_run=not execute)
    if res.get("refused"):
        print(f"  REFUSED [{res.get('phase')}]: {res['refused']}  -> {res.get('detail')}")
        return res
    q = res.get("quote") or {}
    print(f"  quote: {q.get('in_amount')} {q.get('in_symbol')} -> {q.get('out_amount')} "
          f"{q.get('out_symbol')}  (~${res.get('usd'):.4f})  impl_slip={q.get('implied_slippage_pct')}%"
          f"  impact={q.get('price_impact_pct')}%  provider={q.get('provider')}")
    if res.get("dry_run"):
        print("  DRY-RUN: would execute — nothing signed, nothing written.")
    elif res.get("tx_hash"):
        print(f"  SIGNED  tx={res['tx_hash']}  status={res.get('status')}")
        print(f"  bscscan: https://bscscan.com/tx/{res['tx_hash']}")
    elif res.get("error"):
        print(f"  ERROR: {res['error']}  tx={res.get('tx_hash')}")
    return res


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--leg", choices=["buy", "sell", "roundtrip"], default="roundtrip")
    ap.add_argument("--usd", type=float, default=0.40)
    ap.add_argument("--slippage", type=float, default=1.0)
    ap.add_argument("--execute", action="store_true", help="ACTUALLY SIGN (default: dry-run)")
    ap.add_argument("--negative", action="store_true", help="also show an over-cap refusal")
    args = ap.parse_args(argv)
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:  # noqa: BLE001
        pass

    print(f"== live_exec_smoke :: {'EXECUTE (REAL FUNDS)' if args.execute else 'DRY-RUN (quote-only)'} ==")
    print(f"policy: per_trade=${TEST_POLICY.per_trade_usd} daily=${TEST_POLICY.daily_usd} "
          f"lifetime=${TEST_POLICY.lifetime_usd_ceiling} slippage<={TEST_POLICY.max_slippage_pct}%  "
          f"ledger={TEST_LEDGER}")
    print(f"wallet (before): {_balance()}")

    if args.negative:
        print("\n--- NEGATIVE PROOF: a $1.00 intent vs the $0.50 per-trade cap (no network) ---")
        r = execute_trade(TradeIntent("BNB", "USDT", 1.00, "bsc", 1.0), TEST_POLICY,
                          ledger_path=TEST_LEDGER, dry_run=True)
        print(f"  -> refused={r.get('refused')}  phase={r.get('phase')}")

    legs = ["buy", "sell"] if args.leg == "roundtrip" else [args.leg]
    for name in legs:
        _one_leg(name, args.usd, args.slippage, args.execute)

    if args.execute:
        print(f"\nwallet (after): {_balance()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
