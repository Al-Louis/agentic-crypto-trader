"""`execute_trade` — the guardrail-wrapped trade path (the ONLY way a swap gets signed).

Two-phase check (runbook §guardrail skeleton): the *intent* is checked first (refuse early,
no network), then the read-only quote is fetched and **the same caps are re-applied to the
quote's own numbers** — realized USD value, actual route symbols, implied slippage — because
the quote is the truth and the intent is a wish. Out-of-policy ⇒ refused with coded reasons,
never adjusted. Any failure to compute state (ledger unreadable/unwritable, quote missing
fields, twak error) ⇒ refuse with STATE_UNAVAILABLE — fail closed.

Ledger discipline: the attempt row lands on disk BEFORE the swap is signed (if it cannot be
written, the trade is refused), so the daily/lifetime caps survive a crash mid-trade. TWAK's
own `--slippage` is belt-and-suspenders *under* these checks, never instead of them.

Return contract (all success/dry-run keys read via .get() by callers — additive keys are safe):
  in-policy + landed   -> {"tx_hash", "status", "usd", "quote", "out_amount", "out_symbol"}
  in-policy + dry_run  -> {"dry_run": True, "would_execute": True, "status", "usd", "quote"}
  out-of-policy        -> {"refused": [codes], "detail": [...], "phase": intent|quote|state}
  passed checks, swap/confirm failed -> {"error": ..., "tx_hash": maybe} (spend stays counted)
`out_amount`/`out_symbol` are the REALIZED output leg (lets a caller capture the exact quantity
received — e.g. the compliance BUY's BNB, to unwind it precisely via `execute_swap_amount`).
`execute_swap_amount` (below) mirrors this for an exact-AMOUNT input swap.
"""

from __future__ import annotations

import time
from dataclasses import asdict
from pathlib import Path

from trader.execution import twak_cli
from trader.risk import SPIKE_POLICY, Policy, TradeIntent, check_trade, ledger
from trader.risk.checks import STATE_UNAVAILABLE, Verdict

POLL_ATTEMPTS = 24
POLL_INTERVAL_S = 5.0

_HASH_KEYS = ("txHash", "transactionHash", "hash", "txid", "tx_hash")


def extract_tx_hash(d: dict) -> str | None:
    """Find a 0x… tx hash in swap output (top level or one nested object deep)."""
    for k in _HASH_KEYS:
        v = d.get(k)
        if isinstance(v, str) and v.startswith("0x"):
            return v
    for v in d.values():
        if isinstance(v, dict):
            h = extract_tx_hash(v)
            if h:
                return h
    return None


def parse_tx_status(d: dict) -> str:
    """-> 'confirmed' | 'failed' | 'pending' from a `twak tx --json` payload (tolerant)."""
    # twak v0.19.0 emits booleans ({"confirmed": true, "pending": false, "failed": false}),
    # not a status string (observed live on the 2026-06-11 dust trade).
    if d.get("failed") is True:
        return "failed"
    if d.get("confirmed") is True:
        return "confirmed"
    s = str(d.get("status", "")).strip().lower()
    if s in {"success", "successful", "confirmed", "1", "0x1", "true"}:
        return "confirmed"
    if s in {"failed", "fail", "reverted", "error", "0", "0x0", "false"}:
        return "failed"
    return "pending"


def _refusal(verdict: Verdict, phase: str, intent: TradeIntent, path: Path) -> dict:
    """Record the refusal (audit trail; counts no spend) and build the refusal return."""
    try:
        ledger.append({"kind": "refusal", "phase": phase, "intent": asdict(intent),
                       "refusals": list(verdict.refusals)}, path)
    except Exception:  # noqa: BLE001,S110 — we are refusing anyway; never refuse-to-refuse
        pass
    return {"refused": verdict.codes, "detail": list(verdict.refusals), "phase": phase}


def _unavailable(detail: str, phase: str, intent: TradeIntent, path: Path) -> dict:
    v = Verdict(False, ({"code": STATE_UNAVAILABLE, "detail": detail[:300]},))
    return _refusal(v, phase, intent, path)


def execute_trade(intent: TradeIntent, policy: Policy = SPIKE_POLICY, *,
                  ledger_path: Path = ledger.LEDGER_PATH, cli=twak_cli,
                  poll_attempts: int = POLL_ATTEMPTS, poll_interval_s: float = POLL_INTERVAL_S,
                  sleep=time.sleep, dry_run: bool = False) -> dict:
    """Check → quote → re-check → record → swap → confirm. `cli` is injectable for tests.

    `dry_run=True` performs the full two-phase guardrail check (intent + realized quote) but
    stops BEFORE the ledger attempt row, the swap, and the confirm — nothing is written and no
    money moves. It is the safe end-to-end validation: a refusal still returns the refusal dict
    (so the caller sees exactly what a real call would block); a pass returns
    `{"dry_run": True, "would_execute": True, "quote", "usd"}`."""
    # 1) intent-phase check against persisted state (refuse early — no network on refusal).
    state = ledger.state_from_ledger(ledger_path)
    verdict = check_trade(policy, intent, state)
    if not verdict.allowed:
        return _refusal(verdict, "intent", intent, ledger_path)

    # 2) read-only quote, then re-check the caps on the QUOTE's numbers.
    try:
        parsed = twak_cli.parse_quote(
            cli.quote(intent.from_asset, intent.to_asset, intent.usd,
                      chain=intent.chain, slippage_pct=intent.slippage_pct))
    except Exception as e:  # noqa: BLE001 — quote failure/missing fields: fail closed
        return _unavailable(f"quote unavailable: {type(e).__name__}: {e}", "quote",
                            intent, ledger_path)
    if parsed["usd_value"] is None:
        return _unavailable("quote cannot be valued in USD (no price line, no stable leg)",
                            "quote", intent, ledger_path)
    # The asset IDENTITY for the allowlist is the INTENT's (what we actually swap): a contract-pinned
    # assetId can't route to a different token, and its on-chain symbol LABEL may differ from the
    # universe symbol (e.g. BANANAS31's contract reports "BANANA"). Only the realized USD value +
    # slippage are re-derived from the quote — those are the load-bearing two-phase truth.
    quote_intent = TradeIntent(
        from_asset=intent.from_asset, to_asset=intent.to_asset,
        usd=parsed["usd_value"], chain=intent.chain,
        slippage_pct=max(parsed["implied_slippage_pct"], parsed["price_impact_pct"]))
    verdict = check_trade(policy, quote_intent, state)
    if not verdict.allowed:
        return _refusal(verdict, "quote", quote_intent, ledger_path)

    # dry-run stops here: every cap passed on the realized quote, but write nothing and sign
    # nothing — report what WOULD execute. The safe pre-flight (no ledger row, no swap, no poll).
    if dry_run:
        return {"dry_run": True, "would_execute": True, "status": "dry_run",
                "usd": quote_intent.usd, "quote": parsed}

    # 3) the attempt row must be ON DISK before signing — no record, no trade.
    try:
        ledger.append({"kind": "attempt", "intent": asdict(intent), "quote": parsed,
                       "usd": quote_intent.usd}, ledger_path)
    except Exception as e:  # noqa: BLE001
        return _unavailable(f"ledger unwritable: {type(e).__name__}: {e}", "state",
                            intent, ledger_path)

    # 4) sign + broadcast (password via keychain inside twak — never in our argv).
    # Result rows are best-effort: the attempt row above already counted the spend, and once
    # money may have moved a ledger hiccup must not destroy the outcome/tx-hash return.
    try:
        swap_out = cli.swap(intent.from_asset, intent.to_asset, intent.usd,
                            chain=intent.chain, slippage_pct=intent.slippage_pct)
    except Exception as e:  # noqa: BLE001 — outcome unknown; attempt spend stays counted
        _append_result({"kind": "result", "tx_hash": None, "status": "swap_error",
                        "error": f"{type(e).__name__}: {e}"[:300]}, ledger_path)
        return {"error": f"swap failed: {type(e).__name__}: {e}"[:300], "tx_hash": None}
    tx_hash = extract_tx_hash(swap_out)
    if not tx_hash:
        _append_result({"kind": "result", "tx_hash": None, "status": "unknown",
                        "error": "no tx hash in swap output"}, ledger_path)
        return {"error": "no tx hash in swap output — verify on bscscan before retrying",
                "tx_hash": None}

    # 5) poll confirmation; record the outcome either way.
    status = _poll_confirmation(cli, tx_hash, intent.chain, poll_attempts, poll_interval_s, sleep)
    out_amount, out_symbol = parse_swap_output(swap_out)
    _append_result({"kind": "result", "tx_hash": tx_hash, "status": status,
                    "usd": quote_intent.usd, "out_amount": out_amount}, ledger_path)
    return {"tx_hash": tx_hash, "status": status, "usd": quote_intent.usd, "quote": parsed,
            "out_amount": out_amount, "out_symbol": out_symbol}


def parse_swap_output(swap_out: dict) -> tuple[float | None, str | None]:
    """The REALIZED `(amount, symbol)` a swap delivered, from its `output` leg (e.g.
    '0.00068 BNB'). `(None, None)` if absent/unparseable — never raises. Lets a caller capture
    the exact token quantity received (e.g. the BNB a compliance BUY landed, to unwind precisely)."""
    try:
        m = twak_cli._AMOUNT_SYM.match(str((swap_out or {}).get("output", "")))
        if m:
            return float(m.group(1)), m.group(2).upper()
    except Exception:  # noqa: BLE001
        pass
    return None, None


def _poll_confirmation(cli, tx_hash: str, chain: str, attempts: int, interval_s: float,
                       sleep) -> str:
    """Poll `twak tx` until confirmed/failed (or attempts exhausted). Transient RPC errors keep
    polling (a just-broadcast tx may 404 briefly). Shared by the USD- and amount-based paths."""
    status = "pending"
    for i in range(max(1, attempts)):
        try:
            status = parse_tx_status(cli.tx_status(tx_hash, chain=chain))
        except Exception:  # noqa: BLE001 — transient RPC trouble: keep polling
            status = "pending"
        if status in ("confirmed", "failed"):
            break
        if i < attempts - 1:
            sleep(interval_s)
    return status


def _append_result(row: dict, path: Path) -> None:
    """Best-effort result append (spend was already counted by the attempt row)."""
    try:
        ledger.append(row, path)
    except Exception:  # noqa: BLE001,S110 — never mask a live trade outcome behind a disk error
        pass


def execute_swap_amount(from_asset: str, to_asset: str, amount: float, policy: Policy = SPIKE_POLICY,
                        *, ledger_path: Path = ledger.LEDGER_PATH, cli=twak_cli,
                        chain: str = "bsc", slippage_pct: float = 1.0, decimals: int | None = None,
                        poll_attempts: int = POLL_ATTEMPTS, poll_interval_s: float = POLL_INTERVAL_S,
                        sleep=time.sleep, dry_run: bool = False) -> dict:
    """Swap an EXACT token AMOUNT (not a USD notional) through the SAME guardrails as execute_trade.

    Needed to unwind a held position precisely (the compliance SELL sells the exact BNB the BUY
    acquired, preserving the wallet's gas buffer — a USD-sized sell can over/under-shoot on an
    intraday price move). Cap enforcement happens on the QUOTE's realized USD (the out leg must be
    valuable, e.g. a stable): amount-in has no pre-quote USD, so the single (quote-phase) check IS
    the guardrail. Same return shape as execute_trade (+ `amount`). Signature kept parallel; `cli`
    injectable for tests."""
    rec = TradeIntent(from_asset=from_asset, to_asset=to_asset, usd=0.0, chain=chain,
                      slippage_pct=slippage_pct)   # placeholder for refusal-record asdict() only

    # 1) read-only amount-in quote, then check the caps on the QUOTE's realized numbers.
    try:
        parsed = twak_cli.parse_quote(cli.quote_amount(from_asset, to_asset, amount, chain=chain,
                                                       slippage_pct=slippage_pct, decimals=decimals))
    except Exception as e:  # noqa: BLE001 — quote failure/missing fields: fail closed
        return _unavailable(f"amount quote unavailable: {type(e).__name__}: {e}", "quote",
                            rec, ledger_path)
    if parsed["usd_value"] is None:
        return _unavailable("amount quote cannot be valued in USD (no stable leg)", "quote",
                            rec, ledger_path)
    quote_intent = TradeIntent(                              # allowlist on the assets WE swap (the
        from_asset=from_asset, to_asset=to_asset,            # contract); realized USD + slippage are
        usd=parsed["usd_value"], chain=chain,               # the re-checked truth (see execute_trade)
        slippage_pct=max(parsed["implied_slippage_pct"], parsed["price_impact_pct"]))
    verdict = check_trade(policy, quote_intent, ledger.state_from_ledger(ledger_path))
    if not verdict.allowed:
        return _refusal(verdict, "quote", quote_intent, ledger_path)
    if dry_run:
        return {"dry_run": True, "would_execute": True, "status": "dry_run", "amount": amount,
                "usd": quote_intent.usd, "quote": parsed}

    # 2) attempt row on disk BEFORE signing (records the amount + realized USD).
    try:
        ledger.append({"kind": "attempt", "intent": {"from_asset": from_asset, "to_asset": to_asset,
                       "amount": float(amount), "chain": chain}, "quote": parsed,
                       "usd": quote_intent.usd}, ledger_path)
    except Exception as e:  # noqa: BLE001
        return _unavailable(f"ledger unwritable: {type(e).__name__}: {e}", "state", rec, ledger_path)

    # 3) sign + broadcast (amount-in), then poll.
    try:
        swap_out = cli.swap_amount(from_asset, to_asset, amount, chain=chain,
                                   slippage_pct=slippage_pct, decimals=decimals)
    except Exception as e:  # noqa: BLE001 — outcome unknown; attempt spend stays counted
        _append_result({"kind": "result", "tx_hash": None, "status": "swap_error",
                        "error": f"{type(e).__name__}: {e}"[:300]}, ledger_path)
        return {"error": f"swap failed: {type(e).__name__}: {e}"[:300], "tx_hash": None}
    tx_hash = extract_tx_hash(swap_out)
    if not tx_hash:
        _append_result({"kind": "result", "tx_hash": None, "status": "unknown",
                        "error": "no tx hash in swap output"}, ledger_path)
        return {"error": "no tx hash in swap output — verify on bscscan before retrying",
                "tx_hash": None}
    status = _poll_confirmation(cli, tx_hash, chain, poll_attempts, poll_interval_s, sleep)
    out_amount, out_symbol = parse_swap_output(swap_out)
    _append_result({"kind": "result", "tx_hash": tx_hash, "status": status,
                    "usd": quote_intent.usd, "out_amount": out_amount}, ledger_path)
    return {"tx_hash": tx_hash, "status": status, "usd": quote_intent.usd, "amount": amount,
            "quote": parsed, "out_amount": out_amount, "out_symbol": out_symbol}
