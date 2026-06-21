"""Subprocess wrapper over the `twak` CLI (`--json`) — quote / swap / tx / balance.

Custody discipline (vault "Security and Encryption"):
  * **`--password` is NEVER passed** — wallet-password resolution happens inside twak via
    the OS keychain (Windows Credential Manager on this host) or `TWAK_WALLET_PASSWORD` in
    the host environment. A structural guard refuses any arg list containing it.
  * **argv and the environment are never logged or embedded in errors** — error text
    carries only the twak subcommand name and a capped stderr excerpt.
  * Every call has a hard timeout; args are validated against a tight charset before they
    reach the shell shim (npm installs `twak` as a `.cmd` on Windows, so a hostile token
    symbol would otherwise be a command-injection vector).

Observed quirk (captured 2026-06-11, fixture `tests/fixtures/twak_quote_*.txt`): even with
`--json`, `twak swap --quote-only` prints a human line BEFORE the JSON object, e.g.
`$1 USD ≈ 0.001652520591019684 BNB (@ $605.1361813185954)` — and the JSON itself has **no
USD field** (`input`/`output`/`minReceived` are "<amount> <SYMBOL>" strings + `provider`,
`priceImpact`). `parse_quote` therefore values the quote in USD from the `(@ $price)`
prefix and/or the stable leg, taking the conservative max; if neither exists it raises and
the caller refuses (fail closed).
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess

TWAK = "twak"
QUOTE_TIMEOUT_S = 90.0
SWAP_TIMEOUT_S = 240.0
READ_TIMEOUT_S = 60.0

# Tight charset for everything we splice into argv (symbols, chain keys, hashes, numbers).
_SAFE_ARG = re.compile(r"^[A-Za-z0-9@._:/-]+$")

# Treated as $1 for quote valuation (BSC majors; the spike only ever sees USDT).
STABLE_SYMBOLS = frozenset({"USDT", "USDC", "DAI", "BUSD", "FDUSD", "TUSD"})

_AMOUNT_SYM = re.compile(r"^\s*([0-9.eE+-]+)\s+([A-Za-z0-9._-]+)\s*$")
_AT_PRICE = re.compile(r"\(@\s*\$([0-9.eE+-]+)\)")


class TwakError(RuntimeError):
    """twak invocation failed (non-zero exit / timeout / missing binary / bad output)."""


class QuoteParseError(TwakError):
    """The quote output is missing or malformed in a load-bearing field."""


def _twak_bin() -> str:
    return shutil.which(TWAK) or TWAK


def _check_args(args: list[str]) -> list[str]:
    """Structural guards: never --password (any form), never a shell-hostile arg."""
    for a in args:
        if a == "--password" or a.startswith("--password="):
            raise TwakError("refusing to pass --password (keychain/host-env resolution only)")
        if not _SAFE_ARG.match(a):  # flags like --quote-only pass; whitespace/metachars don't
            raise TwakError(f"unsafe CLI argument rejected for 'twak {args[0]}'")
    return args


def run_text(args: list[str], *, timeout: float = READ_TIMEOUT_S) -> str:
    """Run `twak <args>` and return raw stdout. Errors never include argv or env."""
    args = _check_args(list(args))
    sub = args[0]
    try:
        proc = subprocess.run([_twak_bin(), *args],  # noqa: S603 — args charset-validated above
                              capture_output=True, text=True, timeout=timeout)
    except FileNotFoundError as e:
        raise TwakError("twak CLI not found on PATH") from e
    except subprocess.TimeoutExpired as e:
        raise TwakError(f"twak {sub} timed out after {timeout:.0f}s") from e
    if proc.returncode != 0:
        raise TwakError(f"twak {sub} failed (rc={proc.returncode}): "
                        f"{(proc.stderr or proc.stdout).strip()[:300]}")
    return proc.stdout


def extract_json(text: str) -> dict:
    """Parse the first JSON object out of mixed CLI output (human prefix + JSON)."""
    i = text.find("{")
    if i < 0:
        raise TwakError("no JSON object in twak output")
    try:
        obj, _ = json.JSONDecoder().raw_decode(text[i:])
    except json.JSONDecodeError as e:
        raise TwakError(f"malformed JSON in twak output: {e}") from e
    if not isinstance(obj, dict):
        raise TwakError("twak JSON output is not an object")
    return obj


def run_json(args: list[str], *, timeout: float = READ_TIMEOUT_S) -> dict:
    return extract_json(run_text(args, timeout=timeout))


# --- pure arg builders (unit-tested: provably no --password, no execution flags) -----------

def quote_args(from_asset: str, to_asset: str, usd: float, *, chain: str,
               slippage_pct: float) -> list[str]:
    return ["swap", "--usd", f"{usd:g}", str(from_asset), str(to_asset),
            "--chain", chain, "--slippage", f"{slippage_pct:g}", "--quote-only", "--json"]


def swap_args(from_asset: str, to_asset: str, usd: float, *, chain: str,
              slippage_pct: float) -> list[str]:
    return [a for a in quote_args(from_asset, to_asset, usd, chain=chain,
                                  slippage_pct=slippage_pct) if a != "--quote-only"]


# --- the call surface execute_trade consumes ------------------------------------------------

def quote(from_asset: str, to_asset: str, usd: float, *, chain: str = "bsc",
          slippage_pct: float = 1.0, timeout: float = QUOTE_TIMEOUT_S) -> str:
    """Read-only quote. Returns RAW stdout (human prefix + JSON) for `parse_quote`."""
    return run_text(quote_args(from_asset, to_asset, usd, chain=chain,
                               slippage_pct=slippage_pct), timeout=timeout)


def swap(from_asset: str, to_asset: str, usd: float, *, chain: str = "bsc",
         slippage_pct: float = 1.0, timeout: float = SWAP_TIMEOUT_S) -> dict:
    """EXECUTES a swap (signs + broadcasts; password via keychain). Returns parsed JSON."""
    return run_json(swap_args(from_asset, to_asset, usd, chain=chain,
                              slippage_pct=slippage_pct), timeout=timeout)


def tx_status(tx_hash: str, *, chain: str = "bsc", timeout: float = READ_TIMEOUT_S) -> dict:
    return run_json(["tx", str(tx_hash), "--chain", chain, "--json"], timeout=timeout)


def wallet_balance(*, chain: str = "bsc", timeout: float = READ_TIMEOUT_S) -> dict:
    return run_json(["wallet", "balance", "--chain", chain, "--json"], timeout=timeout)


# --- amount-in swaps (swap an EXACT token quantity, not a USD notional) ----------------------
# The `twak swap <amount> <from> <to>` form (no --usd). Needed to unwind a held position
# precisely — e.g. the compliance SELL sells the exact BNB the BUY acquired, preserving the
# wallet's gas buffer (a USD-sized sell can over/under-shoot on an intraday price move).

def _fmt_amount(amount: float) -> str:
    """Plain decimal token amount — never scientific notation (twak won't parse `6.8e-4`)."""
    return f"{float(amount):.18f}".rstrip("0").rstrip(".") or "0"


def quote_amount_args(from_asset: str, to_asset: str, amount: float, *, chain: str,
                      slippage_pct: float, decimals: int | None = None) -> list[str]:
    args = ["swap", _fmt_amount(amount), str(from_asset), str(to_asset), "--chain", chain,
            "--slippage", f"{slippage_pct:g}", "--quote-only", "--json"]
    if decimals is not None:                       # for tokens not in twak's registry
        args += ["--decimals", str(int(decimals))]
    return args


def swap_amount_args(from_asset: str, to_asset: str, amount: float, *, chain: str,
                     slippage_pct: float, decimals: int | None = None) -> list[str]:
    return [a for a in quote_amount_args(from_asset, to_asset, amount, chain=chain,
                                         slippage_pct=slippage_pct, decimals=decimals)
            if a != "--quote-only"]


def quote_amount(from_asset: str, to_asset: str, amount: float, *, chain: str = "bsc",
                 slippage_pct: float = 1.0, decimals: int | None = None,
                 timeout: float = QUOTE_TIMEOUT_S) -> str:
    """Read-only quote for an exact-amount swap. Raw stdout (human prefix + JSON) for parse_quote."""
    return run_text(quote_amount_args(from_asset, to_asset, amount, chain=chain,
                                      slippage_pct=slippage_pct, decimals=decimals), timeout=timeout)


def swap_amount(from_asset: str, to_asset: str, amount: float, *, chain: str = "bsc",
                slippage_pct: float = 1.0, decimals: int | None = None,
                timeout: float = SWAP_TIMEOUT_S) -> dict:
    """EXECUTES an exact-amount swap (signs + broadcasts; password via keychain). Parsed JSON."""
    return run_json(swap_amount_args(from_asset, to_asset, amount, chain=chain,
                                     slippage_pct=slippage_pct, decimals=decimals), timeout=timeout)


# --- quote parsing (pure) --------------------------------------------------------------------

def _leg(q: dict, key: str) -> tuple[float, str]:
    if key not in q:
        raise QuoteParseError(f"quote missing {key!r}")
    m = _AMOUNT_SYM.match(str(q[key]))
    if not m:
        raise QuoteParseError(f"unparseable quote leg {key}={q[key]!r}")
    try:
        return float(m.group(1)), m.group(2).upper()
    except ValueError as e:
        raise QuoteParseError(f"non-numeric amount in {key}={q[key]!r}") from e


def parse_quote(stdout_text: str) -> dict:
    """Normalize raw `--quote-only` stdout into the numbers the guardrails re-check.

    Raises QuoteParseError on ANY missing/odd load-bearing field — the caller refuses.
    `usd_value` is the conservative MAX of the available valuations (`(@ $price)` prefix on
    the input leg; stable legs at $1) or None if the quote cannot be valued in USD.
    `implied_slippage_pct` = (1 - minReceived/output)·100 — empirically exactly the
    requested `--slippage` tolerance (verified at 1% and 0.5%).
    """
    head = stdout_text[:max(stdout_text.find("{"), 0)]
    q = extract_json(stdout_text)
    in_amt, in_sym = _leg(q, "input")
    out_amt, out_sym = _leg(q, "output")
    min_amt, _min_sym = _leg(q, "minReceived")
    if out_amt <= 0 or in_amt <= 0 or min_amt < 0:
        raise QuoteParseError("non-positive quote amounts")
    if "priceImpact" not in q:
        raise QuoteParseError("quote missing 'priceImpact'")
    try:
        impact_pct = float(q["priceImpact"])
    except (TypeError, ValueError) as e:
        raise QuoteParseError(f"non-numeric priceImpact={q['priceImpact']!r}") from e

    valuations = []
    m = _AT_PRICE.search(head)
    if m:
        valuations.append(in_amt * float(m.group(1)))     # input leg at the quoted unit price
    if in_sym in STABLE_SYMBOLS:
        valuations.append(in_amt)
    if out_sym in STABLE_SYMBOLS:
        valuations.append(out_amt)

    return {
        "in_amount": in_amt, "in_symbol": in_sym,
        "out_amount": out_amt, "out_symbol": out_sym,
        "min_received": min_amt,
        "provider": q.get("provider"),
        "price_impact_pct": impact_pct,
        "implied_slippage_pct": round((1.0 - min_amt / out_amt) * 100.0, 4),
        "usd_value": max(valuations) if valuations else None,
    }
