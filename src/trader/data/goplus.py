"""GoPlus Security API — BSC token rug/honeypot forensics (the survival gate).

The structural-risk classifier from the project's adversarial-market thesis (vault
"Security and Encryption" / "Trading Strategies"): on BSC the dominant risk is not
drawdown but a -100%-in-one-block rug / honeypot, which price-based risk metrics
cannot see. This module screens a token's *contract* for those risks before it can
enter the tradeable set, and grades them block / warn / ok.

Keyless (BSC chain_id=56). Pure scoring helpers are testable; fetch is stdlib urllib.
"""

from __future__ import annotations

import json
import time
import urllib.parse
import urllib.request

BASE = "https://api.gopluslabs.io/api/v1"
BSC = "56"

# Any present => instant veto (catastrophic, irreversible).
BLOCK_FLAGS = {
    "is_honeypot": "honeypot (cannot sell)",
    "cannot_sell_all": "cannot sell all",
    "cannot_buy": "cannot buy",
    "hidden_owner": "hidden owner",
    "can_take_back_ownership": "owner reclaimable",
    "selfdestruct": "self-destruct",
    "is_honeypot_with_same_creator": "creator linked to a honeypot",
}
# Dev holds a dangerous power (weight 3).
HIGH_FLAGS = {
    "is_mintable": "mintable supply",
    "transfer_pausable": "transfers pausable",
    "is_blacklisted": "blacklist function",
    "slippage_modifiable": "tax modifiable",
}
# Softer concerns (weight 1).
MED_FLAGS = {
    "is_proxy": "upgradeable proxy",
    "external_call": "external-call risk",
}

# Thresholds / weights — tunable; risk-averse for the 30% drawdown DQ gate.
HIGH_TAX, WARN_TAX = 0.10, 0.05
OWNER_CONC = 0.50
LP_LOCK_MIN = 0.50
# LP-lock only signals rug risk on small/new tokens; established tokens hold LP
# unlocked via market-makers/CEX (e.g. Binance-Peg majors), which is normal.
LP_CONTEXT_HOLDERS = 10_000
W_HIGH, W_MED, W_NOT_OSS, W_OWNER, W_LP, W_TAX_HIGH, W_TAX_WARN = 3, 1, 2, 2, 1, 3, 1
BLOCK_SCORE = 6


def _get(url: str, timeout: int = 25) -> dict:
    req = urllib.request.Request(url, headers={
        "User-Agent": "act-forensics/0.1", "Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def _chunks(seq, n):
    for i in range(0, len(seq), n):
        yield seq[i:i + n]


def fetch_token_security(addresses, chain_id: str = BSC, timeout: int = 25,
                         sleep: float = 1.5, retries: int = 1, logger=None) -> dict:
    """addr(lower) -> raw GoPlus security dict.

    Keyless GoPlus is one-address-at-a-time (batches return partial) and analyzes
    uncached tokens lazily (empty on first hit), so we query singly, throttle, and
    retry the empties once.
    """
    out: dict[str, dict] = {}
    want = [a.lower() for a in addresses if a]
    for _ in range(retries + 1):
        pending = [a for a in want if a not in out]
        if not pending:
            break
        for addr in pending:
            try:
                d = _get(f"{BASE}/token_security/{chain_id}"
                         f"?contract_addresses={urllib.parse.quote(addr)}", timeout=timeout)
                val = (d.get("result") or {}).get(addr) or (d.get("result") or {}).get(addr.lower())
                if val:
                    out[addr] = val
            except Exception as e:  # noqa: BLE001 — one bad token shouldn't kill the run
                if logger:
                    logger(f"    {addr[:10]} err {e!r}")
            time.sleep(sleep)
    return out


# --- pure parsing / scoring ----------------------------------------------

def _b(v) -> bool:
    return str(v) == "1"


def _f(v, default: float = 0.0) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def _lp_locked_pct(raw: dict) -> float:
    holders = raw.get("lp_holders") or []
    return round(sum(_f(h.get("percent")) for h in holders if _b(h.get("is_locked"))), 4)


def _lp_unlocked_risk(sec: dict) -> bool:
    """Unlocked LP is a rug signal only on small/new tokens (see LP_CONTEXT_HOLDERS)."""
    return (sec["lp_holder_count"] > 0 and sec["lp_locked_pct"] < LP_LOCK_MIN
            and sec["holder_count"] < LP_CONTEXT_HOLDERS)


def parse_security(raw: dict) -> dict:
    """Normalize the GoPlus fields we score on (strings -> bool/float)."""
    if not raw:
        return {"available": False}
    sec = {
        "available": True,
        "is_open_source": _b(raw.get("is_open_source")),
        "buy_tax": _f(raw.get("buy_tax")),
        "sell_tax": _f(raw.get("sell_tax")),
        "holder_count": int(_f(raw.get("holder_count"))),
        "owner_percent": _f(raw.get("owner_percent")),
        "creator_percent": _f(raw.get("creator_percent")),
        "lp_holder_count": int(_f(raw.get("lp_holder_count"))),
        "lp_locked_pct": _lp_locked_pct(raw),
    }
    for k in (*BLOCK_FLAGS, *HIGH_FLAGS, *MED_FLAGS):
        sec[k] = _b(raw.get(k))
    return sec


def risk_flags(sec: dict) -> list[str]:
    if not sec.get("available"):
        return ["no_data"]
    flags = []
    for group in (BLOCK_FLAGS, HIGH_FLAGS, MED_FLAGS):
        flags += [desc for k, desc in group.items() if sec.get(k)]
    if not sec["is_open_source"]:
        flags.append("not open-source")
    if sec["sell_tax"] >= WARN_TAX:
        flags.append(f"sell tax {sec['sell_tax'] * 100:.0f}%")
    if sec["owner_percent"] >= OWNER_CONC:
        flags.append(f"owner holds {sec['owner_percent'] * 100:.0f}%")
    if _lp_unlocked_risk(sec):
        flags.append(f"LP only {sec['lp_locked_pct'] * 100:.0f}% locked")
    return flags


def score_security(sec: dict) -> int:
    s = W_HIGH * sum(1 for k in HIGH_FLAGS if sec.get(k))
    s += W_MED * sum(1 for k in MED_FLAGS if sec.get(k))
    s += 0 if sec["is_open_source"] else W_NOT_OSS
    s += W_TAX_HIGH if sec["sell_tax"] >= HIGH_TAX else (W_TAX_WARN if sec["sell_tax"] >= WARN_TAX else 0)
    s += W_OWNER if sec["owner_percent"] >= OWNER_CONC else 0
    s += W_LP if _lp_unlocked_risk(sec) else 0
    return s


def verdict(sec: dict) -> dict:
    """-> {verdict: block|warn|ok|unknown, score, flags}."""
    flags = risk_flags(sec)
    if not sec.get("available"):
        return {"verdict": "unknown", "score": None, "flags": flags}
    # hard veto: any catastrophic flag, or a tax so high the token is untradeable
    if any(sec.get(k) for k in BLOCK_FLAGS) or sec["sell_tax"] >= 0.50:
        return {"verdict": "block", "score": 99, "flags": flags}
    score = score_security(sec)
    grade = "block" if score >= BLOCK_SCORE else ("warn" if score >= 1 else "ok")
    return {"verdict": grade, "score": score, "flags": flags}
