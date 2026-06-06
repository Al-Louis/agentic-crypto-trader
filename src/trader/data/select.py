"""Universe selection — turn the DexScreener screen into a risk-tiered candidate set.

Applies the corrected criteria from the data spike (vault "Simulated Market"):

- **Rank by real traded volume + turnover**, not liquidity magnitude — liquidity
  ranking surfaces parked/fake pools (KOGE: $54.7M liq, 0.4% turnover).
- **Exitability floor** on liquidity (can we get out at size?).
- **Drop stale** (no price movement) and **parked** (turnover below floor) tokens.
- **Flag ambiguous** symbol resolutions for CMC contract verification (35% of the
  universe resolved with a close runner-up — likely wrong contract).

Pure and testable; consumed by `scripts/select_universe.py`. Final selection is a
human call — this produces a *proposal* to adjust, not a lock.
"""

from __future__ import annotations

LIQ_FLOOR = 100_000.0     # USD: min pool liquidity for exitability
TURNOVER_FLOOR = 0.05     # vol_h24 / liq: below this, liquidity is parked/facade


def enrich(rows: list[dict]) -> list[dict]:
    """Resolved rows with turnover + stale flags (copies; input untouched)."""
    out = []
    for r in rows:
        if r.get("status") != "resolved":
            continue
        r = dict(r)
        liq = r.get("liq_usd") or 0.0
        r["turnover"] = (r.get("vol_h24") or 0.0) / liq if liq else 0.0
        r["stale"] = (r.get("vol_proxy") or 0) == 0
        r["needs_verification"] = bool(r.get("ambiguous"))
        out.append(r)
    return out


def is_candidate(r: dict, liq_floor: float = LIQ_FLOOR,
                 turnover_floor: float = TURNOVER_FLOOR) -> bool:
    return (
        not r.get("is_stable")
        and not r.get("stale")
        and (r.get("liq_usd") or 0.0) >= liq_floor
        and r.get("turnover", 0.0) >= turnover_floor
    )


def candidates(rows: list[dict], liq_floor: float = LIQ_FLOOR,
               turnover_floor: float = TURNOVER_FLOOR) -> list[dict]:
    return [r for r in enrich(rows) if is_candidate(r, liq_floor, turnover_floor)]


def tier(cands: list[dict], n_major: int = 7, n_mid: int = 7,
         n_degen: int = 6) -> list[dict]:
    """Spread across risk tiers: deepest-liquidity = major; thin+volatile = degen.

    Majors and mids are the top liquidity bands (best exitability); degens are the
    highest-volatility names among the thinner remainder — the deliberate
    'ranging in risk factors' spread.
    """
    ranked = sorted(cands, key=lambda r: r["liq_usd"], reverse=True)
    majors, mids, rest = ranked[:n_major], ranked[n_major:n_major + n_mid], ranked[n_major + n_mid:]
    degens = sorted(rest, key=lambda r: r.get("vol_proxy", 0), reverse=True)[:n_degen]
    for r in majors:
        r["tier"] = "major"
    for r in mids:
        r["tier"] = "mid"
    for r in degens:
        r["tier"] = "degen"
    return majors + mids + degens


def select(rows: list[dict], n_major: int = 7, n_mid: int = 7, n_degen: int = 6,
           liq_floor: float = LIQ_FLOOR, turnover_floor: float = TURNOVER_FLOOR) -> list[dict]:
    """End-to-end: enrich -> gate -> tier. Returns the tiered candidate proposal."""
    return tier(candidates(rows, liq_floor, turnover_floor), n_major, n_mid, n_degen)
