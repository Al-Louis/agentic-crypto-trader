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
    """End-to-end (liquidity tiering): enrich -> gate -> tier."""
    return tier(candidates(rows, liq_floor, turnover_floor), n_major, n_mid, n_degen)


# --- quality tiering by CMC rank (the chosen approach) --------------------
# On BSC, liquidity != safety (ETH is parked, memes are liquid), so the risk
# tier is driven by CMC rank (establishment / rug-survival), not pool depth.
# See vault "Trading Strategies" / "Market Conditions".
ANCHOR_RANK = 60           # CMC rank <= this: established L1/L2/DeFi major (low risk)
MID_RANK = 200             # <= this: established midcap; above/None: new/meme (high risk)
QUALITY_LIQ_FLOOR = 50_000.0   # relaxed floor so traded modest-liquidity majors qualify


def risk_bucket(r: dict) -> str:
    rank = r.get("cmc_rank")
    if rank is None:
        return "high"
    if rank <= ANCHOR_RANK:
        return "low"
    if rank <= MID_RANK:
        return "mid"
    return "high"


def tier_by_quality(cands: list[dict], n_anchor: int = 7, n_mid: int = 7,
                    n_meme: int = 6) -> list[dict]:
    """Risk spectrum by CMC rank, tradeability by liquidity.

    The *tier* (risk class) is set by CMC rank — established major / midcap /
    new-or-meme. Within each tier we pick the **most liquid** names, so every
    selected token is actually tradeable in its risk class.
    """
    for r in cands:
        r["risk"] = risk_bucket(r)
        r["tier"] = None
    by_liq = lambda rs: sorted(rs, key=lambda r: r.get("liq_usd", 0), reverse=True)
    anchor = by_liq(r for r in cands if r["risk"] == "low")[:n_anchor]
    mid = by_liq(r for r in cands if r["risk"] == "mid")[:n_mid]
    meme = by_liq(r for r in cands if r["risk"] == "high")[:n_meme]
    for r in anchor:
        r["tier"] = "anchor"
    for r in mid:
        r["tier"] = "mid"
    for r in meme:
        r["tier"] = "meme"
    return anchor + mid + meme


def select_quality(rows: list[dict], n_anchor: int = 7, n_mid: int = 7, n_meme: int = 6,
                   liq_floor: float = QUALITY_LIQ_FLOOR,
                   turnover_floor: float = TURNOVER_FLOOR) -> list[dict]:
    """End-to-end quality/risk tiering: enrich -> gate -> tier by CMC rank."""
    return tier_by_quality(candidates(rows, liq_floor, turnover_floor),
                           n_anchor, n_mid, n_meme)
