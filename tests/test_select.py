"""Tests for the universe selection logic (pure, no network)."""

from trader.data import select as sel


def _row(symbol, liq, vol, vp=1.0, status="resolved", stable=False, amb=False):
    return {
        "symbol": symbol, "status": status, "liq_usd": liq, "vol_h24": vol,
        "vol_proxy": vp, "is_stable": stable, "ambiguous": amb,
        "pair_address": f"0x{symbol}",
    }


def test_enrich_computes_turnover_and_flags():
    rows = [_row("A", 1_000_000, 500_000, vp=2.0, amb=True)]
    e = sel.enrich(rows)[0]
    assert e["turnover"] == 0.5
    assert e["stale"] is False
    assert e["needs_verification"] is True


def test_enrich_skips_unresolved():
    rows = [_row("A", 1, 1, status="unresolved"), _row("B", 1, 1, status="error")]
    assert sel.enrich(rows) == []


def test_is_candidate_gates():
    # parked liquidity (low turnover) rejected even with huge liquidity
    parked = sel.enrich([_row("KOGE", 54_000_000, 200_000)])[0]   # turnover ~0.0037
    assert sel.is_candidate(parked) is False
    # stale (vp=0) rejected
    stale = sel.enrich([_row("DEAD", 1_000_000, 500_000, vp=0)])[0]
    assert sel.is_candidate(stale) is False
    # thin (below liq floor) rejected
    thin = sel.enrich([_row("TINY", 50_000, 40_000)])[0]
    assert sel.is_candidate(thin) is False
    # stable excluded
    usd = sel.enrich([_row("USDT", 5_000_000, 5_000_000, stable=True)])[0]
    assert sel.is_candidate(usd) is False
    # genuine: deep + traded
    good = sel.enrich([_row("ETH", 9_000_000, 30_000_000)])[0]
    assert sel.is_candidate(good) is True


def test_select_tiers_and_counts():
    rows = []
    # 20 genuinely-traded tokens with descending liquidity
    for i in range(20):
        rows.append(_row(f"T{i:02}", liq=10_000_000 - i * 100_000,
                         vol=5_000_000, vp=float(i + 1)))  # vp>0 so none are stale
    chosen = sel.select(rows, n_major=7, n_mid=7, n_degen=6)
    assert len(chosen) == 20
    tiers = [r["tier"] for r in chosen]
    assert tiers.count("major") == 7
    assert tiers.count("mid") == 7
    assert tiers.count("degen") == 6
    # majors are the most liquid
    majors = [r for r in chosen if r["tier"] == "major"]
    assert max(r["liq_usd"] for r in chosen) == majors[0]["liq_usd"]


def test_risk_bucket_by_cmc_rank():
    assert sel.risk_bucket({"cmc_rank": 6}) == "low"      # major
    assert sel.risk_bucket({"cmc_rank": 120}) == "mid"    # midcap
    assert sel.risk_bucket({"cmc_rank": 450}) == "high"   # new/meme
    assert sel.risk_bucket({"cmc_rank": None}) == "high"  # unranked -> riskiest


def test_tier_by_quality_assigns_risk_tiers_by_rank_picks_by_liquidity():
    rows = [
        _row("XRP", 1_000_000, 500_000), _row("LINK", 700_000, 300_000),   # low
        _row("SKYAI", 9_000_000, 5_000_000), _row("SIREN", 8_000_000, 1_000_000),  # mid
        _row("BabyDoge", 7_000_000, 400_000), _row("COAI", 1_000_000, 500_000),    # high
    ]
    ranks = {"XRP": 6, "LINK": 17, "SKYAI": 118, "SIREN": 72, "BabyDoge": 348, "COAI": 450}
    cands = sel.enrich(rows)
    for r in cands:
        r["cmc_rank"] = ranks[r["symbol"]]
    chosen = sel.tier_by_quality(cands, n_anchor=2, n_mid=2, n_meme=2)
    by_sym = {r["symbol"]: r["tier"] for r in chosen}
    assert by_sym["XRP"] == "anchor" and by_sym["LINK"] == "anchor"
    assert by_sym["SKYAI"] == "mid" and by_sym["SIREN"] == "mid"
    assert by_sym["BabyDoge"] == "meme" and by_sym["COAI"] == "meme"


def test_select_drops_parked_and_stale():
    rows = [
        _row("ETH", 9_000_000, 30_000_000),         # keep
        _row("KOGE", 54_000_000, 200_000),          # parked -> drop
        _row("DEAD", 2_000_000, 1_000_000, vp=0),   # stale -> drop
        _row("USDT", 5_000_000, 5_000_000, stable=True),  # stable -> drop
    ]
    chosen = sel.select(rows)
    syms = {r["symbol"] for r in chosen}
    assert syms == {"ETH"}
