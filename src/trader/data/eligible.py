"""Competition eligible-token universe (Track 1).

The BEP-20 symbols listed on CoinMarketCap for the BNB "AI Trading Agent Edition"
(149 in the rules prose; deduped here). Source: vault note
"BNB Hackathon/BNB Hack - AI Trading Agent Edition".

The rules give **symbols only** — no contract addresses — so symbol -> BSC-contract
resolution happens downstream (see `scripts/screen_universe.py`). The prose also
contains an exact duplicate (``SLX`` twice); ``dict.fromkeys`` dedupes while
preserving order. For production, resolve canonically via CoinMarketCap contract
addresses rather than symbol search.
"""

_RAW = [
    "ETH", "USDT", "USDC", "XRP", "TRX", "DOGE", "ZEC", "ADA", "LINK", "BCH",
    "DAI", "TON", "USD1", "USDe", "M", "LTC", "AVAX", "SHIB", "XAUt", "WLFI",
    "H", "DOT", "UNI", "ASTER", "DEXE", "USDD", "ETC", "AAVE", "ATOM", "U",
    "STABLE", "FIL", "INJ", "币安人生", "NIGHT", "FET", "TUSD", "BONK", "PENGU", "CAKE",
    "SIREN", "LUNC", "ZRO", "KITE", "FDUSD", "BEAT", "PIEVERSE", "BTT", "NFT", "EDGE",
    "FLOKI", "LDO", "B", "FF", "PENDLE", "NEX", "STG", "AXS", "TWT", "HOME",
    "RAY", "COMP", "GWEI", "XCN", "GENIUS", "XPL", "BAT", "SKYAI", "APE", "IP",
    "SFP", "TAG", "NXPC", "AB", "SAHARA", "1INCH", "CHEEMS", "BANANAS31", "RIVER", "MYX",
    "RAVE", "SNX", "FORM", "LAB", "HTX", "USDf", "CTM", "BDX", "SLX", "UB",
    "DUCKY", "FRAX", "BILL", "WFI", "KOGE", "ALE", "FRXUSD", "USDF", "GOMINING", "VCNT",
    "GUA", "DUSD", "SMILEK", "0G", "BEAM", "MY", "SLX", "SOON", "REAL", "Q",
    "AIOZ", "ZIG", "YFI", "TAC", "lisUSD", "CYS", "ZAMA", "TRIA", "HUMA", "PLUME",
    "ZIL", "XPR", "ZETA", "BabyDoge", "NILA", "ROSE", "VELO", "UAI", "BRETT", "OPEN",
    "BSB", "TOSHI", "BAS", "ACH", "AXL", "LUR", "ELF", "KAVA", "APR", "IRYS",
    "EURI", "XUSD", "BARD", "DUSK", "SUSHI", "PEAQ", "COAI", "BDCA", "XAUM",
]

# Order-preserving dedupe (drops the duplicate ``SLX``).
ELIGIBLE_SYMBOLS = list(dict.fromkeys(_RAW))

# Stable-ish assets in the universe — the risk-off / "cash" leg, NOT directional
# trades. Used to keep capital deployed under the drawdown gate (see vault
# "Market Conditions" / "Trading Strategies").
STABLES = {
    "USDT", "USDC", "DAI", "USD1", "USDe", "USDD", "TUSD", "FDUSD", "USDf",
    "USDF", "FRAX", "FRXUSD", "DUSD", "lisUSD", "XUSD", "EURI", "STABLE",
}


def eligible_symbols(include_stables: bool = True) -> list[str]:
    """The eligible universe, optionally excluding the stablecoin leg."""
    if include_stables:
        return list(ELIGIBLE_SYMBOLS)
    return [s for s in ELIGIBLE_SYMBOLS if s not in STABLES]
