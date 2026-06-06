# Token Universe

How the competition's **149 eligible BEP-20 tokens** were narrowed to a **20-token,
risk-tiered tradeable set**, and the theory behind every step. The pipeline lives in
`src/trader/data/` (findings in [[Simulated Market]]); the decision theory that *trades*
this universe is in [[Trading Strategies]]; regime context in [[Market Conditions]].

## The selection pipeline

Four stages — each its own module, each its own finding:

1. **Screen** — `trader.data.dexscreener` (keyless). For every eligible symbol, find its
   deepest BSC pool and read real on-chain **liquidity, 24h volume, turnover, age**.
2. **Resolve** — `trader.data.cmc` (`CMC_API_KEY`). Map each symbol → canonical CMC id →
   **BNB Smart Chain contract**. Fixes the 35% ticker-collision ambiguity.
3. **Gate** — `trader.data.goplus` (keyless). Screen each contract for **structural
   rug/honeypot risk**; hard-veto the catastrophic ones.
4. **Tier** — `trader.data.select`. Rank by **CMC rank** (establishment) into a risk
   spectrum; take the most-liquid names within each tier.

(This supersedes the original `cmc_history`/BscScan sketch — see [[Tech Stack]] for the
as-built, mostly-keyless data sources.)

## Theory 1 — rank by turnover, not liquidity (liquidity ≠ safety)

The intuitive "more liquidity = safer/better" is **inverted on BSC.** Ranking the universe
by pool liquidity surfaces *parked, fake* pools, not quality:

| token | liquidity | 24h volume | turnover |
|---|--:|--:|--:|
| KOGE | $54.7M | $200k | **0.4%** |
| DUCKY | $36.6M | $105k | 0.3% |
| ETH (Binance-Peg) | $16.0M | **$93** | **0.0%** |
| TRX | $3.45M | $349 | 0.0% |
| — vs — XRP | $1.24M | $621k | **50%** |

**ETH on BSC trades $93/day against $16M of liquidity** — a *held peg*, not a trading
vehicle; trying to trade it is instant slippage death. 22 of 78 tradeable tokens had <5%
turnover — parked, facade liquidity. So selection ranks on **real 24h volume + turnover
(volume ÷ liquidity)**, with liquidity only an *exitability floor*. The parked majors
(ETH, TRX, UNI) are correctly excluded.

Caveat: turnover can be *wash-traded* too. Turnover filters **parked** liquidity; the
forensic gate (below) filters **manufactured** volume — two different lies.

## Theory 2 — risk-tier by CMC rank, not by anything price-based

Because liquidity inverts on BSC (the most-liquid tokens are memes; the established
blue-chips that *do* trade have only modest liquidity), pool depth is a poor risk proxy.
**CMC rank** is the clean objective signal for **establishment / rug-survival**: a top-60
token (even an established meme) has a longer track record and lower abandonment risk than
a rank-3000 newcomer. So:

- **🟢 Anchor** (rank ≤ 60) — established majors, the low-risk leg.
- **🟡 Mid** (rank 60–200) — established midcaps.
- **🔴 Meme** (rank > 200 / unranked) — new, high-risk, but still liquid.

Within each tier we take the **most-liquid** names, so every pick is tradeable in its risk
class. Tradeability gate: turnover ≥ 0.05, liquidity ≥ $50k (relaxed so traded
modest-liquidity majors like LTC/XRP/LINK qualify), non-stale, non-stablecoin. This gives
the deliberate **"ranging in risk factors"** spread — rank-6 XRP to rank-450 COAI — rather
than a bucket of whatever is momentarily most liquid.

## Theory 3 — the forensic survival gate (structural risk is the real tail)

On these tokens the dominant risk is **not drawdown — it's −100% in a single block**: a rug
(LP pulled), a honeypot (buy but can't sell), a mint-and-dump, a blacklist freeze.
Price-based risk metrics (vol, VaR, max-drawdown) **cannot see this** — it's a *contract*
property. With a hard 30% drawdown DQ, **one rugged position ends the entire entry.** So
every token passes a GoPlus structural screen before it can trade:

- **Hard veto:** honeypot, cannot-sell, hidden owner, owner-reclaimable, blacklist
  function, self-destruct, extreme tax.
- **Weighted warn:** mintable, upgradeable proxy, modifiable tax, owner concentration,
  unlocked LP (scored *only* for low-holder tokens — established pegs hold LP unlocked via
  market-makers, which is normal, not a rug).

**Live result:** the gate removed **BAS** (hidden owner) and **FORM** (blacklist function).
The "mintable" warns on the Binance-Peg majors (XRP, LINK, ADA, LTC, ZEC) are *benign* —
pegged tokens are mintable by the bridge operator, by design. See [[Security and Encryption]].

## The locked 20 (2026-06-06)

Three swaps from the raw auto-selection: **SHIB→LTC** (manual — SHIB untradeable at
$17k/day), **BAS→TAC** and **FORM→SFP** (forensic). All forensically clean (0 block):

| Tier | Tokens (CMC rank) |
|------|-------------------|
| **🟢 Anchor** | ASTER (42) · ZEC (15) · XRP (6) · XAUt (31) · LINK (17) · ADA (14) · LTC (25) |
| **🟡 Mid** | SKYAI (118) · SIREN (72) · FF (126) · BANANAS31 (180) · B (121) · TAG (163) · SFP (169) |
| **🔴 Meme** | BabyDoge (348) · UB (216) · HUMA (304) · COAI (450) · Q (284) · TAC (261) |

Stablecoins (USDT/USDC/…) are **excluded from the trading set** but reserved as the
risk-off "cash" leg for the drawdown-defense overlay ([[Market Conditions]]).

## Watch items / caveats

- **Thin-holder tokens:** XAUt (916 holders), TAC (~2k), HUMA (~4k) — tradeable but
  concentration risk; cap position size.
- **Peg-mintable is benign**, but it means the anchor majors are Binance-custodied wrappers,
  not the native asset — fine to trade, relevant for any on-chain reasoning.
- **History depth = listing age, ≤ 6 months** (GeckoTerminal); newer tokens have weeks, not
  months — reinforces a shared/universal policy over per-token models ([[AI Training]]).
- **Re-screen before the live window.** Liquidity, turnover, and forensic status drift;
  `scripts/select_universe.py` and `scripts/forensics.py` are re-runnable and the downloader
  is resumable, so re-locking near June 22 is cheap.

## Reproduce

```
python scripts/screen_universe.py                          # 149 -> DexScreener screen
python scripts/resolve_contracts.py                         # -> canonical BSC contracts (CMC)
python scripts/select_universe.py --exclude SHIB,BAS,FORM --pin LTC:anchor   # -> data/selection.json
python scripts/forensics.py                                 # -> data/forensics.json (GoPlus gate)
python scripts/download_ohlcv.py --selection data/selection.json --timeframes day,hour
```
