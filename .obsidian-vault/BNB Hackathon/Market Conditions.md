# Market Conditions

The honest market-context layer for the [[Project Overview|agent]]: what regime the live
week is in, how to read it from the available data, and how that reading becomes a *risk
overlay* rather than a return-chasing signal. How regimes drive entries/exits belongs to
[[Trading Strategies]]; backtesting across regimes and the metrics suite belong to
[[Simulated Market]]; live PnL/drawdown tracking belongs to [[Real-time Monitoring]].

## Why this note exists — the single-week problem

The competition is scored on **one held-out week (June 22–28)**. We do not know which regime
will occur, and we cannot pick the window. Whatever happens — a trend, a chop, a vol spike, a
liquidation cascade — we are hostage to it. Two consequences drive everything below:

1. **Regime-robustness beats regime-fitting.** A strategy tuned to last month's regime can
   score zero (or DQ) if the live week differs. We optimize for *surviving any plausible
   regime*, not maximizing return in the most likely one.
2. **The ~30% max-drawdown gate is a hard DQ, not a dent.** A single adverse regime can
   disqualify the entry outright regardless of headline PnL ([[Project Overview]] scoring).
   So regime detection is first and foremost a **drawdown-defense** tool.

A one-week ranking is high-variance by construction. We treat risk-adjusted *survival* as a
first-class objective alongside return, and we say so plainly rather than implying the
backtest "edge" will reproduce live.

## What "market conditions" means here

Four distinct, separately-measurable axes. They are not interchangeable; conflating them is a
common source of false signal.

| Axis | Question it answers | Primary source |
|------|--------------------|----------------|
| **Macro regime** | Bull / bear / chop across the market? | `cmc_market` (global metrics), BTC/ETH trend via `cmc_history` |
| **Volatility regime** | Calm, normal, or stressed? Vol expanding or contracting? | `cmc_history` (realized vol from OHLCV) |
| **Liquidity regime** | Can *this token* be traded at acceptable cost? | BscScan pool depth + on-chain volume; `simulate_trade` slippage |
| **Sentiment / positioning** | Crowded? Fearful? Funding stretched? | Fear & Greed, funding rates, derivatives positioning via CMC |

The hackathon examples explicitly cite this surface: an agent combining **funding rates +
Fear & Greed**, and **regime detection that switches strategy based on derivatives
positioning** ([[BNB Hack - AI Trading Agent Edition]]). The data exists; the discipline is in
not over-reading it.

## Data sources — which surface gives what

- **`cmc_market`** — latest quote + global market metrics (total cap, dominance, 24h volume).
  The macro-regime read. CLI analog: `cmc metrics`, `cmc markets`.
- **`cmc_history`** — OHLCV candles. The substrate for **realized volatility**, trend slope,
  and drawdown-from-recent-high per token. Note: **hourly / sub-hourly intervals may be
  tier-gated** on the CMC plan (the CLI flags hourly + 5m as paid-plan features). Confirm our
  tier before assuming intraday granularity — see Open questions.
- **`cmc_token_info`** — profile + chain stats per token; context for thin/illiquid names.
- **`cmc_news`** — news/trending feed; a coarse event filter. Breaking-event scanning is
  deferred to [[Social Media Scanner]] (`social_scan`).
- **Derivatives positioning / funding rates** — surfaced by the CMC Agent Hub (CLI analog:
  `cmc pairs <asset> --category derivatives`, `cmc top-gainers-losers`, `cmc trending`). These
  are **CEX-derived** — see the gap section below.
- **BscScan tools** (`bscscan_wallet_txs`, `bscscan_transfers`, `bscscan_token_holders`) —
  the *on-chain* truth for BSC: actual transfer flow, holder concentration, and the raw
  material for real liquidity assessment on the tokens we actually trade.

## Regime detection as a risk overlay

The overlay's job is to modulate **exposure and sizing**, not to generate alpha. Concretely,
the regime read should set:

- **Position size** — smaller in high-vol / stressed regimes; the drawdown gate makes
  oversizing into a volatile regime the single biggest DQ risk.
- **Gross exposure / cash buffer** — how much capital is deployed vs held back (subject to the
  rule that an hour starting ≤ $1 scores 0%, so we never fully de-risk to dust).
- **Strategy selection** — e.g. momentum logic only when a trend regime is confirmed; mean-
  reversion or stand-down in chop. The *signals* for this live in [[Trading Strategies]].
- **Per-token gating** — refuse or shrink trades in tokens whose liquidity regime can't absorb
  the size at acceptable slippage (`simulate_trade` is the gate).

A useful framing: the overlay's first output is "how much can I afford to be wrong right now?"
— and it feeds the **hard drawdown stop** in `risk/`, which is enforced in code, not by the
model ([[MCP Server]] safety tiers).

### Crude, robust regime buckets (a starting taxonomy)

Prefer a few stable, explainable buckets over a finely-tuned classifier that overfits one
month of history:

| Regime | Rough read | Overlay response |
|--------|-----------|------------------|
| **Trend (up/down)** | Sustained directional drift, vol normal | Allow directional strategy, normal size |
| **Chop / range** | No net drift, mean-reverting | Reduce size or stand down; avoid whipsaw fees |
| **High-vol / stress** | Realized vol spike, Fear extreme, funding stretched | Cut size hard, widen slippage tolerance, protect drawdown |
| **Thin / illiquid (per token)** | Low on-chain depth/volume | Gate or shrink trades on that token specifically |

Boundaries between buckets are themselves uncertain; a regime classifier that flips state
every few hours will churn fees and likely hurt. Hysteresis (slow to switch) is the
conservative default.

## Liquidity is token-specific, not market-wide

The 149-token eligible list ([[BNB Hack - AI Trading Agent Edition]]) is dominated by thin
BEP-20 names. Market-wide "liquidity is fine" tells us nothing about whether a given token can
absorb our trade. This is a **per-token, per-hour** property, read from BscScan pool depth and
on-chain volume — not from CMC's aggregate CEX volume, which can be an order of magnitude
larger than the BSC DEX pool we actually route through. The honest liquidity number is the one
`simulate_trade` returns against real pool depth.

### Empirical: liquidity ≠ safety on BSC (2026-06-06 spike)

Measured across the eligible universe, **liquidity magnitude is actively misleading** as a
quality or tradeability signal — it surfaces *parked* pools. The honest read is **turnover**
(24h volume ÷ liquidity):

| token | liquidity | 24h volume | turnover |
|---|--:|--:|--:|
| KOGE | $54.7M | $200k | **0.4%** |
| ETH (Binance-Peg) | $16.0M | **$93** | **0.0%** |
| TRX | $3.45M | $349 | 0.0% |
| — vs — XRP | $1.24M | $621k | **50%** |

**ETH on BSC trades $93/day against $16M of liquidity** — a held peg, untradeable. 22 of 78
tradeable tokens showed <5% turnover (facade liquidity). Consequence for regime/risk: the
*liquidity regime* of a token must be read from **turnover and real DEX depth**, never pool
size — and the most-liquid BSC tokens are memes, not blue-chips, so liquidity also fails as a
risk proxy (risk-tiering uses **CMC rank** instead). See [[Token Universe]]. The factor read
that drives entries (**BTC + BNB** betas / residual) lives in [[Trading Strategies]].

## The CEX-signal vs on-chain-reality gap

A recurring trap worth stating explicitly:

- **CMC funding rates, derivatives positioning, and aggregate volume are CEX-derived.** They
  describe the global / perps market, which is informative for *macro and sentiment regime*
  but **does not describe the BSC DEX liquidity** our trades execute against.
- A token can look liquid on CMC (high CEX volume) and be untradeable on BSC at our size. The
  reverse also happens. **CMC for regime/sentiment context; BscScan + `simulate_trade` for
  the cost of the actual fill.** Never let a CEX signal authorize a trade whose on-chain cost
  hasn't been checked.

## What is measurable vs noise — be a skeptic

- **Realized volatility** from OHLCV: measurable and reasonably robust, granularity permitting.
- **Trend / drift**: measurable but regime-fragile; a trend signal fit to last month is the
  classic overfit. Validate across multiple historical windows in [[Simulated Market]].
- **Fear & Greed**: a coarse, lagging sentiment gauge. Useful as a slow-moving filter, weak as
  a timing signal. Treat extremes as size modifiers, not entry triggers.
- **Funding rates / derivatives positioning**: meaningful for crowding/squeeze risk, but CEX-
  scoped (see gap above) and noisy at short horizons.
- **On-chain flow (BscScan)**: the most *honest* signal for BSC reality, but sparse and noisy
  on thin tokens — easy to mistake one whale's transfer for a "flow regime."

Default posture: **assume a candidate regime signal is noise until a backtest across several
distinct historical windows shows it survives costs applied equally to strategy and
baseline.** One flattering window is not evidence.

## Open questions

- **Intraday data granularity / tier.** Are hourly (and finer) `cmc_history` intervals
  available on our CMC plan, or daily-only? Scoring is hourly, so intraday vol/regime reads
  may be limited by tier. Confirm before building hour-scale regime logic.
- **Funding-rate / derivatives coverage on eligible tokens.** Most of the 149 BEP-20 names
  have no liquid perps market; funding/positioning signals likely exist only for the majors
  (ETH, a few large caps). Verify which tokens actually carry these fields.
- **On-chain liquidity read latency.** Is BscScan pool/volume data fast and cheap enough to
  drive a per-trade liquidity gate in the live loop? (Shared blocker — [[Real-time Monitoring]].)
- **Regime-label ground truth.** We have no labeled regimes; any classifier is unsupervised
  and unvalidated against a held-out future. Keep buckets crude and explainable.
- **Single-week variance — quantified (2026-06-06).** Answered for the baselines via 7-day-window
  resampling (`trader.sim.resample`): a passive low-turnover book over the eligible 20 has a
  **median weekly return ~+0.7%** (p5 −9%, p95 +18%), **median weekly max-drawdown ~7%**, and
  **P(DQ)≈0%** over a week — the DQ gate is *not* the weekly binding constraint for a diversified
  low-turnover book (the 62%/34% drawdowns seen earlier were 7-*month*, not weekly). The week is
  ≈ a coin-flip; the leaderboard reward lives in the **upper tail** — see the [[Trading Strategies]]
  tournament objective. Caveat: bull-conditioned sample; a bear live week shifts this down.
