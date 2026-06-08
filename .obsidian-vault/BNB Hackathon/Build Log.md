# Build Log

Chronological record of what's been built and the decisions behind it. The authoritative
*why* lives in the linked topic notes; this is the timeline. See [[Index]] for navigation
and [[Project Overview]] for scope.

## 2026-06-05 — Foundation (Phase 1)

- CLAUDE.md, agent roster, `/orient`, repo/git init, Python `src/trader/` skeleton, `trader`
  MCP server stub.
- Vault: the 8 empty topic stubs developed into full notes.

## 2026-06-06 — Data layer + token universe

### Strategy theory (design discussions → [[Trading Strategies]])

Worked out the *edge* before the plumbing:

- **"Bitcoin is King" factor model** — `r_alt = α + β·r_btc + ε`; the residual ε is the
  idiosyncratic signal; two-factor BTC+BNB; time-varying/downside β; lead-lag.
- **Microstructure edges** — front-runners (alt leads BTC), stop-hunts (liquidity grabs),
  played as **resting orders at pre-computed prices** to beat on-chain latency.
- **Reflexivity / second-order** — indicators map where the crowd's orders sit; trade the
  reaction, not the indicator.
- **Adversarial-market thesis** — BSC tokens are dev-controlled, negative-sum; the edge is
  **risk discrimination**, not fearlessness.
- **Sequencing decision** — *training-first*: build the simulated-market / strategy core
  before the live execution layer. The June-16 internal milestone is reframed to "a working
  trained agent"; the live on-chain loop (TWAK signing, on-chain registration) is a separate
  later spike that still must land before **June 22**.

### Data layer (built, tested, committed)

- **Data sourcing validated and re-routed** (mostly keyless): **GeckoTerminal** (OHLCV) +
  **DexScreener** (screen) + **CMC** (contract resolution) + **GoPlus** (forensics).
  BscScan dropped — Etherscan unified to a V2 key whose **free tier is Ethereum-only; BSC is
  paid** ([[Tech Stack]]).
- `trader.data.downloader` — resumable, cached **Parquet** OHLCV backfill (per-page manifest
  checkpoint, exponential 429 backoff). Proven live + offline (crash-resume, 429 tests).
- `trader.data.cmc` — CMC contract resolver; **corrected 22 wrong pools** vs symbol-search
  (147/148 resolved, fixing 35% ambiguity).
- `trader.data.goplus` — forensic rug/honeypot gate; **removed BAS (hidden owner) + FORM
  (blacklist)**; resumable cache for GoPlus's flaky keyless tier.
- `trader.data.select` — turnover-ranked, CMC-rank-tiered selection with `--exclude`/`--pin`
  manual overrides.
- **Locked the 20-token universe** → [[Token Universe]].
- **OHLCV backfill** (daily + hourly) — **complete for all 20** (~181d daily, ~200d hourly,
  cached to Parquet). 1-minute subset for the liquid names is next.
- **46 tests passing.**

### TradeSim handoff analysis (`tradesim_handoff_seed/`)

Analyzed the prior project's lean handoff; verdict captured in [[Simulated Market]] /
[[AI Training]]:
- **Most valuable artifact = the lessons** — esp. *entry timing never beat random; exits /
  risk-management carried performance* — logged as a research question against our
  entry-centric edge thesis.
- **Ports clean:** leakage guard, metrics suite, benchmarks/backtester, indicator registry
  (+71-col feature schema), grouped-attention extractor.
- **Adapt:** broker (AMM slippage), dataset (→ multi-asset), reward (→ ruin-aware).
- **Data caveat:** the seed's BTC slice is Sep 2024–Apr 2025 only and does **not** overlap our
  alt window → still need a fresh ccxt BTC+BNB pull for the factor model.
- **Discipline to adopt:** real (tested) regime curriculum, fee-blind reward, benchmark gate
  before versioning, smoke test before full runs.

### Factor model + IC gate

- Pulled **BTC + BNB anchor** (ccxt / Binance.US, 0-gap 1m/1h/1d) — the factor data
  (`trader.data.anchor`).
- Ported the TradeSim **indicator pipeline** (71-col, leakage guard) + **metrics suite**;
  verified vs the stored BTC parquet to ~1e-9.
- Built the **two-factor residual model** (`trader.features.factor`): R²-classifier and
  BTC/BNB betas validated empirically (majors→BTC, ecosystem→BNB, XAUt uncorrelated).
- **IC gate refuted the residual-momentum *continuation* hypothesis** — negative IC at every
  horizon (mean-reversion, not continuation; ≈ naive momentum). Factor model → *risk* tool,
  not a selection alpha. Reinforces the post-mortem ([[Trading Strategies]]).

### Cost-aware backtest + 7-day resampling

- Built the **AMM cost broker** + **cross-sectional backtester** + **7-day-window resampler**
  (`trader.sim.{broker,backtest,resample}`).
- **Entry alpha is dead here**: momentum/reversal churn thin pools (200–290× turnover, >100%
  cost drag) and lose; the IC reversal is confirmed untradeable. Only **low-turnover** survives.
- **DQ is not the weekly binding constraint** for diversified low-turnover (P(DQ)≈0% over a
  week; the 62%/34% drawdowns were 7-*month*, not weekly). A week ≈ a coin-flip (median +0.7%).
- **Tournament reframing**: the prize rewards a top-5 finish, not the median — so optimize the
  **upper tail (P(big week) s.t. low P(DQ))**, not minimum variance ([[Trading Strategies]]).

### Upper-tail sweep + activity DQ

- Built the **upper-tail tournament sweep** (`scripts/tail_sweep.py`): rank static tilts by
  P(week > +15% AND not DQ'd).
- **Modeled the ≥1-trade/day rule** as a second DQ gate (`trader.sim.resample`) — **buy-and-hold
  is disqualified for inactivity** (P(DQ)=100%); strategies must rebalance ≥ daily.
- **Candidate found:** daily-rebalanced **`vol-top8`** (8 highest-vol tokens, equal-weight) — 26%
  contender rate at **1% P(DQ)**; volatility tilt ≫ beta tilt; daily rebalance also cuts
  drawdown, so compliance is free. ([[Trading Strategies]] tournament objective.)

- **OOS-validated the vol tilt** (`scripts/oos_validate.py`, 60/40 split): vol-rank persists
  (Spearman +0.66); train-selected vol-top8 **doubles the contender rate on held-out test
  windows** (42% vs all-20's 21%, 0% DQ), ~no skill lost OOS. The tilt is real.

### Regime overlay (BTC risk-on/off gate)

- Built `btc_risk_on` (close > trailing EMA) + `regime_gated` (hold tilt risk-on, cash risk-off);
  resampler now conditions outcomes on each window's BTC return.
- **Honest finding:** real insurance in bear weeks (halves drawdown, eliminates bear DQ) but
  **overpriced** in the bull-conditioned sample — cuts the tournament rate in half (27%→13%) and
  bull upside (+15%→+8%); the insured DQ is only ~2%. Sample has no real crash (under-values the
  insurance); the all-or-nothing 72h gate is too blunt. Stance: ungated vol-top8 = bull bet, gate
  = toggle-able insurance ([[Trading Strategies]]).

### In flight / next

- **Refined overlay** — partial de-risk (50% vs full cash) / hysteresis / extreme-stress-only
  gating, to keep upside while capping the tail. Plus synthetic-crash stress tests (the sample
  has no real crash).
- **1-minute data banked** (9/10 liquid tokens, ~182d; SIREN to re-fill; sparse on thin names,
  ~320–1,350 candles/day). Front-run/sweep features **deprioritized** — entry alpha is dead;
  available if we ever revisit micro-structure.
- **Walk-forward** OOS (multiple splits) for extra robustness.
- **BTC + BNB anchor series** (ccxt) for the factor model.
- Feature engineering → residual/factor model → [[Simulated Market]] broker → backtest.

## Phase status (vs [[Project Overview]] build path)

- ✅ **Phase 1** — Foundation.
- 🔄 **Phase 3/4** — Decision logic + offline validation: data layer + universe done; honest
  broker, features, and backtest are the active work.
- ⬜ **Phase 2** — Stack spike / live on-chain loop: **deferred** under the training-first
  plan; a focused execution+custody spike (TWAK dust trade, registration dry-run) is still
  required before June 22.
- ⬜ **June 16 PoC gate** — reframed internally to "trained agent"; the live-loop gate itself
  is not yet met (no on-chain trade landed).
