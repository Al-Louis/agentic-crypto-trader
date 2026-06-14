# Simulated Market

The offline environment where strategy logic is validated **before any live capital is at
risk** — a broker that refuses to flatter the agent, leakage defended at two layers, and an
honest metrics suite tied to the competition's risk gate. Owned with [[AI Training|rl-ml-trainer]].
Strategy definitions live in [[Trading Strategies]], regime/scenario context in
[[Market Conditions]], live execution and custody in [[Security and Encryption]].

## Why this exists

CLAUDE.md mandates: *validate offline before live capital*; the decision core is a **pure,
testable module** exercised against recorded/simulated data. The thesis carried over from
[[TradeSim]] (see [[apentic_tradesim_case_study]]) is blunt — **trading environments are
built to fool you.** One line of look-ahead leakage, a reward that pays off luck, or a
simulator that quietly under-charges costs all hand you a beautiful backtest for a strategy
that loses real money on June 22. The engineering problem is not the model; it is building a
sim *honest enough that the numbers mean something*. A one-week live ranking is also
high-variance, so the sim's job is to estimate risk-adjusted survival, not just a headline
return that may be noise.

This work is the 🟡 SIMULATE tier of the [[MCP Server]], shipping in **Phase 4**. It is
strictly offline; it does **not** satisfy the June 16 PoC gate, which requires a real
on-chain trade ([[Tech Stack]]).

## What "honest" means — the reusable TradeSim broker

The broker/portfolio from [[TradeSim]] ports largely intact. Its non-negotiable properties:

| Mechanism | What it prevents |
|-----------|------------------|
| **Next-candle-open execution** (one-candle delay) | Same-candle look-ahead — the agent can never trade on the bar it is being judged against |
| **Cost model** (slippage + taker fee) | A sim that under-charges and reports phantom edge |
| **2% dead-zone** vs micro-rebalancing | Fee-churning the cost model into oblivion |
| **Minimum trade size** | Unrealistic infinitesimal fills |
| **Forced session-close / mark-to-market** | Unbooked paper gains masking true exposure |

The **portfolio** layer tracks cash, position, mark-to-market value, peak/drawdown, realized
and unrealized PnL, and **FIFO trade records** (the basis for win rate and profit factor).

### Two-layer leakage guard — the single most important discipline

1. **`validate_no_lookahead()`** — recomputes every indicator on truncated history
   `df[:i+1]` and asserts the value at row `i` matches the full-series value. Proves features
   are causal. Runs inside `prepare_dataset`; a failure blocks the dataset.
2. **Next-candle execution** (above) — structural, independent of (1).

Leakage is *tested and structurally prevented*, not assumed away. Any new BSC indicator must
pass layer (1) before it enters a feature set.

## Baselines through the same broker

Any reported edge must come from **decisions, not a rigged sim** — so identical costs apply
to the agent and to every baseline. `run_baseline` runs each through the *same* broker:

- **Buy & Hold** — the honest bar; on the eligible list this is often a strong, low-effort
  competitor and the first thing to beat.
- **SMA crossover**, **RSI mean-reversion** — naive technical baselines.
- **Random** — the sanity floor; underperforming it is a documented failure mode
  (`diagnose_run`, [[MCP Server]]).

A strategy that beats Buy & Hold on net return but loses on Sharpe, or only wins inside one
cherry-picked window, has not cleared the bar. The skeptic's default is *overfit until shown
otherwise*.

## Metrics suite and the competition risk gate

`backtest_report` emits the full panel on a **held-out test period** (train/validation/test
split; no tuning on test):

- Return: total, annualized · **Sharpe, Sortino, Calmar**
- **Max drawdown and its duration** · 95% **VaR / CVaR**
- **Win rate** (FIFO round-trips) · **profit factor** · **fee drag**

These are not generic — they must mirror exactly how the live week scores ([[BNB Hack - AI Trading Agent Edition]]):

- **Drawdown DQ gate (~30%, hard).** Model max-drawdown in-sim as a *disqualifier*, not a
  soft penalty: a run that breaches the cap scores **zero**, regardless of return. Report the
  full drawdown *distribution* across scenarios and the probability of breach, since a single
  live week is a small sample. Calmar (return ÷ maxDD) is the headline risk-adjusted number.
- **Hourly ≤$1 → 0% rule.** Returns are measured **hour by hour**, and any hour beginning
  with portfolio value ≤ $1 records 0% for that hour. The sim must compute returns on the
  **same hourly grid** and flag dust-out, so a strategy that drains capital is penalized
  in-sim exactly as live. (TradeSim's continuous, no-episode-boundary simulator — equity
  compounding across days — is the right base for this.)
- **≥1 trade/day** activity floor — the sim should assert it over any candidate live window.

## Data pipeline → causal feature set

`prepare_dataset` builds a clean, causally-validated dataset per eligible token:

1. **Resolve & screen** — map each `eligible_tokens` symbol to its canonical BSC contract
   (CMC contract address; **symbol search alone is unreliable — see below**), then screen
   on-chain via **DexScreener** (liquidity, 24h volume, turnover, pool age).
2. **Source OHLCV** via **GeckoTerminal** (CoinGecko on-chain), keyed by the token's
   deepest-liquidity pool address — `day`/`hour`/`minute`, ≤1000 candles/request, ~6mo
   history. Replaces TradeSim's ccxt/CEX ingestion and the earlier `cmc_history` plan
   (DexScreener has **no** OHLCV; CMC history is tier-gated and CEX-centric). **BscScan
   swap-event reconstruction** is the unlimited-depth / forensic fallback.
3. **Clean** — duplicate removal, gap handling, OHLC-consistency, flash-move flagging.
4. **Enrich** — ~28 technical indicators (the TradeSim registry) plus BSC second-order
   features (BTC/BNB residual, lead-lag, liquidity), each passing the causal test.
5. **Store** — Parquet, partitioned (token/interval/period), as in [[TradeSim]]; the
   download is **cached and resumable** (GeckoTerminal rate-limits hard — see below).

### Data sourcing — spike findings (2026-06-06)

A screening spike (`scripts/screen_universe.py`, `trader.data.dexscreener` /
`trader.data.geckoterminal`) ran the full eligible list and validated the stack end to end
(`data/universe_screen.*`). Headline results:

- **Coverage.** 130/148 symbols resolved to a BSC pair; 88 tradeable (liq ≥ $50k); 15
  stables. 18 unresolved/error (single-letter tickers `M`/`H`/`U`, `LTC`, `BARD`, …).
- **Resolution is unreliable — 35% ambiguous.** 46/130 had a runner-up pair within 25%
  liquidity (same ticker, different contract). `DOGE` resolved to a dead pair ($24/day
  volume); even `ETH` was ambiguous. **→ CMC contract-address resolution is mandatory** in
  production; symbol search is a screening heuristic only.
- **Rank by turnover, not liquidity.** Liquidity-magnitude ranking surfaces *fakes*: KOGE
  ($54.7M liq, **0.4%** turnover), DUCKY ($36.6M, 0.3%), a price-frozen SMILEK, and the dead
  DOGE pair all rank *above* ETH (turnover 326%). 22/78 tradeable tokens show <5% turnover —
  parked/facade liquidity. **Selection ranks on real 24h volume + turnover (vol/liq), with
  liquidity as an exitability floor**; the forensic gate ([[Security and Encryption]];
  `twak risk` / `check_token_security`) then filters wash-traded volume. Turnover filters
  *parked* liquidity, forensics filters *manufactured* volume — two different lies.
- **History source confirmed.** GeckoTerminal daily returns a clean ~181-day (6-month)
  series for established tokens; depth = `min(6mo, listing age)`, so newer tokens have weeks,
  not months — reinforcing the shared/universal policy in [[AI Training]].
- **1-minute is available but sparse** — only minutes-with-trades exist (a liquid token
  returned ~400 traded-minutes/day of a possible 1,440). Front-run/sweep micro-features
  ([[Trading Strategies]]) are feasible only on the liquid subset and must handle gaps.
- **Rate limit is the binding constraint.** The free tier returns HTTP 429 after ~6–8 rapid
  calls regardless of the documented 30/min, so bulk history needs a **slow, resumable,
  cached (Parquet)** downloader with backoff — a one-time cost. A keyed CoinGecko plan
  removes the bottleneck if speed is needed.

`simulate_trade` ([[MCP Server]], custody side) is the bridge to live: a dry-run of a single
trade returning projected fill, route, slippage, cost, and a guardrail pass/fail — the same
cost assumptions the backtester uses, applied to one prospective order.

## The CEX → on-chain porting gap (the key new work)

TradeSim's broker is calibrated to **CEX candles (Binance/Bitstamp)** with **volume-scaled
slippage** against deep order books. The competition executes **on-chain on BSC via DEX
aggregators (Amber / Rango)** against **AMM pools** — often *thin*. Reusing the CEX cost
model unchanged would produce a dishonest, edge-flattering backtest. What must be rebuilt:

| TradeSim (CEX) | This project (on-chain BSC) |
|----------------|-----------------------------|
| Volume-scaled slippage on a deep book | **AMM / pool-depth slippage** — fill cost is a function of trade size vs pool reserves (constant-product price impact), which is severe on thin eligible tokens |
| Implicit/zero on-chain cost | **Gas** per swap, modeled explicitly (cheap on BSC but non-zero, and it compounds against the ≥1-trade/day floor) |
| Single-venue taker fee | **Aggregator routing** + DEX LP fee + price impact; the relevant fill is the *aggregator's quoted* fill, ideally fed from real quotes |
| Liquidity assumed ample | **Liquidity as a per-token property** — many of the 149 eligible BEP-20s are thin; pool depth must gate position size and be a first-class sim input (see [[Market Conditions]]) |

The honest version calibrates slippage to **real BSC pool depth**, not CEX volume — this is
called out as an open constraint in [[Project Overview]]. The closer the sim's cost model
matches `simulate_trade`'s live quotes, the more the backtest predicts the live week.

## Reuse from TradeSim (2026-06-06 handoff analysis)

The `tradesim_handoff_seed/` package was analyzed for what ports here. The code is clean and
well-tested — but the **lessons are worth more than the code**.

**Ports ~clean** (into `src/trader/`): the **leakage guard + preprocessor**
(`validate_no_lookahead`, gap handling, session segmentation, flash/wick flags); the **metrics
suite** (Sharpe/Sortino/Calmar/maxDD+dur/VaR/CVaR/FIFO win-rate/profit-factor/fee-drag);
**benchmarks + backtester** (Buy&Hold/SMA/RSI/Random through the *same* broker); the
**indicator registry** + its **71-column feature schema** (incl. divergence features
`div_rsi/macd/obv` that map onto our residual/divergence edge — [[Trading Strategies]]); and
the `GroupedIndicatorExtractor` ([[AI Training]]).

**Adapt** (don't port verbatim): the **broker** (rebuild slippage as AMM price-impact — note
the seed's `default.yaml` still ships the *discredited* `volume_based` model); the **dataset /
episode index** (single-asset → **cross-sectional multi-asset**); the **reward** (rebuild
portfolio-level + ruin-aware, not the 8-layer accretion).

### The slippage lesson (sharpens the porting gap above)

TradeSim's worst data bug: **Bitstamp had 59% zero-volume 1-minute candles**, and volume-based
slippage on them produced *fantasy* $300K fills. Their fix was fixed-spread slippage at retail
size (impact ≈ 0 for liquid BTC). **Our case is the mirror image:** 1-minute DEX candles are
*sparse* (minutes-with-trades only) **and** the pools are *thin*, so price-impact is **real and
large** — we need a genuine AMM constant-product model **and** must **discard or down-weight
low-volume candles** so we never backtest fills that couldn't happen. Same failure class,
opposite remedy.

### The BTC slice — a reference asset, not the factor anchor

The seed's BTC_USDT 1m is **Sep 2024 – Apr 2025 only** (~8 months) and **does not time-overlap
our alt window** (~Nov 2025 – Jun 2026), so it **cannot** be the "Bitcoin-is-King" factor
anchor — that still needs a fresh **ccxt BTC + BNB** pull. Its value: a clean, indicator-rich
**offline dataset** to build and unit-test the ported broker / leakage-guard / backtester
against *before* the messy sparse DEX data, and a **feature blueprint** (its 71 columns).

## MCP tools (this note)

`prepare_dataset` → `run_backtest` → `backtest_report`, with `run_baseline` for the honest
comparison set and `simulate_trade` as the live-cost bridge. A `/workflows` loop can drive
*build dataset → backtest → report → diagnose → iterate* deterministically ([[MCP Server]]).

## Checkpoint replay — the cross-timeframe simulator (`scripts/simulate.py`, 2026-06-14)

Once policies are **persisted** (`policy.zip` + `vecnormalize.pkl`, from `e681c4d` onward — every
pre-2026-06-12 policy was lost on process exit), a saved checkpoint can be **replayed over arbitrary
windows** without retraining. This is the diagnostic that maps **where a model holds or breaks across
horizons** — the input to curriculum design (diagnose first, design second; never design blind).

- **It is the trainer's own eval, not a second code path.** `simulate.py` loads the policy from disk
  (CPU), reads the checkpoint's OWN provenance (`metrics.json`) to rebuild the **exact** env config it
  trained with (so obs shape / action space match), then grades each window through
  `train_event.evaluate_and_gate` — so a simulation's numbers are **identical** to a training-time eval
  over the same window, baselines included (rung-0, universe-matched Buy&Hold, Random).
- **Per timeframe** (6mo/3mo/1mo/1wk/1d — *not* 1yr; only ~5123 bars/~7mo of hourly data exist): a
  trailing N+warmup window (warmup served contiguously → tradeable from bar 0), its **own** voltopk
  universe picked at the window start (so the basket **evolves** across horizons), and an
  **in-sample/OOS label** (`oos_frac` = fraction after the train split's end — windows that overlap
  training look optimistic and must be flagged). Publishes one `kind:"portfolio"`, `simulation:true`
  bundle per timeframe ([[Apentic Data Contract]] §Simulation run).
- **Determinism makes this trustworthy:** the rdLe4 config reproduces bit-identically (val
  0.35299690480869833 to 17 decimals, three times), so a replayed checkpoint is the *same* agent, not
  an approximation.
- **Serving = precompute-to-CDN for presets, NOT an on-demand EC2 API.** The stack is static-JSON-on-
  CDN and the frontend already renders portfolio bundles, so a model+timeframe selector over
  precomputed bundles is zero new infra and no security surface. On-demand (arbitrary date ranges) is a
  v2 — and must **never** run on the custody/trading EC2 (it holds signing keys; no public surface).
- **First diagnostic (s0, the val one-trick):** outside its memorized val pocket it is a defensive
  underperformer — fails to ride bulls (its discretion destroys value vs holding the same risk-parity
  basket), loses to its own rung-0 rule OOS, churns in chop; only bear capital-preservation works.
  Full table → [[Experiment Log]]; curriculum implications → [[AI Training]].

### The weekly competition simulator (`scripts/simulate_weekly.py`, 2026-06-14)

The competition-faithful variant for the Apentic "Simulated Trades" dashboard (contract:
[[Apentic Data Contract]] §weekly; design in `.design-export-simulated/HANDOFF.md`). Each session is one
**Mon-00:00-UTC week**, fresh **$10k** (no cross-week compounding), with the **vol-top-8 universe +
risk-parity weights re-selected before each week** (so the basket evolves week to week). Published
per-model so the page can offer a model selector.

- **Exact PnL by construction — the LEDGER pattern.** The dashboard derives PnL itself from
  `qty*(exit-entry)`, but the env is notional/_px-index based and discards real per-coin prices, so
  reconstructing exact round-trips from markers is fragile. Fix: `EventRungEnv.token_pnls()` reports the
  exact per-token realized+open PnL, and the export snaps each asset's positions to it (recon $0.00 all
  weeks). **Lesson: don't infer PnL from reconstructed prices — carry the env's own ledger.**
- **The methodological finding (the reason this matters).** The weekly structure is *honest about
  deployment*, and it exposed that **evaluating in a continuous multi-week episode FLATTERS the agent**:
  s0's continuous-eval star (ZEC +$2,747) collapses under cold weekly sessions (ZEC trades 2/17 weeks,
  skips its big-move ignition by choice). The model overfit windows; it doesn't generalize. **Eval (and
  training) structure must match deployment** — a first-class requirement for the next training phase
  ([[AI Training]] §the-fork, [[Experiment Log]] §2026-06-14).

## Open questions

- **Pool-depth data source.** DexScreener now gives per-pair **liquidity (USD)** for free
  (validated in the spike) — a usable first cut for sizing. Reserve-level depth for precise
  AMM price-impact still TBD: CMC on-chain stats, BscScan, or the aggregator's quote endpoint.
- **Quote fidelity.** Can `simulate_trade` pull *real* Amber/Rango quotes offline, or must
  the sim approximate price impact from reserves? Determines how trustworthy costs are.
- **Hourly grid vs candle interval.** Scoring is hourly; CMC history interval may differ.
  Confirm we can build a clean hourly return series matching the live metric.
- **Scenario selection.** Which historical windows / regimes form the test set so results
  aren't a single lucky period? Coordinate with [[Market Conditions]].
- **One-week variance.** Quantify how much of any backtested edge survives a 7-day sample —
  bootstrap/block-resample the live window to size the confidence interval before trusting a
  ranking. Coordinate reward/eval with [[AI Training]].
