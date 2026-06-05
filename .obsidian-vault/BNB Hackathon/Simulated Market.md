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

1. **Source OHLCV** via `cmc_history` (CMC Agent Hub) for tokens from `eligible_tokens`.
   Replaces TradeSim's ccxt/CEX ingestion.
2. **Clean** — duplicate removal, gap handling, OHLC-consistency, flash-move flagging.
3. **Enrich** — ~28 technical indicators (the TradeSim registry), each passing the causal
   test.
4. **Store** — Parquet, partitioned (token/interval/period), as in [[TradeSim]].

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

## MCP tools (this note)

`prepare_dataset` → `run_backtest` → `backtest_report`, with `run_baseline` for the honest
comparison set and `simulate_trade` as the live-cost bridge. A `/workflows` loop can drive
*build dataset → backtest → report → diagnose → iterate* deterministically ([[MCP Server]]).

## Open questions

- **Pool-depth data source.** Where does live BSC reserve/liquidity data come from — CMC
  on-chain stats, BscScan, or the aggregator's quote endpoint? Needed to calibrate AMM
  slippage. Unverified.
- **Quote fidelity.** Can `simulate_trade` pull *real* Amber/Rango quotes offline, or must
  the sim approximate price impact from reserves? Determines how trustworthy costs are.
- **Hourly grid vs candle interval.** Scoring is hourly; CMC history interval may differ.
  Confirm we can build a clean hourly return series matching the live metric.
- **Scenario selection.** Which historical windows / regimes form the test set so results
  aren't a single lucky period? Coordinate with [[Market Conditions]].
- **One-week variance.** Quantify how much of any backtested edge survives a 7-day sample —
  bootstrap/block-resample the live window to size the confidence interval before trusting a
  ranking. Coordinate reward/eval with [[AI Training]].
