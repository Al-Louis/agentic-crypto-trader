# Trading Strategies

The decision core is an **open design space** — not yet committed. This note maps the candidate
strategy families, how each maps to available data surfaces, and the competition constraints that
shape every viable option. See [[Project Overview]] for the overall objective; [[Market Conditions]]
and [[Simulated Market]] for regime/backtest detail; [[AI Training]] for RL mechanics.
The tradeable token set and its selection theory live in [[Token Universe]].

---

## The edge thesis — beyond the indicators

Anyone can read a book and compute RSI/MACD. The market *reacts* to those indicators
precisely because everyone uses them — so a book-indicator strategy is a **crowded** one,
and crowded edges decay. The real edge is **second-order (reflexivity):** the standard
indicator is not a signal, it is a **map of where the crowd's orders are parked** (a
Schelling point); the edge is trading the *reaction* to those focal points. The developed
edges below are all **engineered features a learned core can weight** — RL won't discover a
10-minute lead-lag from raw candles, but it can learn to combine hand-built signals.

> **⚠ Honest prior (TradeSim post-mortem).** A prior RL project ground through 64 runs and
> found **entry timing never clearly beat random** on single-asset BTC — *exits / risk
> management* carried what little edge there was, and the honest ceiling was bull-only
> breakeven. Our edge here is **cross-sectional selection** (*which* alts), a different and
> better-documented claim than single-asset entry timing — but take the warning: **validate
> any entry edge against a Buy&Hold / cross-sectional-momentum baseline behind an honest gate
> before trusting it**, and weight real effort toward exits and the survival overlay. See the
> [[AI Training]] post-mortem.

### "Bitcoin is King" — a factor model
Each alt is modeled against the market driver: **`r_alt = α + β·r_btc + ε`**. β is how much
the alt amplifies BTC (β≈2 ⇒ BTC −3% → alt −6%). The residual **ε** — the part BTC does
*not* explain — is the **idiosyncratic signal**: a positive residual while BTC bleeds is
hidden strength (accumulation). Refinements: a **two-factor BTC + BNB** model (BNB is the
chain's reserve asset; R² measures how factor-driven vs dev-controlled a token is);
**time-varying, asymmetric *downside* β** (correlations → 1 in a crash — the number that
disqualifies you); and **lead-lag** (BTC leads alts by minutes — the catch-up is the
tradeable form). The cross-sectional **residual rank** across the universe is the *selection*
signal — which alts to hold.

> **⚠ Empirically tested (2026-06-06) — continuation refuted.** Built live (BTC+BNB anchor, 20
> alts, hourly), the model validates as a **risk** tool: R² cleanly separates factor-driven
> majors (LINK 0.56, ADA 0.49, XRP 0.45) from dev-driven memes (UB/Q/SIREN ≈0.03), and the
> two-factor split is real (majors load **BTC**, BSC-ecosystem load **BNB**; XAUt/gold
> uncorrelated). **But the residual-momentum *selection* hypothesis failed an IC gate:** its
> Information Coefficient is *negative* at every horizon (1h −0.044, t=−11; decays to
> insignificance by 72–168h) — recent idiosyncratic strength predicts *lower* forward returns
> (short-horizon **mean reversion**, not continuation), and it's indistinguishable from naive
> price momentum, so the factor decomposition adds **no selection edge**. The short-horizon
> reversal is statistically strong but suspect as tradeable (microstructure + sparse-data
> forward-fill) — pending the cost-aware broker. **Takeaway:** use the factor model for
> *risk / exposure* (beta sizing, BTC/BNB, regime), not as a standalone entry alpha.
> Reinforces the [[AI Training]] post-mortem.

### Front-runners — alts that lead BTC
Certain tokens *front-run* BTC's reversals (sell into the bottom, diverge, then turn up
minutes before BTC). Formally a **lead-lag with the alt leading** (cross-correlation /
Granger). Used as an **early regime-flip trigger** — one reliable front-runner rotates the
whole book risk-on a few minutes early. **Caveats (the most overfit-prone idea):**
rediscover blind in data, never encode remembered tokens; lead relationships **decay** →
test persistence; only multi-minute leads survive on-chain latency.

### Stop hunts — liquidity grabs
In high volatility, price is driven into the obvious liquidity pool (stops just below a
swing low) on a velocity spike, triggering a cascade, then snaps back. The tradeable object
is the **sweep-and-reclaim**; the reclaim is the *whole* signal (it separates a hunt from a
real breakdown). **Reactive execution loses the on-chain latency race** — so we play it as a
**resting limit order at a pre-computed sweep price** (derived from the alt's β to BTC),
which both beats the latency problem *and* is honestly backtestable.

### The unifying principle
The **regime read selects the mode:** trending + broad breadth → momentum/rotation; high-vol
+ at-support + velocity spike → mean-reversion (fade the flush). And the line for on-chain
realizability: **an edge expressible as a pre-computed price survives (resting order); an
edge that needs reacting to a live print does not.**

### Discipline — creative entries, conservative survival
Non-consensus alpha is higher-variance, and the 30% drawdown gate punishes variance with
*death*, not a dent — so the exotic edge rides on top of a survival-first sizing base. The
simulator's freedom is freedom to *explore*, not to *believe*: wild in hypothesis generation,
brutal at acceptance (out-of-sample across many week-slices, honest fills, persistence tests).
That discipline is what turns "intuition that goes against logic" into edge, not mirage
([[Simulated Market]]). And the market itself sets the posture: these are dev-controlled,
low-float, often wash-traded tokens — a **negative-sum** game after costs — so the edge is
not fearlessness but **superior risk *discrimination*:** bold on the recoverable moves that
scare retail, first out the door on the structural tail (rug/honeypot — gated in
[[Token Universe]] / [[Security and Encryption]]).

---

## Decision-core interface (architecture contract)

Strategy logic lives behind a single, swappable interface in `src/trader/strategy/`. The
surrounding layers — execution (TWAK), custody, and guardrails (`src/trader/risk/`) — are
strategy-agnostic. The interface contract:

- **Inputs:** structured market snapshot (price, volume, indicators, regime tag, on-chain
  signals, sentiment flags) assembled by the data layer from `cmc_market`, `cmc_history`,
  `cmc_token_info`, and `bscscan_*` tools.
- **Output:** a typed decision (`BUY | SELL | HOLD`, token, size fraction) — pure, deterministic,
  no side effects.
- **Offline testability:** the decision function is exercisable against recorded snapshots or
  simulated data ([[Simulated Market]]) before live capital is at risk. Strategies are registered
  via `register_strategy` and evaluated via `evaluate_strategy`, which runs `run_backtest` and
  returns a `backtest_report` including Sharpe/Sortino/max-drawdown vs baselines.

No strategy may bypass the guardrail layer (allowlist, per-trade/daily caps, slippage, drawdown
stop). See [[Security and Encryption]].

---

## Constraint envelope — what every strategy must survive

The scoring rules and execution realities impose hard limits before any return optimization matters:

| Constraint | Detail | Strategic implication |
|---|---|---|
| **30% max-drawdown DQ** | Hard disqualification gate (not a points deduction) | Survival is the first objective; position sizing and stop logic are non-negotiable |
| **≥1 trade/day, 7 days** | Minimum to qualify — a **hard activity DQ** | **Buy-and-hold is disqualified** (1 trade). Rebalance ≥ daily (or force a daily ping trade). Modeled as a second gate in `trader.sim.resample`; daily rebalancing also trims drawdown, so it's free here. |
| **Hourly ≤$1 → 0%** | Any hour starting below $1 scores zero | Keep capital deployed; avoid routing all funds into illiquid positions or excessive gas burn |
| **149 BEP-20 eligible tokens only** | Fixed allowlist; trades outside it don't count | Token universe is the hard filter for any signal or copy-trade target |
| **Thin-token slippage** | Many eligible tokens have low DEX liquidity; Amber/Rango aggregators route fills but can't manufacture depth | Prefer liquid names or size positions small enough that `simulate_trade` (slippage preview) passes before committing |
| **`twak swap` defaults to Ethereum** | Must pass `--chain bsc` explicitly on every swap call | An integration detail, but a live-capital bug if missed |

The risk-first posture implied here: **survive the week, then optimize return**. A strategy
that clips 5% while staying under 15% drawdown beats one that peaks at 40% and gets DQ'd.

> **⚠ Tournament objective + cost-aware backtest (2026-06-06).** Two empirical results sharpen
> this posture (`trader.sim.{broker,backtest,resample}`):
> - **Entry alpha is dead here; only low turnover survives.** Through the AMM cost broker,
>   cross-sectional momentum and reversal churn the thin pools to death (200–290× turnover,
>   >100% cost drag) and lose money; the IC-suggested reversal is **confirmed untradeable**.
>   Only **low-turnover** books (Buy&Hold, rebalanced equal-weight) survive costs.
> - **It's a leaderboard, so median return loses — optimize the upper tail.** A 7-day-window
>   resampling shows low-turnover books almost never breach the 30% gate over a *week*
>   (P(DQ)≈0%, median weekly maxDD ~7%), but a typical week is a coin-flip (median +0.7%, p5
>   −9%, p95 +18%). The prize rewards a **top-5 finish**, not the median, so **survival is
>   necessary but not sufficient**: the real target is **maximizing P(big positive week) subject
>   to a low P(DQ)** — which favors *some* concentration / upside variance, not minimum variance.
>   (Caveat: the sample is bull-conditioned; a bear live week raises DQ risk, so survival logic
>   stays as insurance.) See [[Market Conditions]] single-week variance.
> - **≥1 trade/day is a hard *activity* DQ — buy-and-hold is disqualified.** Modeled in
>   `trader.sim.resample` as a second gate (a strategy must rebalance ≥ daily; buy&hold trades
>   once → **P(DQ)=100%**). Conveniently a **daily rebalance also *improves* the risk profile**
>   (it trims winners), so compliance is not a tax here. **Current best candidate (compliant,
>   both gates): daily-rebalanced equal-weight of the ~8 highest-volatility eligible tokens
>   (`vol-top8`)** — a 26% chance of a >+15% week at only **1% P(DQ)** (p95 +40%); `vol-top5` is
>   the more aggressive sibling (26% / 9% DQ, median +5%). **Volatility tilt ≫ beta tilt.**
>   Pending out-of-sample validation + a regime overlay (the vol tilt is a bull bet).

---

## Candidate strategy families

These are not mutually exclusive and share the same execution/custody infrastructure.
The right choice — or combination — depends on validation results, not prior conviction.

### 1. Technical-indicator / momentum

**Logic:** entry/exit rules derived from price and volume indicators computed on `cmc_history`
candles. Classic candidates from the [[TradeSim]] indicator library (~28 TA indicators):

- **RSI** (overbought/oversold, divergence) — reliable in ranging conditions, noisy in trending.
- **MACD** (crossover, histogram slope) — trend-following; higher latency but fewer whipsaws.
- **Moving averages** (EMA crossovers, Golden/Death Cross) — simple, legible, widely robust.
- **Volatility regime** (ATR, Bollinger bandwidth) — gates whether momentum or mean-reversion
  rules fire. See [[Market Conditions]] for regime classification.

**Data surface:** `cmc_history` (OHLCV; hourly intervals require a paid CMC plan — confirm
availability) via the CMC Agent Hub MCP. `cmc_market` for real-time quote between candle
refreshes.

**Spec shape for validation:** parameterized thresholds (e.g. RSI oversold < 30, overbought > 70,
lookback window N, ATR-scaled stop at K×ATR). `evaluate_strategy` benchmarks against Buy&Hold,
SMA crossover, and random-trade baselines before any live use.

**Risks:** indicator signals computed on thin-token candles can be noisy. Confirm that hourly
`cmc_history` is available at the required plan tier before building on it.

### 2. Wallet-monitoring / copy-trade

**Logic:** monitor on-chain activity of known-profitable wallets via `bscscan_wallet_txs` and
`bscscan_transfers`; mirror qualifying trades through the agent's own risk filters. Holder
concentration (`bscscan_token_holders`) can flag early accumulation.

**Data surface:** BscScan REST (`bscscan_wallet_txs`, `bscscan_transfers`, `bscscan_token_holders`).
No real-time push — polling latency means fills will trail the copied wallet.

**Risks:** BSC on-chain data is public but noisy; identifying genuinely skilled wallets vs lucky
ones requires historical validation. Execution lag is structural — the agent is always second.
Thin-token fills may be worse than the source wallet's due to slippage. Feasibility of reliable
wallet-signal extraction should be confirmed with `onchain-custody-engineer` early.

### 3. Sentiment / news-driven

**Logic:** act on breaking signals — large price dislocations correlated with CMC news, Fear &
Greed extremes, or social heat — before they fully propagate. CMC Agent Hub provides funding
rates, Fear & Greed index, and `cmc_news`; the optional `social_scan` tool covers X.com/news
for breaking events.

**Data surface:** `cmc_token_info` (chain stats, social metrics), `cmc_news`, CMC Fear & Greed
(available via Agent Hub MCP), `social_scan` (Phase 3+). See [[Social Media Scanner]] for
scanning detail.

**Risks:** signal latency and false-positive rate are hard to quantify without a live feed test.
Sentiment signals work best as an overlay (e.g., suppress trading during extreme fear, or amplify
a technical signal when sentiment aligns) rather than as a standalone trigger.

### 4. Learned RL policy

**Logic:** a policy trained against the [[Simulated Market]] environment — reward shaped for
risk-adjusted PnL with an explicit drawdown penalty — decides position changes at each step.
`start_training`, `training_status`, `evaluate_model`, and `diagnose_run` drive the train →
evaluate → diagnose loop.

**Data surface:** same feature set as technical indicators + any on-chain/sentiment features
encoded as observations.

**Risks:** RL adds training time and an overfitting surface. The [[TradeSim]] curriculum/regime
work provides a starting point. An RL policy must pass `evaluate_model` on a held-out period
and clear `diagnose_run` checks (under-random, over-trading, fee drag, drawdown) before live
use. See [[AI Training]] for training mechanics.

### 5. Risk- / regime-aware overlay

**Logic:** not a standalone strategy but a meta-layer applicable to any of the above. A regime
classifier (trending / ranging / high-volatility) derived from ATR or Bollinger bandwidth tags
each time-step; strategies switch parameters or go flat based on the tag. Position sizing scales
inversely with realized volatility; a running drawdown tracker stops trading if the soft threshold
(e.g. 20%) is approached, well before the hard 30% DQ gate. See [[Market Conditions]].

**Leaning:** whatever strategy family is chosen, a regime/risk overlay is worth building in.
The drawdown DQ is asymmetric — a missed trade costs points, a DQ costs everything.

---

## Data-to-strategy mapping (summary)

| Signal source | MCP tool(s) | Best-fit strategy family |
|---|---|---|
| Price / OHLCV | `cmc_market`, `cmc_history` | Technical / RL |
| Global market metrics | `cmc_market` (global flags) | Regime overlay |
| Token profile / chain stats | `cmc_token_info` | Sentiment, regime filter |
| CMC news + Fear & Greed | `cmc_news` (+ Agent Hub MCP) | Sentiment overlay |
| Wallet transactions | `bscscan_wallet_txs`, `bscscan_transfers` | Copy-trade |
| Holder concentration | `bscscan_token_holders` | Copy-trade, accumulation signal |
| X.com / breaking events | `social_scan` | Sentiment overlay (Phase 3+) |

---

## Validation pipeline

Before any strategy touches live capital:

1. **Register** the parameterized spec: `register_strategy`.
2. **Evaluate** offline: `evaluate_strategy` → `backtest_report` (Sharpe, Sortino, Calmar,
   max-drawdown, fee-adjusted). Must beat baselines (`run_baseline`: Buy&Hold, SMA, RSI, random).
3. **Diagnose** failure modes: `diagnose_run` checks under-random performance, over-trading,
   fee drag, drawdown events, negative Sharpe.
4. **Simulate** individual trades pre-execution: `simulate_trade` checks route, slippage, and
   guardrail pass/fail before `execute_trade` is called.
5. **Forward-run** on live data (paper or dust) before the June 22 live window.

Backtest numbers alone do not satisfy the June 16 PoC gate — only the live on-chain loop does
(see [[Tech Stack]] Phase 2).

---

## Open questions

- **Hourly `cmc_history` availability:** does the active CMC plan support hourly OHLCV for all
  149 eligible tokens? Daily candles may be the practical ceiling for some; confirm before
  building indicator logic that depends on intraday history.
- **BscScan polling latency for copy-trade:** how stale is wallet-tx data in practice? A
  multi-minute lag on a fast BSC token move may make copy-trade structurally unviable.
- **Regime classifier accuracy on thin tokens:** ATR-based regime detection on low-volume
  tokens may produce noisy tags. May need a liquidity filter before applying regime logic.
- **CMC Fear & Greed endpoint:** confirm which Agent Hub MCP tool surfaces the Fear & Greed
  index specifically (likely `cmc_token_info` or a dedicated sentiment endpoint — not yet
  confirmed from the CLI reference).
- **RL training time vs June 16 gate:** RL adds a training phase that the other families skip.
  If the execution loop isn't stable by ~June 12, RL is likely out of scope for the live window.
