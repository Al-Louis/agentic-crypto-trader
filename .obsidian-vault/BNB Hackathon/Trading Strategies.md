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
> - **Out-of-sample validated (2026-06-06).** On a 60/40 chronological split, the vol *ranking*
>   persists (Spearman train→test +0.66, 5/8 top-8 overlap), and a **train-selected** vol-top8
>   **doubles the contender rate on the held-out test split** (TOURNEY 42% vs all-20's 21%, 0%
>   DQ) — matching the test-selected ceiling, i.e. ~no skill lost OOS. The tilt is real, not an
>   in-sample fluke. Remaining caveat: one split and both periods are bull-ish — **a bear week is
>   still untested**, so a **regime overlay** (hold vol-top8 risk-on, rotate to stables risk-off)
>   is the next piece. Live, select the vol set from recent pre-competition data.
> - **Regime overlay tested (2026-06-06) — overpriced insurance in the bull sample.** A BTC
>   trend-gate (`btc_risk_on`: close > 72h EMA → hold vol-top8, else cash) is **real insurance in
>   bear weeks** (halves mean drawdown 12%→6%, eliminates bear-week DQ 3%→0%) but sits in cash
>   56% of the time, **cutting the tournament rate in half (27%→13%) and bull-week upside
>   (+15.5%→+7.6%)** — net-negative for a leaderboard *in this crash-free sample*, where the
>   insured DQ is only ~2%. Two honest gaps: the sample has **no real crash** (so it
>   *under-values* the insurance), and the **all-or-nothing 72h gate is too blunt**. Working
>   stance: **ungated vol-top8 is the bull bet; the gate is toggle-able bear insurance** keyed to
>   the live-regime forecast.
> - **Refined overlay swept + candidate codified (2026-06-06).** Partial de-risk (`trend 50%`)
>   dominates the blunt full-cash gate (TOURNEY 21% vs 13%, **0% bear-week DQ**); **extreme-
>   stress-only (`stress 50%`)** keeps the *full* tournament rate (27%) and de-risks only on a
>   genuine crash — the ideal "insure the tail, keep the upside" design — but stays **dormant
>   (unvalidated)** in this crash-free sample (pending a synthetic-crash stress test). **The
>   decision core is now codified: `trader.strategy.build_candidate`** — daily-rebalanced
>   equal-weight vol-top8 + regime overlay (default `stress50`; `trend50` = validated hedge;
>   `none` = pure bull bet).
> - **Crash stress test (synthetic, 2026-06-06) — overlay validated; default reconsidered.**
>   Splicing a crash week (BTC drops, high-vol alts amplify via a 1.5× stress beta) confirms the
>   gates **really protect**: BTC −25% linear → ungated 43% DD / 90% DQ vs **`trend50` 24% / 15%**
>   (`stress50` 27% / 40%). **But `trend50` protects far better than the codified `stress50`
>   default** — stress50's −8%/3-day threshold is too lax (barely fires on a slow bleed: BTC −15%
>   → stress50 55% DQ vs trend50 0%), and half-exposure is too little. **Nothing half-exposed
>   survives BTC −50%** (~40% DD / 100% DQ → needs full cash). The upside↔protection tradeoff is
>   now quantified **both ways** (tournament sweep + crash test). **Open decision:** `trend50` is
>   the most robust don't-know-the-regime default; the ideal is a **severity-scaled stress gate**
>   (dormant in calm, scales to full cash by crash depth) — to build. `trader.sim.crash`,
>   `scripts/crash_test.py`.
> - **Severity gate built + measured (2026-06-06).** `severity` (`severity_exposure`, −5%→−20%
>   trailing) keeps **~full upside** (TOURNEY 26% vs ungated 27% — dormant 98% of the time) and
>   **uniquely survives a deep slow crash** (BTC −50% linear → 20% DD / 5% DQ, where `trend50` is
>   42% / 100% and ungated 68% / 100%). **But it does not dominate `trend50`** — it *under-protects*
>   moderate/sharp crashes (it re-invests as the drop ages out of the trailing window: BTC −25%
>   linear → 50% DQ vs trend50's 15%). **Net:** the overlay is a fully-mapped tradeoff —
>   **`trend50` (locked default)** = robust moderate/sharp protection at −6 pts upside; **`severity`**
>   = full upside + deep-tail insurance with a moderate-crash gap; `none` = pure bull. A combined
>   trend+depth gate would dominate `trend50` but is diminishing returns vs the unaddressed on-chain
>   execution work — **strategy core is done; pivot to the Phase-2 execution spike.**

---

## Committed candidate v1 (2026-06-09) — disciplined trend-hold on vol-top8

The first thing we actually build (**rung 0** of a rules→learned ladder). **Honest framing, grounded
in this note's prior findings:** this is *not* a momentum-*selection* alpha play — that was tested
here and failed (residual-momentum IC negative / mean-reverting; cross-sectional momentum churns
thin pools to death and loses to costs; *"entry alpha is dead, only low turnover survives"*). It
targets the **documented edge — exits + low turnover** — by wrapping the validated **vol-top8**
selection in a discretionary **hold / exit / re-entry discipline** distilled from the user's manual
trading. It is the vol-top8 baseline made *less churny and better at exits* — the two axes the
cost-aware backtests say actually decide returns here.

**Why this can beat the vol-top8 baseline (and why our RL lost to it).** Naive vol-top8 holds all 8
equal-weight and rebalances daily — it churns and rides dead tokens down. SIREN (clean test bundle,
2026-06): +84% runup to $1.28 on May 9, then bled **below its ~$0.69 origin** and chopped
$0.45–0.53 for two weeks — and our RL *churn-traded that corpse 8+ times and FOMO-bought the $1.28
peak*, running **3× the baseline's turnover**. This candidate forbids exactly that: ride the trend,
exit on the rollover, then **stand aside in cash** — no churn, no FOMO re-entry.

### Per-token state machine (the rules)

| State | Trigger → action | Trader's rule |
|---|---|---|
| **Watch** (flat) | fresh confirmed uptrend (close > trend-EMA, new higher high) → **enter**, record **origin** = entry price | enter on a real move |
| **Hold** | trend intact (price > peak·(1−k), above trend-EMA) → **hold, don't trim** | *let winners run* |
| **Exit** | rollover: price drops k% off peak **or** close < trend-EMA → **sell to 0** | *if it stalls, sell and walk; never drawdown* |
| **Cooldown** | after exit, **no re-entry** for M bars **and** until a *new* higher high above the prior peak | *no FOMO re-entry without fresh data* |
| **Dead-zone** | price < origin **and** no fresh uptrend → **inactive: no position, no trades** | *never churn sideways below the origin* |

### Starting thresholds (rung 0, hand-set — RL tunes these at rung 1)

- **Trend filter / entry:** close > 24h EMA **and** a new N-bar high (N≈24). *Entry is a state gate,
  not an alpha claim — per the IC finding it won't pick winners; its job is* when *to deploy.*
- **Exit / trailing stop:** k ≈ 10–12% off the rolling peak, or close back below the 24h EMA.
- **Cooldown:** M ≈ 24–48h after exit, plus a fresh higher-high above the prior peak to re-arm.
- **Dead-zone reset:** cleared only by a new higher high above the prior cycle's peak.

### Portfolio / chassis layer

- **Active set ⊆ vol-top8:** hold only the trending members; dead-zone / exited members → **cash**
  (a stablecoin — eligible, not dust, and the de-risk swap satisfies ≥1-trade/day).
- **Sizing:** equal- or conviction-weight across the active set, per-token cap ≈ 25%.
- **≥1 trade/day:** if a day produces no signal trade, a small compliance rebalance.
- **Drawdown backstop:** hard de-risk to cash at **~25%** portfolio drawdown — a **rarely-fired**
  safety net; primary drawdown management is the per-token exits + dead-zone (its trigger rate is a
  *policy health metric*). See [[Security and Encryption]] / `src/trader/risk/`.

### The ladder + validation

- **Rung 0** = the hand-set thresholds above — fully interpretable; becomes the **new
  baseline-to-beat** (replacing plain vol-top8).
- **Rung 1+** = RL tunes the thresholds (entry confirmation, exit k, cooldown M, sizing/exposure),
  kept **only if it beats rung 0 on the frozen test** ([[Experiment Log]] / OOS rig).
- **Honest success criterion:** lower turnover + cleaner exits → higher OOS return at a survivable
  worst-seed drawdown vs the vol-top8 baseline. We do **not** claim entry alpha.

### Built + measured (2026-06-09)

Rung 0 is built (`trader.strategy.rung0`) and evaluated on the same windows + cost model as the RL:

- **The discipline validates.** On the frozen **test** split: **+17.0% @ 12.3% maxDD** (best Sharpe
  2.81, lowest turnover) vs vol-top8 hold +22.5% @ **34.6% — DQ** / trend50 +25.7% @ 24.1%. It rides
  the runup then stands aside (SIREN: held one day, then cash — no churn). The exit + low-turnover
  edge is real, out-of-sample.
- **But it's too conservative** — uses only ~12% of the 30% DD budget, so it doesn't beat trend50 on
  return. The obvious move is to spend the risk budget.
- **Single-window threshold tuning OVERFITS** (`sweep_rung0.py`): the best val config (+167% @ 25.8%)
  collapses to **−17% @ 44% DD** on test — the same trap as the RL. **Robust thresholds require
  walk-forward / multi-window selection, not one val window** (`trader.sim.resample`).
- **Status:** the conservative default is a gate-safe, generalizing baseline; robust aggression is
  pending the walk-forward sweep. See [[Experiment Log]], [[Build Log]].

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
