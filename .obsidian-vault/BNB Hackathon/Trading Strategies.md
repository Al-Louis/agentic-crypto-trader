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
- **Walk-forward sweep — the disciplined rules lose to vol-top8 on the tournament objective.**
  `sweep_rung0_wf.py` scored all 144 configs across ~120 random 7-day windows (train+val), selecting
  by P(week > +15%) at P(DQ) < 5%. It correctly **rejected the single-window overfit** (only 36/144
  gate-safe; the aggressive `maxW 0.60` winners breach too often). But on frozen-test windows (all 0%
  weekly DQ): **vol-top8 plain hold 15% tourney > trend50 9% > rung-0 tuned-pick 6% > rung-0 default
  3%.** The tuned pick beat the default (robust tuning *did* help within the rule set), but the rule
  ceiling sits **below the baseline**. **Why:** the prize rewards upside *variance* (P(a big week));
  risk discipline suppresses variance, so it clips the fat right tail the contest pays for — the
  discipline optimizes for *real-trading* risk-adjusted return, not the tournament objective.
- **Verdict:** **second hypothesis to lose to vol-top8** (after RL-from-scratch). The vol-top8
  *selection* is the edge; complexity on top hasn't beaten it *for the competition*. For the contest:
  ship **vol-top8 + trend50** (gate-safe, bear-insured, best realistic tourney rate) or `none` for
  max upside. The rung-0 discipline is kept as a **real-trading / risk-adjusted** asset, and the next
  *competition* edge must come from a genuine **upside signal** (sentiment/regime), not more
  discipline. See [[Experiment Log]], [[Build Log]].

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

## Intraday breakout-reversal — a candidate HARVEST signal (2026-06-10)

Origin: a dashboard-driven hypothesis off the [[Apentic Data Contract|market_metrics]] vol/correlation
view, characterized empirically on train+val (test frozen), 67k pooled token-bars. **Status: candidate
feature — validated in-sample as a signal, NOT yet net-of-cost or OOS. Do not hard-code it.**

**Hypothesis (refined through three rounds).** In a multi-timeframe downtrend (30d↓, 7d↓), a short-term
up-move (3d↑, 24h↑) is usually a *dead-cat bounce* — but one that **breaks structure** (makes a fresh
short-window high) may be a genuine reversal worth sizing up on.

**What the data says:**
- **Bare reversal** (30d−, 7d−, 3d+, 24h+): forward-24h **−0.31%** mean (median −0.63%, win 41%) — a
  *dead cat*, below the +0.44% baseline. Hard-coding it would have lost money.
- **+ fresh 3-day-high confirmation**: flips to **+0.58–0.77%** mean (win ~50%). The breakout filter
  separates real bounces from dead cats. (5d-high: negative; the literal "exceed the 30d high" never
  occurs in a downtrend — so the signal is *short-window* by nature, and **definition-fragile**.)
- **Horizon is the crux.** The edge **peaks at a ~4–6h hold (+0.65% over baseline) and goes negative by
  48–72h** (≈ −0.5%): it is an **intraday momentum capture that fades into the dead cat on multi-day
  holds.** `IC(trailing-24h, forward-H)` is negative and *strengthens* with horizon (−0.05 @1h → −0.13
  @48h) — the universe mean-reverts on average, but the breakout condition selects a *momentum-
  continuation sub-population*. A real **nonlinearity** (a linear signal can't capture it; RL can).

**The binding constraint — cost.** Round-trip AMM friction is **~1.0%**; the peak *gross* edge is
**+0.77% @6h — below cost.** Median ≈ 0 / win ≈ 50% ⇒ full cost is paid on the coin-flip majority.
**Net of friction the bucket average is slightly negative; the profit lives entirely in the convex tail**
(the minority that rip several % intraday). This is exactly the marginal-gross-vs-cost case the honest
gate (net-of-cost, OOS, through the env) exists to adjudicate.

**Why it matters now — it is the missing HARVEST lever (the GATE-2 gap).** GATE 2 ([[Experiment Log]],
[[AI Training]]) found the regime-adaptive policy learned to *de-risk* well (3/4 seeds survived an 82%
crash; s1 +5.8%) but is **defensive-everywhere** — it fails to *ramp up* in bulls (lost money in a +27%
market by holding cash). This breakout-reversal is a **size-UP trigger**: paired with the existing
universe-**breadth** obs feature, it is the other half of a regime-adaptive pair — *breadth-high +
confirmed breakout → harvest; breadth-collapse → de-risk.*

### Concrete feature + exit spec (2026-06-10, owned by market-indicator-expert)

The four questions below translate the signal characterization into an implementable, leakage-free spec
for `rl-ml-trainer` to wire into the obs vector. They assume the current `EventRungEnv` obs layout
(`OBS_DIM=13`, indices 0–12) and the discrete-action substrate from GATE-2.

#### 1. Breakout-distance feature — exact causal definition

**Compute at each bar, for the token under consideration, using only data available at that bar.**

```
bkout_dist_N(tok, bar) = px[bar, tok] / rolling_max(px[:bar], N) - 1.0
```

where `rolling_max(px[:bar], N)` is `px[bar-N+1 : bar].max()` — a standard right-anchored rolling
maximum, **excluding the current bar from nothing** (the current bar's close is available at decision
time; this is a snapshot of where price sits *relative to its recent high*, taken at the event trigger).

The quantity is naturally **continuous and negative-or-small-positive**: deeply below the N-bar high
it is e.g. −0.15; at a fresh breakout it is ≈0 or slightly positive; it cannot be positive by more
than the current bar's intraday range from the prior high.

**Why continuous rather than a Boolean flag.** The empirical finding is definition-fragile (3d-high
works, 5d-high negative, 30d-high never fires in a downtrend). A continuous ratio lets the RL learn
*how far* from the high matters without baking in a single threshold that may not generalize OOS. The
Boolean "is a fresh 3d-high" is the signal's skeleton; the distance is the dial the agent can grade.

**Which N to use.** Offer **two windows in the obs**, not one:

| feature name | N | rationale |
|---|---|---|
| `bkout_dist_short` | 24 bars (24h at hourly data) | captures the intraday breakout characterization finds peaked at 4–6h; this is what the signal is |
| `bkout_dist_med`   | 72 bars (3 days)           | the window where the Boolean "3d-high" confirmation was measured; continuous analog |

Do not add the 5d or 30d windows — those were either negative or degenerate. Two features add 2 to
`OBS_DIM` (13 → 15). Both are computed from the same `_px` array already precomputed in the env,
so they are free to add: `rolling_max = np.max(self._px[max(bar-N+1,0):bar+1, j])`.

**Leakage check.** `_px` is a cumulative-product forward-filled price index built from `returns` with
no shift — it is the end-of-bar close at `bar`, which is the value available at decision time (the
event fires on the bar's data). No future bar enters the computation. Causal: confirmed.

**Orthogonality to `cush`.** `cush` (obs index 1) is `px / ema_72 − 1`, the distance above/below the
72h *smoothed* trend. `bkout_dist_short` is distance from the *raw rolling max* over the last 24 bars.
These are structurally different: a token can be above its EMA but well below its 24h high (recent
pullback in an uptrend) or at a fresh high while below its EMA (early recovery). The validation bar
below requires incremental IC over `cush` — both features must earn their obs slot.

#### 2. Harvest + de-risk pair — how this composes with breadth in the obs

The GATE-2 gap is precise: the agent learned to respond to *low* breadth (de-risk) but not to *high*
breadth (harvest). The fix is not to hard-code "breadth-high + breakout → big size" — that bakes in
a threshold and is exactly what the RL should *learn* from the obs signals. The fix is to make the
relevant signals **simultaneously visible** in the obs so the interaction is learnable.

**What the agent needs to see (additions to the obs vector):**

In addition to `bkout_dist_short` and `bkout_dist_med`, add the following **multi-timeframe return
context** features, computed causally from `_px` at event-trigger time for the token under consideration:

```
r_24h  = px[bar, tok] / px[bar-24, tok]  - 1.0   # already available via cush/surge; add as explicit scalar
r_3d   = px[bar, tok] / px[bar-72, tok]  - 1.0   # 3-day return
r_7d   = px[bar, tok] / px[bar-168, tok] - 1.0   # 7-day return
r_30d  = px[bar, tok] / px[bar-720, tok] - 1.0   # 30-day return (720 hourly bars)
```

Clip all return features to `[-RET_CLIP, RET_CLIP]` (already the env convention). These four features,
combined with the existing `breadth` (obs index 12) and the two `bkout_dist_*` features, give the agent
the full conditional structure the signal characterization describes:

- **breadth high (0.7+) + bkout_dist_short near 0 + r_24h > 0 + r_3d > 0**: the harvest condition — an
  upside breakout in a bullish basket. The RL can learn to size UP here.
- **breadth low (0.3−) + r_7d < 0 + r_30d < 0**: the dead-cat condition — even if r_24h > 0, these
  contradict. The RL can learn to stay SMALL here.
- The interaction is NOT hard-coded. Both signals are in the obs; PPO's MLP can represent the joint
  conditioning. The discrete action space (4 size levels) means the output is bucketed, which is exactly
  right for "size small / skip / medium / max" decisions.

**Total new obs features:** `bkout_dist_short`, `bkout_dist_med`, `r_24h`, `r_3d`, `r_7d`, `r_30d` — 6
features, `OBS_DIM` 13 → 19. This is moderate and preserves sample efficiency; a 19-dim obs is still
fully tractable for an MLP at 1M–4M timesteps.

**One important note on the GATE-2 crash-survival.** The 3-of-4 seeds that de-risked well in GATE-2 saw
breadth collapse and responded. Adding bull-harvest signals must not destroy that. The test is: after
adding these features, retrain with the same crash injection and check that crash-regime behavior
degrades no more than ~1 seed. The reward-rebalance (lower `dd_lambda`) is the separate lever for the
defensiveness problem — do not conflate them.

#### 3. Short-hold exit pairing — how to bias the exit-override decision

**The problem.** The edge peaks at 4–6h and goes negative by 48–72h. Rung-0's trailing-stop/EMA exit
fires at a 25% peak-drawdown or EMA cross, which on a volatile alt can mean holding for many bars after
the intraday pop has faded. The RL's exit-override action (obs is-exit=1, action = keep-fraction) is
the lever, but GATE-1 and GATE-2 showed the exit-override path has at most 39 decisions per episode —
sparse — and `corr(giveback, post-exit fwd-24h) = −0.058` (flat IC).

**The fix is two-pronged, not just an obs feature:**

First, add **`held_hours`** as a direct, easily-readable scalar in the obs when `is_exit=1`:

```
held_hours = (bar - pos[tok]["entry_bar"])   # in hourly bars, already available as held_frac scaled
```

`held_frac` (obs index 4) already encodes this as a fraction of `episode_bars`. At 168-bar episodes,
`held_frac=0.03` means ~5 hours held. The agent *can* already read this. The gap is that the signal
characterization data (edge peaks 4–6h, goes negative 48–72h) has not been in the training signal — the
agent hasn't been rewarded for the timing of exits, only for the portfolio-level outcome.

**Second and more important**: the `bkout_dist_short` and multi-timeframe return features are observed
at *exit time* too (the obs is constructed at the decision bar regardless of event type). At exit time
after a breakout-reversal entry:
- If `held_frac` is low (4–6h), `bkout_dist_short` may have moved positive (the pop materialized), and
  `r_24h` has turned up — take profit signal. The RL can learn: short hold + pop occurred → take profit
  (action = 0, full exit).
- If `held_frac` is high (>48h), `r_3d` may have turned negative — the dead cat thesis. The RL can
  learn: long hold + multi-day fade → don't override the stop.

**What this means for obs wiring at exit events.** The six new features (`bkout_dist_short`,
`bkout_dist_med`, `r_24h`, `r_3d`, `r_7d`, `r_30d`) should be **computed for the exiting token** at
exit time, exactly as for entry events. The obs construction in `_obs()` already takes `tok` and
computes `cush`, `surge`, `unreal`, `held_frac`, `giveback` for that token; the new features slot in
with the same logic. There is no architectural change required — just extend the obs vector.

**No separate exit-timer hard-code.** Do not add a rule that forces an exit at 6h. That bakes in the
fragile threshold. Instead, give the agent `held_frac`, `bkout_dist_short` at exit time, and
`r_24h`/`r_3d` at exit time — it has all the information needed to learn the 4–6h pop-and-exit
pattern *if the reward exists to teach it*. The reward is the portfolio-level return (net of cost),
which will penalize holding into the 48–72h fade. The signal is there; training it requires adequate
exit-event count. If the exit-IC remains flat after adding the new features (post-training diagnostic),
escalate to `rl-ml-trainer` for the recurrence lever.

#### 4. Selectivity to clear the cost wall — ranking within the breakout bucket

**The problem.** Gross peak +0.77%, cost ~1.0% → full-bucket average is net-negative. The profit is in
the convex tail: the subset of breakout entries that rip several percent within 4–6h. To select into
that tail the agent needs a *rank* within the breakout bucket, not just a Boolean trigger.

**The ranking signal: a composite of `cush`, `surge`, and `bkout_dist_short`.**

From the existing probe data (`probe_obs_alpha.py`, `probe_subset_ic.py`):
- `cush` has robust negative IC at ignition time: **stretched** ignitions (price pushed far above EMA)
  revert; **tight** ignitions (just reclaimed EMA, low cush) continue. Inverse cush = a size-up signal.
- `surge` is the volume-ignition strength: high surge = conviction in the move. Positive IC for
  continuation.
- `bkout_dist_short` (new): near-zero = fresh breakout (price just crossed the 24h high, momentum
  is live); deeply negative = price broke out a long time ago and has since pulled back (momentum stale).

The three signals together form a **composite rank**: the highest-quality breakout entry is one with
**low cush (tight, just reclaimed EMA), high surge (volume backing), and near-zero bkout_dist_short
(fresh, not stale)**. Formally:

```
rank_score(tok) = -cush[tok] + alpha_s * surge[tok] + alpha_b * (1 - abs(bkout_dist_short[tok]))
```

where `alpha_s` and `alpha_b` are learned implicitly by the RL policy (they should not be hand-tuned).
The RL sees all three in the obs and — given the `entry_forward` reward landscape that's already been
shown to produce a correct-discriminator argmax at γ=0.10 — will learn the combination that predicts
forward returns within the breakout bucket.

**Why the `entry_forward` reward + `ungate=True` is already the right mechanism here.** The exp5
landscape (preflight_selector, γ=0.10) passed the in-env gate with the `[cush, surge, btcT]` predictor
achieving OOS coef `[−0.336, +0.008, +0.977]`. Adding `bkout_dist_short` to the OLS fit and to the obs
is a direct extension — if `bkout_dist_short` has incremental IC over `cush` within the breakout bucket,
the preflight will show it; if it doesn't, it won't improve the reward landscape and should be dropped.

**The cost-wall mechanism in the discrete-action substrate.** The discrete action (4 levels: 0 = skip,
1/3 = small, 2/3 = medium, 1 = max) with risk-parity per-token caps means:
- Skip (level 0): pays zero cost, zero return. The baseline.
- Max (level 3) × token cap (e.g. 5% for a high-vol token): a $500 bet on a $10k portfolio. Break-even
  requires a +1% net move just to cover cost. Only makes sense on the tail entries that are expected to
  rip 3–5%+.
- The RL learns to size level-3 only when the composite rank is highest — exactly the selectivity needed.

**The honest gate for this mechanism.** After obs extension + retraining:
1. Run `diag_deviation_alpha.py` on the trained policy. Require `corr(dev, fwd_ret) > +0.10` (the gate
   exp5 set but didn't achieve with only `cush/surge/btcT`).
2. Verify that the avg entry-level chosen on breakout-bucket events (where `r_3d > 0 and r_7d < 0`)
   is higher than on non-breakout ignitions in the same regime. If not, the composite rank isn't being
   used — re-examine the obs normalization or the forward-horizon window.
3. The net-of-cost OOS gate on frozen test: seed-mean return > rung-0 (+18%), worst-seed DD < 25%,
   **in the bull regime specifically** (the GATE-2 gap), while crash-survival degrades by at most 1 seed.

#### Sequencing: before or after the reward-rebalance/recurrence levers?

**Verdict: add the obs features FIRST, before the reward-rebalance or recurrence sweeps.**

Reasoning:
1. The GATE-2 defensiveness has two separable causes: (a) `dd_lambda` too high → trains blanket caution;
   (b) no harvest signal in the obs → nothing to ramp up on even if the reward permitted it. These are
   *complementary*, not redundant. If you fix (a) without (b), the agent becomes less cautious but still
   has no selective harvest trigger — it will just take *more* mediocre entries. Fix (b) first so that
   when the reward rebalance opens up sizing, the agent has signals to discriminate on.
2. Adding 6 obs features (OBS_DIM 13→19) is cheap: a 3-day training run on the desktop, no architectural
   change. The reward-rebalance (`dd_lambda` 1.0→0.5) and recurrence (RecurrentPPO) are larger levers
   with larger risk surfaces — they should be gated on a stable obs foundation.
3. The in-env landscape check (`preflight_selector.py`) must be re-run with the new obs before any sweep.
   If `bkout_dist_short` + the MTF returns improve the correct-discriminator's in-env score, the gate
   passes and the sweep is green-lit. If not, the features don't help at this leverage point — stop.

**Concrete handoff to `rl-ml-trainer`:**
- Extend `EventRungEnv._obs()` to add the 6 features; update `OBS_DIM = 19`.
- Precompute `_bkout_short[bar, j]` and `_bkout_med[bar, j]` alongside `_px`, `_cush`, `_surge` in
  `__init__` (rolling max over N=24 and N=72 bars respectively, shifted by 1 to be causal).
- Precompute `_r24[bar,j]`, `_r3d[bar,j]`, `_r7d[bar,j]`, `_r30d[bar,j]` from `_px` (ratio at
  `bar` to `bar-N`, shifted to be causal at bars < N).
- Re-run `preflight_selector.py` with the new obs; confirm in-env gate PASS before sweeping.
- Keep the same `entry_forward` + `ungate=True` + `γ=0.10` reward config from exp5.
- After sweep: run `diag_deviation_alpha.py` and the three-point honest gate above.

#### Single biggest risk

**The exit-IC wall.** The breakout-reversal edge is an *intraday* phenomenon (4–6h). The env's event
engine fires exit prompts only when rung-0's trailing-stop or EMA-cross fires — which on a volatile alt
can be hours or days after entry, well into the negative multi-day zone. Even with the new obs features,
if the exit-prompt count stays at ~39/episode and the post-entry multi-bar drift swamps the 4–6h pop
in the reward signal, the agent will be trained on the full-hold outcome (which the characterization
says is negative) and will learn to skip the breakout bucket entirely rather than size up for a
short-hold harvest.

**Mitigation:** add an explicit **time-based exit prompt** — at 6h (6 bars) after a breakout-reversal
entry (identified by `r_3d > 0 and r_7d < 0 and bkout_dist_short > -0.02` at entry time), force an
exit event prompt regardless of whether the trailing stop has fired. This is NOT a hard-coded exit rule;
it is an additional decision point that gives the agent the *opportunity* to take profit at the 4–6h
mark. The agent still chooses the action (keep 0% / 33% / 67% / 100%); the forced prompt just ensures
the gradient sees a decision at the right time. This is the minimum architectural support needed to
teach the intraday-pop-and-exit pattern.

**Validation bar (before it enters the obs):** must hold **OOS (frozen test)**, survive **transaction
costs**, and be **incremental over `cush`** (already in the obs vector). Owned by `market-indicator-expert`
(signal + exit pairing) with `rl-ml-trainer` (obs integration + training).

## Signal-level findings from the rung-1b forensics arc (2026-06-10/11)

Probe-validated rules and refutations from the user's per-token chart reviews — each measured
against the panel before touching the rule:

1. **Take-profit rungs are a real missing degree of freedom.** rung-0 (and the event env) only
   sells on WEAKNESS (trailing stop / EMA-break) — selling into strength was inexpressible. Adding
   profit-take prompts at +25/50/100/200% unrealized (default = let-winners-run, so rung-0 is
   preserved) raised the hindsight ceiling from +74.8% to **+95.5% val while halving its drawdown
   (12.1%→7.1%)**. Selling high is worth more than the discipline of never trimming.
2. **The detonation blacklist (the Q pattern) is real and expires.** A massive volume surge WHILE
   price collapses (surge ≥8×, rising ≤−15%) marks a token: its subsequent ignitions are poison
   (fwd48 **−8.4% train / −24.3% val**, win 8–21%, n=121) until ~4 weeks out, where they revert to
   baseline. Built as `det_blacklist` (672 bars) in the ignite precompute — applies to the agent
   AND the rule mirror. Probe: `scripts/probe_detonation.py`.
3. **The low-rising "false flag" entry filter is REFUTED.** The intuition (huge surge + weak price
   progress = distribution → filter it) fails at the population level: the kept bucket (rising
   ≥15%) has *worse* forward returns than the killed one, and the explicit surge≥8×/rising<15%
   corner is *positive* on both splits. Extended movers are the statistical poison — the same
   mean-reversion (`cush`-negative) finding every probe has produced. Q's Mar 28 disaster was real
   but its damage came from the EXIT OVERRIDE (riding −45% at 2× size), not the entry; the fix was
   the **loss floor** (no override below entry−20%), a behavioral guardrail, not a signal. Probe:
   `scripts/probe_false_flag.py`.
4. **The calm half of a vol-stratified universe is pure bleed.** broad-12's XRP/LINK/LTC/BabyDoge
   tier lost in every regime across seeds; reverted to **voltop8** (risk-parity caps stay ON —
   concentration is what DQ'd GATE-1).

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

## PARKED — wallet-attributed token personality (post-competition expansion, 2026-06-12)

The user theory chain, held for after the live window: the token-personality probe found a real
but price-only-unusable kernel (cross-family sign persistence ~2/3, magnitude unpredictable,
entry-payoff REFUTED — see [[Experiment Log]]). The expansion: an **on-chain trade-execution
logger** (BscScan transfers/holders — the [[Real-time Monitoring]] surface) attributing each
token's flow to specific WALLETS: the resident MM/dev wallets that *create* the stable
personality, vs disrupting actors (a new wallet accumulating -> pump precursor; an aged wallet
distributing -> local-dump precursor; the MM going quiet -> detonation precursor). This is the
concrete mechanism behind the quant consult's Addition-3 (liquidity/flow state), which was
DATA-GATED — the logger IS the missing time-varying data source. Probe targets when built:
(a) does wallet-cohort flow lead price by enough hours to act on; (b) does MM-wallet behavior
change predict personality breaks; (c) detonation early-warning. Strictly post-competition:
needs live collection infrastructure, not derivable from the recorded OHLCV panel.

**Addendum (2026-06-12, order-book depth concept):** a third independent arrow at this same parked
target. The CLOB depth-leads-price mechanism (vanishing resting bids, book imbalance, ~11s lead)
does NOT exist on our AMM venue (no book; deterministic constant-product slippage - already
modeled in the broker; and sub-minute edges die to our ~0.7-1% round-trip cost). The on-chain
ANALOGS are real and belong to the same logger: **LP Mint/Burn events** (liquidity withdrawal
precedes the dump - plausibly the mechanism behind the detonation pattern) and **net swap-flow
imbalance** (the AMM book-imbalance analog, probe-able for reversion stats at cost-survivable
timescales). The pool-event stream (Sync/Mint/Burn/Swap logs) is the single missing instrument
behind all three ideas: wallet attribution, liquidity/flow knowledge, depth-leads-price.

**Addendum (2026-06-12, instrument BUILT):** the pool-event logger now exists —
`src/trader/chain/` + a historical backfill over the recorded OHLCV window, which un-parks the
*probing* of all three ideas without waiting for live collection (the "strictly post-competition"
constraint applied to live infrastructure; the backfill route was the unlock). Data contract,
RPC findings, and the three pre-registered probes: [[Pool-Event Data Layer]]. Probe results land
in [[Experiment Log]]; integration into training stays gated on a probe PASS, per the law.
