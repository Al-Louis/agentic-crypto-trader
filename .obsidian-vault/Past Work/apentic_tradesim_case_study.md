# Reinforcement learning in a domain built to fool you

**Apentic / TradeSim — a full reinforcement-learning pipeline for crypto day-trading
agents: a market simulator that refuses to cheat, a risk-adjusted reward, ~28 technical
indicators, a custom attention-based policy network, a rigorous evaluation suite run
against honest baselines, and a Next.js frontend that live-simulates trained agents.
Roughly 18,000 lines across Python and TypeScript, designed and built solo.**

---

## The short version

Trading is the cruelest place to do reinforcement learning, because the environment is
*built to fool you*. A single line of look-ahead leakage, a reward that pays off luck, or
a simulator that quietly ignores fees will all hand you a beautiful backtest for a
strategy that loses real money. The actual engineering problem isn't the model — it's
building infrastructure honest enough that the numbers mean something.

TradeSim is that infrastructure, built end to end:

- A **data pipeline** that turns raw exchange candles into clean, causally-validated
  feature sets — with an explicit test that proves no indicator can see the future.
- A **market simulator** with a realistic broker (next-candle execution, volume-scaled
  slippage, fees, execution delay) that applies the *same* costs to the agent and to
  every benchmark, so any measured edge comes from decisions, not from a rigged sim.
- An **RL training pipeline** (Stable-Baselines3: PPO / RecurrentPPO / SAC) with a
  risk-adjusted reward and a custom attention-based network that learns which signals to
  trust in which market conditions.
- An **evaluation suite** that scores agents on held-out periods against real baselines
  with a full risk-metrics panel — the part most trading projects skip.
- An **MCP server** that exposes the whole train→evaluate→diagnose→retrain loop as tools,
  so an AI assistant can run the experiment cycle.
- A **Next.js frontend** that renders trained runs and streams a live, candle-by-candle
  simulation of an agent trading.

---

## What it is

TradeSim (product-branded *Apentic* / *AgentApe.ai*) is a local, single-developer Python
system covering the full lifecycle of building an RL trading agent: acquire historical
OHLCV data, clean and enrich it, simulate a market that charges realistic costs, train
agents against it, and evaluate them honestly against benchmark strategies — with a web
frontend that visualizes results and live-simulates a chosen model.

It is, deliberately, a **research-and-training platform, not a deployed product.** The
core pipeline is real and runs locally; the dashboard and the live WebSocket simulation
work on `localhost`. The surrounding product wrapper — landing, pricing, tournaments, a
model marketplace, accounts, payments, managed cloud training, live exchange execution —
is designed in the project's docs and partly mocked in the UI, but not built. Nothing is
deployed, and there are no users. The honest version of this project is the engine and
the discipline around it, and that is what's worth showing.

---

## The hard problems — and how they were solved

Each of these is a specific way trading systems lie to their builders, solved in the
actual code.

**Not fooling yourself (look-ahead leakage).** This is the single most common way a
backtest deceives, so it's defended at two independent layers. First, a
`validate_no_lookahead()` test recomputes every indicator on truncated history `df[:i+1]`
and asserts the value at row `i` matches the full-series value — proving the features are
causal. Second, the broker executes orders at the *next* candle's open, so the agent can
never trade on the same candle it's currently being judged against. Leakage isn't assumed
away; it's tested and structurally prevented.

**A reward that pays for durable skill, not luck.** Optimizing raw profit teaches an
agent to chase a few lucky, high-variance trades. The reward instead uses an incremental
**Differential Sharpe Ratio** (Moody & Saffell, 1998) — a dense, per-step, risk-adjusted
signal — with EMA tracking, asymmetric weighting that punishes losses harder than it
rewards gains, a quadratic drawdown penalty, an explicit per-trade fee penalty, and a
holding cost, all clipped to a stable range. The git history shows roughly thirty reward
and config iterations behind that design: this is the part that was *earned*, not
guessed.

**A simulator that doesn't cheat.** The broker models volume-scaled slippage, a taker
fee, execution delay, a 2% "dead zone" that suppresses fee-churning micro-trades, and a
forced position-close at session end. Crucially, those exact costs are applied identically
to the agent *and* to all four benchmarks (Buy & Hold, SMA crossover, RSI mean-reversion,
Random) — so a reported edge can only come from better decisions, never from execution
asymmetry. The portfolio layer tracks cash, position, mark-to-market value, peak/drawdown,
and FIFO trade records.

**Learning which signals to trust.** An early version leaned on hard-coded "guardrail"
rules and they were deliberately torn out (the commit is literally named *"no guardrails"*)
in favor of a learned approach: a `GroupedIndicatorExtractor` that gives each indicator
group its own MLP head and combines them with multi-head attention, so the policy learns
to weight signals differently across market regimes rather than being told how. A
curriculum callback ramps difficulty from low-volatility through mixed and high-volatility
to full data plus noise as training progresses.

**Evaluation you can actually trust.** The metrics suite goes well past "did it make
money": annualized return, Sharpe, Sortino, Calmar, max drawdown and its duration,
95% VaR/CVaR, win rate via FIFO round-trip pairing, profit factor, average win/loss, max
consecutive losses, and fee drag — all measured on a held-out test period and against the
same baselines. A continuous simulator removes episode boundaries so equity compounds
across days, closer to live conditions than clean bounded episodes.

**The experiment loop as something an AI can drive.** The whole cycle is exposed through
an MCP server's fourteen tools spanning data, training, evaluation, analysis, and config.
Training runs launch as background subprocesses so long jobs don't block; a
`diagnose_issues` tool encodes known RL-trading failure modes as rule-based checks
(under-performs random, over- or under-trading, fee drag, large drawdowns, negative
Sharpe) and returns actionable recommendations — so an assistant can run
train→evaluate→diagnose→retrain programmatically.

---

## The stack

| Domain | Tools |
|--------|-------|
| Data | Python 3.11, ccxt (exchange access), `ta` indicators, pandas/numpy, Parquet (pyarrow), Pydantic config; ~28 indicators, causal-validation test, episode index |
| Simulator | Gymnasium env(s) + custom broker/portfolio; next-candle execution, volume-scaled slippage, fees, execution delay, dead-zone, session accounting |
| RL / ML | Stable-Baselines3 + sb3-contrib (PPO / RecurrentPPO / SAC), PyTorch; Differential Sharpe reward; curriculum + early-stopping callbacks; TensorBoard |
| Networks | Custom feature extractors: grouped per-indicator MLPs + multi-head attention; 1D-CNN over the 60-candle lookback |
| Evaluation | Full risk-metrics suite, four benchmark strategies through the same broker, backtester, continuous compounding simulator |
| Automation | MCP server (14 tools, stdio), background-subprocess training, rule-based diagnostic |
| Frontend | Next.js (App Router), React, TypeScript, Tailwind, lightweight-charts, Zustand; dashboard + live WebSocket simulation view |

About **18,000 lines** total — ~7,100 Python across ~40 modules, ~2,200 lines of CLI
scripts, and ~9,000 lines of TypeScript/TSX across 8 pages and 41 components.

---

## Scale & status

Built over roughly **two weeks of focused, experiment-driven iteration** as a solo
effort, with a clear arc visible in the git history: core build and first reward/action
experiments, observation and regime work, the strategy framework and dashboard, the
modular indicator library and continuous simulation, then the recurrent-policy and
reward overhaul.

The scale of iteration is the tell that this was real research, not a demo: about **64**
training-run directories, **2,134** checkpoints, **23** finalized self-describing models,
and roughly **30+** reward/config versions — each validated against benchmarks before
moving on. The data layer holds ~**1.1 GB** of processed 1-minute candles across
BTC/ETH/SOL/XRP (Binance) and BTC (Bitstamp), spanning 2019–2026 with a clean
train/validation/test split. Test coverage is **135** test functions across 15 files,
including dedicated unit tests for slippage direction, fee application, the reward's
asymmetry and drawdown penalty, and the look-ahead guard.

Status, stated plainly: the data pipeline, simulator, training, evaluation, MCP server,
and the data-driven parts of the frontend are **built and run locally**. The marketing
and product pages, backend API, database, auth, payments, managed cloud training, and
live exchange trading are **designed and partly mocked, not implemented**. Nothing is
deployed. The objective the system is built to test honestly — beating baselines like
Buy & Hold and Random on held-out periods — is exactly that: a question the platform lets
you answer rigorously. Historical run results noted in commit messages are treated as
unverified notes, not performance claims. The point of this project is the machinery that
would let you find out the truth, and the refusal to pretend otherwise.

---

## Why this matters if you're hiring

In RL for markets, almost anyone can wire up Stable-Baselines3 and produce a chart that
goes up. The thing that's genuinely hard — and genuinely rare — is the engineering and
the discipline that make such a chart *mean* anything: causal feature validation, a
simulator that charges honest costs to everyone equally, a reward designed to resist
reward-hacking, evaluation on held-out data against real baselines, and the judgment to
keep iterating against those baselines instead of against your own hopes.

This project is evidence of that discipline applied end to end — domain-honest ML
engineering, careful reward and observation design, rigorous evaluation, a custom
attention-based network, and a full TypeScript frontend to visualize it, all wrapped in
an MCP tooling layer that makes the experiment loop agent-drivable. It's the same
strength that runs through my reinforcement-learning work generally: treating reward
design and honest evaluation as the real problem, not an afterthought.

If you need someone who can build an ML system that tells you the truth — and who knows
the difference between a result and a mirage — that's the work I do.

**Malexy** · [your contact / malexy.com]
