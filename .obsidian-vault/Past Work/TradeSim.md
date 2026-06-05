# TradeSim (Apentic / AgentApe.ai)

Indexed in [[Current Projects]]. Related skills in [[Skills]].

**Type:** Personal RL project — cryptocurrency day-trading agents
**Role:** Sole developer
**Status:** Built and working locally; shelved (~April 2026); nothing deployed; no users
**Timeline:** Built over ~16 days, 2026-04-02 to 2026-04-17
**Scale:** ~7,100 lines of Python (~40 modules) + ~2,200 lines of CLI scripts + ~9,000 lines of TS/TSX frontend

---

## What it is

A Python reinforcement-learning system for training and evaluating cryptocurrency day-trading agents, paired with a Next.js web frontend that visualizes and live-simulates them. A local, single-developer project. Conceptually a follow-on to [[Dynamic Train Labs]]: it reapplies the same RL-plus-MCP architecture learned from DTL to a different domain (crypto trading and technical analysis), an area of personal expertise.

Covers the full lifecycle of building an RL trading agent: acquire historical candlestick data, clean and enrich it with technical indicators, simulate a market with a realistic broker, train agents, evaluate against benchmark strategies, and serve results into a web dashboard and a real-time streaming simulation.

---

## Components

### Data Pipeline
- Python 3.11; ccxt (exchange access), ta (indicators), pandas, numpy, pyarrow (Parquet), pydantic (config), loguru.
- Downloads paginated OHLCV candles from any ccxt exchange with retry/backoff; cleans (duplicate removal, gap handling, OHLC consistency, flash-move flagging), segments into sessions, appends ~28 technical indicators, and builds a validated episode index guaranteeing sufficient lookback and contiguous in-session future.
- **Leakage guard:** a look-ahead validator recomputes indicators on truncated data and asserts causal values match full-series values.
- Parquet storage, partitioned by exchange/symbol/timeframe/year.

### Trading Environment + Simulated Broker
- Gymnasium environments (two generations: continuous target-allocation action and discrete Hold/Buy/Sell).
- Broker executes orders at the next candle's open (one-candle delay, preventing same-candle look-ahead), with volume-based slippage, taker fees, a dead-zone against fee-churning, and a minimum trade size.
- Portfolio with cash/position accounting, mark-to-market, drawdown tracking, realized/unrealized PnL, and FIFO trade records.

### RL Training Pipeline
- Stable-Baselines3 + sb3-contrib (RecurrentPPO), PyTorch, TensorBoard.
- Algorithms: RecurrentPPO (MlpLstmPolicy, LSTM-256), PPO, SAC.
- SubprocVecEnv parallel environments; callbacks for trading metrics, curriculum (low-vol → mixed → high-vol → full+noise), and early stopping.
- **Custom networks:** a grouped indicator feature-extractor giving each indicator group its own MLP head combined via multi-head attention; a 1D-CNN extractor over the lookback window.

### Reward Design
- Incremental Differential Sharpe Ratio (Moody & Saffell, 1998) with EMA tracking instead of raw PnL, plus asymmetric loss weighting, quadratic drawdown penalty, per-trade fee penalty, and holding cost. ~30+ reward/config iterations behind the final design.

### Evaluation and Serving
- Full metrics suite: total/annualized return, Sharpe, Sortino, Calmar, max drawdown, VaR/CVaR, win rate, profit factor, fee drag.
- Benchmarks (Buy & Hold, SMA crossover, RSI mean-reversion, Random) run through the same broker for fair comparison.
- Backtester and a continuous (no-episode-boundary) simulator where the portfolio compounds across days.
- Export script produces dashboard JSON; a Python websockets server streams a model stepping through historical data in real time.

### MCP Server
- 14 tools across data, training, evaluation, analysis, and config, exposing the full train → evaluate → diagnose → retrain loop so an AI assistant can drive it programmatically.
- Training runs launch as background subprocesses; a rule-based `diagnose_issues` tool encodes known RL-trading failure modes (under-performing random, over/under-trading, fee drag, large drawdowns, negative Sharpe) and returns recommendations.

### Web Frontend
- Next.js 16 (App Router), React 19, TypeScript, Tailwind 4, lightweight-charts, Zustand, TanStack Table. 8 pages, 41 components.
- Built and data-driven: a dashboard rendering exported runs (candlesticks, indicator overlays, equity curve, trade table, metrics) and a live trading view streaming the local WebSocket simulation.
- Marketing pages (landing, pricing, tournaments, marketplace, learn) are built UI driven by mock data, previewing the intended product.

---

## Notable Technical Decisions

- Look-ahead leakage defended at two independent layers (causal indicator test + next-candle execution).
- Differential Sharpe reward over raw PnL for a dense, risk-adjusted per-step signal.
- Indicators precomputed into Parquet to keep the training loop fast over millions of rows.
- Modular indicator registry with per-group MLP + attention, replacing brittle hard-coded rules with learned signal weighting.
- Two environments (episodic for training, continuous for serving) sharing one broker/portfolio.
- MCP server as the agent-drivable automation surface, mirroring the DTL approach.
- Config-as-data (Pydantic + YAML), editable through an MCP tool.

---

## Status and Scale

- ~7,100 lines of Python, ~9,000 lines of frontend TS/TSX, 14 MCP tools, 135 pytest test functions.
- ~1.1 GB of processed Parquet across BTC, ETH, SOL, XRP (Binance) and BTC (Bitstamp), 1-minute candles, 2019–2026.
- ~64 training run directories, 2,134 checkpoints, 23 finalized models with metadata — evidence of heavy experiment-driven iteration.
- Runs entirely locally (Python venv; Next.js dev/build; localhost dashboard and WebSocket). Nothing deployed, no accounts, no live trading.
- Roadmap (designed, not built): managed web platform for users to train their own agents, backend API/database, managed cloud GPU training, accounts/payments, tournaments, model marketplace, live exchange trading.

---

## Relevance

This project is the clearest demonstration of transferable RL/ML system-building: the same architecture pattern from [[Dynamic Train Labs]] (physics/market simulator + Gymnasium env + Stable-Baselines3 + MCP-driven train/eval loop) applied to an entirely different domain, showing the DTL approach was not a one-off. It also draws on personal cryptocurrency and technical-analysis knowledge (see also [[Solana Monitoring System]]).
