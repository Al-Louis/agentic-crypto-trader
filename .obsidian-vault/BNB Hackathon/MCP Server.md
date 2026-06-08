# MCP Server — Command-Set Design

The project's **isolated MCP server** (`mcp-server/`, registered in `.mcp.json` as `trader`).
It exposes this project's operations as deterministic tools so the [[Index|agents]] and
`/workflows` drive the build → data → simulate → train → execute loop programmatically —
the same pattern that made [[TradeSim]]'s 14-tool server effective. See [[Project Overview]]
and [[Tech Stack]] for context.

## Design principles

- **One operation = one tool.** Small, composable, machine-readable I/O (JSON in/out).
- **Three safety tiers, enforced by the server** (see below). Read is free; execution is
  hard-gated.
- **Deterministic and inspectable.** Every tool supports a `dry_run` where meaningful and
  returns structured results a workflow can branch on.
- **The server enforces guardrails**, not the model. [[Allowlist,]] per-trade/daily caps,
  slippage, and the drawdown stop live in the `execute_trade` code path; out-of-policy calls
  are refused, not negotiated.
- **Secrets never in the server or repo.** Keys/API keys load from a local `.env` / secure
  store at runtime; self-custody signing stays local via TWAK.
- **Python**, stdio transport — matches the BNB SDK runtime and the [[TradeSim]] lineage.

## Safety tiers

| Tier | Meaning | Rule |
|------|---------|------|
| 🟢 **READ** | No side effects (data, status, reports) | Always allowed; auto-approved. |
| 🟡 **SIMULATE** | Dry-run / local compute, no chain writes (backtests, trade simulation, training) | Allowed; may be long-running (background). |
| 🔴 **EXECUTE** | Mutates chain / spends real funds (live trade, registration) | Hard-gated: guardrails enforced server-side, explicit enable flag required, small caps in dev, every tx hash logged. |

## Tool catalog

### Data — CMC Agent Hub + BscScan  🟢
*Owner: [[quant-analyst|quant-analyst]], [[onchain-custody-engineer|onchain-custody-engineer]] · Phase 2*

| Tool | Purpose | Key inputs → output |
|------|---------|---------------------|
| `eligible_tokens` | The fixed competition token universe + metadata | — → token list (id, symbol, address, liquidity) |
| `cmc_market` | Latest quote/metrics for token(s) | ids, convert → price, volume, market regime |
| `cmc_history` | OHLCV history | id, period, interval → candle series |
| `cmc_token_info` | Profile + chain stats | id → info, chain stats |
| `bscscan_wallet_txs` | Transactions for an address | address, range → tx list |
| `bscscan_transfers` | Token transfers for address/token | address|token → transfer list |
| `bscscan_token_holders` | Holder distribution | token → holders, concentration |

### Sentiment / News  🟢
*Owner: [[quant-analyst]] (later `sentiment-scanner`) · Phase 3*

| Tool | Purpose | Key inputs → output |
|------|---------|---------------------|
| `cmc_news` | Latest CMC news/trending | filters → items |
| `social_scan` | X.com / news scan for breaking events (hacks, adoption) | query, window → flagged events |

### Simulation / Backtest — Simulated Market  🟡
*Owner: [[quant-analyst]], [[rl-ml-trainer]] · Phase 4 (see [[Simulated Market]])*

| Tool | Purpose | Key inputs → output |
|------|---------|---------------------|
| `prepare_dataset` | Build clean, causally-validated feature set | token, period → dataset ref (+ leakage check) |
| `run_backtest` | Run a strategy through the simulated broker | strategy ref, period → run id |
| `backtest_report` | Metrics suite vs baselines | run id → Sharpe/Sortino/Calmar/maxDD/VaR/fees |
| `run_baseline` | Buy&Hold / SMA / RSI / Random through same broker | baseline, period → metrics |

### Training — AI Training  🟡
*Owner: [[rl-ml-trainer]] · Phase 4+ (see [[AI Training]])*

| Tool | Purpose | Key inputs → output |
|------|---------|---------------------|
| `start_training` | Launch a training run (background subprocess) | config → run id |
| `training_status` | Progress + live metrics of a run | run id → status, metrics |
| `list_models` / `model_info` | Enumerate / describe finalized models | — / model id → metadata |
| `evaluate_model` | Held-out eval vs baselines | model id, period → metrics |
| `diagnose_run` | Rule-based failure-mode checks → recommendations | run id → issues (under-random, over/under-trading, fee drag, drawdown, neg-Sharpe) |

### Strategy — Trading Strategies  🟡
*Owner: [[market-indicator-expert]] · Phase 4 (see [[Trading Strategies]])*

| Tool | Purpose | Key inputs → output |
|------|---------|---------------------|
| `register_strategy` | Store a strategy spec behind the decision-core interface | spec → strategy ref |
| `evaluate_strategy` | Backtest + report a spec in one call | strategy ref, period → report |

### Execution / Custody — TWAK  🔴 (read parts 🟢)
*Owner: [[onchain-custody-engineer]], [[principal-engineer]] · Phase 2–3 (see [[Security and Encryption]])*

| Tool | Tier | Purpose | Key inputs → output |
|------|------|---------|---------------------|
| `wallet_status` | 🟢 | Agent wallet balances/positions | — → holdings, value |
| `guardrails_get` | 🟢 | View active hard limits | — → allowlist, caps, slippage, drawdown stop |
| `guardrails_set` | 🟡 | Configure limits (config mutate, not chain) | limits → confirmation |
| `simulate_trade` | 🟡 | Dry-run a trade: route, slippage, cost, guardrail check | token, side, size → projected fill, pass/fail |
| `execute_trade` | 🔴 | Sign via TWAK + submit on BSC, **behind guardrails** | token, side, size → tx hash (or refusal + reason) |
| `competition_register` | 🔴 | `twak compete register` before June 22 | — → registration tx hash |

### Monitoring — Real-time  🟢
*Owner: [[onchain-custody-engineer]] · Phase 3 (see [[Real-time Monitoring]])*

| Tool | Purpose | Key inputs → output |
|------|---------|---------------------|
| `watch_wallets` | Register addresses to monitor | addresses → watch id |
| `recent_activity` | Recent monitored on-chain events | watch id → events |
| `portfolio_pnl` | Current PnL / hourly returns (mirrors scoring) | — → pnl, hourly series, drawdown |

> **Vault knowledge tools deliberately omitted.** Agents read/write the vault with native
> file tools; no MCP layer needed for that. Add one only if workflows need programmatic note
> access.

## Build phasing (incremental, from Phase 2)

| Phase | Tools to ship | Unlocks |
|-------|---------------|---------|
| **2 · Stack spike** | `eligible_tokens`, `cmc_market`, `cmc_history`, `bscscan_wallet_txs`, `wallet_status`, `guardrails_get`, `simulate_trade`, `execute_trade` (dust), `competition_register` (dry) | The **June 16 PoC**: a real guarded trade on-chain |
| **3 · Loop + monitoring** | `guardrails_set`, `watch_wallets`, `recent_activity`, `portfolio_pnl`, `cmc_token_info`, `bscscan_transfers`/`token_holders`, `cmc_news` | Autonomous loop with live risk + PnL visibility |
| **4 · Strategy + training** | `prepare_dataset`, `run_backtest`, `backtest_report`, `run_baseline`, `register_strategy`, `evaluate_strategy`, `start_training`, `training_status`, `evaluate_model`, `diagnose_run`, `list_models` | The full train → evaluate → diagnose loop, workflow-drivable |
| **Later** | `social_scan` | Sentiment overlay |

## How workflows use this

A `/workflows` script can fan agents across tools deterministically — e.g. *evaluate a
strategy*: `market-indicator-expert` → `register_strategy` → `evaluate_strategy` →
`quant-analyst` verifies `backtest_report` against baselines → `diagnose_run` → loop until
the report clears the bar. Execution-tier tools stay behind the guardrail gate even inside a
workflow.

> **Open items:** confirm the CMC Agent Hub MCP vs the `cmc` CLI as the data backend (x402
> lives in the Agent Hub MCP — see [[Tech Stack]]); confirm the exact `twak compete register`
> surface; decide whether `execute_trade` wraps the `twak` CLI or the TWAK MCP directly.
