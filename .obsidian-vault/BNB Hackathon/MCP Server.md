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

### Training / RL experiment loop — AI Training  🟡
*Owner: [[rl-ml-trainer]] · Phase 4+ (see [[AI Training]], [[Experiment Log]], [[Remote Capabilities]])*

The verbs of the **automated reward-shaping loop** (Phase A of the experiment-loop automation). The
MCP server runs on the **laptop**, drives the keyless training desktop over SSH, and reads results
from the CDN (`data.alexlouis.dev`) — it holds **no state itself** (fire-and-poll, survives a
restart). It wraps `remote_train` + the committed diagnostics (`compare_seeds`,
`diag_deviation_alpha`, `probe_obs_alpha`).

| Tool | Tier | Purpose | inputs → output |
|------|------|---------|-----------------|
| `rl_obs_probe` | 🟢 | Cheap reward-bound vs capacity-bound check, **no training** | split, horizon → OOS IC, per-feature corr, verdict |
| `rl_train` | 🟡 | **Launch a sweep, safely** — the guard tool (see below) | reward_config, seeds, split, timesteps, smoke, final_verdict → prefix, run_ids, state |
| `rl_status` | 🟢 | Which seeds published + liveness (CDN + one tiny ssh) | prefix → published[], running, load |
| `rl_compare` | 🟢 | Per-seed + mean return / DD / Sharpe vs the rung-0 baseline | prefix, seeds → per_seed, mean, spread, worst_dd, gate_pass |
| `rl_diagnose` | 🟢 | Deviation-alpha corr + trade count + action dist + the full gate | prefix → corr, straddle, the verdict packet the agent reads |
| `rl_kill` | 🟡 | **Stop a sweep by specific PID** (driver bash + train main) | prefix → killed_pids, load_after |
| `experiment_record` / `experiment_champion` | 🟢 | Append to the ledger / current best + repro command | … → ledger id / champion |

**Discipline baked into the tools (so an autonomous loop can't re-learn it the hard way — these are
the scars from [[Remote Capabilities]]):**
- **SSH via the Windows OpenSSH** (the MSYS/bash ssh can't route the tailnet — it hangs); every
  status reply kept **< ~2 KB** (the path-MTU black hole). One `_ssh()` helper enforces both.
- **`rl_train` is launch-once by construction:** refuses if a sweep is already running (`pgrep`
  guard), syncs + preflights data, `mkdir`s `runs-rl`, launches detached, then **waits 60–90 s and
  verifies exactly one clean run** (`load ≈ n_envs`) — auto-aborting on a stacked run. This makes the
  Vmmem-stacking incident *structurally impossible*.
- **`rl_train(split="test")` refuses without `final_verdict=True`** — the meta-overfitting guard at
  the tool layer: tuning runs go to **val**, the frozen test can't be auto-burned ([[Experiment Log]]).
- **`rl_train` runs the 100 k smoke + smoke-gate first** (alive + straddle) and won't sweep a dud.
- **`rl_kill` targets specific PIDs, never `kill -- -<PGID>`** (the group-kill that took *tailscaled*
  down and dropped the box off the tailnet).

These **supersede** the placeholder `start_training` / `training_status` / `evaluate_model` /
`diagnose_run`. Phase B composes them into a `run_rl_experiment` workflow (probe-gate → smoke-gate →
sweep → diagnose → verdict); Phase C is the agent-driven `/loop` with a compute budget + the
[[quant-analyst]] block-bootstrap CI gate (so a lucky seed can't be crowned champion).

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
| **4 · Strategy + training** | `prepare_dataset`, `run_backtest`, `backtest_report`, `run_baseline`, `register_strategy`, `evaluate_strategy` | Backtest/strategy evaluation, workflow-drivable |
| **4A · RL experiment loop** | `rl_obs_probe`, `rl_train`, `rl_status`, `rl_compare`, `rl_diagnose`, `rl_kill`, `experiment_record`/`champion` | The safe **probe → smoke → sweep → diagnose** loop; the substrate for the automated reward-shaping search (workflow in 4B, agent-`/loop` in 4C) |
| **Later** | `social_scan` | Sentiment overlay |

## How workflows use this

A `/workflows` script can fan agents across tools deterministically — e.g. *evaluate a
strategy*: `market-indicator-expert` → `register_strategy` → `evaluate_strategy` →
`quant-analyst` verifies `backtest_report` against baselines → `diagnose_run` → loop until
the report clears the bar. Execution-tier tools stay behind the guardrail gate even inside a
workflow.

The **RL experiment loop** is the marquee case (mirrors the manual cycle in [[Experiment Log]]):
`rl_obs_probe` (cheap reward-vs-capacity gate) → `rl_train` (safe launch, smoke-gated, on **val**)
→ poll `rl_status` → `rl_diagnose` → `experiment_record`. The [[rl-ml-trainer]] reads the verdict +
`experiment_champion` and proposes the next `reward_config`; the [[quant-analyst]] adjudicates the
gate (mean > rule, corr > 0, gate-safe, CI-validated). A `/loop` drives this across iterations under
a compute budget, promoting a champion and ending when one beats the rung-0 rule on the **frozen
test** (`final_verdict=True`) — the one place the test split is spent.

## As-built — 4A modernized to the rd-era (2026-06-11, during the rdL sweep)

The 4A tier as first built predated the whole rung-1b arc; brought up to the pipeline that
actually runs (one manual-day = one loop-iteration parity):

- **`rl_train`** — `reward_config` whitelist extended to every rd-era knob (substrate:
  `rule_default`/`exit_commit`/`dust_usd`/`tp_rungs`/`loss_floor`/`det_blacklist`; curriculum:
  `crash_train`/`crash_eval`/`universe_mode`/`k`/`vol_target`; obs: `harvest_obs`/`eval_prepad`;
  arch: `recurrent`/`lstm_size` + the rest). Run-ids are **sha-stamped on the box**
  (`{prefix}-${SHA}-s<seed>`, the ec1e487 convention) so an automated launch can never recreate
  the overwrite bug, and the sha-only leaderboard includes its runs. Smoke gate is now
  **discrete-aware** (levels-used, not the continuous mean-cap that would have refused every
  healthy rd policy) and its `[eval]`-line parser was fixed for the `primary=<split>` format (a
  latent break — the old regex matched nothing current). Recurrent smokes get a longer timeout.
- **`rl_verdict`** *(new)* — the per-regime (val/test/crash) table from each bundle's `regimes`
  block: per-seed rows, seed-means, worst-seed DD, per-regime mean gate + `overall_pass`. The
  exact table every manual sweep verdict used; the loop's primary read.
- **`rl_forensics`** *(new)* — `diag_token_events` as a tool (prompt-by-prompt entry/skip/cool
  labels, component breakdown at given timestamps, the rule's own trades) — the truth-teller
  behind the veto/false-flag/detonation findings, now agent-callable before any rule change.
- **`experiment_record`** — `sha_only` (default: the post-ec1e487 valid era only) +
  `publish=True` (push leaderboard.json to the data host + CloudFront invalidation).

Tests: `tests/test_mcp_loop.py` (the full rdL config dict accepted; sha-stamping + sequencing
asserted; discrete smoke pass/fail; regime-verdict means/binding/DQ on fixtures).

## As-built — the loop DRIVER (4B/4C, 2026-06-11)

The piece that strings the 4A tools into the autonomous iterate loop:

- **`trader.experiment.driver`** — the stateful state machine (`experiments/loop_state.json`):
  `idle+queue → launched → running → verdict → record → decide → idle/halted`. Mechanical steps in
  code (launch rides the guarded `rl_train` flow; poll = published-count + one tiny ssh; verdict =
  the per-regime `regime_verdict`; record = sha-only ledger; decide = `loop_control.decide` with
  the drift alarm + iteration budget). **The judgment step is deliberately absent**: a tick that
  returns `needs_proposal=True` is the driving agent's cue to analyze + queue ONE config. The
  north star is the WORST regime's margin-vs-Buy&Hold (`result_from_verdict`); a refused/dead
  launch or drift alarm HALTS for human review. The loop never spends the frozen test (val only;
  `promote` fires only from a human `final_verdict` run).
- **Surfaces:** CLI `scripts/rl_loop.py {status,step,propose,reset}`; MCP tools `rl_loop_status` /
  `rl_loop_step` / `rl_loop_propose` / `rl_loop_reset`.
- **The driving agent:** the `/rl-loop` project skill — one wake = one tick; on `verdict` it logs
  standings to [[Experiment Log]], runs `rl_forensics` when behavior looks wrong, consults the
  refuted-levers record so nothing dead is re-proposed, and proposes one single-variable config.
  Driven by `/loop` (self-paced wakes ≈ sweep duration) or a cron; a `halted` tick stops the loop
  and notifies the human. Hard rules restated in the skill: val-only, one variable per config,
  never launch around the driver, the desktop is shared.
- Tests: `tests/test_loop_driver.py` (full offline cycle, drift-alarm halt + soft reset,
  refused-launch requeue, dead-sweep halt, worst-regime margin distillation).

> **Open items:** confirm the CMC Agent Hub MCP vs the `cmc` CLI as the data backend (x402
> lives in the Agent Hub MCP — see [[Tech Stack]]); confirm the exact `twak compete register`
> surface; decide whether `execute_trade` wraps the `twak` CLI or the TWAK MCP directly.
