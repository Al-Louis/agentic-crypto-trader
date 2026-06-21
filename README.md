# agentic-crypto-trader

An **autonomous, self-custody crypto trading agent** for the BNB Chain "AI Trading Agent
Edition" hackathon (Track 1). It reads on-chain market data, decides with a learned policy,
and **signs and lands its own swaps on BSC** via the Trust Wallet Agent Kit (TWAK) — hands-off,
inside hard external guardrails.

The decision core is a **selective volatility-ignition reinforcement-learning agent**
(RecurrentPPO, LSTM-256), trained and certified offline on a held-out frozen **test** split,
then run live against a fixed eligible-token universe. Execution, custody, data, and guardrails
are strategy-agnostic infrastructure around it.

## Status — what's proven

- **Self-custody, autonomous.** The agent signs and lands its own BSC swaps via TWAK; keys
  never leave the local wallet (`~/.twak/wallet.json`, encrypted). No human in the signing loop.
- **Live execution loop proven on-chain.** The guardrail-wrapped signing path has landed a real
  **BNB↔USDT round-trip on BSC mainnet** on the dev wallet (`scripts/live_exec_smoke.py`).
- **Committed strategy.** A selective volatility-ignition RL agent (RecurrentPPO/LSTM). The
  deployed champion (`ppo-event-rdLe4-sbq-3c84b4a-s1`, "sbq-s1") was selected on a validation
  split and certified on a held-out **frozen test** split (never used for tuning).
- **Hard guardrails, fail-closed.** Allowlist, per-trade and daily USD caps, slippage bound, and
  a max-drawdown stop wrap every signing call — checked in **two phases** (intent first, then
  re-applied to the realized quote). Out-of-policy ⇒ refused with coded reasons, never silently
  adjusted. Any state it can't compute ⇒ refuse.
- **Running now.** The live event-driven harness runs on an always-on host (EC2) in **paper**
  mode against the champion; a triple-gated live-signing flip is staged for the scored window.
- **Results publish** as static JSON to a dashboard/leaderboard (`data.alexlouis.dev`).

> The agent defaults to **paper**. Real signing requires all three of
> `TRADER_MODE=live`, `AGENT_ALLOW_LIVE=1`, and `AGENT_LIVE_CONFIRM=1` — miss any one and it
> stays paper.

## Evidence

- **On-chain execution proof** — a real BNB↔USDT round-trip, signed by the agent's guardrail
  path and landed on BSC mainnet (dev wallet, dust-sized):
  [SELL BNB→USDT](https://bscscan.com/tx/0xac75ff719de08e81fcfad6f838931b372613ac1137c4db6a64094da84a83f380)
  · [BUY USDT→BNB](https://bscscan.com/tx/0xf4b5dc29dd191298622a7a0daa6942a3493e213b0b8f380bca9846cb6b3d4501).
- **Competition entry** — the live wallet
  [`0x08a0…d65C`](https://bscscan.com/address/0x08a08ba2BBB2100A3760C244CbB84BA0202fd65C) is
  registered on-chain (`twak compete register`,
  [registration tx](https://bscscan.com/tx/0x9f6f3ceef515549f74294527c0c644dee1f2d9275991b343e7ae5bc95fbc1dc1))
  and funded.
- **Strategy certification** — champion `sbq-s1` on a held-out **frozen TEST** split (5 cold
  weeks, fresh $10k each, never used for tuning): **+58.6% sum · +11.7%/wk mean · 5-of-5
  winning weeks · 8.8% worst-week drawdown · DQ-safe**.
- **Live results** — published to **[data.alexlouis.dev](https://data.alexlouis.dev)**.

## How it works

```
on-chain pool data (GeckoTerminal / chain pool-event layer)
        → features / indicators
        → RL decision core (RecurrentPPO + LSTM; selective volatility ignition)
        → hard guardrails (allowlist · caps · slippage · drawdown, two-phase)
        → TWAK signs + lands the swap on BSC
        → telemetry published to the dashboard
```

Market data is **on-chain DEX pool OHLCV from GeckoTerminal** — keyless by default, or keyed via
`COINGECKO_API_KEY` to lift rate limits (identical pools either way). This is the **same source
for training and live serving**, deliberately chosen over CEX-spot data to avoid train/serve
skew (the champion was trained on this exact feed). CoinMarketCap is used for universe/contract
resolution and the x402 surface, not as the price feed; ccxt/Binance.US supplies only the
BTC/BNB factor anchor.

## Orientation

- **Read first:** `CLAUDE.md` (auto-loaded), then run `/orient`.
- **Knowledge base:** `.obsidian-vault/BNB Hackathon/` — start at `Index.md` → `Project Overview`,
  `Tech Stack`, then `Build Log`, `Experiment Log`, `Trading Strategies`, `AI Training`,
  `Dashboard Leaderboard`, `Live Forward-Run Harness`.
- **SDK docs:** mirrored under `.obsidian-vault/References/`.

The repo has two halves: **`.obsidian-vault/`** is the living design record and results narrative;
**`src/`** is the agent codebase — the deliverable.

## Quickstart

The dev laptop is Windows; the live host is Linux/EC2. Both paths are below.

**Linux / macOS (the deploy target):**

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev,data,training]"     # add ,remote for the S3 publish path
cp .env.example .env                        # then fill in (git-ignored)
pytest                                       # ~540 tests
```

**Windows PowerShell:**

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -e ".[dev,data,training]"
Copy-Item .env.example .env
pytest
```

Install extras:
- `dev,data` — offline research + backtest pipeline (pandas, pyarrow, ta, ccxt).
- `training` — the RL surface: gymnasium, stable-baselines3, sb3-contrib (+ CPU torch).
  **Required** to run/evaluate the champion or the event harness. RL **training** runs on the
  desktop trainer only (the laptop's Python has no torch wheel); evaluation/serving runs anywhere
  `training` is installed.
- `remote` — boto3, for publishing run bundles to S3 / the dashboard.

The project MCP server is registered in `.mcp.json` (its `command` is a Windows venv path —
change to `.venv/bin/python` on POSIX):

```bash
python -m trader.mcp_server      # or the `trader-mcp` console entry point
```

It exposes ~20 tools: the RL **train → evaluate → diagnose** loop (`rl_train`, `rl_loop_step`,
`rl_diagnose`, `rl_verdict`, `rl_forensics`, …), the experiment registry (`experiment`,
`experiment_champion`, `diagnose_run`), plus `health` and an `eligible_tokens` stub.

## Run the agent

```bash
# Train the RL policy (desktop trainer; needs [training]) and publish its eval bundle
python scripts/train_rl.py --timesteps 300000 --n-envs 8 --run-id ppo-exposure-001

# Evaluate a saved checkpoint across trailing windows (where it holds / breaks)
python scripts/simulate.py --run-dir runs-rl/<run-id>
python scripts/simulate_weekly.py --run-dir runs-rl/<run-id>

# Drive the autonomous iterate loop (train → evaluate → diagnose → propose next config)
python scripts/rl_loop.py step

# Live event-driven harness — PAPER (safe; no signing). This is what runs on EC2.
python -m trader.agent.event_agent --run-dir runs-rl/<champion> --once

# Live harness with the guardrail signing path — DRY-RUN routes every fill through the
# quote-only check so you can watch guardrails fire without spending funds
python -m trader.agent.live_event_agent --run-dir runs-rl/<champion> --once --dry-run

# Real signing requires ALL THREE gates set; otherwise it refuses and stays paper:
#   TRADER_MODE=live AGENT_ALLOW_LIVE=1 AGENT_LIVE_CONFIRM=1

# Prove the signing path end-to-end on BSC (dry-run by default; --execute spends a dust amount)
python scripts/live_exec_smoke.py

# Publish results to the dashboard/leaderboard (needs [remote] + creds)
python scripts/publish_leaderboard.py
```

## Layout (as built)

```
src/trader/
  config.py        .env loader
  data/            universe + on-chain OHLCV (geckoterminal, cmc, dexscreener, goplus, select)  [built]
  features/        indicators + factor + regime (leakage-guarded)                                [built]
  sim/             metrics, broker, backtest, strategies, IC, crash                              [built]
  strategy/        decision core: rung0 baseline + candidate                                     [built]
  risk/            hard guardrails: checks (fail-closed), ledger, policy                          [built]
  execution/       TWAK live signing: execute (two-phase guardrail path) + twak_cli   [built · proven on BSC]
  agent/           live event-driven harness: event_agent, event_runner,
                   live_event_agent, event_live, publish, compliance, feed, …          [built · EC2 paper]
  chain/           read-only on-chain pool-event layer (rpc, events, collector, panels)          [built]
  train/           RL stack: event_env, event_reward, curriculum, weekly_eval, gym_env           [built]
  experiment/      iterate-loop driver, champion selection, remote launch                         [built]
  report/          Apentic dashboard/leaderboard publisher (static JSON → data.alexlouis.dev)     [built]
  mcp_server/      project ops server: RL train/diagnose loop + experiment tools + health         [built]
  monitoring/      wallet/PnL watching                                                            [stub]
scripts/           CLIs: train_rl, train_event, simulate, simulate_weekly, rl_loop,
                   live_exec_smoke, publish_leaderboard, + offline research pipeline
data/              generated caches (ohlcv / anchor / features) — git-ignored
deploy/            EC2 runbook: systemd units (event-agent, live-event-agent, daily-scan),
                   IAM policies, host-verify script, env template
tests/             ~540 pytest functions across 54 files
```

## Offline research pipeline (early / universe-build)

The original two-factor "Bitcoin-is-King residual" research line still resolves and is used for
universe selection and forensics. It is **secondary** to the RL agent above.

```bash
python scripts/screen_universe.py
python scripts/resolve_contracts.py
python scripts/select_universe.py --exclude SHIB,BAS,FORM --pin LTC:anchor
python scripts/forensics.py
python scripts/download_ohlcv.py --selection data/selection.json --timeframes day,hour
python scripts/download_anchor.py            # BTC/BNB factor anchor (ccxt / Binance.US)
python scripts/build_factor_features.py
python scripts/run_backtest.py
python scripts/tail_sweep.py
python scripts/oos_validate.py
```

## Security

- **Secrets never committed.** Keys, mnemonics, API keys, and wallet files live in a local,
  git-ignored `.env` (and `~/.twak/`). Self-custody signing stays local — TWAK never exposes the
  mnemonic.
- **Guardrails are hard, external limits** in code around the signing call — not prompt
  suggestions. They fail closed.
- **Validate offline before live capital.** Strategy logic is pure and tested against
  recorded/simulated data before it touches mainnet.
- The training-host SSH target is configured via `TRAINER_SSH_HOST` in your local `.env`
  (the committed default is a non-routable placeholder); there is no working default to clone.

## License

No license has been added yet. Until one is, treat this code as **all rights reserved** — no
permission is granted to use, copy, modify, or distribute it.
