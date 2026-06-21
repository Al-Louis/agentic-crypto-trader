# Project Overview

A neutral starting point for the [[BNB Hack - AI Trading Agent Edition]] Track 1 entry. See
[[Index]] for the full map of content. This note frames *what the project is and the
decisions behind and in front of it*. The strategy question — once an open design space — is
now **decided**: a selective volatility-ignition reinforcement-learning agent (see
[[Trading Strategies]], [[AI Training]]).

## What this is

An **autonomous, self-custody trading agent** that runs on BNB Smart Chain (BSC). The core
loop is: read market and on-chain data → decide → sign and execute its own transactions →
repeat, hands-off, inside rules it can't violate. During the live window it trades real
assets on BSC and is scored on the result.

The agent is a decision core wrapped around a few mandated surfaces (see [[Tech Stack]]).
The trading logic is kept modular behind a clean interface — the surrounding infrastructure
(execution, custody, data, guardrails) is strategy-agnostic — but the decision core itself is
now committed: the learned volatility-ignition policy described below.

## How it's scored (Track 1)

Objective and mechanical — worth designing toward directly:

- **Ranked on live PnL** over the held-out trading window (June 22–28).
- **Max drawdown cap is a hard gate** — exceed it (the brief gives ~30% as an example) and
  the entry is disqualified regardless of return.
- **Minimum activity:** at least one trade per day across the week.
- **Must hold in-scope assets at the start** and keep capital deployed (an hour beginning
  at ≤ $1 scores 0%). Only the fixed eligible-token list counts.
- **Discretionary special prizes** also exist (best use of each SDK), judged on technical
  execution, originality, real-world relevance, and demo — a separate, lower-variance path
  to recognition alongside the live-PnL leaderboard.

A useful one-line summary of the objective: **the most return without breaching the
drawdown gate, traded hands-off with self-custody intact.** Survival is a first-class goal,
not an afterthought — see [[Security and Encryption]] for the custody side.

## The stack — four surfaces

The "three SDKs" framing undercounts; in practice the build touches four distinct surfaces.
Full detail belongs in [[Tech Stack]]; the shape:

| Surface | Role |
|---------|------|
| **Trust Wallet Agent Kit (TWAK)** | Execution + self-custody signing. The sole execution layer; trades and competition registration go through the `twak` CLI / MCP. |
| **CoinMarketCap AI Agent Hub (MCP)** | Market / derivatives / on-chain / social / news data **and x402** pay-per-request. (x402 lives in the Agent Hub MCP, not in the standalone `cmc` CLI.) |
| **BNB AI Agent SDK** (Python) | Agent runtime + on-chain **identity** (ERC-8004) and agentic-commerce protocols — *not* an execution layer. Trades route through TWAK, not this. |
| **BscScan API** | On-chain analytics — wallet activity, transfers, token/holder data. The source for any wallet-monitoring or on-chain-signal logic (see [[Real-time Monitoring]]). |

Runtime is **Python** (the BNB SDK is a Python toolkit), which suits a data/ML-leaning
build. Reference docs for all of the above are mirrored under `References/`.

## Architecture approach

Neutral patterns that hold regardless of strategy:

- **Event-driven loop:** monitor → evaluate → decide → execute → confirm. A proven shape
  for real-time on-chain work (see [[Real-time Monitoring]]).
- **Modular, swappable decision core.** Keep the strategy logic behind a clean interface so
  it can be replaced or tuned without touching execution, custody, or guardrails.
- **Decision logic as a pure, testable module** — deterministic, exercised against recorded
  data and fixtures so behavior can be validated offline (see [[Simulated Market]]) before
  any live capital is at risk.
- **Guardrails as hard, external limits** enforced in code around the signing call (token
  allowlist, per-trade and daily caps, slippage protection, drawdown stop) — not as
  suggestions the model can talk itself out of.
- **MCP-driven orchestration.** Both the CMC Agent Hub and TWAK expose MCP, so the agent's
  tool surface is uniform and inspectable.

## Strategy — decided

The decision core is a **selective volatility-ignition reinforcement-learning agent**
(RecurrentPPO, LSTM-256), trained on the simulated market and certified on a held-out frozen
TEST split; the deployed champion is `sbq-s1` (detail in [[Trading Strategies]], [[AI Training]]).
It was reached by evaluating these candidate directions on merit:

- Technical-indicator / momentum strategies (RSI, MACD, regime detection).
- Wallet-monitoring / copy-style strategies driven by on-chain activity.
- Sentiment- or news-driven signals (e.g. a social scanner for breaking events).
- **Learned policies (RL/ML) trained against a simulated market — the direction taken.**
- Risk-/regime-aware overlays and filters on any of the above.

The surrounding infrastructure is identical regardless of strategy, which is why the execution
loop was built in parallel while the decision core was settled.

## Key constraints & open questions

The genuine unknowns to resolve early — several gate everything downstream:

1. ~~**Autonomous signing while self-custodial (blocker).**~~ **RESOLVED 2026-06-11:** proven
   live — a guardrailed dust trade signed unattended via TWAK (keychain password resolution)
   and confirmed on BSC. See [[Security and Encryption]] / [[TWAK Spike Runbook]].
2. **Hosting & key management (decided, build pending).** Live-week host = **AWS EC2** (small
   Linux instance, systemd, hardened env-file); the competition wallet is created *on the
   box* so keys never transit. Decided 2026-06-11 after a desktop WSL stall demonstrated the
   residential-host failure mode. See [[Remote Capabilities]] and [[Security and Encryption]].
3. **On-chain data reach.** How much wallet/transfer/holder detail is cheaply and quickly
   available via BscScan (and CMC chain stats) for whatever on-chain logic the strategy
   needs? See [[Real-time Monitoring]]. (BscScan free tier is ETH-only — BSC reads route via
   GoPlus + public RPC; see [[Tech Stack]].)
4. **x402 path.** What's the minimal genuine pay-per-request use in the loop (data or
   inference) via the Agent Hub MCP?
5. ~~**Registration mechanics + deadline.**~~ **RESOLVED 2026-06-11:** `twak compete
   register/status` confirmed (on-chain deadline reads **June 25**; June 22 stays the working
   deadline since the scored window starts then). ERC-8004 identity alignment proven on
   `bsctestnet` — identity, registration, and trading all sign from one TWAK wallet, zero key
   export. The mainnet mint requires a hosted agent-card `--uri` first.
6. **Liquidity & slippage.** Many eligible tokens are thin; execution cost assumptions must
   match real BSC pool depth, not centralized-exchange volume. Note: **BSC testnet has no
   real DEX liquidity**, so testnet "trading" validates nothing about fills — the validation
   ladder is paper-on-live-data → mainnet *dust* trades through the guardrails.

## Build path

A front-loaded sequence — stand up the unfamiliar execution/custody layer first; leave the
familiar strategy logic for later. Aligned to the [[Index]] timeline:

1. **Stack spike.** ✅ *Custody half done 2026-06-11* — a real guardrail-checked dust trade
   landed on BSC via TWAK ([[TWAK Spike Runbook]]). *Remaining:* the Agent Hub returns data;
   the BNB SDK agent runs (identity already proven via TWAK's native `erc8004`).
2. **Execution loop.** 🔄 *The active build* — wire read → decide (stub/champion) → sign →
   execute → confirm in autonomous mode, with the guardrail scaffold (built) and drawdown
   tracking enforced in code. Paper-run locally first; **EC2 provisioning in parallel**.
3. **Decision logic.** 🔄 Implement and validate the chosen strategy against a simulated /
   recorded-data environment (see [[Simulated Market]], [[AI Training]] — RL tuning active).
4. **June 16 — Track 1 proof of concept (go/no-go).** The on-chain half is met (a real trade
   signed and landed, guardrails active); the continuous loop is what remains by the gate.
5. **Harden + forward-run (June 16–21).** Deploy the loop to **AWS EC2 in paper mode** on
   live data; wire the `trading/` publish + `/apentic/trading` monitoring page with a
   dead-man heartbeat; validation ladder is **paper → mainnet dust** (testnet trading ruled
   out — no real liquidity); one dust trade from the production host as the end-to-end smoke.
6. **Register + submit.** Create the competition wallet **on the EC2 box**, host the agent
   card, mint the ERC-8004 identity, register **before June 22** (on-chain deadline reads
   June 25 — don't lean on the slack); public repo, demo video, strategy writeup.
7. **Live window (June 22–28).** Fund the live bankroll (user-sized); operate hands-off:
   ≥1 trade/day, stay under the drawdown cap, keep capital deployed. Post-stability:
   sponsor-tool expansion for special-prize coverage.

## Reusable prior work

Relevant groundwork to draw on, without constraining the design — see [[TradeSim]]:
RL training pipelines, simulated-market / backtesting infrastructure, honest evaluation
against baselines, real-time on-chain monitoring, and MCP server development.
