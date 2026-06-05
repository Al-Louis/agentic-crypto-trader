# Project Overview

A neutral starting point for the [[BNB Hack - AI Trading Agent Edition]] Track 1 entry. See
[[Index]] for the full map of content. This note frames *what the project is and the
decisions in front of it* — it deliberately does not commit to a single trading strategy;
that's an open design space (see [[Trading Strategies]]).

## What this is

An **autonomous, self-custody trading agent** that runs on BNB Smart Chain (BSC). The core
loop is: read market and on-chain data → decide → sign and execute its own transactions →
repeat, hands-off, inside rules it can't violate. During the live window it trades real
assets on BSC and is scored on the result.

The agent is a decision core wrapped around a few mandated surfaces (see [[Tech Stack]]).
The trading logic that drives it is intentionally modular and swappable — the surrounding
infrastructure (execution, custody, data, guardrails) is the same regardless of which
strategy is chosen.

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

## Open design space — strategy

The decision core is *not yet chosen*. Candidate directions, to evaluate on merit (detail
in [[Trading Strategies]], [[AI Training]], [[Market Conditions]], [[Social Media Scanner]]):

- Technical-indicator / momentum strategies (RSI, MACD, regime detection).
- Wallet-monitoring / copy-style strategies driven by on-chain activity.
- Sentiment- or news-driven signals (e.g. a social scanner for breaking events).
- Learned policies (RL/ML) trained against a simulated market.
- Risk-/regime-aware overlays and filters on any of the above.

These are not mutually exclusive, and the surrounding infrastructure is identical for all
of them — which is why strategy choice can stay open while the execution loop is built.

## Key constraints & open questions

The genuine unknowns to resolve early — several gate everything downstream:

1. **Autonomous signing while self-custodial (blocker).** Can the agent sign unattended for
   a full week with custody staying local? Gates the whole hands-off premise. See
   [[Security and Encryption]].
2. **Hosting & key management (blocker).** A trade-a-day-for-a-week requirement means the
   agent must run on a reliable always-on host — which puts self-custody signing keys on a
   remote box. How are they stored and unlocked there safely? See [[Remote Capabilities]]
   and [[Security and Encryption]].
3. **On-chain data reach.** How much wallet/transfer/holder detail is cheaply and quickly
   available via BscScan (and CMC chain stats) for whatever on-chain logic the strategy
   needs? See [[Real-time Monitoring]].
4. **x402 path.** What's the minimal genuine pay-per-request use in the loop (data or
   inference) via the Agent Hub MCP?
5. **Registration mechanics + deadline.** On-chain registration must land before June 22;
   confirm the exact `twak compete register` flow. Worth probing whether the BNB SDK's
   ERC-8004 agent identity aligns with the competition's agent-address registration.
6. **Liquidity & slippage.** Many eligible tokens are thin; execution cost assumptions must
   match real BSC pool depth, not centralized-exchange volume.

## Build path

A front-loaded sequence — stand up the unfamiliar execution/custody layer first; leave the
familiar strategy logic for later. Aligned to the [[Index]] timeline:

1. **Stack spike.** Stand up all four surfaces: a real (dust-sized) trade lands on BSC via
   TWAK; the Agent Hub returns data; BscScan returns wallet/transfer data; the BNB SDK
   agent runs.
2. **Execution loop.** Wire read → decide (stub) → sign → execute → confirm in autonomous
   mode, with the guardrail scaffold and drawdown tracking enforced in code.
3. **Decision logic.** Implement and validate the chosen strategy against a simulated /
   recorded-data environment (see [[Simulated Market]]).
4. **June 16 — Track 1 proof of concept (go/no-go).** The PoC should demonstrate the **live
   execution loop end to end on-chain** (a real trade signed and landed, guardrails active),
   not just an offline backtest. If the live loop isn't real by this gate, **switch to
   Track 2** per the [[Index]] timeline.
5. **Harden + forward-run.** Enforce the drawdown gate, daily-trade scheduling, and
   BSC-tuned slippage handling; run forward over live/paper ticks to observe real behavior.
6. **Register + submit.** On-chain registration before June 22; public repo, demo video,
   strategy writeup.
7. **Live window (June 22–28).** Operate hands-off: ≥1 trade/day, stay under the drawdown
   cap, keep capital deployed.

## Reusable prior work

Relevant groundwork to draw on, without constraining the design — see [[TradeSim]]:
RL training pipelines, simulated-market / backtesting infrastructure, honest evaluation
against baselines, real-time on-chain monitoring, and MCP server development.
