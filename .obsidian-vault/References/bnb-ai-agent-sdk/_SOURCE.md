# Source / provenance — BNB AI Agent SDK

- **Source:** https://github.com/bnb-chain/bnbagent-sdk
- **Pulled:** 2026-06-05 (shallow clone, `.git` removed — point-in-time snapshot)
- **Language:** Python (`pyproject.toml`, `uv.lock`)

## What it actually is (read before architecting)

A Python toolkit for **on-chain AI agents on BNB Chain** — wallet management, a plugin
module system, off-chain storage abstraction, and two protocol modules:

- **ERC-8004** — on-chain **identity registry** for AI agents (register / discover / resolve
  endpoints).
- **ERC-8183** — **agentic commerce** stack: job lifecycle + escrow, an evaluator router,
  and an optimistic (UMA-style) dispute/voting policy.

**This is the agent runtime + identity + commerce layer, NOT a trade-execution SDK.**
Execution and self-custody signing belong to TWAK. Use this SDK for the agent framework,
wallet-signing abstraction, and (potentially) on-chain agent identity tied to competition
registration.

## Key docs in this mirror

- `README.md` (32K) — main SDK guide
- `ARCHITECTURE.md` (16K) — module layout, dependency direction, code map
- `bnbagent/{core,erc8004,erc8183,storage,wallets}/README.md` — per-module
- `examples/{agent-server,client,voter}/` — runnable examples
- `.env.example` — required configuration surface

## Refresh

```bash
git clone --depth 1 https://github.com/bnb-chain/bnbagent-sdk bnb-ai-agent-sdk
rm -rf bnb-ai-agent-sdk/.git
```

> Snapshot only — verify against the live repo before relying on any specifics.
