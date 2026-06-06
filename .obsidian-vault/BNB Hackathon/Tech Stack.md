# Tech Stack

Project scaffolding, environment, build, and SDK/API reference for the
agentic-crypto-trader build. See [[Project Overview]] for scope and [[MCP Server]] for the
tool-surface design. SDK docs are mirrored under `.obsidian-vault/References/`.

## Runtime & tooling

- **Language:** Python ≥ 3.11 (matches the BNB AI Agent SDK runtime and the [[TradeSim]] lineage).
- **Env/build:** `venv` + `pip` (editable install: `pip install -e ".[dev]"`). Package
  name `trader`, `src/` layout.
- **MCP server:** `mcp` (FastMCP), stdio transport. Registered in `.mcp.json` as `trader`,
  launched via the venv interpreter: `.venv\Scripts\python.exe -m trader.mcp_server`
  (POSIX: `.venv/bin/python`).
- **Secrets:** local `.env` (git-ignored). Keys never committed; self-custody signing stays
  local ([[Security and Encryption]]).

## Project layout

```
agentic-crypto-trader/
├── CLAUDE.md            # scope + pipeline (auto-loaded)
├── .mcp.json            # registers the `trader` MCP server
├── pyproject.toml       # pip/hatchling project
├── .env.example         # required config surface (copy to .env)
├── src/trader/
│   ├── execution/       # TWAK self-custody signing + BSC submission
│   ├── data/            # CMC Agent Hub (data + x402) + BscScan
│   ├── strategy/        # swappable decision core (interface)
│   ├── risk/            # guardrails: allowlist, caps, slippage, drawdown stop
│   ├── agent/           # read→decide→sign→confirm loop
│   ├── monitoring/      # wallet/tx watching + PnL
│   └── mcp_server/      # the project MCP server (tools per [[MCP Server]])
└── tests/
```

## The four surfaces (reference)

| Surface | Package / tool | Auth | Docs |
|---------|----------------|------|------|
| **TWAK** (execution + self-custody signing) | `@trustwallet/cli` (`twak`), TWAK MCP | API key + HMAC; local keys | `References/trust-wallet-agent-kit/` |
| **CMC Agent Hub** (data + **x402**) | Agent Hub MCP (x402 lives here, not the `cmc` CLI) | `CMC_API_KEY` | `References/coinmarketcap-cli/` |
| **BNB AI Agent SDK** (runtime + identity) | `bnbagent` (Python); ERC-8004/8183 — **not execution** | `.env` | `References/bnb-ai-agent-sdk/` |
| **BscScan** (on-chain analytics) | BscScan REST API | `BSCSCAN_API_KEY` | bscscan.com/apis |

> **Data sources — as built (2026-06-06 spike).** The on-chain data story diverged from
> this original sketch once tested (see [[Simulated Market]]):
> - **OHLCV history** → **GeckoTerminal** (CoinGecko on-chain, *keyless*) by pool address;
>   DexScreener has no history, CMC history is CEX-centric/tier-gated. Cached to resumable
>   Parquet (`trader.data.downloader`).
> - **Screening** → **DexScreener** (*keyless*): liquidity / volume / turnover / pool age.
> - **Contract resolution** → **CMC** `cryptocurrency/map`+`info` (`CMC_API_KEY`): symbol →
>   canonical BSC contract, fixing the 35% symbol-search ambiguity (`trader.data.cmc`).
> - **Forensics / rug gate** → **GoPlus** Security API (*keyless*, BSC `chain_id=56`):
>   honeypot, mintable, holder count, buy/sell tax, LP. **Replaces BscScan** here.
> - **⚠ BscScan/Etherscan caveat.** Etherscan unified all chains under one **V2** key, but
>   the **free tier covers Ethereum only — BSC requires a paid plan** (`"Free API access is
>   not supported for this chain"`). The standalone `api.bscscan.com` V1 endpoint is
>   deprecated. So `BSCSCAN_API_KEY` (an Etherscan key) is ETH-only on free; BSC on-chain
>   reads route via **GoPlus** + a **public BSC RPC** (`BSC_RPC_URL`), both free.

---

## Phase 2 — Stack spike (the critical first build)

**Objective:** stand up all four surfaces and prove the **live execution loop on-chain** —
the artifact the **June 16 Track 1 PoC gate** requires (a real, guarded, dust-sized trade
signed and landed on BSC). This is the unfamiliar, blocking layer; it is built before any
strategy logic. Owner: `principal-engineer` with `onchain-custody-engineer`.

### Steps

1. **Environment up.** `python -m venv .venv`, activate, `pip install -e ".[dev]"`; copy
   `.env.example` → `.env`; obtain `CMC_API_KEY`, `BSCSCAN_API_KEY`, and TWAK credentials;
   create/import the agent wallet (custody local).
2. **TWAK signs.** Confirm `twak` CLI/MCP can sign and submit a trivial BSC transaction with
   local self-custody keys. *(Resolves blocker: autonomous self-custody signing.)*
3. **Data reads.** CMC Agent Hub MCP returns market data (with an x402 pay-per-request in the
   path); BscScan returns wallet/transfer data for an address. *(Resolves blocker: on-chain
   data reach.)*
4. **BNB SDK runs.** A minimal BNB AI Agent SDK agent initializes; probe whether its ERC-8004
   identity aligns with the competition's agent-address registration.
5. **Guardrails first.** Implement the `risk/` limits (allowlist, per-trade/daily caps,
   slippage, drawdown stop) and wire them around the signing call **before** any live trade.
6. **Dust trade.** `execute_trade` signs via TWAK and lands a real, tiny, guardrail-checked
   trade on BSC; capture the **tx hash**.
7. **Registration dry-run.** Exercise the `twak compete register` / `competition_register`
   flow (don't miss the June 22 deadline).

### MCP tools to ship this phase (see [[MCP Server]])

`eligible_tokens` · `cmc_market` · `cmc_history` · `bscscan_wallet_txs` · `wallet_status` ·
`guardrails_get` · `simulate_trade` · `execute_trade` (dust) · `competition_register` (dry).

### Blockers to resolve in Phase 2 (gating)

- **Autonomous self-custody signing** — can the agent sign unattended while custody stays
  local? ([[Security and Encryption]])
- **Hosting & keys** — the live week needs an always-on host; design how signing keys live
  there safely. ([[Remote Capabilities]], [[Security and Encryption]])
- **On-chain data reach** — is BscScan's funding/transfer/holder data fast and cheap enough
  for whatever on-chain logic the strategy needs? ([[Real-time Monitoring]])

### Done / go-no-go

Frame the gate as a built-in goal:

```
/goal a guardrail-checked dust trade is signed via TWAK and confirmed on BSC with a tx hash,
and CMC + BscScan reads succeed
```

If this isn't real by **June 16**, switch to Track 2 per the [[Index]] timeline. Backtest
numbers do **not** satisfy this gate — only the live on-chain loop does.
