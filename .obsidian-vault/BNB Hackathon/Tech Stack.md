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

## Project layout (as built — 2026-06-06)

```
agentic-crypto-trader/
├── CLAUDE.md            # scope + pipeline (auto-loaded)
├── pyproject.toml       # pip/hatchling; [data] extra = pandas, pyarrow, ta, ccxt
├── .env.example         # CMC_API_KEY (used) · BSCSCAN (Etherscan, ETH-only) · TWAK (later)
├── src/trader/
│   ├── config.py        # ✅ .env loader (CMC key, etc.)
│   ├── data/            # ✅ universe + OHLCV: dexscreener · geckoterminal (OHLCV) ·
│   │                    #    cmc (contract resolution) · cmc_market (LIVE quotes — loop read) ·
│   │                    #    goplus (rug gate) · eligible · select · downloader · anchor
│   ├── features/        # ✅ indicators (71-col + leakage guard) · factor (BTC/BNB residual)
│   ├── sim/             # ✅ metrics · broker (AMM cost) · backtest · strategies · resample · ic
│   ├── execution/       # ⬜ stub — TWAK self-custody signing + BSC submission (Phase 2)
│   ├── strategy/        # ⬜ stub — the validated candidate lands here (vol-tilt + regime overlay)
│   ├── risk/            # ⬜ stub — guardrails: allowlist, caps, slippage, drawdown stop
│   ├── agent/           # ✅ autonomous loop: loop · decide (DecisionCore+HoldCore) ·
│   │                    #    feed (CmcPriceFeed/FakeFeed) · paper (AMM-cost fills) · store
│   │                    #    (crash-safe ledger) · __main__ (paper default, live double-gated)
│   ├── monitoring/      # ⬜ stub — wallet/tx watching + PnL
│   └── mcp_server/      # 🟡 skeleton — health + eligible_tokens stub (catalog in [[MCP Server]])
├── scripts/             # research CLIs: screen · resolve · select · forensics ·
│                        #   download_ohlcv · download_anchor · build_factor_features ·
│                        #   ic_analysis · run_backtest · resample_eval · tail_sweep · oos_validate
├── data/                # generated caches, git-ignored: ohlcv/ · anchor/ · features/ · *.json
└── tests/               # ~89 pytest functions
```

> **As-built note.** The research stack (`data/` → `features/` → `sim/`) is plain Python modules
> + `scripts/` CLIs, **not yet [[MCP Server]] tools** — that catalog remains the planned wrapper.
> The decision core (`strategy/`) is empty; the evidence-backed candidate (daily-rebalanced
> vol-top8 + a regime overlay — see [[Trading Strategies]]) lands there next. Live execution
> /custody (`execution/`, `risk/`, `agent/`) is the deferred Phase-2 on-chain spike.

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
>   Parquet (`trader.data.downloader`). **Also the LIVE forward-run feed** (`trader.agent.live_data`,
>   hourly append + 429 backoff) — parity-locked to GeckoTerminal because ef-s2 trained on these
>   BSC-pool candles; CMC was rejected as the live feed (CEX-aggregated → out-of-distribution for
>   the frozen model). See [[Live Forward-Run Harness]].
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

> **Status 2026-06-11 — custody half + data half DONE** ([[TWAK Spike Runbook]], steps 0–8;
> [[Build Log]] 2026-06-11 loop entry): steps 1, 2, 3, 5, 6, 7 below are ✅. A live
> guardrail-checked dust trade landed on BSC (tx `0x739bb1…7c96`), `risk/` + `execution/` +
> **`agent/` (the autonomous loop)** built (**343 tests**), registration dry-run done
> (on-chain deadline reads **June 25**; June 22 stays the working deadline), wallet
> unification proven on `bsctestnet`, auto-lock re-unlock confirmed, and **CMC live reads
> proven** (full 147-symbol eligible universe in one batched `quotes/latest` call, 1 credit,
> 150k/mo budget — `trader.data.cmc_market`). **Remaining:** step 4 only (BNB SDK runtime
> probe — OPTIONAL). The autonomous loop runs end-to-end in **paper mode** (read→decide(stub)
> →paper-fill→persisted PnL/heartbeat), with live double-gated behind `mode="live"` +
> `AGENT_ALLOW_LIVE=1`.
> **Plan forward (agreed 2026-06-11):** build the autonomous loop now with EC2 provisioning
> in parallel; paper-mode forward-run on AWS June 16–21; competition wallet created **on the
> EC2 box** and registered before June 22; validation ladder paper → mainnet dust
> ([[Build Log]] plan-forward entry, [[Remote Capabilities]]).

### Steps

1. **Environment up.** `python -m venv .venv`, activate, `pip install -e ".[dev]"`; copy
   `.env.example` → `.env`; obtain `CMC_API_KEY`, `BSCSCAN_API_KEY`, and TWAK credentials;
   create/import the agent wallet (custody local).
2. **TWAK signs.** Confirm `twak` CLI/MCP can sign and submit a trivial BSC transaction with
   local self-custody keys. *(Resolves blocker: autonomous self-custody signing.)*
3. ✅ **Data reads.** **DONE 2026-06-11** — CMC `quotes/latest` returns live USD prices for the
   **whole eligible universe in one batched call (1 credit)**; the loop's read step uses **plain
   CMC Pro REST** (`trader.data.cmc_market`), not the Agent Hub MCP. **x402 is recon-only** — the
   same data is reachable via the Agent Hub MCP with x402 pay-per-call, kept as the fallback only
   if a future need exceeds the credit tier; no payment path is wired. Keyless GeckoTerminal/
   DexScreener stays the vendor-independent fallback feed. BscScan wallet/transfer reads remain
   for monitoring (Phase 3). *(Resolves blocker: on-chain data reach.)*
4. ⬜ **BNB SDK runs (OPEN, optional).** A minimal BNB AI Agent SDK agent initializes; probe whether
   its ERC-8004 identity aligns with the competition's agent-address registration. **Not yet done** —
   identity ≠ execution, so it doesn't gate the loop; deferred.
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

- ~~**Autonomous self-custody signing**~~ — **RESOLVED 2026-06-11**: proven live (keychain
  resolution, guarded dust trade confirmed on BSC). ([[Security and Encryption]])
- **Hosting & keys** — **DECIDED 2026-06-11**: AWS EC2, systemd + hardened env-file,
  competition wallet created on the box; build pending. ([[Remote Capabilities]],
  [[Security and Encryption]])
- **On-chain data reach** — is the GoPlus + public-RPC + CMC route fast and cheap enough for
  whatever on-chain logic the strategy needs? (BscScan free tier is ETH-only — see the data
  caveat above.) ([[Real-time Monitoring]])

### Done / go-no-go

Frame the gate as a built-in goal:

```
/goal a guardrail-checked dust trade is signed via TWAK and confirmed on BSC with a tx hash,
and CMC + BscScan reads succeed
```

If this isn't real by **June 16**, switch to Track 2 per the [[Index]] timeline. Backtest
numbers do **not** satisfy this gate — only the live on-chain loop does.

**2026-06-11: the trade half is met** — tx
`0x739bb1516c99e56237c7585a449455d90a7f0b027ef9f252a5275b67e4757c96` confirmed on BSC
through the guardrails. The data-reads half (CMC Agent Hub; on-chain reads via the GoPlus/RPC
route) is the remaining go/no-go input.
