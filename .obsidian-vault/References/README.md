# references — SDK / docs mirrors

Local, point-in-time mirrors of the three hackathon stack pillars, for the
[[bnb-rug-aware-copytrader]] build. **Outside the vault** — import into Obsidian as you see
fit. All pulled **2026-06-05**.

| Folder | Source | What it is | Lang |
|--------|--------|------------|------|
| `trust-wallet-agent-kit/` | github.com/trustwallet/developer | TWAK — execution & self-custody signing layer (+ MCP, Skills) | CLI (`twak`) |
| `bnb-ai-agent-sdk/` | github.com/bnb-chain/bnbagent-sdk | BNB AI Agent SDK — agent **identity & commerce** runtime | Python |
| `coinmarketcap-cli/` | github.com/openCMC/CoinMarketCap-CLI | `cmc` CLI — CoinMarketCap **data** access + Skills | Go |

## Two things worth knowing before you build

1. **The BNB AI Agent SDK is not your execution layer.** Its headline protocols are
   **ERC-8004 (on-chain agent identity registry)** and **ERC-8183 (agentic commerce:
   jobs/escrow/voting)** — plus wallet-signing abstraction and a plugin system. It's the
   agent *runtime and identity* scaffolding. **Trade execution is TWAK's job.** Don't wire
   swaps through the BNB SDK; use it for the agent framework, wallet abstraction, and
   on-chain agent identity (which may even relate to the competition's agent-address
   registration).

2. **The CMC CLI is data-only, and its provenance is unconfirmed.** `cmc` pulls price /
   history / markets / news / pairs / chain-scoped lookups from the CMC Pro API (needs
   `CMC_API_KEY`) and ships Skills for Claude Code/OpenClaw. But:
   - The org is **`openCMC`**, *not* an obviously first-party CoinMarketCap org — its own
     `CLAUDE.md` even says "do not reintroduce `coinmarketcap/*` as the primary release
     target." Treat it as **likely community/independent** until confirmed, and verify it's
     the tool the hackathon sanctions.
   - **No x402 and no execution here.** x402 (scored for the "Best Use of TWAK" special)
     lives in the broader CMC **Agent Hub / MCP**, not in this CLI. This repo is the *data*
     and *Skills* surface only.
   - **Sybil/rug funding-graph tracing is not a CMC capability** — CMC gives you market +
     pairs + chain stats; the funding-graph forensics need **BscScan** (concept §6, OQ #2).

> Each folder has a `_SOURCE.md` with exact URLs and a refresh command. The
> `coinmarketcap-cli/` repo ships its own `CLAUDE.md` — rename it on import if you don't
> want a stray project-instruction file in your notes.
