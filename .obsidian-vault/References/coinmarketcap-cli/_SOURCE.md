# Source / provenance — CoinMarketCap CLI (`cmc`)

- **Source:** https://github.com/openCMC/CoinMarketCap-CLI
- **Pulled:** 2026-06-05 (shallow clone, `.git` removed — point-in-time snapshot)
- **Language:** Go (binary name `cmc`)
- **Auth:** needs `CMC_API_KEY` (CoinMarketCap Pro API); some intervals (5m/hourly history)
  are paid-tier gated.

## What it is — and three caveats

`cmc` is a terminal-native CoinMarketCap **data** client: `resolve`, `price`
(`--with-info` / `--with-chain-stats`), `search --chain --address`, `markets`, `pairs`,
`metrics`, `news`, `history`, `trending`, `top-gainers-losers`, a polling `monitor`, and a
TUI. It ships **Skills** under `skills/` for Claude Code and OpenClaw.

1. **Provenance unconfirmed.** Org is `openCMC`, not a clearly first-party CoinMarketCap
   org; its `CLAUDE.md` says not to use `coinmarketcap/*` as the release target. **Likely
   community/independent** — confirm it's the hackathon-sanctioned tool before depending on
   it. The hackathon's "CMC AI Agent Hub" (coinmarketcap.com/api/agent) is broader
   (MCP + x402 + hosted Skills); this repo is the CLI + Skills slice.
2. **Data only — no x402, no execution.** x402 (scored for the TWAK special) is **not**
   here; it lives in the Agent Hub MCP/hosted layer.
3. **No funding-graph forensics.** Good for market/pairs/chain-stats signals feeding the
   rug filter; the sybil funding-graph trace needs **BscScan**, not CMC.

## Useful for this build

- **Track 1 data/signal layer:** chain-scoped lookups, pairs (incl. derivatives), market
  regime, news/trending — candidate signal inputs.
- **Track 2 reference:** `skills/cmc-cli/SKILL.md` et al. are working examples of a
  CMC-authored Skill — the exact deliverable shape if the June 16 gate fails and you pivot.

## Refresh

```bash
git clone --depth 1 https://github.com/openCMC/CoinMarketCap-CLI coinmarketcap-cli
rm -rf coinmarketcap-cli/.git
```

> This repo ships its own `CLAUDE.md` — rename on import if you don't want a stray
> project-instruction file. Snapshot only; verify against the live repo.
