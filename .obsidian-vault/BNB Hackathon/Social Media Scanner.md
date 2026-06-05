# Social Media Scanner

An optional, later-phase signal source for the trading agent. Feeds into the decision core
described in [[Trading Strategies]]; macro regime context lives in [[Market Conditions]];
on-chain flow that pairs with social signals comes from [[Real-time Monitoring]].

## What it is for

The scanner's value is narrow and specific: **breaking events that materially change a
token's risk profile before price fully reflects them.** Examples:

- Protocol hacks or exploits (immediate exit signal)
- Major exchange listings or delistings
- Regulatory actions on a token or its issuer
- Large institutional adoption announcements
- Coordinated FUD or pump campaigns (a manipulation flag, not a trade signal)

This is a **risk-overlay and gate function** more than a directional alpha source. The
primary use case is: if a credible hack report surfaces for a held position, trigger an exit
before waiting for the next price-based signal. Secondary use: filter entries when sentiment
and on-chain flow diverge in a suspicious way.

It does not replace price-derived signals. Sentiment is noisy, delayed, and easily gamed;
the technical indicators and on-chain signals in [[Trading Strategies]] remain the primary
decision inputs.

## Data sources

| Source | Surface | What it provides | Status |
|--------|---------|-----------------|--------|
| CMC Agent Hub MCP | `cmc_news` tool | CMC-curated news and trending items; machine-readable, no x402 required | Available Phase 3 |
| CMC Agent Hub MCP | Social / KOL data | Social heat metrics and KOL activity from the Agent Hub's social layer | Available via Agent Hub MCP; exact fields to verify |
| `cmc` CLI | `cmc news` command | Same CMC news feed, data-only (no x402 path in the standalone CLI) | Available now, data only |
| External X.com | API or scrape | Real-time tweets from KOLs and protocol accounts | Open question — see below |

The CMC Agent Hub MCP is the practical first path: it bundles social, KOL, and news data
under a single auth (`CMC_API_KEY`) and is already wired into the stack. The standalone
`cmc news` command surfaces the same news feed and works without MCP if an Agent Hub
connection is unavailable, but it does not carry x402 capability.

## The divergence signal

The hackathon brief explicitly names **"sentiment-divergence"** as a candidate signal: social
heat and on-chain flow disagreeing. The pattern in practice:

- **Social hot, on-chain cold:** KOL chatter rising but on-chain volume or holder activity
  flat — potential astroturfing; avoid entry or tighten stops.
- **On-chain hot, social cold:** unusual wallet activity (large transfers, holder
  concentration changes from [[Real-time Monitoring]]) with no public narrative — worth
  flagging as a pre-announcement or insider-movement candidate.
- **Both hot:** higher conviction for momentum entry, but also higher manipulation risk on
  thin tokens.

This divergence check uses `social_scan` / `cmc_news` for the social side and BscScan
wallet/transfer data (via `bscscan_transfers`, `bscscan_token_holders`) for the on-chain
side. How the combined signal feeds a decision is defined in [[Trading Strategies]]; offline
validation of it belongs in [[Simulated Market]].

## MCP tools

| Tool | Tier | Purpose |
|------|------|---------|
| `cmc_news` | READ | CMC news feed with filters; Phase 3 |
| `social_scan` | READ | X.com / news scan for breaking events (hacks, listings); Phase Later |

`cmc_news` ships in Phase 3 (loop + monitoring phase). `social_scan` is explicitly deferred
to the "Later" build phase in [[MCP Server]] — it is not on the critical path.

## Honest risk assessment

Sentiment signals carry specific failure modes that are distinct from price-based indicators:

- **Latency.** By the time a news item is in a feed and the agent has polled it, price has
  often already moved. Reaction windows on BSC are seconds, not minutes.
- **Noise ratio.** Most social volume for eligible tokens is low-signal chatter; a simple
  keyword hit on "hack" is more likely to be false positive than real.
- **Manipulation and astroturfing.** Many eligible tokens have thin liquidity and small
  communities — coordinated pump or FUD campaigns are easy to run and will pattern-match
  against naive sentiment rules.
- **Coverage gaps.** CMC news covers larger tokens better than micro-cap BEP-20s; many of
  the 149 eligible tokens will have sparse coverage.

Given these risks, the scanner should be wired as a **veto / gate** (block or exit on
confirmed bad news) rather than a **generator** (do not enter positions solely on social
signal without corroborating price or on-chain evidence).

## Phasing

The `sentiment-scanner` agent is marked optional in CLAUDE.md and this note's data surface
is non-blocking for the June 16 PoC gate. Reasonable sequence:

1. **Phase 3:** Wire `cmc_news` into the monitoring loop as a passive background feed.
   Flag items matching held-token symbols; log but do not act automatically.
2. **Phase 4 / later:** Validate divergence signal against historical data in [[Simulated
   Market]]. Only promote to an active veto or entry filter if it clears a meaningful
   bar — negative Sharpe contribution from noise drag is a real cost.
3. **If X.com access is resolved:** re-evaluate `social_scan` scope and latency; keep it
   optional and downstream of the price/on-chain core.

## Open questions

- **X.com API access.** Direct API access requires a paid developer plan (pricing and tier
  limits unverified). Scraping is fragile and likely against ToS. Until cost and access are
  confirmed, `social_scan` depends on CMC social data only.
- **CMC Agent Hub social / KOL fields.** The Agent Hub MCP advertises social and KOL data;
  the exact field names, update frequency, and token coverage for the eligible BEP-20 list
  need to be verified against the live API.
- **Latency floor.** What is the realistic polling interval and event-to-agent latency for
  `cmc_news`? If it exceeds 5–10 minutes, the hack-exit use case weakens considerably.
- **Divergence threshold calibration.** What constitutes "social hot" in quantitative terms
  for a thin BEP-20 token? Needs measurement on real data, not assumed.
