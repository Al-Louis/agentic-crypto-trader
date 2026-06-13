# Pool-Event Data Layer

The on-chain instrument behind three independent ideas that converged in
[[Trading Strategies]] §"PARKED — wallet-attributed token personality": wallet-attributed
flow, liquidity/flow knowledge (the quant consult's data-gated Addition-③), and the AMM
analogs of order-book depth (LP pulls precede dumps; swap-flow imbalance ≈ book imbalance).
Built 2026-06-12 as `src/trader/chain/` — a **fully isolated, read-only** collector for
PancakeSwap pair/pool events over the 20-token [[Token Universe]].

**ISOLATION CONTRACT (non-negotiable):** this module reads the chain and writes
`data/chain/`. Nothing in `trader/train/`, the env, the rewards, or the live trading path
imports it or is touched by it. Integration happens ONLY if a probe passes, later, through
the training loop's process ([[Agent Communication Contract]]).

## Why backfill is the key move

The sim's liquidity is **static** (one `liq_usd` per token) — that's what data-gated the
liquidity/flow knowledge direction. The backfill reconstructs **time-varying** pool state
over the *same recorded OHLCV window every prior probe ran on* (Nov 2025 → Jun 2026), so
the flow probes are directly comparable to the price-only probes in [[Experiment Log]].

## RPC reality (probed 2026-06-12 — this took the morning; don't re-derive)

BscScan V2 free tier is **Ethereum-only** ([[Tech Stack]]), so collection rides public
JSON-RPC `eth_getLogs`. The free-endpoint landscape:

| Endpoint | Verdict |
|---|---|
| `bsc.therpc.io` | **The backfill workhorse.** Serves *deep* historical logs (Nov 2025 ✓). ~10-20k-block spans OK (502 above that), >12k logs per response observed, ~7-20s/call at hot windows. |
| `bsc-rpc.publicnode.com` | Fast, generous spans, **but log history pruned to ~1 day**. Use for live tail / `eth_call` / block lookups. Needs a browser-ish User-Agent (403s python-urllib). |
| `bsc-dataseed.binance.org` (.env default) | `eth_getLogs` always "limit exceeded" at ANY span. Fine for `eth_call`/blocks. |
| `1rpc.io/bnb`, `drpc`, `meowrpc`, `ankr`, … | 50-block caps, no getLogs, auth walls, or pruned. Dead ends. |

`trader.chain.rpc.BscRpc` encodes this: ordered endpoints `[therpc, publicnode, .env]`,
failover on prune/permission errors, backoff on 429, and a `SpanTooWide` signal so the
collector owns block-span adaptation (start 10k, halve on error, creep back up).

## The universe's pools (registry)

`data/chain/_pools.json`, built by `trader.chain.registry` from `data/selection.json` +
on-chain probes (`slot0()` ⇒ V3, `getReserves()` ⇒ V2; `token0/1`; ERC-20 decimals):
**16 V3 pools + 4 V2** (ADA, SKYAI, B, BabyDoge). Non-18 decimals exist (XAUt 6,
BabyDoge 9, HUMA 6) — never assume. PancakeSwap **V3's Swap topic0 differs from
Uniswap's** (two protocol-fee fields appended): `0x19b47279…`, confirmed empirically on Q.

## Data contract

```
data/chain/                      (git-ignored)
  _pools.json                    # registry: version, token_side, decimals, fee, quote
  _manifest.json                 # scan cursor — resumable, downloader-pattern
  blockindex/samples.parquet     # (block, ts) samples; per-log ts by interpolation
                                 #   (BSC block time moved 3s → 0.45s across the window)
  logs/<SYM>_<pool10>/p_<from>_<to>.parquet   # unified decoded rows
  panels/hour/<SYM>.parquet      # hourly panel aligned to the returns index
```

**Unified event rows** (`trader.chain.events`): V2 Sync/Swap/Mint/Burn + V3
Swap/Mint/Burn/Collect, one sign convention: **positive = into the pool**. A trader buy ⇒
token side negative. V3 `Burn` is the LP-removal *decision* (no token transfer), `Collect`
is the withdrawal transfer — both kept, distinct labels. Amounts decimal-normalized
float64 (research precision). Wallet fields: `sender`/`recipient` topics; `recipient` is
the user wallet on simple router swaps (tx.from enrichment deferred — probe-gated).

**Hourly panels** (`trader.chain.panels`, epoch-second hours = the returns index keys):
`n_swaps, vol_token, vol_quote, net_token_in, net_quote_in, n_mints/burns/collects,
lp_add/remove_token/quote, liquidity_end, reserve_token_end, reserve_quote_end, price_end,
unique_swappers`. V2 reserves from Sync; V3 virtual in-range reserves from the last swap's
`liquidity` + `sqrtPriceX96`.

## Operations

```
python scripts/chain_backfill.py --init      # registry + block range (once)
python scripts/chain_backfill.py             # run/resume the backfill
python scripts/chain_backfill.py --tail      # extend to current head
python scripts/chain_backfill.py --panels    # build hourly panels
```

Backfill scale: ~35.9M blocks, ~20-25M logs, **~10-14h laptop wall-clock** (density decays
from ~12k logs/10k-blocks in Dec to ~4-5k by Mar). Fully resumable at the manifest cursor;
safe to ctrl-C and re-run. Desktop training box NOT involved.

**Live tail & EC2:** the tail is the same scan extended to head. The EC2 trading host
([[EC2 Trading Host Runbook]]) is the natural always-on home — a systemd timer running
`--tail && --panels` hourly via publicnode (recent blocks only ⇒ pruning irrelevant) would
keep panels current through the live window. *Evaluation:* low risk (read-only, no keys
touched, ~negligible CPU), but deferred until the paper-loop forward-run is stable — do not
co-deploy new software onto the trading host during its own validation window.

## Backfill + probes — RESULT (2026-06-13)

Backfill completed: **36.1M events** across all 20 pools, blocks 66.98M→102.87M
(Nov-2025→Jun-2026), ~14h laptop. Decoders validated against ground truth — panel `price_end`
tracks the recorded OHLCV at **corr 0.91–0.99 on the V3 pools** (the two V2 pools, ADA/BabyDoge,
~0.70, consistent with WBNB-vs-USD quote / GeckoTerminal sampling a different pool). Thinnest:
XAUt (78 hours, listed April); all others 4.1k–5.2k hours.

**All three pre-registered targets refuted** (full numbers in [[Experiment Log]]) — the
cross-sectional-rank / personality null shape: real structural facts that don't convert. (1)
LP-pull raises detonation odds **×4.5 both splits** but is flat on the DRAWDOWN target, recall
25-39% / precision 4-7%; (2) flow-imbalance IC at/below the noise bar; (3) wallet-cohort flow
flips sign OOS / wrong-sign kernel / MM-quiet contradictory. The probes saved three builds; the
reactive det-blacklist stays the guardrail; integration stays gated on a PASS through the
training loop's process (none earned one).

**Quant cross-check (2026-06-13, full numbers in [[Experiment Log]]):** independent re-analysis
agreed all three refutals and tested six more functionals. **Depth-normalized turnover**
(`vol_quote / reserve_quote_end`) is the correct risk functional — fwd48-worst IC −0.19/−0.39,
real incremental signal over price-vol on partial-IC — but **failed the matched-frequency
de-risking overlay**: dominated by the trailing realized-vol the agent already observes (the
redundancy null, fifth occurrence). Verdict: **STOP — route none into the obs** (dilution risk per
the rdLc spent-move precedent); the binding constraint is RETURN, not drawdown (already DQ-safe).
**Keep warm as OPS TELEMETRY only:** turnover spiking 5σ on a held token = a human-eyeball alert on
the EC2 live tail, not a model input or validated guardrail.

## Pre-registered probes (the law: probe-before-build, OOS, honest framing)

Scripts ship with the layer; run after backfill+panels; train/val only (test frozen);
grading per [[verify-claims-with-run-data]]:

1. **`probe_lp_pull.py`** — LP-PULL → DETONATION LEAD. Do Burn/withdrawal events precede
   the detonation signature (surge≥8× & rising≤−15%, exactly the det-blacklist
   construction) and by how many hours? **Graded on forward DRAWDOWN** (worst trough — the
   DQ-relevant target): lead-time distribution, conditional fwd damage, det precision/recall.
2. **`probe_flow_imbalance.py`** — FLOW-IMBALANCE → REVERSION. Trailing-24h
   `net_quote_in / vol_quote` vs fwd {24,48,72}h returns + worst trough, quintiles + IC.
   Horizons ≥24h only (round-trip ~0.7-1% — sub-minute is dead on arrival).
3. **`probe_wallet_cohort.py`** — WALLET-COHORT LEAD. New-wallet accumulation / aged-wallet
   distribution → fwd returns IC; MM-wallets-going-quiet → P(detonation ≤48h) lift.
   Recipient-proxy attribution; cross-pool addresses dropped as router infrastructure.

Findings land in [[Experiment Log]]; the build record in [[Build Log]].
