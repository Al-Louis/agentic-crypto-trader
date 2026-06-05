# Real-time Monitoring

The observability surface for the autonomous loop — what the agent watches on-chain and off,
how it confirms its own trades, and the **hourly PnL + drawdown tracker that mirrors scoring**
and feeds the drawdown-stop guardrail. Custody/signing lives in [[Security and Encryption]];
how signals become trades lives in [[Trading Strategies]]; regime context in
[[Market Conditions]]; where the always-on watcher runs in [[Remote Capabilities]].

## Why monitoring is risk-critical, not cosmetic

The [[Project Overview|scoring]] is mechanical and unforgiving, and monitoring is the
instrument that keeps the entry inside it:

- **Returns measured hourly.** Any hour that *begins* with portfolio ≤ $1 scores 0% — a
  drained-to-dust wallet is treated as having no capital at work.
- **~30% max drawdown is a hard DQ gate.** Breach it and the entry is disqualified regardless
  of headline return.
- **≥1 trade/day** across the live week (7 trades) is required to qualify.

So the tracker is an **early-warning system** for the DQ gate and the activity floor, not a
dashboard. It must surface a worsening drawdown *before* it crosses ~30%, with margin for the
drawdown-stop guardrail to halt trading. The drawdown stop is enforced in `risk/` around the
signing call ([[Project Overview]]); monitoring supplies the running drawdown figure it reads.

## What gets monitored

| Subject | What | Source |
|---------|------|--------|
| **Own wallet balances/positions** | Native + token holdings, USD value across chains | TWAK `wallet portfolio`; `wallet_status` |
| **Hourly PnL + drawdown** | Hourly return series, running peak, current drawdown vs ~30% gate | `portfolio_pnl` (derived) |
| **Executed-trade confirmation** | Each trade's tx landed, balances changed as expected | TWAK `tx <hash>`, `history`; `recent_activity` |
| **Daily trade count** | Trades-today vs the ≥1/day floor | own trade log + `bscscan_wallet_txs` |
| **Fund transfers in/out** | Unexpected inbound/outbound transfers on the agent wallet | `bscscan_transfers`, `bscscan_wallet_txs` |
| **Target wallets** *(copy strategies only)* | Activity of wallets a copy strategy mirrors | `watch_wallets`, `recent_activity`, `bscscan_wallet_txs` |
| **Token health** *(optional)* | Holder concentration / rug signals on held tokens | `bscscan_token_holders`, TWAK `risk` |

Target-wallet monitoring only matters if a copy-style decision core is chosen; the *signal*
formed from that activity belongs to [[Trading Strategies]], the *watching* belongs here.

## Data sources and tradeoffs

Two complementary sources; neither is push/streaming, so the loop **polls** on its event tick.

- **BscScan REST** — authoritative on-chain analytics: wallet tx history, ERC-20/BEP-20
  transfers, holder distribution and concentration. Free tier is rate-limited (commonly
  ~5 req/s, daily cap); calls must be batched and cached, and polling cadence tuned to stay
  under quota. Latency is block-confirmation plus indexer lag. The source of record for
  *confirmation* and *transfer* detection.
- **TWAK** — `wallet portfolio` gives USD-valued holdings across chains in one call (good for
  PnL without assembling per-token prices); `history` and `tx <hash>` confirm specific
  transactions; `alert create/list/check` provides simple price alerts. TWAK alerts and DCA/
  limit automations only fire while a watcher is live — `twak serve --watch` (tune with
  `--watch-interval`) or `twak watch`; without it, alerts and automations are saved but never
  run. The background runner uses the local agent wallet, which is the self-custody path
  ([[Security and Encryption]]).

| Concern | BscScan REST | TWAK portfolio/history/alerts |
|---------|--------------|-------------------------------|
| Model | Poll REST | Poll + watcher-run alerts/automations |
| Strength | Raw on-chain truth (txs, transfers, holders) | One-shot USD portfolio; native tx confirm; price alerts |
| Latency | Block + indexer lag | Block + provider lag |
| Limits | Rate/daily quota — batch & cache | API/HMAC auth; watcher must stay up |

Rule of thumb: **TWAK for USD-valued portfolio state and price alerts; BscScan for raw
event/transfer/holder truth and trade confirmation.** Cross-check the two when they disagree.

## The hourly PnL + drawdown tracker (`portfolio_pnl`)

The core monitoring artifact, mirroring the scoring mechanic so the agent sees what the judges
see. On each hourly boundary it records:

- **Portfolio USD value** (from `wallet portfolio` / `wallet_status`), and the hour-over-hour
  **return**; any hour opening at ≤ $1 is recorded as 0% (matches scoring).
- **Running peak equity** and **current drawdown** = `(peak − value) / peak`, the figure the
  `risk/` drawdown stop reads.
- **Daily trade count** vs the ≥1/day floor.

Output (per [[MCP Server]]): `pnl`, an `hourly` return series, and `drawdown`. It is a 🟢 READ
tool — derived, no side effects. Lineage: [[TradeSim]] carried PnL/drawdown tracking and
streaming-sim infrastructure that this reuses.

## Alerting

Thresholds that turn the tracker into action; alarms are logged and can gate the loop.

| Alert | Trigger | Response |
|-------|---------|----------|
| **Drawdown alarm** | Drawdown approaches the soft band below the ~30% gate | Warn early; at the stop threshold, `risk/` halts new trades |
| **Daily-trade-count check** | No qualifying trade yet within the day's window | Flag so the loop schedules the mandatory ≥1/day trade |
| **Abnormal-transfer alert** | Unexpected inbound/outbound transfer on the agent wallet | Raise immediately — possible key compromise ([[Security and Encryption]]) |
| **Price alert** *(optional)* | TWAK `alert` above/below a level on a held/watched token | Feed the decision core as a signal input |

The drawdown alarm is the most important: it must fire with enough margin that the stop
engages **before** the hard DQ line, not at it.

## Closing the execution loop

Monitoring is the final stage of the [[Project Overview|event-driven loop]]
(monitor → evaluate → decide → execute → **confirm**). `execute_trade` is a 🔴 EXECUTE tool
that returns a **tx hash**; monitoring closes the loop by:

1. Confirming the tx landed (TWAK `tx <hash>` / `history`; cross-check `bscscan_wallet_txs`).
2. Verifying the **balance change** matches the intended fill (`wallet_status` /
   `wallet portfolio`, against `simulate_trade`'s projection).
3. Updating `portfolio_pnl` (equity, hourly return, drawdown) and the daily trade count.
4. Re-arming alerts for the next tick.

A trade isn't "done" at broadcast — it's done when confirmed landed and reconciled against
expected balances. Unconfirmed or mismatched fills are surfaced, not assumed successful.

## MCP tools (this surface)

`watch_wallets` · `recent_activity` · `portfolio_pnl` · `wallet_status` ·
`bscscan_wallet_txs` · `bscscan_transfers` · `bscscan_token_holders`. All 🟢 READ. Shipped in
**Phase 3** (loop + monitoring) per [[MCP Server]], building on Phase 2's `wallet_status`.

## Open questions

- **Polling cadence vs BscScan quota.** What hourly/tick cadence keeps confirmation + transfer
  detection fresh while staying under the free-tier rate/daily caps? Confirm exact limits.
- **PnL valuation source.** Is TWAK `wallet portfolio` USD valuation timely and consistent
  enough to drive the drawdown gate, or should CMC quotes (`cmc_market`) price holdings
  independently? Decide the authoritative pricing path.
- **Scoring-mirror fidelity.** Confirm the exact hourly boundary/valuation convention the
  competition uses so `portfolio_pnl` matches it (avoid drifting from the judged figure).
- **Watcher liveness.** `twak serve --watch` must stay up for the full week for TWAK alerts/
  automations to run; how this is hosted and supervised is owned by [[Remote Capabilities]].
- **Confirmation depth.** How many confirmations before a trade is treated as final on BSC for
  reconciliation purposes?
