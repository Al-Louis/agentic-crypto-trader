# Apentic Data Contract

The static-JSON contract between the training pipeline (producer) and the **Apentic frontend**
(consumer, `alexlouis-site`). Bundles are published to `PUBLIC_APENTIC_DATA`
(`https://data.alexlouis.dev`). The frontend reads `manifest.json`, then a run's files. See
[[Remote Capabilities]] for how bundles get there. **Two run kinds** — branch on the manifest
entry's `kind`.

## manifest.json
```
[ { id, model_name, timestamp, n_episodes, regime, symbol, simulation,
    kind?: "portfolio",                 // absent ⇒ single-asset (legacy demo)
    universe?: [{ symbol, slug }] } ]   // portfolio only
```

## leaderboard.json — training-progress overview (top-level)

Published by `scripts/build_ledger.py --publish` (separate from the per-run bundles). A
self-contained summary for the frontend's **overview / leaderboard screen**. Fetch
`https://data.alexlouis.dev/leaderboard.json` (no-cache; invalidated on each publish).

```
{ generated, dd_gate, totals:{ runs, configs },
  baseline:        { name, return_pct, window },        // vol-tilt overlay, same window
  champion:        { config_label, mean_return, mean_maxdd, worst_maxdd, mean_sharpe, reproduce, … }
                   | null,                              // NULL when no config has passed OOS — handle it
  champion_criterion: "…",                              // passed frozen test: split=test, beats test base, worst-seed under gate
  configs: [ {                                          // sorted by mean_return desc
     config_label, timesteps, n, seeds:[…],
     split,                 // "val" = tuning window; "test" = frozen OOS verdict — DON'T compare across
     baseline,              // this config's own split baseline (val and test differ — never share a column)
     mean_return, mean_maxdd, worst_maxdd, mean_sharpe, mean_pf,
     legal_mean,            // mean DD under the gate
     gate_safe_worst,       // WORST seed under the gate — the deployment-honest badge
     beats_baseline, git, reproduce,
     seeds_detail: [ { seed, return, maxdd, sharpe, pf, run_id } ]   // per-seed drill-down
  } ] }
```

All `*_return` / `*_maxdd` are **fractions** (×100 for display). Suggested render: a leaderboard
table sorted by `mean_return`, the `champion` highlighted, a `baseline.return_pct` reference line,
a **gate-safe badge** from `gate_safe_worst` (not `legal_mean` — worst-seed is the honest bar), and
an expandable per-config seed breakdown from `seeds_detail`.

## market_metrics.json — volatility / correlation dashboard (top-level)

Published by `scripts/publish_market_metrics.py [--publish]` (sibling of `leaderboard.json`,
no-cache, invalidated each publish). A self-contained market-structure snapshot for the
**volatility / correlation screen** — it validates the risk picture the agent design hinges on:
the alts span ~8× the median vol, they **decouple from BTC** (alts pump while BTC bleeds), and
the token×token correlation is near-zero (so risk-parity sizing collapses portfolio drawdown).
Producer: `trader.report.market_metrics.compute_market_metrics` (pure, tested).

```
{ generated, kind?: in window,
  window:      { start, end, bars, hours_per_year, vol_window, excursion_window, kind },  // secs
  btc:         { ret_window, ann_vol },                       // fractions; ann_vol annualized
  tokens: [ {                                                 // sorted by ann_vol DESC (display order)
     symbol, slug,
     ann_vol,                       // annualized realized vol (full window)
     ret_window,                    // cumulative return over the window (fraction)
     vol_by_window: { "24h", "7d", "30d" },   // annualized vol over each trailing window
     max_runup, max_drawdown,       // largest rolling excursion over `excursion_window` (14d)
     corr_btc, beta_btc,            // correlation + beta to BTC  (the decoupling story)
     avg_corr_peers,                // mean corr to the other tokens (low ⇒ diversifiable)
     vol_series: [ { time, ann_vol } ]   // rolling-vol sparkline (downsampled, ~150 pts)
  } ],
  correlation: { symbols: [ …tokens…, "BTC" ], matrix: [[…]] },  // (n+1)² symmetric, unit diagonal
  summary:     { n_tokens, avg_pairwise_corr, median_pairwise_corr, max_pairwise_corr,
                 universe_ew_return, regime_label: "bull"|"bear"|"flat" } }
```

All `*_return` / `ret_window` / `*_runup` / `*_drawdown` are **fractions** (×100 for display).
Suggested render: a **vol-ranked token table** (ann_vol + vol_by_window + the `vol_series`
sparkline), a **correlation heatmap** from `correlation.matrix`/`symbols` (BTC as the last
row/col makes the decoupling visible), and a `corr_btc`/`beta_btc` scatter. `--window`
(full|train|val|test|last) selects the slice; the `last` window on a schedule gives a rolling view.

## Single-asset run (`kind` absent) — the demo / TradeSim-style
Per `<run_id>/`: `trades.json` (`RoundTrip[]`), `metrics.json` (`MetricsReport`),
`candles.json` (`CandleData[]` of the one symbol), `equity_curve.json` (`EquityPoint[]`),
`run_info.json`. Renders as candlesticks + entry/exit markers + trade table + equity + metrics
(the current dashboard).

## Portfolio run (`kind: "portfolio"`) — RL allocator
A cross-sectional strategy has **no single symbol** — its story is the *allocation over time*
plus per-token behaviour. Per `<run_id>/`:

| File | Shape | Use |
|------|-------|-----|
| `metrics.json` | `MetricsReport` | portfolio risk panel. **All `*_pct` fields are fractions** (×100 for display), incl. `avg_win_pct`/`avg_loss_pct` (per-round-trip *return*, not $); `total_trades` = individual trades (not rebalance days) |
| `equity_curve.json` | `EquityPoint[]` | portfolio NAV |
| `weights.json` | `[{ time, weights: { SYMBOL: frac } }]` per step | **allocation-over-time** (cash = 1 − Σ). Stacked-area / heatmap. |
| `run_info.json` | `{ model_name, kind:"portfolio", action_mode, universe:[{symbol,slug}], regime, … }` | the held universe + model meta |
| `tk_<slug>_candles.json` | `CandleData[]` | that token's OHLCV over the window |
| `tk_<slug>_trades.json` | `[{ time, price, side:"buy"\|"sell", usd, weight }]` | the agent's buy/sell **markers** on that token (`weight` = resulting allocation) |

`slug` is a URL-safe token name (`run_info.universe[i].slug`); fetch `tk_<slug>_*.json`.

### Frontend rendering (portfolio)
- **Overview:** weights stacked-area/heatmap (`weights.json`) + equity curve + metrics + the
  universe list. No candlestick at the portfolio level (there's no single price).
- **Per-token drill-down** (for fine-tuning — spotting the agent's trade logic): for each
  `universe` token, render `tk_<slug>_candles.json` as candlesticks with the
  `tk_<slug>_trades.json` buy/sell markers overlaid (same marker placement as the single-asset
  view: `time` + `price` + `side`).

## Producer side (this repo)
- Single-asset: `trader.report.export_run` (+ `roundtrips_from_position`).
- Portfolio: `trader.report.export_portfolio_run`; `scripts/train_rl.py` records per-step
  weights + per-token trades (env `step` info) and loads per-token OHLCV for the candles.
- Both publish via `trader.report.publish_run` → S3 + CloudFront invalidate.

## `trading/` prefix — live-trading telemetry (planned 2026-06-11, schema TBD)

A third top-level surface for the **live agent loop** (consumer: the planned
`/apentic/trading` page — design in [[Real-time Monitoring]] §public monitoring surface).
Differs from the run bundles above in producer and cadence:

- **Producer is the EC2 trading host itself** (put-only instance role scoped to `trading/*`
  — the laptop-credential publish path is not involved; the no-delete posture carries over).
- **Continuously updated** (per loop tick / hourly), not a one-shot bundle: equity +
  drawdown series, trade log (tx hashes + guardrail refusals), daily trade count, and a
  `generated` **heartbeat** the frontend ages for the dead-man indicator.
- A `mode: "paper" | "live"` field distinguishes the June 16–21 forward-run from the scored
  window. Exact file shapes land here when the loop's publisher is built.
- **Freshness: a CloudFront cache behavior on `trading/*` (managed CachingDisabled), NOT
  invalidations** — `CreateInvalidation` can't be path-scoped and would bloat the put-only
  instance role; the run-bundle invalidate-on-publish pattern does not apply to this prefix
  ([[EC2 Trading Host Runbook]] Phase F).

### As-built source rows (2026-06-11) — the loop's ledger the publisher will project

The loop (`trader.agent`) already emits the raw telemetry; the EC2 publisher (NOT yet built)
projects these append-only rows from `data/agent_ledger.jsonl` (`trader.agent.store`) into the
static `trading/` JSON. Row kinds (every row carries a UTC `ts` and `mode`):

```
{ kind:"fill",      mode, from, to, usd_in, usd_out, cost_usd,
                    units_from, units_to, price_from, price_to, reason, tx_hash? }  // tx_hash live only
{ kind:"equity",    mode, tick, equity_usd, peak_usd, drawdown_pct, below_dust }    // hourly PnL mark
{ kind:"heartbeat", mode, tick, equity_usd }                                        // dead-man input
{ kind:"refusal",   mode, intent:{from,to,usd}, refusals:[CODE,…] }                 // guardrail audit
```

Provisional projection into the published shapes (refine when the publisher lands):
- **equity/drawdown series** ← `equity` rows; **trade feed** ← `fill` rows (tx → BscScan in live)
  + `refusal` rows; **daily trade count** ← `fill` rows per UTC day vs the ≥1/day floor;
  **heartbeat** ← newest `heartbeat`/`equity` `ts`.
- **Convention note:** the loop emits `drawdown_pct` as a **percent** (e.g. `4.2` = 4.2%), unlike
  the run-bundle `*_pct` **fractions** — the publisher normalizes to the contract's fraction
  convention before writing `trading/` JSON.
