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
