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

Published two ways: ad-hoc `scripts/publish_market_metrics.py [--publish]` (laptop/desktop, with
CloudFront invalidation), and — **as of 2026-06-17 — a DAILY automated scan on the EC2 box**
(`trader.agent.daily_scan`, systemd `trader-daily-scan.timer` at 00:10 UTC). A self-contained
market-structure snapshot for the **volatility / correlation screen** — it validates the risk
picture the agent design hinges on: the alts span ~8× the median vol, they **decouple from BTC**
(alts pump while BTC bleeds), and the token×token correlation is near-zero (so risk-parity sizing
collapses portfolio drawdown). Producer: `trader.report.market_metrics.compute_market_metrics` (pure,
tested).

**`selected` (added 2026-06-17) — the model's ACTUAL traded set.** The daily EC2 scan appends a
`selected` block read from the SAME env the live harness trades (`eval_universe_and_caps` over the
current cold-week window): the **vol-top-8 ef-s2 is trading this week** + its risk-parity caps /
USD allocation. ef-s2 selects **WEEKLY** (causal trailing-168h vol at the Monday open, fixed for the
week — how it trained), so `selected` changes weekly while the dashboard metrics refresh daily. It
is informational — it does NOT drive the model (the model self-selects internally). `vol_rankings`
is the evolving-pool research view; `selected` is the authoritative live pick. The EC2 publish uses
the instance role (a scoped `market_metrics.json` PutObject grant — `deploy/iam/market-metrics-put-policy.json`)
+ `no-cache`; freshness ideally backed by a CloudFront CachingDisabled behavior on `market_metrics.json`
(like `trading/*`). See [[Live Forward-Run Harness]].

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
  vol_rankings: { "24h"|"7d"|"30d"|"90d"|"180d": { ranked:[{symbol,ann_vol,rank}], top8:[…] } },  // evolving-pool view
  summary:     { n_tokens, avg_pairwise_corr, median_pairwise_corr, max_pairwise_corr,
                 universe_ew_return, regime_label: "bull"|"bear"|"flat" },
  selected:    { method, run_id, week_start, as_of,        // the MODEL's ACTUAL current vol-top-8
                 tokens:[…8…], caps:{tok:frac}, alloc_usd:{tok:usd} } }   // weekly cadence — see below
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

### Simulation run (`simulation:true`) — a saved checkpoint replayed over a timeframe

A **simulation** is a portfolio run produced by `scripts/simulate.py` ([[Simulated Market]]): a
*saved* policy replayed over a trailing window (6mo/3mo/1mo/1wk/1d), one bundle per timeframe. **Same
files as a portfolio run** (incl. `tk_<slug>_candles.json` with full per-token OHLCV + markers — a
complete candlestick chart is renderable), so the existing portfolio renderer works unchanged; the
only new work is a model+timeframe selector. It adds extra **manifest** + **metrics** fields:

**Manifest entry** — the last three are the selector keys (added so dropdowns need zero per-bundle fetches):
```
{ id:"ppo-event-rdLe4r-68b268f-s0-sim-3mo", kind:"portfolio", simulation:true,
  symbol:"PORTFOLIO", model_name, timestamp, regime:"bull"|"bear"|"flat", universe:[{symbol,slug}],
  source_run:"ppo-event-rdLe4r-68b268f-s0",   // THE MODEL  (group dropdown 1)
  timeframe:"3mo",                            // 6mo|3mo|1mo|1wk|1d  (dropdown 2)
  oos_frac:0.949 }                            // out-of-sample fraction -> in-sample/OOS badge
```
Selector: filter manifest `simulation===true && kind==="portfolio"`; model dropdown = distinct
`source_run`, timeframe dropdown = `timeframe`; bundle id = `${source_run}-sim-${timeframe}`.

**`metrics.json`** adds the comparison bars + a `simulation` block:
```
baseline_return,   // hand-coded rung-0 rule on the same window
buyhold_return,    // buy & hold of the SAME risk-parity basket
random_return,     // random-action floor
regime:{ btc_return, universe_ew_return, label },
simulation:{ source_run, timeframe, window_bars, window_start, window_end,
             oos_frac, train_frac, val_frac, test_frac, git_commit }
```
`baseline_return`/`buyhold_return`/`random_return` drive a "policy vs B&H vs rung-0 vs random" bar;
`oos_frac` drives the honesty badge — windows that overlap the train split look optimistic, so a
viewer MUST see e.g. "48% in-sample" vs "100% OOS". All `*_return` are fractions (×100 for display).

### Frontend rendering (portfolio)
- **Overview:** weights stacked-area/heatmap (`weights.json`) + equity curve + metrics + the
  universe list. No candlestick at the portfolio level (there's no single price).
- **Per-token drill-down** (for fine-tuning — spotting the agent's trade logic): for each
  `universe` token, render `tk_<slug>_candles.json` as candlesticks with the
  `tk_<slug>_trades.json` buy/sell markers overlaid (same marker placement as the single-asset
  view: `time` + `price` + `side`).

## Weekly simulation (`{meta, weeks[]}`) — the "Simulated Trades" dashboard

A SEPARATE single-file contract (NOT the manifest/bundle shape above) for the locked Apentic "Simulated
Trades" page (design + page code in `.design-export-simulated/`). Produced by `scripts/simulate_weekly.py`
([[Simulated Market]]), published **per-model** at `<run-id>/simulated_trades.json`, with a top-level
`simulated_models.json` index (`[{id, model_name, path, n_weeks, window_start, window_end, generated}]`)
so the page can offer a **model selector**. The page derives every metric itself from per-asset
`candles` + `positions` (the producer emits only those — never PnL).

```jsonc
{ "meta": { "start_capital":10000, "candle_interval_seconds":3600, "drawdown_limit":-0.30,
            "universe_size":20, "source_run":"<run-id>", "window_start":<sec>, "window_end":<sec>,
            "n_weeks":28, "generated":"ISO" },
  "weeks": [ { "index":0, "label":"W01", "start":<Mon 00:00 UTC sec>, "end":<+7d sec>,
               "portfolio_start":10000,                 // EVERY week resets to $10k (no compounding)
               "assets": [ {                            // the week's causal vol-top-8 (re-picked weekly)
                  "symbol":"ZEC", "class":"alt|major|peg", "vol_rank":1, "alloc_usd":1002.25,
                  "candles":[ {"t":<sec>,"o":,"h":,"l":,"c":,"v":} ],   // 168 hourly bars, real OHLCV
                  "positions":[ {"entry_t":,"entry_price":,"exit_t":,"exit_price":,"qty":,"kind":"core"} ]
               } ] } ] }
```
Competition semantics baked in: weekly **Mon-00:00-UTC** sessions, **$10k reset each week**, **per-week
universe**. The page computes weekly PnL/equity, intra-week drawdown (Rule 2, `drawdown_limit`), and
daily-activity DQ (Rule 1, ≥1 trade/day) from the positions. **Fidelity rule (do NOT bend data to
schema):** position prices are cost-baked and SNAPPED to the env's exact per-token PnL ledger
(`token_pnls()`) so the page's `qty*(exit-entry)` equals the sim's true equity (recon $0). Note: this
revealed the agent's continuous-eval results don't survive weekly sessions — see [[Experiment Log]]
§2026-06-14 and [[AI Training]] §the-fork.

### INVARIANT (2026-06-19) — no empty-candle assets in a published `simulated_trades.json`

**A published asset MUST carry non-empty `candles`.** The frontend's `computeBacktest`
(`../alexlouis-site/src/apentic`, `backtest.ts` line 261) does `const t0 = candles[0].t` with no
guard, so an asset with `candles: []` is `undefined.t` → the whole Simulations page crashes (the
`SimulationsClient` defaults to the NEWEST model, so a single bad bundle takes the page down).

How the empty-candle assets got there: a **fixed / forced universe** (the closed fixed-13 branch,
`universe_mode="fixed"`) can place a token into the basket in a week BEFORE that token had OHLCV
(e.g. ASTER/HUMA/SIREN/ZEC in early weeks were not-yet-listed) → `candles` came back `[]`. The
causal vol-top-k selector never picks a dataless token, so this only surfaced with a forced universe.

**Producer guard (fix 1):** `scripts/simulate_weekly.py` now **skips any asset with empty candles** —
a dataless token has no trades and 0 PnL, so dropping it leaves the per-week recon balanced (still
$0). This is the authoritative fix going forward.

**De-list mechanism (fix 2) — `scripts/delist_sim_model.py`:** to pull a bad/old run off the page,
rewrite the top-level `simulated_models.json` **without** that run-id and invalidate CloudFront.
Note the no-delete posture (see [[Remote Capabilities]] / [[Apentic Data Contract]]'s `trading/`
section): the S3 publisher can **PUT but not byte-delete**, so this is a **de-list** — the run's
`<run-id>/simulated_trades.json` bytes remain in the bucket, just unreferenced by the index.

Incident: the `eff-s1` (fixed-13) bundle shipped with 11 empty-candle assets and crashed the page;
it was de-listed, then re-published clean (0 empty-candle assets) after the producer guard. See
[[Experiment Log]] §2026-06-19.

### 2026-06-19 — compliance overlay schema fields (`assets[].compliance`, `weeks[].compliance_pnl`)

The `simulate_weekly.py` bundle gains two fields so the dashboard can show the **≥1-trade/day
compliance overlay** (the forced daily BNB↔USDT rebalance that satisfies Rule-1 — a deploy
guardrail, not a strategy signal; see [[Live Forward-Run Harness]] and [[AI Training]]). Both are
**additive** and leave the existing weekly-simulation shape above unchanged.

```jsonc
"assets": [ { …, "compliance": true } ],   // bool — true ONLY on the BNB compliance asset
"weeks":  [ { …, "compliance_pnl": -74.0 } ]   // float — the sleeve's realized PnL for the week
```

- **`assets[].compliance`** (bool) is `true` **only** on the single BNB compliance asset appended
  to each week; it is absent/false on every strategy (vol-top-8) asset.
- The compliance asset carries the **same `candles` + `positions` shape as a strategy asset** —
  BNB hourly OHLCV (from the BNB anchor parquet) plus the daily 01:00-UTC-buy → 23:00-UTC-sell
  round-trips (cost baked into the prices, the `simulate_weekly.fold_positions` convention). So the
  page **derives its trades itself** the same way it does for any asset, and — because it always
  has non-empty candles — it never trips the empty-candle crash (the INVARIANT above holds).
- **Its PnL is a SEPARATE SLEEVE, not in the env book.** The compliance asset is **NOT** added to
  `recon_pnl` / `eq` / `weeks[].return` / `weeks[].dd` / the `weekly_score`. The strategy env stays
  at exactly **$10k for fill/obs-parity**, so the leaderboard rank is **unchanged** (no silent
  re-grade). The sleeve's realized PnL is reported separately as **`weeks[].compliance_pnl`**.
- **`weeks[].compliance_pnl`** (float) is the compliance sleeve's realized PnL for that week,
  distinct from `weeks[].return`/`weeks[].dd` (which remain the strategy book). It is **directional
  drag/gain** — a 22-hour daily long-BNB exposure (a sample week realized **−$74 = −0.74%** of the
  $10k book), so it tends to drag in a bear week. Producer: `scripts/simulate_weekly.py` (commit
  `b43d0e2`); the live counterpart records the same round-trips as `fill` rows + a separate
  `compliance_pnl_usd` in the equity ledger row (`trader.agent.event_runner`, commit `d936101`).
- **Not yet verified end-to-end on the desktop** — the dashboard render of the compliance asset is
  pending a `simulate_weekly` re-run after the sbq sweep.

### 2026-06-20/21 — three SYSTEMIC export-path fixes (thin-token attribution)

Three bugs on the **shared export path** that builds the weekly-simulation bundles — all
**systemic**: they are properties of the export code, so they affect **every published model that
holds a thin / low-liquidity token**, not one run. In each case the **week TOTAL was always correct**
(the env equity is the source of truth) — these were **mis-attribution / placement** errors at the
per-token round-trip layer, **not lost PnL**. Surfaced and fixed while republishing the curated
keepers; see [[Experiment Log]] §2026-06-21.

1. **Marker drift from gappy candle arrays.** A thin token has missing OHLCV hours (e.g. SIREN W11
   carried **124 of 168** candles); the dashboard placed trade markers by **array index** assuming a
   dense one-bar-per-hour series, so markers drifted (~14h off). **Fix:** `ap.densify_candles` fills
   internal gaps with **flat zero-volume bars** (`o=h=l=c=prev_close`, `v=0`) so the array is
   contiguous one-bar-per-hour. Applied in `build_portfolio_artifacts`. (commit `6896557`)
2. **Corrupt negative `exit_price` from a `fold_positions` dust crumb.** `fold_positions` (FIFO
   round-trip reconstruction) left a float dust crumb (qty ~`1e-12`) that the ledger-snap divided
   into, producing a **negative** `exit_price` (−0.124 / −230% on SIREN). **Fix:** drop sub-$0.01 dust
   positions before the snap + snap the residual onto the **largest-notional** position. (commit
   `6896557`)
3. **Forced end-of-week close showed +$0.** Positions held to the session end were recorded at
   `exit_price = entry_price` (0 PnL), and the ledger-snap mis-attributed their real gain to another
   row. **Fix:** mark held-to-end lots at the **week-end close price** (`end_px`). (commit `c019556`)

Same fidelity posture as the INVARIANT above (do NOT bend data to schema): the per-token reconstruction
must net to the env's true equity (recon $0). After the fixes, the curated keepers were republished
clean (see [[Experiment Log]] §2026-06-21).

## Producer side (this repo)
- Single-asset: `trader.report.export_run` (+ `roundtrips_from_position`).
- Portfolio: `trader.report.export_portfolio_run`; `scripts/train_rl.py` records per-step
  weights + per-token trades (env `step` info) and loads per-token OHLCV for the candles.
- Both publish via `trader.report.publish_run` → S3 + CloudFront invalidate.

## `trading/` prefix — live-trading telemetry (publisher built 2026-06-12)

A third top-level surface for the **live agent loop** (consumer: the planned
`/apentic/trading` page — design in [[Real-time Monitoring]] §public monitoring surface).
**As of 2026-06-17 the producer is the event-driven RL champion forward-run**
(`trader.agent.event_runner`, multi-token vol-top-8, paper $10k cold-weekly) — the row kinds
below are unchanged; see [[Live Forward-Run Harness]].
Differs from the run bundles above in producer and cadence:

- **Producer is the EC2 trading host itself** (put-only instance role scoped to `trading/*`
  — the laptop-credential publish path is not involved; the no-delete posture carries over).
- **Continuously updated** (per loop tick / hourly), not a one-shot bundle: equity +
  drawdown series, trade log (tx hashes + guardrail refusals), daily trade count, and a
  `generated` **heartbeat** the frontend ages for the dead-man indicator.
- A `mode: "paper" | "live"` field distinguishes the June 16–21 forward-run from the scored
  window.
- **Freshness: a CloudFront cache behavior on `trading/*` (managed CachingDisabled), NOT
  invalidations** — `CreateInvalidation` can't be path-scoped and would bloat the put-only
  instance role; the run-bundle invalidate-on-publish pattern does not apply to this prefix
  ([[EC2 Trading Host Runbook]] Phase F).

### As-built source rows (2026-06-11) — the loop's ledger the publisher projects

The loop (`trader.agent`) emits the raw telemetry; the publisher (`trader.agent.publish`,
built 2026-06-12) projects these append-only rows from `data/agent_ledger.jsonl`
(`trader.agent.store`) into the static `trading/` JSON. Row kinds (every row carries a UTC
`ts` and `mode`):

```
{ kind:"fill",      mode, from, to, usd_in, usd_out, cost_usd,
                    units_from, units_to, price_from, price_to, reason, tx_hash? }  // tx_hash live only
{ kind:"equity",    mode, tick, equity_usd, peak_usd, drawdown_pct, below_dust }    // hourly PnL mark
{ kind:"heartbeat", mode, tick, equity_usd }                                        // dead-man input
{ kind:"refusal",   mode, intent:{from,to,usd}, refusals:[CODE,…] }                 // guardrail audit
```

### Published shapes (as built — `trader.agent.publish.project`, re-PUT every tick)

```
trading/heartbeat.json  { generated, mode, tick, equity_usd }
trading/status.json     { generated, mode, tick, equity_usd, peak_usd, drawdown,
                          below_dust, trades_today, daily_floor_ok, n_fills, n_refusals }
trading/equity.json     { generated, mode, series: [{ ts, equity_usd, drawdown }] }
trading/trades.json     { generated, mode, fills: [<fill rows + time/time_utc>], refusals: [...] }
```

- **`generated` is the newest `heartbeat`/`equity` row `ts`, never the wall clock** — a
  stopped loop publishes *as* stale, so the frontend's dead-man aging is honest by
  construction.
- **Trade TIME = the bar, in UTC (fixed 2026-06-19).** Each published fill carries `time`
  (unix seconds, exact UTC hour) + `time_utc` (`2026-06-17T16:00:00Z`), and its `ts` is
  **overwritten to the trade time** so any consumer reading `ts` shows when the trade happened;
  the original write time is kept as `recorded_ts`. The raw ledger `ts` is the wall-clock time
  the row was *written* during the weekly replay (≈ now on a restart) — NOT the trade time, so
  it must never be shown as the trade time. `trades_today`/`daily_floor_ok` likewise count by the
  **trade bar's** UTC day (not the write `ts`) vs the ≥1/day floor — else a post-restart
  re-record would mark the whole week's trades as "today".
- **Convention note:** the loop's ledger emits `drawdown_pct` as a **percent** (e.g. `4.2`
  = 4.2%); the published `drawdown` fields are **fractions**, normalized in `project()` and
  nowhere else.
- Wiring: `python -m trader.agent` builds the publisher iff `APENTIC_PUBLISH_TARGET` is set
  (the host env-file sets `s3://alexlouis-apentic-data/trading`); the loop's hook is
  fail-safe — a broken put warns on stderr and never stops a tick.

## PARKED SPEC — configurable simulation frontend (user design, 2026-06-12)

Site page: pick {trained model, historical window, universe-vol-lookback} -> simulated
performance bundle. Enablers BUILT: (1) trainer now persists `policy.zip` +
`vecnormalize.pkl` per run (CRITICAL gap found: every pre-2026-06-12 policy was lost on
process exit — weights begin persisting from the next launch; the in-flight 12-seed pool is
deliberately unpatched to keep the draws identical); (2) `universe_lookback` env param +
`--universe-lookback` (the current selection window is trailing 168h/7d — NOT 1yr; the ladder
24h/168h/720h/2160h/4320h is an untested axis). REMAINING to build: a `simulate` job (load
saved policy + VecNormalize, run evaluate_event_policy over an arbitrary window/lookback,
publish a `kind:"simulation"` bundle — the manifest field already exists and the dashboard
already filters on it) + a precomputed matrix or an EC2-hosted API for on-demand runs (static
site cannot trigger compute — choose per Apentic architecture). The vol-lookback experiment
ALSO runs offline without the frontend: rung-0/B&H across lookbacks is a cheap probe;
model-based requires the new weight artifacts.
