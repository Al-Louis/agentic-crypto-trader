# Live Forward-Run Harness

How the trained RL champion (**ef-s2**, `ppo-event-rdLe4-ef-503b784-s2`) runs **live on BSC in
paper mode** on the EC2 host, forward-testing before the June 22–28 scored window. Built
2026-06-17 (branch `feat/live-event-harness`). The host itself is [[EC2 Trading Host Runbook]];
the model + training are [[AI Training]]; the telemetry surface is [[Apentic Data Contract]] §trading/.

## The core principle — reuse the validated loop, don't reimplement

The model is only valid fed the **exact** observation `EventRungEnv` produced in training — any
train/serve skew sends it out-of-distribution. So the harness **re-runs the validated inference
path verbatim** (`scripts/train_event.evaluate_event_policy` + `EventRungEnv` +
`simulate.make_predict`) and swaps only **two layers**: recorded panel → live rolling panel, and
the offline replay cadence → an hourly tick. Nothing about the strategy (vol-top-8 selection,
ignition timing, trailing-stop / loss-floor / tp-rungs / intrabar-floor / rule-overlay, fills) is
re-coded — it all stays inside `EventRungEnv`, which is reused unchanged.

**Why this is safe (the weekly-replay insight):** `EventRungEnv` precomputes its signals over a
panel and is deterministic given that panel, and the deployment cadence is already defined by
`simulate_weekly` — each **calendar week (00:00 UTC Monday)** is an independent COLD session:
fresh **$10k**, vol-top-8 reselected at the week open from causal trailing vol, LSTM reset, no
cross-week compounding. So each hour the harness re-runs `evaluate_event_policy` over the current
cold-week window (warmup prepad → now) and **diffs the fills** — any fill on the newest bar is
this hour's decision. Closed bars never change, so the replay for past bars is stable; only the
newest bar can introduce a new event. Obs-parity thus reduces from "did I reimplement the obs
correctly" to "does my live panel equal the recorded panel" — a diffable data check.

**$10k cold-weekly is mandatory, not a choice.** The env prices fills on its internal index;
running ef-s2 at a different capital (e.g. $1k) changes the AMM-cost/liquidity fraction → fill
skew → out-of-distribution. So the paper book IS the env's per-week $10k equity, exactly as the
model was validated and the competition scores.

## Modules (`src/trader/agent/`)

- **`event_live.py`** — `cold_week_window()` (Monday-open + 168-bar warmup prepad, truncated at
  now), `LiveEventTrader` (holds the checkpoint + provenance, rebuilds the exact `env_kwargs` via
  `env_kwargs_from_provenance`, runs one hourly week-evaluation with a **cold-LSTM-per-week**
  predictor), and `new_fills()` (the deterministic hourly fill-diff).
- **`live_data.py`** — the hourly data updater. Keeps the `data/` panels fresh by appending the
  just-closed bar and regenerating `r_alt` via the **exact training producers**
  (`trader.data.anchor.download_anchor`, the OHLCV part-file cache, `trader.features.factor`), so
  the live files are byte-compatible with recorded. Two invariants: **finalization** (only closed
  bars appended, `ts+3600 ≤ now`) and **append-immutability** (past bars never rewritten) — the
  pair that lets the weekly replay reproduce past decisions.
- **`event_runner.py`** — records the env's fills through the hard [[Security and Encryption]]
  guardrails (`trader.risk`: allowlist + per-trade/daily caps + slippage + 30% drawdown stop),
  logs every decision to the agent ledger, marks within-week equity/drawdown + heartbeat, tracks
  the ≥1-trade/day floor, and publishes. The env IS the paper book; the runner does not
  re-simulate. `forward_run_policy` caps are paper placeholders (Phase G sets live caps); the
  allowlist + DD-stop are binding.
- **`event_agent.py`** — `python -m trader.agent.event_agent --run-dir <ckpt>`: the hourly loop
  (ticks on start, then at HH:03), clean SIGTERM shutdown. **PAPER-ONLY** — the event harness has
  no TWAK signing path yet, so live mode refuses loudly.

**Obs-parity gate (offline, `tests/test_live_data.py`):** truncate a recorded token, replay later
bars forward, prove `r_alt` + volume match recorded EXACTLY (`rtol 1e-9`) on the appended bars and
past bars stay byte-identical. The train/serve-skew guarantee, proven without the box.

## Live data source — GeckoTerminal (parity-locked), NOT CMC

The live 1h OHLCV feed is **GeckoTerminal BSC-pool candles** — because that is what ef-s2 trained
on (the recorded data came from `download_ohlcv` → GeckoTerminal pools). CMC was considered and
**rejected**: it is CEX-aggregated spot (different price/volume/microstructure), which would push
the frozen model out-of-distribution; and the existing `cmc_market` only does `quotes/latest`, no
1h history. Decided with the user 2026-06-17. (Switching feeds would require retraining on the new
source.) BTC/BNB anchor stays ccxt/Binance.US (the factor anchor, as in training).

## Private model store — weights are not public

Trained weights are core IP; the results bucket (`data.alexlouis.dev`) is CloudFront-public, so
weights never go there. A dedicated private bucket **`s3://alexlouis-act-private/models/<run-id>/`**
(Block Public Access + SSE, no CDN behavior) holds `policy.zip` + `vecnormalize.pkl`; the EC2
instance role gets a scoped `s3:GetObject`. The provenance `metrics.json` comes from the **public**
bundle (eval metadata isn't secret). Full procedure + IAM policies: `deploy/private-model-store.md`,
`deploy/iam/private-models-{get,put}-policy.json`.

## Deployment (2026-06-17, LIVE)

`trader-event-agent.service` (`deploy/trader-event-agent.service`, hardened like the HoldCore unit)
runs the loop in paper mode, **replacing** the HoldCore placeholder (`trader-agent.service`,
disabled). torch 2.12.1+cpu + sb3-contrib 2.9.0 + ccxt in `/srv/trader/venv`; checkpoint at
`/srv/trader/models/<run-id>/`. Gate before enable: an on-box dry-run
(`event_agent --run-dir … --once --no-refresh`) proved the model loads, the LSTM threads, and a
tick produces fills. First live tick 2026-06-17T23:49Z; publishing to `data.alexlouis.dev/trading`.

## The 429 degenerate-universe bug (found 2026-06-18, FIXED)

The user spotted **XRP and LINK** in the trade log and questioned the vol-top-8. Diagnosis:
`fetch_alt_latest` had **no 429 retry**, so under GeckoTerminal's rate limit most of the 20 tokens
silently dropped each tick (the WARN was on block-buffered stdout, invisible in the journal). The
vol-top-8 then fell back to whichever ~8 tokens happened to have data — the **low-vol majors**
(XRP/LINK), not the genuinely volatile microcaps. The selection *code* was correct; it was
selecting from a starved pool.

**Fix (`cdbcf03`):** exponential 429 backoff on `fetch_alt_latest` (mirrors the backfill
downloader), WARN → stderr (flushed, visible), pacing 2.5s → 3.0s/token. **Verified:** the true
vol-top-8 is `SIREN, COAI, SKYAI, UB, BANANAS31, B, ZEC, HUMA` (XRP #15, LINK #16 — correctly
excluded); the clean run trades only true-top-8 tokens (first fill HUMA, IGNITION). 19/20 tokens
fresh live; **XAUt** has a genuinely inactive BSC pool (perma-stale, ranks last, never selected).
`deploy/inspect_universe.py` prints the live selection + the whole-pool vol ranking on demand.

**Lessons:** (1) under systemd, log operational warnings to **stderr** (stdout is block-buffered →
invisible); (2) any external data fetch in the hourly loop needs 429 backoff, like the backfill
downloader; (3) a degenerate universe is *silent* — the count (`uni=8`) looked fine because k=8
always; only inspecting the per-token vol ranking revealed the NaN-starved pool. This is the
[[Project Overview]] "thin BSC liquidity" open question made concrete: the live feed must be
rate-limit-resilient and ~1 of 20 pools is effectively dead.

## Steady-state watch
Each hourly tick fetches all 20 tokens (~20 gentle calls/tick with 3s pacing + backoff) to append
the 1 new bar. Self-healing (a one-bar-behind token catches up next tick). If 429s persist in
steady state, the clean fix is a GeckoTerminal API key (higher limits) — surface it rather than let
data silently drift. Also watch the **≥1-trade/day floor**: ef-s2's published gate shows several
low-activity weeks (a real DQ risk for the live window — `daily_floor_ok` in `status.json`).

## What's NOT built yet
- **Live TWAK signing path** for the event harness (paper-only today; live mode refuses). Separate
  from the Phase-G on-chain registration ([[EC2 Trading Host Runbook]]).
- Richer portfolio / per-trade-reasoning telemetry surface (the fills already carry the trigger +
  obs; [[Trade Reasoning Capture]] is the eventual home).
- Branch `feat/live-event-harness` is **unmerged** to main (the box runs the branch).
