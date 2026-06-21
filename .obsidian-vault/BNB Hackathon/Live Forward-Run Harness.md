# Live Forward-Run Harness

How the trained RL champion (**sbq-s1**, `ppo-event-rdLe4-sbq-3c84b4a-s1`) runs **live on BSC in
paper mode** on the EC2 host, forward-testing before the June 22–28 scored window. Built
2026-06-17; the deployed champion is now sbq-s1 (replaced ef-s2 on 2026-06-21 — see §below). The
canonical code line is `main` (the `feat/live-event-harness` branch was merged in and deleted —
see §branch reconciliation). The host itself is [[EC2 Trading Host Runbook]]; the model + training
are [[AI Training]]; the telemetry surface is [[Apentic Data Contract]] §trading/.

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
running the champion at a different capital (e.g. $1k) changes the AMM-cost/liquidity fraction →
fill skew → out-of-distribution. So the paper book IS the env's per-week $10k equity, exactly as
the model was validated and the competition scores.

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

The live 1h OHLCV feed is **GeckoTerminal BSC-pool candles** — because that is what the champion
trained on (the recorded data came from `download_ohlcv` → GeckoTerminal pools). CMC was considered and
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
data silently drift. The **≥1-trade/day floor** — the selective event champion shows several
low-activity weeks (`daily_floor_ok` in `status.json`) — was a real DQ risk; it is now **ADDRESSED** by the
compliance overlay (see §below). The runner already *tracked* the floor; the overlay *satisfies* it.

## Daily market scan (`market_metrics.json`)

A daily EC2 timer (`trader.agent.daily_scan`, `trader-daily-scan.timer` @ 00:10 UTC) refreshes the
top-level **`market_metrics.json`** dashboard (vol/correlation, via `compute_market_metrics`) and
appends a **`selected`** block = the model's ACTUAL current vol-top-8, read from the same env path
the harness trades (`eval_universe_and_caps` over the cold-week window). Because the champion selects
**weekly**, `selected` changes weekly while the metrics refresh daily — it surfaces the real traded
set transparently and does **not** drive the model (a daily re-pick would be OOD for the frozen
model — the explicit design decision). Torch-free. Publishes top-level via the instance role (a
scoped `market_metrics.json` PutObject grant — the role is otherwise `trading/*`-only). Inspect the
live pick on-box with `deploy/inspect_universe.py`. Schema → [[Apentic Data Contract]] §market_metrics.json.

## 2026-06-19 — the ≥1-trade/day compliance overlay (forced daily BNB↔USDT rebalance)

Rule-1 of the competition requires **≥1 trade EVERY day** (a hard DQ axis). The runner already
*tracked* this (`daily_floor_ok` in `status.json`) but nothing *satisfied* it — the event champion
is **selective** (idle between ignitions), so several cold weeks miss the floor. The fix is a deploy
**guardrail, not a strategy signal**: a forced daily rebalance that clears Rule-1, kept out of the
decision core (in [[AI Training]] terms the activity floor was always meant to be a deploy
guardrail, never a strategy discriminator — this is that guardrail, implemented).

**The design (user, 2026-06-19):** each UTC day **BUY 3% of equity into BNB at 01:00** and **SELL
it back to USDT at 23:00** — two recorded trades/day, flat overnight. BNB↔USDT is the deep,
already-allowlisted pair (the Phase-2 spike-trade policy), so it is the cheap, liquid choice and the
ideal first live trade.

**`compliance.py` (`src/trader/agent/`, pure):**
- `compliance_action(now_ts)` → `'buy'` at 01:00 UTC / `'sell'` at 23:00 UTC / `None` (by UTC hour).
- `compliance_cost(usd)` → AMM cost via the same broker (deep BNB liquidity ≈ LP fee + gas, a few
  $/day).
- `compliance_positions(week_start, week_end, px_at, frac, capital)` → the per-week daily round-trips
  for the sim (cost baked into the prices, like `simulate_weekly.fold_positions`).
- constants: `COMPLIANCE_TOKEN=BNB`, `BUY_HOUR=1`, `SELL_HOUR=23`, `DEFAULT_FRAC=0.03`.

**Live runner overlay (`event_runner.py`, commit `d936101`):** `_run_compliance` runs on each hourly
tick and records the buy/sell as `'fill'` ledger rows, so they **count toward `trades_today` / the
daily floor**, routed through the **same `trader.risk` guardrails** (BNB added to the
`forward_run_policy` allowlist). It is kept **OFF the `EventRungEnv` book** — the env must stay at
exactly **$10k** for fill/obs-parity, so the 3% is a **SEPARATE SLEEVE** whose realized PnL is
tracked separately (`compliance_pnl_usd`, in the equity ledger row). **Idempotent off the ledger by
BAR-day** (`bar_ts`, not the wall-clock `ts` the store stamps) so a restart / re-tick never
double-trades (holds under simulated-time replay AND live). BNB price comes from the **BNB anchor
parquet** (`data/anchor/BNB_USDT/1h.parquet`, the source the harness already keeps fresh). Sized by
`compliance_frac` (`0` disables it). `TickResult` gained `compliance_trades`. Tested (14 runner
tests, 6 compliance-specific: schedule, the floor-satisfying round-trip + 3% sizing + allowlisted, same-day idempotency,
sleeve PnL, `frac=0` disable, BNB in the policy).

**`simulate_weekly.py` (commit `b43d0e2`):** replays the **same** overlay into each simulated week so
the dashboard shows the floor-satisfying trades the live runner makes. It appends a **FLAGGED asset**
(`compliance:true`) carrying BNB candles (from the BNB anchor) + the daily round-trip positions.
Still a **SEPARATE SLEEVE**: it is **NOT** added to `recon_pnl` / `eq` / the `weekly_score` — the env
stays $10k for parity, so the **leaderboard rank is UNCHANGED** (no silent re-grade). Each week now
carries a `compliance_pnl` field. Skips cleanly (with a WARNING) if no BNB anchor; the per-week recon
check is unaffected.

**New bundle / dashboard schema fields:** `assets[].compliance` (bool — true ONLY on the BNB
compliance asset) and `weeks[].compliance_pnl` (float — the sleeve's realized PnL for that week,
separate from `weeks[].return` / `weeks[].dd`, which stay the strategy env book). The compliance
asset uses the same candles/positions shape as a strategy asset (so the page derives its trades), but
its PnL is the separate sleeve, not in the week return / DD / `weekly_score`.

**CAVEAT — directional drag (do not omit):** the 01:00→23:00 hold is a **22-hour daily long-BNB
exposure**, so it is **DIRECTIONAL** — it **drags in a down/bear week** (a sample week realized
−$74 = −0.74% of the $10k book) and **gains in an up week**. Given the bearish live-week thesis
([[Market Conditions]] §live-week-read) it will tend to drag. It is the price of Rule-1; tunable via
`BUY_HOUR` / `SELL_HOUR` / `DEFAULT_FRAC` in `compliance.py` (a shorter hold = less directional risk).

**Status (precise):** this is **PAPER/sim logic**, committed + pushed (`d936101` + `b43d0e2`, now on
`main` after the branch reconciliation §below). **LIVE execution of these trades on June 22 still needs the TWAK signing
path** (separate, not built — this fixed BNB↔USDT swap is the ideal first live trade). Live-window
start assumed 2026-06-22 00:00 UTC (an assumption to verify vs the rules). The **end-to-end dashboard
render** of the compliance asset is **NOT yet verified** on the desktop (pending a `simulate_weekly`
re-run after the sbq sweep).

## 2026-06-21 — sbq-s1 deployed, replacing ef-s2 (the surgical 2-file deploy)

The deployed champion is now **sbq-s1** (`ppo-event-rdLe4-sbq-3c84b4a-s1`): the prior config
(`voltopk` k=10, `vol_mult=2.0`) **plus sideways EMA-break suppression** (`shallow_break_max=0.02`,
`consol_vol_max=0.015`), `entry_forward` reward, RecurrentPPO LSTM-256. It was certified on the
held-out **frozen TEST** split (5 cold weeks, fresh $10k each): **+58.6% sum / +11.7%/wk / 5-of-5
winning weeks / worst-week DD 8.8% / DQ-safe** — held up vs validation, no overfit collapse (the
one-shot OOS cert is now CONSUMED; no further tuning to the sbq config). Selection details + the
frozen-TEST certification live in [[AI Training]] and [[Experiment Log]].

**The train/serve-match principle (the deploy invariant).** The serving env must MATCH the
checkpoint's training env, or the model goes out-of-distribution. ef-s2 was trained **without**
suppression, so it was served with the frozen **pre**-suppression env. sbq-s1 was trained **with**
suppression, so it ships **with** the suppression env — matched, no train/serve skew. This is the
[[AI Training]] obs-parity rule applied to a config change, not just a feed change.

**Surgical 2-file deploy.** Only the two **inference** files were updated on the box to the new
code: `src/trader/train/event_env.py` (the suppression env) and `scripts/simulate.py`
(`env_kwargs_from_provenance`, so the live harness rebuilds the suppression `env_kwargs` from the
checkpoint's provenance). The **live harness code was UNTOUCHED** (`event_agent` / `event_runner`
/ telemetry / `compliance`) — the suppression is entirely inside the reused `EventRungEnv`, exactly
per the "reuse the validated loop" principle §above. Steps:
1. Weights (`policy.zip` + `vecnormalize.pkl` + `metrics.json`) pushed to the private bucket
   `s3://alexlouis-act-private/models/ppo-event-rdLe4-sbq-3c84b4a-s1/` (§private model store).
2. EC2 pulled the bundle via **boto3 + the instance role** (no `aws` CLI on the box).
3. **Dry-run gate passed** — model loads, suppression applied, a tick selects `uni=10`.
4. `trader-event-agent.service` repointed (`--run-dir` → sbq-s1) + restarted → active, paper mode,
   publishing `trading/` telemetry.

The leaderboard now ranks #1 sbq-s1; champion = sbq-s1 (see [[Dashboard Leaderboard]]).

## 2026-06-21 — branch reconciliation (one canonical branch: `main`)

`origin/main` (the live-harness line: `event_agent`/`runner`, telemetry, compliance) and
`origin/feat/live-event-harness` (the training / export / leaderboard line) had **diverged** at
merge-base `55cf113`. Root cause: a parallel EC2-deploy chat cherry-picked the compliance commit
onto `main` + added telemetry, deliberately **excluding** the suppression-env commit (to avoid
**ef-s2** train/serve skew — ef-s2 had to keep its pre-suppression env). With sbq-s1 (trained with
suppression) now the champion, that exclusion is moot.

`feat` was effectively a **superset** of `main` (the harness was the same commits cherry-picked to
both; `feat` additionally had all the training/export work + a +32-line `compliance_positions`
superset). Merged `feat` → `main` in an isolated worktree — only 3 conflicts: `compliance.py` +
`test_event_runner.py` (took `feat`'s superset) and `Experiment Log.md` (combined). The merged tree
verified **byte-identical to `feat`**; tested (**493 unit + 28 harness**). Merge commit **`3cfb5aa`**
(2 parents). The box was redeployed from **committed `main`** (clean working tree, no fragile
working-tree mods), running sbq-s1. The redundant `feat/live-event-harness` branch was **deleted** →
`origin/main` (`3cfb5aa`) is the **single canonical branch**.

## 2026-06-21 — pair-freshness audit + the UB "12:00 ignition late?" forensic

**Trigger (user):** UB surged from the daily-session open; the champion bought at **12:00 UTC**,
and the worry was that ignition monitoring watches **USDT** volume while UB's pool is
**USDC**-dominated — so the entry might have been late on a thin/lagging USDT signal.

**Finding — premise false, but the instinct caught a real (separate) bug.** Selection reads each
token's **deepest** pool, not USDT ([[Token Universe]] stage 1); UB is on its **USDC** pool
(~99% of UB liquidity/volume). Reproducing the env ignition formula (`event_env.py:220-226`,
`vol_mult=2.0`) on real GeckoTerminal candles (`scripts/probe_ub_pair_timing.py`):

- USDC `surge` crossed 2.0 at **03:00 UTC** — volume was never the bottleneck.
- The 9h wait to 12:00 was the **price/cushion gate** (`px>EMA72 & ema_up`) holding off after
  UB's −23% 06-19 crash until the trend reclaimed — the anti-falling-knife gate working, not a
  data gap. (The 06-19 13:00 `surge`=4.2 spike was a **dump**: price −23%, gates shut — the
  det-blacklist pattern, correctly *not* an entry.)
- The **USDT** pool `surge` **never reached 2.0** → a USDT-watcher would have **missed** the
  +40% move. **Aggregating** all UB pools fires the **same** bar (USDC dominates; median surge
  lift 0.1%) → the proposed cross-pool aggregation is a **no-op** for UB.

So 12:00 was the **earliest the strategy's own gates allowed**, and pairing was not the cause.
*Caveat:* the gate reconstruction used the raw USDC price as a proxy for the env's
factor-adjusted `_px`; bar-level timing is approximate, but the entry-bar verdict held
identically on USDC and aggregate.

**The real catch — stale frozen pairs** ([[Token Universe]] §pair-freshness): 17/20 fresh; **ZEC
the only live exposure** (traded on a 23× shallower pool); ASTER/XAUt fail safe.

**Decision — FREEZE for the live window.** Train and serve read the **identical** frozen
`pair_address` (`build_volume_panel` in both `train_rl.py` and `event_runner.py`), so there is
**zero train/serve skew today** — the frozen pool *is* sbq-s1's certified distribution.
Repointing a pair or switching to aggregate volume **creates** skew on a checkpoint whose
one-shot TEST is consumed; under the 30% DD gate the asymmetry (downside = DQ; upside small —
thin pools rarely rank into the vol-top-8) says leave it. Log ZEC/ASTER drift as a known
limitation, not a final-hours edit.

**v2 (post-submission):** periodic dominant-pool re-screen → **repoint → retrain → re-cert** as
one unit (never a serve-time patch); cross-pool aggregate-volume **only** if liquidity-floored
(drop pools < ~5% of the deepest) **and** retrained — low priority given the ~0.1% measured lift.
Safe-now option: a serve-time **WARNING** when a *selected* token's frozen-pool liquidity ratio
is below floor (pure telemetry, no behavior change, catches the ZEC case). Tools:
`scripts/audit_pair_freshness.py`, `scripts/probe_ub_pair_timing.py`.

## 2026-06-21 — live TWAK signing path wired into the harness (proven on the dev wallet)

The harness's #1 "not built yet" — the live signing path — is now **built, tested, and proven on
real funds** (the dev spike wallet, NOT the competition wallet). The hard part (`execute_trade`:
check → quote → re-check → ledger → sign → confirm) already existed from the [[TWAK Spike Runbook]];
this wired it into the event-driven runner.

**Design — additive, gated, paper byte-identical (the EC2 service is untouched):**
- `execution/execute.py` gained **`dry_run`**: the full two-phase guardrail check + a REAL quote,
  then stops before the ledger attempt and the swap (writes nothing on a pass). The safe pre-flight.
- `agent/event_runner.py` gained a gated sleeve **`_sign_live`** that routes guardrail-PASSED
  **strategy fills AND the compliance BNB↔USDT trade** through `execute_fn` (= `execute_trade`).
  It fires ONLY when `mode=="live"` **and** an `execute_fn` is wired. Knobs: `live_policy` (the tight
  real-money caps, separate from the loose env-parity `self.policy`), `live_compliance_usd` (the env
  equity is the **$10k cold-weekly book, not the real bankroll**, so `frac*equity` would oversize the
  real swap — this overrides it), `live_dry_run`. Default `execute_fn=None` ⇒ paper unchanged
  (`_exec_summary` adds `tx_hash`/`exec_status` only when signing happened).
- `scripts/live_exec_smoke.py` — the local driver: DRY-RUN by default, `--execute` to sign, tight
  test caps **$0.50/$1.50/$3**, an isolated ledger (`data/risk_ledger_localtest.jsonl`).
- Tests +6 (3 `dry_run`, 3 live-wiring with a mock executor); **full suite 524 passed**.

**Proof (dev wallet `0x2C19…D32E`):** a real BNB↔USDT round-trip, both confirmed on BSC, guardrails
enforced (negative proof: $1.00 > $0.50 → refused at the intent phase, no network):
- SELL BNB→USDT $0.40 — tx `0xac75ff719de08e81fcfad6f838931b372613ac1137c4db6a64094da84a83f380`
- BUY USDT→BNB $0.40 — tx `0xf4b5dc29dd191298622a7a0daa6942a3493e213b0b8f380bca9846cb6b3d4501`

The wallet returned ~flat (few-cent AMM + slippage cost); slippage held at 1.0%, impact 0%.

## What's NOT built yet
- **EC2 live flip + real strategy-fill execution.** The signing path is wired at the runner level and
  proven locally, but: (1) `event_agent._resolve_mode` on the box still REFUSES live (deliberate — the
  box stays paper; the competition-wallet flip is Phase G/H, [[EC2 Trading Host Runbook]]); (2)
  **strategy-fill REAL sizing/routing is Stage 3** — env fills are `frac*$10k` (~$hundreds) and the
  runner hardcodes `CASH_LEG="USDT"`, but most tokens are WBNB/BTCB-quoted ([[Token Universe]]
  §pair-freshness); at a small real bankroll these *safely refuse* on the per-trade cap (fail-safe),
  but proper live needs scale-to-real-bankroll + per-token deepest-pool routing.
- Richer portfolio / per-trade-reasoning telemetry surface (the fills already carry the trigger +
  obs; [[Trade Reasoning Capture]] is the eventual home).
