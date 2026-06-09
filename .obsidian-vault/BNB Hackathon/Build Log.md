# Build Log

Chronological record of what's been built and the decisions behind it. The authoritative
*why* lives in the linked topic notes; this is the timeline. See [[Index]] for navigation
and [[Project Overview]] for scope.

## 2026-06-05 — Foundation (Phase 1)

- CLAUDE.md, agent roster, `/orient`, repo/git init, Python `src/trader/` skeleton, `trader`
  MCP server stub.
- Vault: the 8 empty topic stubs developed into full notes.

## 2026-06-06 — Data layer + token universe

### Strategy theory (design discussions → [[Trading Strategies]])

Worked out the *edge* before the plumbing:

- **"Bitcoin is King" factor model** — `r_alt = α + β·r_btc + ε`; the residual ε is the
  idiosyncratic signal; two-factor BTC+BNB; time-varying/downside β; lead-lag.
- **Microstructure edges** — front-runners (alt leads BTC), stop-hunts (liquidity grabs),
  played as **resting orders at pre-computed prices** to beat on-chain latency.
- **Reflexivity / second-order** — indicators map where the crowd's orders sit; trade the
  reaction, not the indicator.
- **Adversarial-market thesis** — BSC tokens are dev-controlled, negative-sum; the edge is
  **risk discrimination**, not fearlessness.
- **Sequencing decision** — *training-first*: build the simulated-market / strategy core
  before the live execution layer. The June-16 internal milestone is reframed to "a working
  trained agent"; the live on-chain loop (TWAK signing, on-chain registration) is a separate
  later spike that still must land before **June 22**.

### Data layer (built, tested, committed)

- **Data sourcing validated and re-routed** (mostly keyless): **GeckoTerminal** (OHLCV) +
  **DexScreener** (screen) + **CMC** (contract resolution) + **GoPlus** (forensics).
  BscScan dropped — Etherscan unified to a V2 key whose **free tier is Ethereum-only; BSC is
  paid** ([[Tech Stack]]).
- `trader.data.downloader` — resumable, cached **Parquet** OHLCV backfill (per-page manifest
  checkpoint, exponential 429 backoff). Proven live + offline (crash-resume, 429 tests).
- `trader.data.cmc` — CMC contract resolver; **corrected 22 wrong pools** vs symbol-search
  (147/148 resolved, fixing 35% ambiguity).
- `trader.data.goplus` — forensic rug/honeypot gate; **removed BAS (hidden owner) + FORM
  (blacklist)**; resumable cache for GoPlus's flaky keyless tier.
- `trader.data.select` — turnover-ranked, CMC-rank-tiered selection with `--exclude`/`--pin`
  manual overrides.
- **Locked the 20-token universe** → [[Token Universe]].
- **OHLCV backfill** (daily + hourly) — **complete for all 20** (~181d daily, ~200d hourly,
  cached to Parquet). 1-minute subset for the liquid names is next.
- **46 tests passing.**

### TradeSim handoff analysis (`tradesim_handoff_seed/`)

Analyzed the prior project's lean handoff; verdict captured in [[Simulated Market]] /
[[AI Training]]:
- **Most valuable artifact = the lessons** — esp. *entry timing never beat random; exits /
  risk-management carried performance* — logged as a research question against our
  entry-centric edge thesis.
- **Ports clean:** leakage guard, metrics suite, benchmarks/backtester, indicator registry
  (+71-col feature schema), grouped-attention extractor.
- **Adapt:** broker (AMM slippage), dataset (→ multi-asset), reward (→ ruin-aware).
- **Data caveat:** the seed's BTC slice is Sep 2024–Apr 2025 only and does **not** overlap our
  alt window → still need a fresh ccxt BTC+BNB pull for the factor model.
- **Discipline to adopt:** real (tested) regime curriculum, fee-blind reward, benchmark gate
  before versioning, smoke test before full runs.

### Factor model + IC gate

- Pulled **BTC + BNB anchor** (ccxt / Binance.US, 0-gap 1m/1h/1d) — the factor data
  (`trader.data.anchor`).
- Ported the TradeSim **indicator pipeline** (71-col, leakage guard) + **metrics suite**;
  verified vs the stored BTC parquet to ~1e-9.
- Built the **two-factor residual model** (`trader.features.factor`): R²-classifier and
  BTC/BNB betas validated empirically (majors→BTC, ecosystem→BNB, XAUt uncorrelated).
- **IC gate refuted the residual-momentum *continuation* hypothesis** — negative IC at every
  horizon (mean-reversion, not continuation; ≈ naive momentum). Factor model → *risk* tool,
  not a selection alpha. Reinforces the post-mortem ([[Trading Strategies]]).

### Cost-aware backtest + 7-day resampling

- Built the **AMM cost broker** + **cross-sectional backtester** + **7-day-window resampler**
  (`trader.sim.{broker,backtest,resample}`).
- **Entry alpha is dead here**: momentum/reversal churn thin pools (200–290× turnover, >100%
  cost drag) and lose; the IC reversal is confirmed untradeable. Only **low-turnover** survives.
- **DQ is not the weekly binding constraint** for diversified low-turnover (P(DQ)≈0% over a
  week; the 62%/34% drawdowns were 7-*month*, not weekly). A week ≈ a coin-flip (median +0.7%).
- **Tournament reframing**: the prize rewards a top-5 finish, not the median — so optimize the
  **upper tail (P(big week) s.t. low P(DQ))**, not minimum variance ([[Trading Strategies]]).

### Upper-tail sweep + activity DQ

- Built the **upper-tail tournament sweep** (`scripts/tail_sweep.py`): rank static tilts by
  P(week > +15% AND not DQ'd).
- **Modeled the ≥1-trade/day rule** as a second DQ gate (`trader.sim.resample`) — **buy-and-hold
  is disqualified for inactivity** (P(DQ)=100%); strategies must rebalance ≥ daily.
- **Candidate found:** daily-rebalanced **`vol-top8`** (8 highest-vol tokens, equal-weight) — 26%
  contender rate at **1% P(DQ)**; volatility tilt ≫ beta tilt; daily rebalance also cuts
  drawdown, so compliance is free. ([[Trading Strategies]] tournament objective.)

- **OOS-validated the vol tilt** (`scripts/oos_validate.py`, 60/40 split): vol-rank persists
  (Spearman +0.66); train-selected vol-top8 **doubles the contender rate on held-out test
  windows** (42% vs all-20's 21%, 0% DQ), ~no skill lost OOS. The tilt is real.

### Regime overlay (BTC risk-on/off gate)

- Built `btc_risk_on` (close > trailing EMA) + `regime_gated` (hold tilt risk-on, cash risk-off);
  resampler now conditions outcomes on each window's BTC return.
- **Honest finding:** real insurance in bear weeks (halves drawdown, eliminates bear DQ) but
  **overpriced** in the bull-conditioned sample — cuts the tournament rate in half (27%→13%) and
  bull upside (+15%→+8%); the insured DQ is only ~2%. Sample has no real crash (under-values the
  insurance); the all-or-nothing 72h gate is too blunt. Stance: ungated vol-top8 = bull bet, gate
  = toggle-able insurance ([[Trading Strategies]]).

### Strategy candidate codified

- Refined-overlay sweep: **partial de-risk (`trend 50%`)** beats the blunt gate (TOURNEY 21% vs
  13%, 0% bear-week DQ); **`stress 50%`** (extreme-only) keeps full upside (TOURNEY 27%) and
  de-risks only in a crash — ideal but dormant/unvalidated here (no crash in the sample).
- **Committed the decision core → `trader.strategy.build_candidate`** (`src/trader/strategy/`):
  daily-rebalanced equal-weight **vol-top8** + regime overlay (default `stress50`; `trend50` =
  validated hedge; `none` = pure bull). The validated candidate now lives in `strategy/`.

### Crash stress test (synthetic)

- Built `trader.sim.crash` (BTC crash path + high-vol-alt amplification via a stress beta) +
  `scripts/crash_test.py`.
- **Overlay VALIDATED:** both gates cut crash drawdown/DQ hard vs ungated (BTC −25% linear:
  ungated 90% DQ → `trend50` 15%, `stress50` 40%). **But `trend50` ≫ the codified `stress50`
  default** — stress50's threshold too lax (misses slow bleeds), half-exposure too little.
  **Nothing half-exposed survives BTC −50%** (needs full cash). Tradeoff now quantified both ways.

- **`trend50` locked as default**; **`severity` gate built + measured** — keeps ~full upside
  (TOURNEY 26%) and uniquely survives a deep slow crash (BTC −50% → 20% DD) but under-protects
  moderate/sharp crashes; complementary to `trend50`, not dominant. **Overlay frontier fully
  mapped; strategy core done.**

## 2026-06-08 — Apentic training/telemetry pipeline (laptop ↔ desktop ↔ frontend)

Reframed the next phase: train on the **desktop** (CPU-parallel, no keys — this RL workload is
env-stepping-bound, not GPU-bound; torch CPU-only), orchestrate from the
**laptop**, surface results in the **Apentic** web frontend (`alexlouis-site`). Built the
pipeline **first**, decoupled and proven locally before the desktop exists ([[Remote Capabilities]]).

- **`remote_train/`** — a generic, **trading-agnostic** job orchestrator (separate package, lifts
  into its own repo later; test-enforced *no `import trader`*). `JobSpec` → `submit`/`status`/
  `publish`, `progress.json` telemetry, pluggable **`LocalExecutor`** (now) + **`SSHExecutor`**
  (desktop over Tailscale). *Decouple-now, extract-after-second-use* — not a premature repo.
- **`trader.report.export_run`** — the bridge to the dashboard's static-JSON contract (manifest +
  `trades`/`metrics`/`candles`/`equity_curve`/`run_info`). `roundtrips_from_position` folds any
  single-asset exposure series into cost-honest round-trips.
- **End-to-end proven:** `scripts/dispatch_demo.py` runs submit → job (a real HUMA trend backtest)
  → publish → manifest upsert; the bundle renders in Apentic at `/apentic/training`. **+13 tests
  (122 total).** Decisions locked: **R2** publish, **SSH/Tailscale** dispatch, **pipeline-first**.
- **Open fork:** frontend is single-asset; our strategy is portfolio. Demo exercises every panel
  via a heuristic; the trained-agent shape is decided with [[rl-ml-trainer]] (pipeline is identical
  either way).

### ✅ Live end-to-end on AWS (2026-06-08, evening)

The full loop runs hands-off on real hardware: laptop `dispatch_demo.py` → SSH trigger →
**desktop runs the job, exports the bundle, self-publishes to AWS S3, and invalidates
CloudFront** over its own internet (no tailnet haul-back). Verified at
`https://data.alexlouis.dev/manifest.json` — **HTTP 200, `X-Cache: Hit from cloudfront`**.

- **Pivoted publish R2 → the site's existing AWS infra** (the publish code is cloud-agnostic S3,
  so no transport change): S3 `alexlouis-apentic-data` + a **dedicated CloudFront distribution
  `E14F268NIY6WLZ` on `data.alexlouis.dev`** (OAC, managed SimpleCORS, *no* SPA error fallback →
  clean 404s; isolated from the site's `s3 sync --delete`). `.deploy/provision-apentic-data.ps1`
  provisions it idempotently; scoped IAM user `apentic-publisher` (S3 Put/Get/**List** +
  `CreateInvalidation`) creds live in the desktop `.env`.
- **The job self-publishes** (`JobSpec.fetch_artifacts=False`) — which is exactly why the
  **path-MTU black hole** on the haul-back (≤512 B returns OK, ≥4 KB stall and the ssh session
  dies) no longer matters: nothing large crosses the tailnet.
- **Debug trail that got here:** Tailscale on the laptop; MagicDNS didn't resolve inside the ssh
  *subprocess* → use the tailnet IP `100.97.195.65`; tar-stream haul-back hit the PMTU wall →
  self-publish; the desktop `/root` clone tracked a **stale P: drive mirror, not GitHub** →
  fast-forwarded it; first publish needed **`s3:ListBucket`** (missing key returns 403 not 404
  without it).
- **Remaining:** the frontend sets `PUBLIC_APENTIC_DATA=https://data.alexlouis.dev` (cross-origin
  subdomain; SimpleCORS already allows it). Then this same path serves real RL training runs — the
  telemetry half is done.

### ✅ Frontend live + training-loop machinery (2026-06-09)

- **Frontend wired:** `PUBLIC_APENTIC_DATA=https://data.alexlouis.dev`; the dashboard renders
  published runs. **Multi-run verified** — HUMA + ZEC both in the manifest after sequential
  dispatches (manifest merge + CloudFront invalidation both correct).
- **Loop machinery (autonomy Level B), scaffolded on the demo before the RL env** — the
  near-autonomous train → diagnose → tune cycle the user envisions:
  - `trader.train`: **config** (RL-extensible dicts + stable key), **registry** (JSON
    experiment store with config → run → result **lineage**), **diagnose** (deterministic
    gates — drawdown DQ, positive Sharpe, fee drag, beats-baseline, ≥1-trade/day — the honest
    "did it actually improve?" so the loop can't chase a reward-hacked run).
  - `trader.train.loop.run_iteration` + `scripts/train_loop.py`: register → dispatch
    (`remote_train`) → fetch the **published** bundle from the CDN (not the tailnet) → derive
    baseline+days from it → diagnose → record. Proven end-to-end: a HUMA `ema120` config
    dispatched, published, fetched, diagnosed **FAIL** with all 5 gates active.
  - MCP **analysis tools** (🟢 READ): `list_experiments`, `experiment` (+lineage),
    `diagnose_run`. Dispatch stays on the CLI until it gains a background variant for long runs.
  - **Autonomy decision:** Level **B** now (mechanical loop automated; reward/curriculum
    changes Claude-proposed + human-gated; bounded hyperparam sweeps OK), Level **C**
    (scheduled overnight) next. Guardrails: val/frozen-test split + beat-baseline criterion to
    avoid the loop **meta-overfitting**; "improve" means beat the vol-tilt baseline OOS, not
    training reward.
- **+10 tests (150 total).** Next: `remote_train` background submit (long runs) + the **RL env**.

### RL training stack built (2026-06-09)

The full path from "dispatch a config" to "trained policy scored on the dashboard" now exists
(detail: [[AI Training]] as-built). Deliberately simpler than the ported TradeSim design —
beat the baseline first, add complexity only if earned.

- **`trader.train.env.PortfolioEnv`** — cross-sectional **exposure-overlay** env (action C),
  pure numpy/pandas so it's testable without torch (laptop Py3.14 has none). Action = exposure
  ∈ [0,1] on the vol-top8; reward = **differential Sharpe − quadratic drawdown-proximity
  penalty**, AMM cost netted into equity (not in the reward — the post-mortem's fee-blind fix);
  causal universe + features, intra-step drawdown. **8 tests.**
- **`gym_env.GymPortfolioEnv`** — gymnasium adapter for sb3; passes `check_env`.
- **`remote_train` background submit** — `executor.launch`/`read_progress`/`is_alive`
  (Local + SSH via `nohup`) + `submit_background`/`poll`. Fire-and-poll for hours-long runs;
  status from the job's `progress.json` (terminal state wins) + liveness fallback. **+2 tests.**
- **`scripts/train_rl.py`** (DESKTOP-only, torch) — time-split train/val/frozen-test, PPO
  MlpPolicy on `SubprocVecEnv + VecNormalize`, eval on held-out val → Apentic bundle →
  self-publish, `progress.json` throughout. Composes tested modules; **PPO glue pending a
  desktop smoke run** (`--timesteps 5000`). +1 (152 total).
- **Next:** desktop smoke-run the trainer (install `.[training]`), fix glue, then a real run →
  `diagnose_run` scores it vs the vol-tilt baseline. Small wiring: an `rl` config kind so
  `train_loop` dispatches `train_rl` via `submit_background`.

### First RL result: exposure-overlay → cash (2026-06-09)

The RL pipeline is **proven end-to-end on the desktop** — config → PPO (vectorized, CPU) →
eval → published bundle → live on `data.alexlouis.dev`, scored by the loop's gates. The smoke
process found + fixed two real bugs: **NaN obs** (BTC anchor `ffill` left leading NaN →
NaN actions → fixed with `bfill` + `nan_to_num`) and the **differential-Sharpe reward
exploding to ±18k** (near-zero variance estimate → fixed with a denom floor + clip to ±10, the
post-mortem's reward-clipping lesson). Reward is now O(1).

**First honest result — action C (exposure overlay) learns *cash is optimal*.** The
deterministic policy mean is ≤0 (sb3 clips to the 0 floor) for every observation: with a
Sharpe-based, ruin-aware reward, *committing* to the vol-top8 is risk-adjusted-negative, so the
agent stays flat (it earns lucky +reward while exploring, but its best estimate is cash). This
**independently rediscovers the project's core finding** — alpha is scarce; holding these
tokens isn't worth it (cash 0% beats the −40% heuristic baselines). Caveats: always-cash is
**degenerate for the competition** (fails ≥1-trade/day → DQ), and it's one config on one split.
→ Build **action B** (allocate/weight tokens, not just dial exposure) — cash-vs-hold is too
thin a lever to show learning; allocation is where the vol-tilt edge lives.

### Action B (allocation) works — and the regime signal was dead (2026-06-09)

**Action B (per-token weight allocation) produces a non-trivial policy**, unlike the
cash-collapsed exposure overlay: on the held-out val window it allocates (~76% mean invested),
returns **+18.4% net, Sharpe ~2.0** (after fixing a ~5× Sharpe over-annualization — daily eval
steps were annualized hourly), **22.3% maxDD** (under the gate), **34 trades**. The loop's
`diagnose_run` scored it — PASS drawdown / positive-Sharpe / activity, **FAIL fee_drag** (fees
eat 96% of net PnL → it churns). The loop *working*: train → eval → publish → "promising
allocator, but fee-heavy → cut turnover."

**Bug found while validating — the BTC regime features were dead.** Factor returns index is in
**seconds**, the BTC anchor in **milliseconds** → `reindex` made BTC all-NaN → the env's
`btc_trend`/`btc_recent_return` obs were always 0. So the exposure overlay (C) couldn't see the
regime it's meant to gate on (confounds the "cash optimal" result), and `candles.json` published
empty (no baseline gate, blank chart). Fixed (align anchor → seconds). Added the **frozen-test
split** + a real **vol-tilt baseline head-to-head** to the trainer for honest validation.

**Frozen-test verdict (3 seeds) — RL loses to the vol-tilt on every axis.** Policy returns
**−5.8% / +10.0% / +15.6%** (seed-unstable; 2 of 3 breach the 30% DQ at 33.7% / 40.7%) vs the
deterministic **vol-tilt(trend50): +25.7%, Sharpe 2.76, 22.0% maxDD**. So the loop's first real
question — *can RL beat the validated baseline OOS?* — answers **no, robustly**: the simple
strategy wins on return, Sharpe, drawdown **and** stability; the RL allocator churns (fee drag)
and is unreliable. A **well-earned negative result**, consistent with the whole thesis (scarce
alpha, entry timing dead, the post-mortem's bull-only breakeven). The discipline (frozen test +
baseline gate + multi-seed) did exactly its job — it stopped a flattering-but-worse policy
(+18% on one val window) from masquerading as progress. **Stance: the vol-tilt remains the
strategy candidate; RL is a thoroughly-documented null.** Remaining RL lever (low EV): a
turnover penalty to cut the fee drag — unlikely to close a 15-pt gap + the DQ breaches.

### In flight / next

- ✅ **Desktop training host — stood up & verified.** Runs inside a fresh dedicated WSL2 distro
  **`act-trainer`** (Ubuntu 24.04, root, systemd), not native Windows — `SSHExecutor` is POSIX
  and Windows-side Python 3.14 has no torch wheel; WSL gives systemd + Python 3.12 + rsync +
  tailscaled. Machine: **8c/16t, 32 GB**. CPU-only torch venv (**122 tests pass**); **Tailscale
  SSH** at `100.97.195.65` / `act-trainer.tail7214b2.ts.net`; data scp'd in (102 MB); the
  dispatch entrypoint runs on the trainer and emits the full bundle. `dispatch_demo.py` now
  defaults to SSH dispatch (`--local` to opt out) and `SSHExecutor` streams artifacts back as a
  **tar over ssh** (Windows has no rsync). Gotchas (WSL idle-shutdown → keep-alive task, tailnet
  naming, private-repo clone-from-/mnt/p, clock skew) → [[Remote Capabilities]]. Remaining
  laptop-side: `--target` → R2. **Desktop on-disk only (GitHub auth pending) for this commit.**
- ⏭️ **RL env on the desktop** ([[AI Training]] / [[Simulated Market]]) — backtester=env,
  metrics=eval, **vol-tilt=baseline-to-beat**, ruin-aware reward, real regime curriculum.
- ⏭️ **Phase-2 on-chain spike** — TWAK self-custody signing, a dust trade, and on-chain
  registration **before June 22**. The unfamiliar, blocker-laden half; gates a real Track-1 entry.
- (optional) combined trend+depth overlay; walk-forward OOS; 1-min micro-edges (banked).
- **1-minute data banked** (9/10 liquid tokens, ~182d; SIREN to re-fill; sparse on thin names,
  ~320–1,350 candles/day). Front-run/sweep features **deprioritized** — entry alpha is dead;
  available if we ever revisit micro-structure.
- **Walk-forward** OOS (multiple splits) for extra robustness.
- **BTC + BNB anchor series** (ccxt) for the factor model.
- Feature engineering → residual/factor model → [[Simulated Market]] broker → backtest.

## Phase status (vs [[Project Overview]] build path)

- ✅ **Phase 1** — Foundation.
- 🔄 **Phase 3/4** — Decision logic + offline validation: data layer + universe done; honest
  broker, features, and backtest are the active work.
- ⬜ **Phase 2** — Stack spike / live on-chain loop: **deferred** under the training-first
  plan; a focused execution+custody spike (TWAK dust trade, registration dry-run) is still
  required before June 22.
- ⬜ **June 16 PoC gate** — reframed internally to "trained agent"; the live-loop gate itself
  is not yet met (no on-chain trade landed).
