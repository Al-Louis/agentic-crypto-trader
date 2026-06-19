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
- **What ports is the engineering discipline** — the seed is single-asset BTC with a heavy
  technical-indicator stack (a different market and problem), so its value here is the
  engineering practice listed below, not its strategy-level findings.
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

**First frozen-test data point — NOT a verdict.** At the smallest, feature-poorest config
(**50k steps**, **6/26 bare scalar obs, zero technical indicators**, no curriculum, one
timeframe), the policy loses to the vol-tilt across 3 seeds: returns **−5.8% / +10.0% / +15.6%**
(seed-unstable, 2/3 breach the 30% DQ) vs deterministic **vol-tilt(trend50): +25.7%, Sharpe
2.76, 22.0% maxDD**. The frozen-test discipline earned its keep — it caught that the +18% val
number was a mirage (one window + the then-dead regime signal). **But this is the *start line*
of RL exploration, not the finish.** 50k steps is ~1% of TradeSim's converged ~5M; the obs has
none of TradeSim's ~28 indicators + grouped-attention extractor; no staged curriculum, no
timeframe variation. The loop pipeline exists precisely to widen this search — concluding here
would defeat its purpose. **Exploration roadmap** (highest leverage first): (1) **richer
observations** — fold the existing factor features (residual, β, resid_mom, R²) + technical
indicators per token into the obs; (2) **far larger timestep budgets** (300k → 1M → 5M+, via
overnight Level-C background runs); (3) **staged regime curriculum** + synthetic-crash injection
(the post-mortem's #1 lesson); (4) **timeframe / rebalance-cadence** variation; (5) the
**grouped-extractor + RecurrentPPO** architecture once features are rich; (6) **reward
refinement** (turnover penalty for the fee drag). Frozen-test + baseline-gate stays the honesty
backstop throughout — held conclusions, wide search.

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

## 2026-06-09 — Reward-shaping sweep, data-realism audit, experiment ledger

### Frontend honesty pass (→ [[Apentic Data Contract]])
- Fixed `total_trades` (was counting rebalance *days*, not trades — 34 → ~194 real per-token
  trades), added real `win_rate`/`profit_factor` from per-token FIFO round-trips, and corrected
  `avg_win_pct`/`avg_loss_pct` from mislabeled **dollars** to genuine **return fractions** (clipped
  to [-1, +10] to kill dust/fee artifacts). Every bundle now carries the seed in its `model_name`.

### Reward-shaping sweep #1 (→ [[Experiment Log]], [[AI Training]])
- Added `--reward-mode {sharpe,giveback,realized,turnover}` + `--rich-obs`. The env now tracks
  per-token **cost basis** + **high-water unrealized return**: `giveback` penalizes surrendering
  gains from a held position's peak (a learned trailing-stop that *selling* never triggers),
  `realized` rewards locked-in profit, `turnover` penalizes churn. Rich obs add per-token
  unrealized gain + distance-below-recent-high so the policy can *see* the profit it holds.
- 12-run sweep (4 modes × 3 seeds × 100k, identical obs/seed → reward is the only variable).
  **All four modes beat the vol-tilt baseline (+78.7%):** realized +198% / sharpe +152% /
  turnover +127% / giveback +103%. At 20k without rich obs, RL *lost* to the baseline — rich obs +
  steps flipped it. Frontier is **return-vs-DQ**: the high-return modes breach the 30% gate; only
  turnover/giveback clear it on the *mean*, but **every mode's worst seed hits ~40–43% DD** —
  robustness, not return, is the gap.

### Data-realism audit (skepticism on the +100–200% returns)
- Per-token PnL **reconciles** to the equity curve ($22.0k vs $21.1k) → not a frontend bug.
- SIREN's violent path is **real data**: the −81% bar traded **10.2M vs 11.3k median volume**
  (~900×) — a genuine liquidation event; SIREN is **CMC #72**, vetted at **$1.1M/24h**, $9.2M pool.
- The AMM friction (~$18 = **0.36%** on a $5k trade vs a $9.2M pool) is **defensible** constant-
  product math (slippage from pool depth, not daily volume — my earlier "fantasy" framing was
  wrong). Returns are real within a mostly-sound sim; residual gaps: static liquidity under stress,
  and concentration (one token can dominate). Tools: `diag_token_pnl.py`.

### Experiment ledger — the TradeSim lesson, made structural (→ [[Experiment Log]])
- `train_rl` now stamps a full **`provenance`** block (git commit + every hyperparameter) into each
  bundle. `build_ledger.py` rebuilds a committed, append-only `experiments/ledger.jsonl` +
  `experiments/champion.json` (best mean return under the DD gate, with the exact reproduce
  command). Never tweak without a permanent, version-controlled performance trail again.
- **Champion (provisional):** `turnover` +126.5% @ 29.6% mean DD (worst seed 41.1%).

### Thesis recalibration (→ [[Market Conditions]])
- Re-anchored: realized-volatility capture **is** the edge (not the S&P 500); the ~30% drawdown DQ
  gate is the only hard constraint. Stop importing tradfi skepticism / writing approaches off early.

## 2026-06-09 (overnight) — Sweep #2 (1M-step composite frontier) + fee audit

### Composable rewards + the 1M frontier (→ [[Experiment Log]])
- Made the shaping terms composable (`--reward-mode composite` stacks giveback + realized +
  turnover by their lambdas; `--dd-lambda` exposes the drawdown brake). Ran 6 configs × 3 seeds ×
  **1M steps** overnight — realized's engine + drawdown brakes of increasing strength.
- **Headline: more training regularizes the engine.** `realized`@100k (+198%, worst-DD 41.5%, Sh
  4.75) vs the *identical reward* `real`@1M (+83%, worst-DD **26.6%**, Sh **5.12**). The +198% was
  undertrained high-variance froth; convergence trades return for a gate-safe, higher-Sharpe policy.
  **Deployment champion = `ppo2-real`** (+83.1%, all seeds <30% DD). `real-give` is higher (+156.5%)
  but its worst seed breaches (37.8%). Sobering: gate-safe configs now sit ~*at* the +78.7% baseline
  on **val** — so OOS/regime validation is now the decisive next step (frozen test + walk-forward).

### Fee/turnover consistency audit (→ [[Experiment Log]])
- Sweep-#2 fees far lower at similar trade counts → verified **fees track dollar turnover, not trade
  count** (rate ~constant 0.4–0.6% = the AMM cost). 1M policies trade similar-count but **smaller**
  (fee/trade $12 → $3; turnover $440k → $195k). Same convergence fingerprint as the DD drop — the
  trained policy is calmer *and* cheaper (smaller trades cut slippage, 0.6% → 0.4%). Not a bug.

## 2026-06-09 (cont.) — Universe-selection churn + dynamic re-ranking

### Churn diagnostic
- `scripts/diag_universe_churn.py`: the vol-top-k universe is picked **once** at episode start and
  held for the whole window. Measured how fast that goes stale (daily re-picks across the series):
  set churn is gentle (**0.52 names/day**) but the **rank order collapses** — rank corr 0.85 (1d) →
  **0.32 (7d)** → **0.17 (30d)**. Over a 30-step episode ~2.4/8 names rotate and the positional
  slot-map is ≈shuffled. Stable core (Q 93%, SIREN 81%, UB/B/TAG ~72%) + rotating fringe (ZEC 64%,
  COAI/TAC ~50%). This is why ZEC was in the val universe (vol-rank 5) but out of test (rank 10).

### Dynamic re-ranking (opt-in)
- env **`rerank_every`** (0 = once at start; 1 = daily): re-picks the vol-top-k every N rebalances —
  liquidates names leaving the universe to cash, carries retained, starts entrants flat — so the
  positional slot-map tracks the *current* vol leaders instead of a stale snapshot. Tested (rotation
  + no-orphaned-positions + equity-never-minted invariants; 14 env tests pass). `train_rl
  --rerank-every`, in provenance. **Default off**; recommended **daily** for the generalization work
  (also makes the task more stationary across regimes, which should aid OOS transfer).

## 2026-06-09 (cont.) — Frozen-test OOS verdict: the edge did not generalize

The decisive test (→ [[Experiment Log]]). The two configs that beat the val baseline, run on the
**never-touched test split** (a calmer regime; vol-tilt baseline +25.7% @ 22% DD, gate-safe):

- `real`: val +83% → **test +11.1%** (−15 pts vs baseline, all seeds breach the gate).
- `real-give`: val +156% → **test −1.8%** (−27 pts, 39–49% DD).

**Both collapsed OOS — the simple vol-tilt baseline beat the RL agent on both *and* stayed gate-safe.**
The +83–156% val numbers were **regime/era overfitting**, confirmed (the universe-churn finding and
the memorization hypothesis both pointed here). This is the *earned* conclusion of the full pipeline
(rich obs → 1M convergence → multi-seed → clean frozen window), not a premature write-off:
RL-learns-allocation-from-scratch, as built, has **no generalizable edge**. Caught before any capital.

- **Champion = none** (`build_ledger` now requires passing OOS: split=test, beats test baseline,
  worst-seed under gate). Split-aware leaderboard published; `experiments/champion.json` = `null`.
- **Next — generalization redesign:** train across regimes (walk-forward) > the `rerank_every` 0-vs-1
  A/B on test > regularize hard > reframe RL as a *tuner on the baseline* (the baseline is what
  generalizes). Power outage mid-run; resumable runner added so OOS finished cleanly.

## 2026-06-09 (cont.) — Silent re-rank accounting bug + simulation-integrity guard

Caught only by reading the actual trades: BANANAS31 showed a **−$2,144 loss on a −4% price move**.
The re-rank liquidation of a departed token updated cash but **never recorded a sell marker**, so
per-token PnL / win-rate / profit-factor / fees didn't reconcile with the equity curve for
re-ranked runs (**reconciliation gap $3,467**). Headline return/DD/Sharpe were *always* correct
(equity includes the liquidations) — only per-token **attribution** was broken.

- **Fix** (`1d26881`): `_rerank` returns the forced sells; `step` records them as markers. Gap
  **$3,467 → $30**, verified via `diag_token_pnl.py` reconciliation. The A/B rerank arm re-ran clean.
- **Guard:** the per-token-PnL-vs-equity **reconciliation check** is now the gate for this class of
  silent accounting bug — run it on a bundle before trusting any per-token analysis.
- **Open concern (→ trustworthy sim):** other silent integrity issues may lurk and are *not* easily
  spotted by eye — price-series consistency (r_alt vs candles per token), look-ahead leakage, fee
  double-counts, weight/position conservation. **Plan: a conservation/invariant audit suite** run
  across every bundle + as synthetic-data tests, so corruption is caught automatically, not by luck.

## 2026-06-09 (cont.) — Integrity audit finds (and fixes) a silent data bug

The integrity suite paid off immediately. `audit_bundles.py` (invariant #1, per-token PnL
reconciliation) showed the re-rank marker bug was **not isolated** — ~13 static bundles also failed
with $200–1700 bidirectional gaps. **Invariant #2** (`r_alt` vs candle returns, per token) found the
cause:

- **5 tokens' env return series diverged from their candle prices** — ZEC catastrophically
  (+141.5% `r_alt` vs −31.7% candles, **+173pt**); SIREN/UB/SKYAI/Q mildly (7–20pt).
- **Root cause: a spurious opening-bar return per token** — the feature pipeline computed each
  series' first return against a non-existent prior price (ZEC's was a phantom +253.8%). Every other
  bar matched the candle exactly; zeroing the first bar reconciled all five.
- **Fix** (`6ed8412`): `load_data` zeros each token's first valid return (a return with no prior
  price must be 0). All 20 tokens now reconcile. `audit_data.py` is the **invariant-#2 gate** (exits
  non-zero on divergence) — run before trusting a training run.
- **Impact is limited:** the bad bars sit at each series' start (Nov–Dec 2025), **months before the
  val/test windows**, so the eval results (val/OOS verdicts) stand; only training's opening was
  mildly affected. The earlier "val is partly phantom" worry was overstated.

**Integrity suite status:** #1 bundle-PnL reconciliation ✅, #2 data price-consistency ✅ (both now
gates). Still to add: cash/position conservation, fee totals, weight conservation, and the big one —
**look-ahead / causality**.

## 2026-06-09 (cont.) — Strategy pivot: committed candidate v1 (→ [[Trading Strategies]])

After the OOS failure (RL-from-scratch doesn't generalize; vol-tilt baseline beats it) + the rerank
A/B (universe freshness isn't the lever; re-ranking *tripled* turnover), pivoted from "RL learns
allocation" to a **signal-grounded, rule-first** strategy, sketched with the user.

- **Chassis decided:** (1) ≥1 trade/day is wallet-level total (confirmed in the rules: "7 over the
  week") → **hold-by-default** with a no-trade band, killing the forced daily churn; (2) a
  **rarely-fired ~25% drawdown backstop**, with primary DD management *learned* (its trigger rate =
  a policy health metric).
- **Edge = the user's discretionary discipline**, encoded as a per-token state machine: enter on a
  confirmed trend, *let winners run*, exit on the rollover, **no-FOMO re-entry** (cooldown +
  fresh-high), **dead-zone** (never churn sideways below the runup origin). Grounded in the SIREN
  case (our RL FOMO-bought the $1.28 peak and churn-traded the corpse below origin 8+ times).
- **Honest reconciliation with prior work:** momentum *selection* already failed here (negative IC,
  mean-reverting; "entry alpha is dead, only low turnover survives"). The user's rules are **exit +
  anti-churn discipline = the documented edge**, not the refuted selection claim — so v1 is the
  proven **vol-top8** universe + that discipline, *not* a momentum-alpha bet.
- **The ladder:** rung 0 = hand-set rules (interpretable, the new baseline-to-beat); RL tunes the
  thresholds at rung 1+ only if it beats rung 0 OOS — so we never commit to one architecture blind.

## 2026-06-09 (cont.) — Rung 0 built + threshold sweep overfits (→ [[Trading Strategies]])

- **Built rung 0** (`trader.strategy.rung0`) — the per-token state machine (enter on breakout, let
  winners run, exit on rollover, no-FOMO cooldown, dead-zone) as a stateful `run_xs_backtest`
  weights-fn; test pins ride-runup-then-stand-aside; `eval_rung0.py` compares vs the baselines.
- **First read (frozen TEST):** rung0 **+17.0% @ 12.3% DD** (best Sharpe 2.81, lowest turnover) vs
  vol-top8 hold +22.5% @ **34.6% (DQ)** / trend50 +25.7% @ 24.1%. The discipline *works* (SIREN: held
  one day then cash — no churn, vs RL's 8+ churn trades) — but it's **dialed too conservative**: uses
  only ~12% of a 30% DD budget, so it leaves return on the table and doesn't beat trend50.
- **Threshold sweep (rung 0.5) OVERFIT.** Grid-searched the 4 knobs on val, picked best val-return
  under the gate: **+167% on val → −17% @ 44% DD on test** (blows the gate). The conservative default
  *generalizes*; the val-greedy config detonates. **Same trap as the RL** — single-window greedy
  tuning (policy weights *or* rule thresholds) finds the val-noise-fit point. Robust aggression needs
  **walk-forward / multi-window** selection, not one val window. `scripts/{eval,sweep}_rung0.py`.

## 2026-06-09 (cont.) — Walk-forward sweep: discipline loses to vol-top8 on the tourney objective

`sweep_rung0_wf.py` — robust multi-window selection (P(week>+15%) at P(DQ)<5% across ~120 random
7-day windows). It **rejected the single-window overfit** (only 36/144 gate-safe) — the method works.
But on frozen-test windows (all 0% weekly DQ): **vol-top8 hold 15% tourney > trend50 9% > rung-0
pick 6% > default 3%.** The disciplined rules sit *below* the baseline because the prize rewards
upside *variance* and discipline suppresses it (it's the right objective for real trading, the wrong
one for the contest). **Second hypothesis to lose to vol-top8** (after RL-from-scratch) — the
selection is the edge. Strategy side has converged; the open work is the **unbuilt Track-1 execution
loop** (TWAK signing + on-chain registration — the June 16 PoC gate). See [[Trading Strategies]].

## 2026-06-09 — Rung-0 made event-driven, then trade-logic forensics

Pivoted rung-0 from a daily rebalance-to-target to a true **event-driven, intra-day** executor
(act the hour a signal fires; let winners run untrimmed), then published it to the frontend and
read the **actual buy/sell markers on the candles** — the only way to see whether the rules are
too rigid. Built two diagnostics: `trace_gates.py` (per-bar entry/exit gates vs the real candle
close — also catches strategy-space-vs-candle divergence) and `trace_funding.py` (portfolio-level
funding/markers). The forensic read surfaced **four patterns**, three of them fixable bugs:

1. **Capital model (the ZEC mystery) — silent accounting bug.** 20%-per-entry x up to 8 holds x
   never-trim **starved cash after ~5 names**, so a great later ignition couldn't be funded — and
   the state machine flipped it to `held=True` *anyway*, so it **phantom-held** through the whole
   runup owning nothing, then logged a markerless paper-exit. ZEC's perfect May-1 ignition (+28%)
   was lost this way (unfunded at cash=-$5). Fix: moved **all** held/cash/sizing state into
   `run_rung0`; `held=True` only when **funded**; a fresh ignition with no cash does
   **loser-funded rotation** — close the **weakest holding** (lowest price/EMA cushion) *only if*
   it's weaker than the candidate, so winners stronger than the new opportunity are never trimmed.
   Rotation sells recorded as markers (the [[Build Log]] re-rank-marker lesson). `build_rung0` is
   now a **stateless per-bar signal**.
2. **Volume-spike detector lagged ~11h.** The 24-bar trailing-*mean* diluted a sharp spike, firing
   B's entry at +52% instead of the +22% ignition. Replaced with a sharp `vol_fast`(4)-bar surge —
   B now enters at the May-11 06:00 ignition.
3. **Low-quality re-entries whipsawed (SKYAI/Q/TAC/UB 2nd trades).** Brief micro-spikes near a flat
   EMA stopped out in hours. Added a **trend gate** (price above a *rising* EMA) — SKYAI's
   02:00->04:00 whipsaw eliminated.
4. **Dead-zone guard confirmed working** — UB correctly stood aside through its -26% post-runup bleed.

**OOS TEST:** +18.2% -> **+29.0%** (Sharpe 3.74, DD 17.4%) — rung-0 now **beats both vol-top8
baselines on return AND drawdown** on the test split, the first time it has. **Caveat, not buried:**
**VAL is -9.4% / 31.5% DD (a DQ)** — same code, a melt-up regime where stand-aside discipline hurts
(plain-hold made +109%). Regime-dependence, not trained-overfit (these are hand rules), but it
**must be understood before trusting +29%**. Bundle: `rung0-rotation-v4`. See [[Trading Strategies]].

## 2026-06-10 - RL on rung-0: features experiment, then the event-driven pivot

Continued from the rung-0 forensics. Widened rung-0's trailing stop **11% -> 25%** so winners ride
pullbacks instead of whipsawing (test +13.9% -> **+18.2%**, fewer trades). Published rung-0 on the
**VAL** window for visual inspection and confirmed the regime caveat directly: **VAL is a melt-up**
(vol-top8 plain-hold +137%, rung-0 -10.9%) - stand-aside discipline *hurts* when everything trends
up. Also fixed a frontend confusion (trades table renders local time, candle axis UTC -> a 4h
display offset, **not** a PnL bug).

Then took up the user's ask: **train RL with the rung-0 rules.** Two attempts (full reasoning ->
[[AI Training]] "As-built 2026-06-10"):

- **Option A - rung-0 signals as RL features** (`--rung0-obs` on the daily-rebalance `PortfolioEnv`).
  Built, locally validated (causal obs, +3 features/token), and a 4-seed x 1M sweep launched. **Killed
  on inspection:** seed-0 put **all 241 trades at 07:00 UTC** (the env's daily rebalance clock - the
  exact rigidity we reject) and its +137% on val merely **matched plain-hold** (the melt-up, not
  skill). Lesson: features can carry rung-0's *information* into the policy but not rung-0's *intra-day
  execution* - the **daily-rebalance env is the ceiling**, not the signal. Shelved.

- **Option D - event-driven rung-1** (`trader.train.event_env.EventRungEnv`). The pivot. A semi-MDP
  that steps at rung-0's **events** (ignition / stop-or-EMA trigger), advancing bar-by-bar between
  them so execution is intra-day by construction. **rung-0 owns the edge** (timing, exits,
  dead-zone/cooldown, loser-funded rotation); **RL learns the discretion** (entry sizing + exit
  override). Built + validated on the laptop: 8 tests + a real-data smoke (**51 decisions across 20
  of 24 hours** vs all-at-07:00 for Option A); the eval/publish path proven torch-free. Trainer
  `train_event.py` + sweep `run_eventrung_sweep.sh`, baseline = the **rung-0 rule itself**. 4-seed x
  1M event-driven sweep launched on the desktop.

**Operational (cost real time, now in the runbook -> [[Remote Capabilities]]):** two remote-training
incidents - (1) fast post-launch checks fired before the ~30-60s torch+volume-panel startup, so a
"dead" launch was relaunched, **stacking parallel sweeps** that spiked Vmmem and forced a reboot;
(2) stopping a sweep with `kill -- -<PGID>` took **tailscaled** down with it (shared process group)
and dropped the box off the tailnet. Both written up as hard rules (launch-once-wait-verify; stop by
specific PID, never the process group; SSH via PowerShell; tiny SSH output for the tailnet MTU).

## 2026-06-10 (cont.) - RL experiment 1: relative-to-rule reward solves the under-trading

The event-driven rung-1 sweep (above) under-traded (2-4 trades/seed, +9.7% test) and lost to the
rule. **Consulted the [[rl-ml-trainer]] agent** (first use) for a forward plan rather than abandoning
RL - the thesis is a real learned agent, and rung-0 is the **baseline to beat**, not a replacement.

Its diagnosis: the **absolute** reward makes passivity optimal and never references the rule, so the
agent has no signal that skipping the rule's ignitions costs anything. **Experiment 1 (built +
shipped):**

- **Relative-to-rule reward** - a shadow rung-0 equity curve precomputed in-env
  (`EventRungEnv._rule_equity_curve`, mirrors `run_rung0`), reward = agent's interval return MINUS
  the rule's. Parity-verified **VAL 0.0pt / TEST 0.3pt** before any training. + relaxed dd, the
  post-mortem exploration config, 2-week episodes.
- **The smoke caught a dud cheaply** (100k, ~2 min): action mean **0.000, 0 trades** - a
  Gaussian-on-[0,1] dead-gradient collapse to the skip boundary. **Fix 1b:** action reparam to
  **[−1,1]** so neutral trades from init. Re-smoke: action mean 0.649, full range, 239 trades - alive.
- **4-seed × 1M frozen-TEST sweep:** **+8.6% avg (±3.7%), 15.7% DD, ~18 trades/seed.** The
  **under-trading is solved** (16-22 trades vs 0-4), all positive + gate-safe + tight spread - the
  first RL config that behaves like a real active agent. Does **not yet beat the rule** (~+18%
  causal); it learned to *act like* the rule, not yet *out-discriminate* it (a capacity gap).

Process note: the smoke-first discipline paid off (caught the collapse before a 20-min sweep), and
the launch/verify/kill-by-PID runbook held across ~5 desktop sweeps today with no repeat incidents.
Standings → [[Experiment Log]]; mechanics + experiment-2 plan (LSTM + regime obs) → [[AI Training]].

## 2026-06-10 (cont.) - deviation-alpha diagnostic -> RL experiment 2 (per-decision reward)

Before building "experiment 2 = LSTM", asked the [[rl-ml-trainer]] (2nd consult) whether to refine
the reward first. It called the +8.6% gap **reward-bound** and proposed a cheap check: the
**deviation-alpha diagnostic** (`scripts/diag_deviation_alpha.py`) - correlate each executed entry's
over-size-vs-rule with its forward-24h return on the exp1 bundles. **Result: corr = −0.027** (flat;
the agent over-sizes indiscriminately and never sizes below the rule). Confirmed reward-bound, so the
LSTM stays deferred.

Built **experiment 2 (`reward_mode="residual"`)**: reward = the agent's **weight deviations from the
rule** dotted with token returns (`Σ(agent_w − rule_w)·ret`), so shared positions cancel and only the
agent's active bets score. Shadow book now returns per-token weights too; rule-exposure added to the
obs (12-dim); `norm_reward=True`. **Verified locally:** rule-mimic agent nets ~0 residual (+0.013),
max-size agent +0.538 - the missing gradient is now present. 11 env tests pass, eval/publish path
torch-free. Sweep `... test residual` -> `ppo-event-res-test-s<seed>`; gate seed-mean > +18%, DD < 25%.
Full reasoning → [[Experiment Log]] / [[AI Training]].

## 2026-06-10 (cont.) - exp2 smoke -> discrimination probe -> exp2b (residual + R4)

The exp2 residual 100k smoke was alive but **under-sized the rule** (entries 0.03-0.12, below 0.20).
3rd [[rl-ml-trainer]] consult: the **minimal-deviation basin** - the still-present one-sided dd brake
makes under-sizing the expected-reward optimum for a skill-less agent. Before committing the sweep,
ran a **discrimination-headroom probe** (`scripts/probe_obs_alpha.py`, no training): do the obs
features at each ignition predict forward-24h return OOS? **Yes - OOS IC +0.246**, driven by
**`cush = -0.423`** (stretched ignitions revert). So the alpha **is in the obs** -> **reward-bound
confirmed**, LSTM stays deferred.

Built **exp2b = residual + R4** (`--r4-beta`): a one-sided foregone-opportunity penalty
`-beta·Σ max(0, rule_w - agent_w)·max(0, ret)` - charge the surrendered upside when the agent
under-sizes a token that rose. Strictly-negative expected penalty on under-sizing, so it closes the
basin without a new over-size incentive. **Verified:** R4 (β=0.4) drives a min-size agent -0.155 ->
-0.544 while the rule-mimic stays ≈0; 12 env tests pass. Sweep `... test residual` now carries
β=0.4. Full reasoning → [[Experiment Log]] / [[AI Training]].

## 2026-06-10 (cont.) - exp2b verdict: the corner-solution finding

β-tuned exp2b on smokes (β=0.4 under-sized, β=0.8 sized up) then ran 4 seeds x 1M frozen-TEST at
β=0.8: **+15.2% avg** (up from exp1's +8.6%) but **fails the gate** - s0 breaches the 30% DQ (31.8%
DD) and **corr = +0.013** (no skill). The +15.2% is **beta, not discrimination** (over-size
everything -> more return + more drawdown).

**The corner-solution finding:** across all four reward variants (relative oversize-all; residual
β=0 undersize-all; R4 β=0.4 undersize-all; R4 β=0.8 oversize-all) the agent always goes to a sizing
**corner**, never to the `cush`-conditional sizing the probe proved is learnable (IC +0.246). Every
reward so far rewards/penalizes sizing **magnitude**, so tuning β just slides between corners. The
alpha is untouched because **no reward pays for *rank-correct* sizing**. Next: a conditional /
IC-based reward (4th [[rl-ml-trainer]] consult); LSTM still deferred (the alpha is in the obs).
Full data → [[Experiment Log]].

## 2026-06-10 (cont.) - exp3: demeaned-ranked residual (corner is a functional-form problem)

4th [[rl-ml-trainer]] consult reframed the corner: a reward **linear in `dev`** can only learn a
*global* size (constant-direction gradient); β just slides between corners. Fix: **demeaned-ranked
residual** `R = Σ dev·(ret − ret_bar) − res_gamma·Σ dev²` (`reward_mode="residual_ranked"`).
Demeaning kills the drift-corner (skill-less E[ret−ret_bar]=0 → only the obs-predictable part, cush,
is left to earn); the quadratic budget gives an interior optimum (rank-correct sizing). Retires R4;
softens the dd brake (the budget caps per-name tilt → caps DD).

**The process upgrade — a reward-landscape preflight** (`scripts/preflight_residual.py`) run BEFORE
training (we never did this for the 4 prior corners): score scripted agents on the reward over real
ignitions, require the **correct-discriminator (`dev ∝ −cush`) to be the unique argmax**, corners ≤ 0,
IC-hacker loses. **PASSES** — demean alone collapses all-big to 0, the budget makes corners strictly
negative, correct-disc wins (+2.69 at γ=0.1, corr +0.239). The corner is provably gone in the reward
*form* before any compute. 13 env tests pass. Sweep → `ppo-event-rank-test`; **corr now a success
gate** (> +0.10). Full design → [[Experiment Log]] / [[AI Training]].

## 2026-06-10 (cont.) — g2b re-run on the fixed env + PROOF that rung-0 gates the trades

The post-`8ccad69` re-run (`ppo-event-g2b`, 4×1M @ `e466f0e`, launched by the prior session,
landed 19:13Z) — full table + verdict in [[Experiment Log]]. Headline: **the env exit bug was NOT
the plateau's cause** (val −7.6% vs −6.7% invalid), **crash survival regressed** (s2 DQ'd at 63.7%
DD — the pre-fix "4/4 survive" was partly trained by the broken env), and the load-bearing fact is
that **the rule itself loses the bull** (−4.6% vs B&H +27.5%) — the event skeleton's ceiling.

**`scripts/verify_rung0_gating.py` (new)** — answers "is rung-0 actually driving the agent's
entries/exits?" with run data, not claims: rebuilds the env's ignition/cushion signals locally from
the bundle's provenance and cross-checks every published per-token trade marker. Result on all 4
g2b seeds: **86/86 buys land exactly on rung-0 ignition bars; every sell is in the stop/EMA-zone
or a loser-funded rotation; zero unexplained trades.** With `ungate=False` the code has no other
entry path (`_scan_bar` requires `_ignite[bar,tok]`), and the empirical check confirms the build
matches the claim. The one rung-0 discipline the agent CAN defeat is the **exit override**
(action ≥ 0.95 re-anchors the trailing stop to the current price — repeated overrides ratchet the
stop down indefinitely), which is rung-1's intended discretion but is the plausible mechanism for
s2's −59.5% crash blowup. Bypass-flag inventory for any config review: `ungate` (drops
cooldown/dead-zone only — ignition is ALWAYS required), exit-override/partial-trim re-anchoring,
`universe_mode` (the agent's basket ≠ the canonical rung-0 baseline's vol-top-8), risk-parity caps
(scale entry sizes), `crash_train` (training data only).

## 2026-06-10 (late) → 06-11 — the rung-1b "rd ladder": forensics-driven substrate + 8 sweeps

The fastest build-measure cycle of the project (a sweep ≈ 20 min at 1M steps): the user's per-token
chart forensics drove probes, probes drove substrate changes, each change swept same-day. Standings
for every rung: [[Experiment Log]]; design detail: [[AI Training]]. In order:

- **Published-data repair** — `scripts/repair_manifest.py`: retro-fit self-describing model_names
  from provenance, tag then **de-list** the pre-`8ccad69` invalid era; `ec1e487` sha-stamped run-ids
  pushed and in force (`ppo-event-<family>-<sha>-s<seed>`). Leaderboard rebuilt **sha-only**
  (`build_ledger.py` default; `champion.json` stays honestly null — a PREVIEW crown is published
  for frontend styling, clearly labeled).
- **Rung-1b `rule_default`** (@ `df943bf`) — action idx 0 EXECUTES rung-0 (forensics showed the old
  discretion vetoed the rule for free); no stop re-anchor (the ratchet), `exit_commit`, `dust_usd`,
  `--rule-prior` init bias. Gated A (oracle ceiling val +74.6% — exits carry it) / B (parity) / C
  (in-env landscape) before compute.
- **Take-profit prompts** (`tp_rungs`, @ `a4132cc`) — the env could not SELL INTO STRENGTH (exit
  prompts fire on weakness only); profit rungs at +25/50/100/200% unrealized, default = let-run.
  Raised the oracle ceiling to **+95.5% val at 7.1% DD**. + **voltop8** (user: the calm half of
  broad-12 bleeds; cut it).
- **`--eval-prepad`** (@ `edb1af2`) — eval warmup served from the prior split's tail; the published
  window is tradeable from bar 0 (no dead first week on the charts; mirrors live).
- **Lever sweeps** — rd8h (+harvest obs), rd8h0 (`dd_lambda 0` — named + half-confirmed the
  **diet-rule equilibrium**: the dd penalty double-counts risk the substrate already bounds; worst
  seed 26.9% with NO reward brake), rd8h0c1 (`crash_train 4→1` + **loss-floor 0.2**, the Q disaster
  guardrail — **first positive val in the project, +4.7%**).
- **Q probes** — `probe_false_flag.py` REFUTED the low-rising entry filter (population data:
  extended movers are the poison); `probe_detonation.py` CONFIRMED the post-detonation blacklist
  (fwd48 −8/−24% train/val, expires ~4wk) → `det_blacklist` built into the ignite precompute.
- **rd9 @ 5M** — regression everywhere (converged to rule-hugging flat); the scale lever failed →
  the gap is memory, not steps.
- **RecurrentPPO (rdL, @ `a27e469`, sweeping)** — `--recurrent --lstm-size 256` + stateful eval:
  the user's failure classes (re-buying post-pump bleed, not holding winners, "no learned
  experience") are SEQUENCE skills a stateless MLP cannot express. Verdict is behavioral, not just
  returns.
- **MCP 4A modernized** (@ `a27e469`+) — `rl_train` full rd-era whitelist + on-box sha-stamping +
  discrete-aware smoke gate (latent parser break fixed); new `rl_verdict` (per-regime table,
  validated == the manual verdicts) + `rl_forensics`; `experiment_record(sha_only, publish)`.
  The manual day is now one tool-complete loop iteration → [[MCP Server]] §As-built.

## 2026-06-11 — Phase 2 custody spike: a guardrailed dust trade CONFIRMED on BSC

The TWAK execution/custody spike ran start-to-finish in one day — full evidence in
[[TWAK Spike Runbook]] (steps 0–8 all done) and [[Security and Encryption]]:

- **The June-16 gate artifact, 5 days early:** a $1 BNB→USDT swap signed via TWAK
  (password from Windows Credential Manager, unattended) through the new guardrail path and
  confirmed on BSC — tx `0x739bb1…7c96`. Negative proof: a $5 intent refuses
  `PER_TRADE_CAP`; the ledger persists real spend across processes.
- **`src/trader/risk/` + `src/trader/execution/` built** (frozen `SPIKE_POLICY`, 8 refusal
  codes, append-only ledger, two-phase intent→quote re-check, `--password`-free CLI
  wrapper); 325 tests passing. Live run caught + fixed the `tx --json` boolean-status shape.
- **Wallet unification PROVEN on `bsctestnet`:** ERC-8004 agentId 1369 minted from the spike
  wallet via native `twak erc8004`; `owner`/`agentWallet` == the trading address. One
  `~/.twak` store covers trading + registration + identity, zero key export.
- **Registration recon:** `compete status` reads the on-chain deadline as **June 25** (later
  than the assumed June 22; June 22 stays the working deadline). `--uri` is *required* on
  the identity mint — the agent card must be hosted before the mainnet mint.
- **`--auto-lock` re-unlock confirmed transparent** (keychain re-resolution, no human step).

Custody-slice blockers all closed: autonomous self-custody signing ✅, registration
mechanics ✅, unification ✅. Remaining Phase-2 half: CMC Agent Hub reads + BNB SDK runtime
probe (`principal-engineer`).

## 2026-06-11 — Plan forward agreed: AWS live host, paper→dust ladder, /apentic/trading

Discussed and locked the path to the live window (user decision; rationale in
[[Remote Capabilities]] and [[Security and Encryption]]):

- **Live-week host = AWS EC2** (small Linux instance, systemd, hardened env-file). Decided
  after the training desktop's WSL VM silently died mid-sweep — the residential-host failure
  mode in person. The **competition wallet is created ON the EC2 box** (keys born where
  signing happens); the spike wallet stays a laptop throwaway.
- **Validation ladder: paper → mainnet dust — testnet trading ruled out** (no real DEX
  liquidity; TWAK quotes route through mainnet aggregators, so testnet fills would validate
  nothing about slippage/PnL). Testnet remains the rung for tx *mechanics* (as used for the
  ERC-8004 probe).
- **Monitoring page `/apentic/trading`:** the bot publishes its own trading JSON from EC2
  straight to the `data.alexlouis.dev` bucket (put-only IAM role, `trading/` prefix) — no
  laptop in the publish path — plus a **heartbeat** the frontend renders stale if the loop
  goes quiet (the dead-man switch the host design called for). → [[Real-time Monitoring]],
  [[Apentic Data Contract]].
- **Sequence:** (1) now→Jun 16: build the agent loop (data reads → `decide()` → 
  `execute_trade`) + paper-run locally; EC2 provisioning in parallel. (2) Jun 16–21: deploy
  paper mode to AWS, wire publish + frontend, create/register the competition wallet (agent
  card hosted first), one dust trade from the production host. (3) Jun 22–28: live window.
  (4) post-stability: sponsor-tool expansion for special-prize coverage (CMC reads are NOT
  deferred — they're the loop's data feed and the other half of the Phase-2 gate).

## 2026-06-11 (cont.) — Phase 2 data half + the autonomous loop in paper mode

Owner: `principal-engineer`. The remaining half of the Phase-2 gate (CMC reads) **proven**, and
the autonomous loop built end-to-end in paper mode. Tests 325 → **343** (full suite green).

**CMC reads proven (step 3).** Live `quotes/latest` works on the project key: **all 147 ASCII
eligible symbols resolve a USD price in one batched call at 1 credit**, 150k monthly credit
budget (≈720 hourly calls/month — vast headroom). Tier is generous, no rate gate hit. New client
`trader.data.cmc_market` (`fetch_quotes → {SYMBOL: Quote}`, stdlib urllib, no new deps).
- **Bug found + fixed:** the symbol→records collision (`"BNB"` returns 4 records — real BNB rank-4
  *plus* BNB AI / BNBTiger / an inactive null-price BeanBox). A blind `recs[0]` silently dropped
  BNB's price. `parse_quotes` now picks the **canonical** record (active, best `cmc_rank`, valid
  price) — mirrors `cmc.pick_canonical`. A symbol with no priced record is **omitted, never zeroed**
  (the loop reads absence as "no observation").
- **Agent Hub vs plain REST (recon):** same quote data via Pro REST (used), the CMC Go CLI (wraps
  REST), or the Agent Hub MCP (adds **x402** pay-per-call). Loop uses **plain Pro REST** — lowest
  dependency, no MCP runtime, key already covers the universe free. **x402 is recon-only** this
  engagement; it's the fallback only if a future need exceeds the credit tier. **Fallback if CMC
  ever tier-gates the loop:** the proven keyless GeckoTerminal/DexScreener route — a new `PriceFeed`,
  nothing in the loop changes. The loop needs A feed, not a vendor.

**The autonomous loop `src/trader/agent/` (paper mode, first-class).** read → decide → execute →
confirm → monitor, as a long-running process (`python -m trader.agent`):
- **`decide()` behind a clean interface** — `DecisionCore` protocol (`decide(Observation) ->
  list[Intent]`, pure: no I/O, no signing, no clock). Shipped stub `HoldCore` (always holds) so the
  loop runs *now*; the RL champion plugs in via the same two methods — `loop.py` unchanged. **No
  strategy is baked into the loop.**
- **Paper mode is a real simulator, not a mock** — every tick: real CMC read, `decide()`, fills via
  the **same AMM cost model the backtest uses** (`sim.broker.amm_cost_usd`: LP fee + constant-product
  impact + gas), equity/PnL/drawdown marked per tick (scoring mirror), rows persisted. Paper spend
  **debits the same `risk/` caps** the live run will, so a forward-run respects the real budget.
- **Live routes through `execute_trade` and nothing else** — wired but **double-gated**: `mode` must
  be the exact string `"live"` *and* env `AGENT_ALLOW_LIVE=1`; anything else fails closed to paper.
  The loop never signs anything itself. **No live trade executed this engagement.**
- **Crash-safe** — on construction the loop re-derives its full portfolio (positions, peak, tick
  pointer) from `agent_ledger.jsonl` via `store.derive_state`; nothing lives only in memory. A
  malformed ledger **refuses to start** (fail closed). Verified: a fresh `Loop` recovers identical
  state to a pre-crash snapshot.
- **Heartbeat** row every tick (dead-man input for `/apentic/trading`). Clean SIGINT/SIGTERM
  shutdown finishes the current tick then returns.
- **Tests (18 new):** deterministic tick (fake feed + stub core), paper-fill accounting vs the cost
  model, cap-debit, out-of-policy refusal (not obeyed), restart recovery, malformed-ledger refusal,
  live-mode-requires-flag (config + `__main__`), live routes-only-through-`execute_trade`, dust mark.

**Provisional / next:** (a) the `trading/` publisher (EC2 → `data.alexlouis.dev`) is **not** built —
shapes drafted in [[Apentic Data Contract]] §trading/ but the put-only role + serializer is
`onchain-custody-engineer`/infra + a follow-up here. (b) Paper-fill **liquidity is a conservative
$250k default** (the read step has no per-token pool-depth feed yet) — wire DexScreener `liq_usd` per
held token for honest impact when the strategy trades thin tokens. (c) **BNB SDK runtime probe (step
4) skipped — still OPEN** (optional this engagement). (d) Equity valuation uses CMC prices; the
TWAK-portfolio-vs-CMC pricing authority question ([[Real-time Monitoring]] open Qs) stays open until
live.

## Phase status (vs [[Project Overview]] build path)

- ✅ **Phase 1** — Foundation.
- ✅ **Phase 2** — TWAK spike (custody half) + **data half DONE**: guardrailed dust trade confirmed
  on BSC (tx `0x739bb1…7c96`), unification proven, registration recon done ([[TWAK Spike Runbook]]);
  **CMC live reads proven** (full eligible universe, 1 credit/tick) and the **autonomous loop built
  in paper mode** (read→decide→execute→confirm→monitor, crash-safe, live double-gated). **Only open
  item:** BNB SDK runtime probe (step 4, optional).
- 🔄 **Phase 3/4** — Decision logic + offline validation: the rung-1b rd substrate is built and
  structurally DQ-safe (caps, loss floor, blacklist, no-ratchet — worst seed 26.9% DD with zero
  reward brake); honest per-regime gates in code; **RL tuning is the active work** (RecurrentPPO
  rdL sweeping; best learned config rd8h0c1 val +4.7%, rung-0 still the bar).
- ✅ **Phase 4A** — the MCP RL experiment loop tier (probe → guarded launch → poll → per-regime
  verdict → forensics → ledger), tool-complete; the loop driver is the remaining piece.
- 🔄 **June 16 PoC gate** — the on-chain half is **met** (real guarded trade landed via TWAK);
  the autonomous loop (read→decide→sign→confirm, continuous) is the active build, with AWS
  deployment + paper forward-run to follow (see the plan-forward entry above).

## 2026-06-12 — pool-event data layer: the parked instrument gets built

- **`src/trader/chain/` built** (fully isolated, read-only — see [[Pool-Event Data Layer]] for the
  data contract and the isolation contract): `rpc` (multi-endpoint failover — `bsc.therpc.io` is
  the one free endpoint serving deep historical `eth_getLogs`; publicnode prunes to ~1 day;
  dataseed refuses getLogs entirely), `events` (V2 Sync/Swap/Mint/Burn + V3 Swap/Mint/Burn/Collect
  decoders, one pool-perspective sign convention; Pancake V3's Swap topic0 ≠ Uniswap's — confirmed
  on Q), `registry` (16 V3 + 4 V2 pools, probed decimals — XAUt 6, BabyDoge 9, HUMA 6), `collector`
  (downloader-pattern manifest resume, adaptive span, truncation-split), `panels` (hourly frames
  aligned to the returns index). 11 new tests (decoders + aggregation).
- **Why:** retroactively unblocks the data-gated liquidity/flow knowledge direction — the backfill
  covers the SAME Nov-2025→Jun-2026 window every prior probe ran on. Three parked ideas converge on
  this one instrument ([[Trading Strategies]] PARKED + addendums).
- **Backfill launched** (~35.9M blocks, ~20-25M logs, ~10-14h laptop, resumable). Three
  pre-registered probes ship with it: LP-pull→detonation lead (graded on DRAWDOWN), flow-imbalance→
  reversion (≥24h horizons), wallet-cohort lead. Findings -> [[Experiment Log]] when run.

## 2026-06-11 → 06-12 — the autonomous loop runs the lab; the knowledge era opens

- **Loop driver built + armed** (`trader.experiment.driver`, `scripts/rl_loop.py`, `/rl-loop`
  skill, 30-min cron): launch->poll->verdict->record->decide, judgment left to the driving agent.
  Hardened live: smoke-parser fix, fire-and-verify launches (tailscale session-tree wait), setsid
  (retires the kill-PGID hazard), partial-sweep death detection (the WSL-close crash).
- **Six autonomous iterations** -> rdLe4 family champion (val +13.6/test +14.7/crash +13.2,
  worst-DD 10.5%) -> drift-alarm self-halt with the neighborhood mapped. -> [[Experiment Log]].
- **Knowledge era:** trade post-mortem grader + quant-consult rubric; five theories probed
  (1 validated -> `cycle_obs`/rdLc sweeping; 3 refuted pre-compute; 1 data-gated/parked).
  Probes: `probe_knowledge.py`, `probe_personality.py`. -> [[AI Training]] as-built.

## 2026-06-13/14 — checkpoint reproduction, the simulator, and the cross-timeframe diagnostic

A direction shift (user-set): a training run is not the deliverable — a **reproducible checkpoint**
is, to seed **curriculum / warm-start** training. Curriculum + checkpoint warm-start are now
first-class levers; the prior reflex against them was a category error (they govern OPTIMIZATION; the
honest gate governs EVALUATION — orthogonal). Standings/detail → [[Experiment Log]], [[AI Training]].

### Checkpoints are reproducible — bit-identically
- The **pre-2026-06-12 total-loss gap is closed** (`e681c4d`: `model.save(policy.zip)` +
  `venv.save(vecnormalize.pkl)` after `learn()`). Every run from that sha on persists a reloadable policy.
- **s0 reproduced to all 17 decimals, THREE times** (original, a c07bda0 re-run, the save-enabled
  68b268f capture): val 0.35299690480869833 / maxDD 0.07005478355079246. The rdLe4 config's training
  is **fully deterministic** on the box (CPU PPO + fixed seeds) — re-running a seed recaptures its exact policy.
- **Workflow (via the `scripts/rl_loop.py` CLI, fresh process):** `reset` → `propose --config <json>
  --seeds N --sha <save-enabled sha>` → `step`. c07bda0 PREDATES the save, so to capture a pre-e681c4d
  run you re-run its seed at the minimal **training-identical** save-enabled sha (here 68b268f:
  n_epochs/target_kl/cycle_obs defaults match, universe_lookback=0→warmup, the save is post-`learn()`
  so the trajectory is unchanged — proven by byte-identical smoke).
- **Captured checkpoint:** `runs-rl/ppo-event-rdLe4r-68b268f-s0/{policy.zip 7.27 MB, vecnormalize.pkl}`
  on the box AND pushed to `s3://alexlouis-apentic-data/ppo-event-rdLe4r-68b268f-s0/` (durable; runs-rl is local/gitignored).

### `scripts/simulate.py` — replay a checkpoint over arbitrary windows (commit `bcb4750`/`ebfebc1`)
- Loads policy.zip + vecnormalize.pkl from disk (CPU), reads the checkpoint's OWN provenance to
  rebuild the EXACT trained env config, and grades each trailing window through the trainer's own
  `evaluate_and_gate` — so sim numbers == a training-eval over that window. Per timeframe: trailing
  N+warmup bars (tradeable from bar 0), **evolving** voltopk universe (re-ranked at each window start),
  **in/OOS-labeled** (overlap with the train split), one `kind:"portfolio"`, `simulation:true` bundle
  per timeframe with full per-token OHLCV candles + markers. Contract → [[Apentic Data Contract]]
  §Simulation run; mechanism → [[Simulated Market]].
- **Serving decision: precompute-to-CDN for presets, NOT an on-demand EC2 API.** The stack is already
  static-JSON-on-CDN and the frontend already renders portfolio bundles, so a model+timeframe selector
  over precomputed bundles is zero new infra and no security surface. On-demand (arbitrary ranges) is a
  v2 — and must NOT live on the custody/trading EC2 (it signs txs; no public surface there).
- Timeframes 6mo/3mo/1mo/1wk/1d (NOT 1yr — only ~5123 bars/~7mo of hourly data exist).

### Cross-timeframe diagnostic on s0 (the curriculum input → [[Experiment Log]] for the table)
Outside its memorized val pocket, s0 is a **defensive underperformer**: (a) fails to ride bull upside —
its discretion DESTROYS value vs holding the same risk-parity basket (6mo −1.2% vs B&H +127%; 3mo +20%
vs +151%); (b) loses to its OWN rung-0 rule OOS in every window; (c) bleeds/churns in chop (1wk −8.7%,
18 trades). One virtue: bear capital preservation (1mo +0.7% vs B&H −20%). Episodes were 336-bar (2wk)
→ never learned long-horizon holding. **Curriculum targets (a)/(b)/(c)**; design waits on the user.

### Operational (→ [[Remote Capabilities]])
- **The in-session `trader` MCP server's SSH goes STALE after the desktop reboots** — every
  `mcp__trader__rl_*` then fails `subprocess.TimeoutExpired` ("could not reach desktop") while direct
  PowerShell ssh works; `health` still says ok (it does no ssh). **Fix: drive via the `rl_loop.py` CLI**
  (fresh process per call, immune) or reconnect via `/mcp`. Don't re-diagnose the MCP ssh path.
- **sha propagation:** the box's git origin is the stale P: mirror with NO non-interactive GitHub auth —
  only commits up to `origin/main` reach the box; newer unpushed local commits don't. Propose only an
  on-box sha; `scp` a new script to the box auth-free.

## 2026-06-14 (cont.) — weekly competition simulator + the train/deploy reckoning (FORK)

- **`scripts/simulate_weekly.py`** — the Apentic "Simulated Trades" dashboard export (design in
  `.design-export-simulated/HANDOFF.md`): **Mon-00:00-UTC weekly sessions, fresh $10k (no compounding),
  per-week causal vol-top-8**, `{meta, weeks[]}` JSON published **per-model**
  (`<run-id>/simulated_trades.json` + a `simulated_models.json` selector index). [[Apentic Data Contract]] §weekly.
- **The LEDGER pattern (recon fix).** The dashboard derives PnL itself from `qty*(exit-entry)`, but the
  env is notional/_px-index based and discards real per-coin prices, so reconstructing exact round-trips
  from markers was whack-a-mole (intrabar stops, total-loss-to-0, an eq-trace off-by-one: $29k→$420→…,
  never exact). SOLUTION: `EventRungEnv.token_pnls()` = exact per-token realized+open PnL; `fold_positions`
  builds round-trips for structure then **snaps the last position's exit so the token total == ledger**.
  **Recon $0.00 / 28 weeks** (commits `8158651`…`d81f301`). Also fixed: eq_trace records the final bar;
  every sell marker recorded (dropped sub-$1 dust suppression). Gotcha: box clock-skew serves a stale
  `.pyc` — clear `__pycache__` / `PYTHONDONTWRITEBYTECODE=1` when a box code change "doesn't take."
- **The diagnostic outcome → FORK.** The weekly sim showed s0's continuous-eval stardom (ZEC +$2,747)
  does NOT survive cold weekly sessions: ZEC is tradable in 17/28 weeks but trades in 2, and **skipped its
  Apr 6–12 big-move ignition by deliberate policy choice** (action idx 2 = 0× = skip). Cause = flattering
  continuous eval + the known overfit, NOT a bug (recon exact, ignition fires, cash free). **Decision:
  return to the training loop and train+evaluate in the deployment (weekly/cold) structure to the honest
  gate.** Full reckoning → [[Experiment Log]] §2026-06-14, [[AI Training]] §the-fork.

## 2026-06-14 (cont.) — cold-weekly deployment gate + the long-default basket overlay

The fork made concrete in code: train AND grade in the deployment structure, and widen the
substrate so the agent CAN hold the basket (the +13% bull-gap the event-only skeleton bled).
**This overlay direction was subsequently SHELVED on 06-15 (benchmark-driven drift — see below);
recorded here for the engineering trail.**

- **`src/trader/train/weekly_eval.py`** (`c83312b`) — a torch-free **cold-weekly grader**
  (Mon-00:00-UTC weeks, fresh $10k, per-week causal vol-top-8) + a **random-week distribution
  gate** (PAIRED bootstrap policy-vs-baseline, CI-low > 0; activity informational).
  `scripts/eval_weekly_baselines.py` measures the deployment bar (rung-0 trails B&H by the +13.2%
  bull-gap OOS, DQ'd on the ≥1-trade/day rule most weeks).
- **`EventRungEnv.basket_default`** (`c83312b`) — reset buys the risk-parity basket (= B&H), the
  exit/profit tables invert (idx0 = hold), benchmark becomes the held basket, so do-nothing == B&H
  and only correct tilts score. + `no_btc_obs` (the universe is BTC-decorrelated by selection).
  Both flags default OFF → byte-identical. `train_event.py` gains `--eval-mode weekly`,
  `--basket-default`, `--no-btc-obs`; policy return measured from the $10k deposit (not post-cost
  `eq[0]`) so hold == B&H. `launch.REWARD_KEYS` registers the knobs. Validated: hold-overlay == B&H
  to 5 decimals; 380 tests green.
- **`d819025`** — emit the **opening basket-buy markers**: `evaluate_event_policy` only collected
  trades from `step()` (which clears its buffer each call), so the `reset()` basket buy was dropped
  and the dashboard showed every token selling at the start with no preceding buy (HUMA: 10 sells,
  0 buys, yet held from step 0). Emit the reset buy as the first record (overlay-only; off =
  byte-identical). + **`--reexport`**: regenerate + republish a bundle from the SAVED `policy.zip`
  (no retrain), reading config from saved provenance so the eval is byte-identical and only the
  artifacts change — used to repair the already-published OVERLAY-1/2 bundles.

Numbers/standings for the overlay runs → [[Experiment Log]]; the overlay's defensive-basin
failure and its shelving → [[AI Training]].

## 2026-06-15 — horizon curriculum, the B&H demotion (weekly_gate only), drift post-mortem

- **Horizon curriculum** (`e926e2e`) — OVERLAY-1 learned a defensive trim-everywhere basin (gives
  back the bull, per-week gap vs B&H −27..−31%); root cause = a 1-week episode credits trimming a
  dip but truncates the cost (the missed multi-week run). Fix = train on LONG episodes first
  (holding the bull is creditable), anneal to the 1wk deploy shape. Built: `probe_horizon_credit.py`
  (torch-free, no training — PROVES the lever: fwd-return holding from a weakness bar triples in
  bull windows, 10%→30% over 1wk→4wk); `EventRungEnv.set_episode_bars` (shrink-only safe; env built
  at the largest horizon) + gym delegate; `trader.train.curriculum.parse_horizon_schedule`/
  `horizon_at` (torch-free); `HorizonCurriculumCallback` + `--curriculum-horizon` flag + provenance;
  `launch.REWARD_KEYS` knob. **Two ANTI-COSMETIC tests** (the curriculum-was-cosmetic lesson — the sampler
  provably MOVES on `set_episode_bars`; the schedule drives each phase once, descending). Flags
  default OFF (byte-identical); 387 tests green.
- **B&H demoted to a reported reference — in `weekly_gate` ONLY** (`6eda1d5`) — the **DIRECTION
  RESET** (user). The "beat B&H" gate drove the substrate to the buy-everything overlay (it
  structurally rewards holding everything), which abandoned selective-ignition entry and made the
  trap guards (`det_blacklist`) inert (all 4 overlay-curh seeds bought the Q trap, ~−$1k each).
  `weekly_gate` now binds on `survives_dq + beats_rung0` (the SELECTIVE rule is the bar);
  `beats_buyhold` becomes a reported `edge_vs_buyhold`, binding only under `require_buyhold=True`.
  **Note: this fixed only `weekly_gate` — every other gate site still required beat-B&H** (finished
  the next day, `503b784`). The overlay is shelved; return to the selective rd/rdL ignition
  substrate, keep the cold-weekly eval structure.
- **Drift post-mortem + current-direction banner** (`0f7b965`) — captured in [[AI Training]] how the
  session drifted (the benchmark drove us off the thesis), added a top CURRENT-DIRECTION banner
  (selective rd/rdL, beat-the-rule bar, overlay shelved) and the durable lesson: don't let a
  benchmark define the agent; when the metric drives you off the thesis, the metric is wrong.
  → [[AI Training]], [[Experiment Log]] §2026-06-15 DIRECTION RESET.

## 2026-06-16 — universe-regime curriculum, the gym-passthrough catch, and the honest-gate contract refactor

Engineering for this session (numbers/verdicts: [[Experiment Log]] §2026-06-16, [[AI Training]]
§"As-built (2026-06-16)" — do not re-derive them here). Three commits, in build order:

- **Universe-regime curriculum BUILT** (`789979f`) — the **volatility-axis analog** of the horizon
  curriculum (`e926e2e`). Stages the TRAINING universe through regimes over training progress —
  `lowvol` (the k calmest tokens: learn ignition sizing/exit basics on tractable dynamics) →
  `broad` (vol-stratified) → `voltopk` (the k most volatile = the deploy/eval distribution) — then
  anneals into the deploy shape, mirroring the horizon ramp landing on the 1wk deploy episode.
  Built: `trader.train.curriculum.parse_universe_schedule`/`universe_at` + `UNIVERSE_MODES`
  (torch-free); `EventRungEnv.set_universe_mode` (a **between-episode setter** — no `_max_start`
  constraint, categorical, unlike the shrink-only horizon setter); `UniverseCurriculumCallback`
  (`env_method` push) + `--curriculum-universe` flag + provenance in `scripts/train_event.py`;
  `curriculum_universe` in `launch.REWARD_KEYS` (so `/rl-loop` can drive it); anti-cosmetic tests
  (lowvol vs voltopk pick DISJOINT universes; the callback drives each regime once, in order).
  **Invariants:** default `""` = OFF / byte-identical; **EVAL always runs the deploy
  `--universe-mode`** — only the TRAINING envs' START regime is staged, via a `make_env` dict-merge
  override (`env_kwargs.universe_mode` is untouched), so the cold-weekly honest gate stays
  meaningful; the schedule is validated to END at `--universe-mode`. Adversarially reviewed (2
  lenses clean); 394 tests green.
- **Gym-passthrough fix** (`7458aa8`) — the curriculum callback's
  `env_method("set_universe_mode", …)` resolves on the `GymEventRungEnv` wrapper, not the core
  `EventRungEnv`; `gym.Env` does NOT forward unknown attrs to `self.core`, so the call raised
  `AttributeError` inside each `SubprocVecEnv` worker and killed it (parent saw **EOFError**),
  crashing the curriculum sweep's 100k smoke **in setup**. The horizon curriculum worked only
  because `set_episode_bars` already had a passthrough; `set_universe_mode` was missing its twin.
  **The guard held: the smoke caught it before any real launch and the loop HALTED — nothing bad
  ran.** Fix: add the passthrough on `GymEventRungEnv`; + a vec-env regression test (`DummyVecEnv`
  `env_method` reproducing the exact callback path) the adversarial review had flagged as missing
  coverage.
- **Honest-gate contract refactor** (`503b784`) — FINISHED the 06-15 B&H demotion **across ALL gate
  sites** (`6eda1d5` had fixed only `weekly_gate`; every other site still required beat-B&H, so the
  loop mislabeled a rule-beating, DQ-surviving control as FAIL and steered toward PnL-vs-B&H — the
  exact benchmark trap the reset killed). Demoted **Buy&Hold + Random to REPORTED references**
  (still computed + surfaced, never binding) at: `train_event.honest_gate` (beats = {rung-0}, DQ-
  first); `diagnostics.compare_seeds` + `regime_verdict` (checks = {drawdown, rung-0}; B&H/Random in
  bars/fields); `champion._honest_gate` (pass = test-split + survive-DQ + beat-rung-0); `loop_control`
  (north-star metric `margin_vs_buyhold → margin_vs_rung0` — the **rung-0 RULE edge**; drift alarm
  now fires on no edge-vs-rung-0 improvement; `margin_vs_buyhold` retained as a None-safe reported
  field, driver persists both); the `server.rl_diagnose` note; `contract.SUCCESS_METRIC`; and the
  vault **[[Agent Communication Contract]]**. **Champion-contract consequence:** a rule-beating +
  DQ-surviving config that LOSES to B&H can now be champion (the selective thesis); one that loses
  to the rule binds "rung-0". ~20 pinned tests rewritten to the corrected contract; adversarially
  reviewed (completeness/anti-weakening + semantics/backward-compat, both clean); **398 passed, 1
  skipped.**

**Operational notes (→ [[Remote Capabilities]], [[MCP Server]]):**
- **The in-session `trader` MCP server ran STALE code** (loaded pre-fix) until the user restarted
  it — the in-session-MCP-stale lesson recurs. The loop was driven via the **CLI**
  (`scripts/rl_loop.py`, fresh process per call) rather than the MCP read tools, which lag a code
  change until the server restarts.
- **The `entry_forward` reward-shaping sweep (`ppo-event-rdLe4-ef`) was LAUNCHED and is TRAINING**
  (in progress — NO verdict yet; do not read it as a result). Direction + design → [[Experiment Log]]
  §2026-06-16, [[AI Training]] §"As-built (2026-06-16)".

## 2026-06-17/18 — ef-s2 goes live: the paper forward-run harness on EC2 (→ [[Live Forward-Run Harness]])

Deployed the trained RL champion **ef-s2** (`ppo-event-rdLe4-ef-503b784-s2`) to the EC2 host for a
live **paper** forward-run on BSC, ahead of the June 22 window. Branch `feat/live-event-harness`
(unmerged; the box runs the branch). Full subsystem doc → **[[Live Forward-Run Harness]]**.

- **Architecture — reuse the validated loop, don't reimplement.** Each hour re-run
  `train_event.evaluate_event_policy` + `EventRungEnv` VERBATIM over the current **cold-week** window
  (Mon 00:00 UTC open, 168-bar warmup prepad, fresh **$10k**, vol-top-8 reselected at the open, LSTM
  reset — the `simulate_weekly` cadence), swapping only recorded panel → **live rolling panel**, and
  diff the fills (newest-bar fill = this hour's decision). Obs-parity becomes a data check, not a
  reimplementation. **$10k cold-weekly is mandatory** (the env prices fills on its internal index;
  another capital changes the AMM-cost fraction → fill skew). Modules: `event_live` (cold-week window
  + fill-diff + `LiveEventTrader`), `live_data` (hourly updater reusing the exact producers; invariants
  finalization + append-immutability), `event_runner` (env fills → hard `trader.risk` guardrails +
  ledger + telemetry; the env IS the paper book), `event_agent` (hourly loop, **PAPER-ONLY** — live
  refuses, no TWAK signing path yet). **Offline obs-parity gate** (`tests/test_live_data.py`): replayed
  bars match recorded `r_alt`+volume EXACTLY (rtol 1e-9). ~20 new tests; suite green.
- **Live feed = GeckoTerminal BSC pools (parity-locked), NOT CMC.** ef-s2 trained on GeckoTerminal
  pool candles; CMC is CEX-aggregated spot → would push the frozen model out-of-distribution (and the
  existing `cmc_market` has no 1h history). Decided with the user. Anchor stays ccxt/Binance.US.
- **Private model store** (weights are core IP, must not be public): `s3://alexlouis-act-private/models/`
  (Block Public Access + SSE, no CDN), EC2 role scoped `s3:GetObject`; provenance `metrics.json` pulled
  from the public bundle. `deploy/private-model-store.md` + `deploy/iam/private-models-*.json`.
- **Deployed LIVE** (`trader-event-agent.service`, replaces the disabled HoldCore unit). torch
  2.12.1+cpu + sb3-contrib + ccxt on the box; on-box dry-run gate passed; first live tick
  2026-06-17T23:49Z, publishing to `data.alexlouis.dev/trading`.
- **429 degenerate-universe bug (user-caught, FIXED `cdbcf03`).** The user spotted XRP/LINK trades:
  `fetch_alt_latest` had no 429 retry, so GeckoTerminal's rate limit silently starved ~12 of 20 tokens
  each tick (WARN hidden on block-buffered stdout) → the vol-top-8 fell back to the low-vol majors with
  data, not the volatile microcaps. Fix: exp-backoff 429 retry + WARN→stderr + 3s pacing. Verified true
  vol-top-8 = SIREN/COAI/SKYAI/UB/BANANAS31/B/ZEC/HUMA (XRP #15, LINK #16 excluded); clean run trades
  only true-top-8 (first fill HUMA). 19/20 tokens fresh live; **XAUt** pool is genuinely inactive
  (perma-stale, never selected). Lessons: log to stderr under systemd; any hourly external fetch needs
  429 backoff; a degenerate universe is silent (`uni=8` is just k=8) — inspect the per-token vol ranking
  (`deploy/inspect_universe.py`). This makes the [[Project Overview]] "thin BSC liquidity" risk concrete.
- **Daily market-volatility scan automated on EC2** (`trader.agent.daily_scan`, systemd
  `trader-daily-scan.timer` @ 00:10 UTC). Refreshes the top-level `market_metrics.json` (vol/corr
  dashboard, via the existing `compute_market_metrics`) over current data AND appends a **`selected`**
  block = the model's ACTUAL current vol-top-8, read from the SAME env the harness trades
  (`eval_universe_and_caps`). ef-s2 selects WEEKLY, so `selected` changes weekly while metrics refresh
  daily — surfaces the real traded set transparently, **no model change** (the daily-informational
  decision; a daily re-pick would be OOD for the frozen model). Torch-free selection; laptop-tested;
  verified on the box (selected = SIREN/COAI/SKYAI/UB/BANANAS31/B/ZEC/HUMA, wk 2026-06-15). Top-level
  publish needs a scoped `market_metrics.json` PutObject grant (the box role is otherwise `trading/*`-only)
  — `deploy/iam/market-metrics-put-policy.json`. → [[Apentic Data Contract]] §market_metrics.json.
- **Earlier this session (host stand-up, ~06-12):** EC2 host phases A–F completed (provision → harden →
  key ceremony → systemd paper) and the **`trading/` telemetry publisher built** (`trader.agent.publish`,
  put-only role) — see [[EC2 Trading Host Runbook]] (incl. the as-found corrections: TWAK never shows the
  mnemonic; the headless keychain copy is deleted; the SIGTERM clean-stop fix) and [[Apentic Data Contract]]
  §trading/.

## 2026-06-19 — fixed-universe mode, vol_mult provenance, rung-0 demotion, sideways EMA-break suppression

Four commits (build order; numbers/verdicts → [[Experiment Log]] §2026-06-19, [[AI Training]], the
fixed-13 closed branch and the EMA-break investigation — do not re-derive them here). Started from the
"+51% val vs 25% over 6mo" discrepancy: RESOLVED as two graders of different things — "+51%" was the
CONTINUOUS eval (one long episode, universe picked ONCE at val open, returns compound) vs "25%" the
cold-weekly sim (causal vol-top-k RE-PICKED every Monday, fresh $10k per week, summed). The cold-weekly
sim is the deployment-honest grader; the discrepancy traced to FF's Apr 9-10 +100%-plus roundtrip being
in the once-picked continuous basket but NOT in the Apr-6 week's causal top-10 (its trailing vol had
decayed; it re-entered only the following week, after the pump).

- **`abf089b` — sideways EMA-break suppression + rung-0 `exit_ema_span` + the EMA-break probes.** The
  leak: the 72-bar-EMA weakness-exit fires on shallow noise-dips during tight sideways consolidation,
  shaking the agent out before a pump (the FF Apr-9 case: a −0.1% cushion NOISE break, then the 48h
  cooldown locked re-entry through FF's real Apr-10 18:00 rip). Fix: when a break is SHALLOW (cushion >
  −`shallow_break_max`) AND the token is QUIET (24h realized vol < `consol_vol_max`), do NOT fire the
  EMA-break; the loss_floor (−20%) and trailing stop stay fully active, so real breakdowns still cut
  (bounded downside) while the position survives noise to catch the pump (asymmetric: bounded downside
  via the floor, large upside via pump capture). The user chose the "shallow + quiet" definition (most
  surgical — a deep break or a high-vol break still cuts). Both knobs `0` ⇒ OFF, byte-identical (30 env
  tests pass). Wired through `launch.REWARD_KEYS` (`shallow_break_max`, `consol_vol_max`) + `train_event`
  flags + provenance + the `simulate` loader, so it is sweepable via `/rl-loop` and graded honestly.
  Also in this commit: `rung0` gains a separate `exit_ema_span` (additive/default-off) and the EMA-break
  investigation probes. **STATUS: committed, NOT yet retrained** — planned next experiment is to retrain
  ef2 + `shallow_break_max=0.02` + `consol_vol_max=0.015` (the FF-validated thresholds), 4 seeds, graded
  honest cold-weekly vs ef2 (pending the user's green-light; the desktop is shared). Mechanical check on
  the bare rule (FF fixed-13 week): OFF sells FF twice (Apr 7 + Apr 9 EMA_BREAK); ON suppresses both and
  holds FF through the chop. Open co-factor: ROTATION_OUT can still swap a held-but-flat token out before
  its pump (a second shakeout mechanism) — the next thread if suppression helps but rotation caps it.
- **`6db0674` — `simulate_weekly` dataless-asset guard + a `--vol-mult` override + `delist_sim_model.py`.**
  Driven by a FRONTEND CRASH: the eff-s1 fixed-13 `simulated_trades.json` carried 11 empty-candle assets
  (the FIXED universe forced not-yet-listed tokens — ASTER/HUMA/SIREN/ZEC in early weeks — into the
  basket with no OHLCV), so `candles` was `[]` and the simulations frontend crashed in `computeBacktest`
  at `backtest.ts:261` (`const t0 = candles[0].t` → undefined). Three fixes: (1) PRODUCER GUARD —
  `simulate_weekly` now skips any asset with empty candles (a dataless token has no trades and 0 PnL; the
  per-week recon still balances); (2) `scripts/delist_sim_model.py` rewrites `simulated_models.json`
  without a given run-id + invalidates CloudFront (the S3 publisher can PUT but not byte-delete, so this
  is a DE-LIST — bytes remain, just unlisted; → [[Apentic Data Contract]], the no-byte-delete publisher
  fact); (3) eff-s1 was re-published clean. The `--vol-mult` override lets older runs be re-graded at
  their correct vol_mult (see `2345fd6`). Page healed.
- **`8009973` — contract DIRECTION RESET: rung-0 demoted to a reference floor (docs).** rung-0 is
  demoted from the BINDING gate to a REFERENCE floor (the same demotion Buy&Hold got 2026-06-15). The
  corrected gate: a config earns a version iff it (1) SURVIVES the DQ gate (worst single-week maxDD <
  ~30%, still HARD) AND (2) IMPROVES on the previous best iteration (the champion) on the honest
  cold-weekly metric. rung-0, Buy&Hold and Random are all computed/reported references, none binding;
  the loop north-star becomes margin-vs-prior-champion and the drift alarm fires on
  no-improvement-over-champion. Deploy on the best single seed; select configs on the seed-mean. This
  commit wrote the reset into the [[Agent Communication Contract]]; **CODE TODO (not yet shipped):**
  `weekly_gate` / `honest_gate` / `loop_control.decide` / `rl_north_star` still enforce `beats_rung0`.
- **`2345fd6` — fixed-universe mode + record vol_mult in provenance.** `EventRungEnv` gains
  `universe_mode="fixed"` + a `fixed_universe` token list (NO causal weekly re-pick; `k` auto-syncs to the
  list length; obs width unchanged). `train_event` gains `--universe-mode fixed` + `--fixed-universe` and
  now RECORDS `vol_mult` + `fixed_universe` in provenance; `REWARD_KEYS` gains both;
  `simulate.env_kwargs_from_provenance` now reads `vol_mult` (was defaulting to 2.5) and `fixed_universe`.
  This closed a real **vol_mult PROVENANCE BUG**: `train_event` never recorded `vol_mult`, so the loader
  defaulted to the constructor's 2.5 — but ef2 TRAINED at vol_mult 2.0, so every PUBLISHED ef2 sim ran
  the policy OFF-DISTRIBUTION at 2.5, depressing its numbers (ef2-s1 cold-weekly +4.9%/wk at 2.5 vs
  +6.0% at the correct 2.0). All 4 ef2 seeds were re-published at the correct 2.0. The fixed-universe
  mode was built to test a FIXED-13 universe (drop the 7 most-BTC-correlated/lowest-vol tokens so a
  high-vol spike like FF is never selected out mid-week) — that experiment is a CLOSED BRANCH (ef2 beats
  fixed-13 on both seed-mean and best-seed; causal vol-top-k stays the substrate), but the
  `universe_mode="fixed"` machinery remains as tooling. → [[Experiment Log]].

**Operational note (→ [[Remote Capabilities]], [[MCP Server]]):** the PowerShell SSH TOOL was DEAD this
session; the working route to the training desktop was the **Windows OpenSSH binary**
(`C:\Windows\System32\OpenSSH\ssh.exe` / `scp.exe`) invoked from the **Bash tool** — the in-session MCP
`rl_loop_*` tools stay unreliable (drive sweeps via `scripts/rl_loop.py`, fresh process per call).
