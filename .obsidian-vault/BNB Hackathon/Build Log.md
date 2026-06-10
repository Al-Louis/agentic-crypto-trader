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

## Phase status (vs [[Project Overview]] build path)

- ✅ **Phase 1** — Foundation.
- 🔄 **Phase 3/4** — Decision logic + offline validation: data layer + universe done; honest
  broker, features, and backtest are the active work.
- ⬜ **Phase 2** — Stack spike / live on-chain loop: **deferred** under the training-first
  plan; a focused execution+custody spike (TWAK dust trade, registration dry-run) is still
  required before June 22.
- ⬜ **June 16 PoC gate** — reframed internally to "trained agent"; the live-loop gate itself
  is not yet met (no on-chain trade landed).
