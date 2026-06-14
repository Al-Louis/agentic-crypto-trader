# AI Training

The learned-policy candidate for the decision core — an RL pipeline ported from [[TradeSim]]
that trains a trading agent against the [[Simulated Market]], with reward and evaluation tied
to the competition's risk gate. **RL is one option, not a mandate** — weighed here against
simpler robust strategies ([[Trading Strategies]]) for a single, high-variance live week.
Owned by `rl-ml-trainer`. Regime/scenario context: [[Market Conditions]]; training-host
question: [[Remote Capabilities]].

## Where RL sits

The decision core is a **pure module behind a clean interface** ([[Trading Strategies]]). A
learned policy is one implementation of that interface; an SMA/RSI rule or a hand-tuned
heuristic is another. Everything downstream — the [[Simulated Market]] broker, the honest
baselines, [[Security and Encryption|execution and custody]] — is strategy-agnostic and does
not care whether the decision came from a neural net or an `if` statement. So RL can be
developed, evaluated, and **dropped** without touching the rest of the system. That
separation is the precondition for the candid "is RL worth it?" question below.

This is 🟡 SIMULATE-tier work in the [[MCP Server]], shipping in **Phase 4**. It is strictly
**offline and keyless** — training never touches a wallet or the chain. It does **not**
satisfy the June 16 PoC gate, which needs a real on-chain trade ([[Tech Stack]]).

## Post-mortem: hard lessons from TradeSim (carry these, not the optimism)

The prior project shipped a `baseline_handoff.md` (`tradesim_handoff_seed/`) distilling ~40
iterations and 64 runs. It is the most valuable thing in the seed, and it **corrects** some of
the optimism elsewhere in this note. Non-negotiable takeaways:

- **Entry timing never clearly beat random; *exits / risk-management* carried performance** —
  best honest outcome was **bull-regime breakeven**. → Treat **entry alpha as an open research
  question** (the [[Trading Strategies]] edge thesis), weight effort toward exit/risk logic and
  the survival overlay, and make a **baseline (Buy&Hold / cross-sectional momentum) behind an
  honest gate** the first validated thing. Our *cross-sectional selection* claim differs from
  single-asset entry timing — but the skepticism stands.
- **The curriculum was cosmetic.** `CurriculumCallback` only logged phase names; it never
  changed the episode sampler, so the phases were never applied. → Build curriculum as a
  **real, data-driven sampler with a test asserting the sampled distribution shifts per phase**.
  Start from the lever that worked — **regime** (bull → mixed → bear) — and add
  volatility-bucketed / walk-forward phases.
- **Fee-blind reward.** Fees *in the reward* taught the agent to **not trade** at all. → Track
  fees for PnL reporting only; keep them out of the reward.
- **`Discrete(3)` (Hold/Buy/Sell) beat continuous allocation** decisively; the continuous env
  is legacy — consolidate to one discrete env.
- **Reward was an 8-layer accretion — don't port it verbatim.** Rebuild from the clean intent
  (DSR + light per-step shaping), **portfolio-level + ruin-aware** for our −100% rug tail.
- **No gate ⇒ 64 redundant runs** (~95% minor variants of one config). → **Freeze a held-out
  test set; every model must beat Buy&Hold and Random to earn a version; 100K-step smoke test
  before any full run.**
- **Slippage must match the data** ([[Simulated Market]]): volume-based slippage on
  sparse/zero-volume candles produced fantasy fills — our sparse 1-min DEX data is the same
  trap (our fix is an **AMM price-impact** model, not their fixed-spread).
- **Converged config to start from:** PPO, lr **3e-5**, **ent_coef 0.2** (not 0.05 on volatile
  data — low ent_coef collapsed to "always wait"), ~**5M** steps (20M overfit), n_envs 8,
  **position cap 0.3**, min-hold lock, `GroupedIndicatorExtractor` (features_dim 128).

The breakthrough to **keep** is the architecture itself: *modular indicators → grouped
per-group MLP + attention → PPO, no hard-coded guardrails* — its first run was the project's
best result (Sharpe 0.64, 74% win, learned from data).

## The reusable training stack (from TradeSim)

The pipeline ports largely intact — the earned part is the reward and the evaluation
discipline, not the framework wiring.

| Layer | Choice | Notes |
|-------|--------|-------|
| Algorithms | **RecurrentPPO** (sb3-contrib, `MlpLstmPolicy`, LSTM-256), PPO, SAC | LSTM carries state across the lookback window; PPO/SAC as comparators |
| Framework | Stable-Baselines3 + sb3-contrib, PyTorch | Mature, well-tested; we did not reinvent the optimizer |
| Parallelism | **SubprocVecEnv** | Many env copies in parallel for sample throughput |
| Tracking | TensorBoard + per-run dirs/checkpoints | ~64 runs / 2,134 checkpoints / 23 finalized models in TradeSim |
| Callbacks | trading-metrics, **curriculum**, early-stopping | See curriculum below |

**Curriculum** should ramp difficulty so the policy learns a stable core before facing chaos:
**low-vol → mixed → high-vol → full + noise**. Scenario definitions for each stage are owned
by [[Market Conditions]]. Early stopping halts runs that plateau or regress on the validation
metric, keeping the experiment budget on promising configs. **⚠ In TradeSim this callback was
cosmetic — it never changed the sampler (see post-mortem); it must be rebuilt as a real,
tested data-driven sampler, ideally regime-based.**

## Reward design — the earned part

Raw PnL is a poor RL signal: sparse, lucky, and trivially reward-hacked. TradeSim's reward
went through **30+ iterations** to a dense, risk-adjusted, hacking-resistant shape:

- **Incremental Differential Sharpe Ratio** (Moody & Saffell, 1998) with EMA tracking — a
  per-step risk-adjusted signal, not an end-of-episode lump. This is the spine.
- **Quadratic drawdown penalty** — penalty grows with the *square* of drawdown, so deep
  drawdowns hurt disproportionately.
- **Asymmetric loss weighting** — losses punished harder than equivalent gains are rewarded.
- **Per-trade fee penalty + holding cost** — discourages churn and idle exposure. **⚠ But
  TradeSim found fees *in the reward* made the agent stop trading entirely — keep fee
  accounting for PnL reporting, out of the reward (see post-mortem).**
- **Clipping** — bounds reward magnitude against exploit spikes.

The drawdown term maps **directly onto the competition's ~30% max-drawdown HARD DQ gate**: a
run that breaches the cap scores zero regardless of return, so training the policy to fear
drawdown is aligned with survival, not just return. The [[Simulated Market]] models that gate
as a disqualifier (Calmar as the headline number); the reward and the evaluation agree on
what "good" means. **A new reward shape is unverified until forward-validated** — a curve
that climbs may just mean the agent learned to game the reward.

## Observation / feature design

Observations are the causally-validated feature set from [[Simulated Market]]'s
`prepare_dataset` (~28 indicators, each passing the look-ahead test) over a lookback window.
Feature *selection* is coordinated with `market-indicator-expert` ([[Trading Strategies]]);
any new BSC/on-chain feature must clear the leakage guard before it enters an observation.

Two custom extractors carry over:

- **`GroupedIndicatorExtractor`** — each indicator group gets its own MLP head, combined via
  **multi-head attention**, so the policy *learns which signals to trust per regime* rather
  than relying on hard-coded weights. An earlier hard-coded-guardrail approach was
  deliberately removed in favor of this learned weighting.
- **1D-CNN** over the lookback window — local temporal patterns as an alternative front end.

## Honest evaluation + the diagnose loop

Training curves are **not** performance claims. A model is judged only by `evaluate_model` on
**held-out periods**, through the *same* [[Simulated Market]] broker, against the four
baselines — **Buy & Hold, SMA, RSI, Random** — with identical costs. Beating Random is the
floor; beating Buy & Hold on a risk-adjusted basis (Sharpe, Calmar) is the bar.

`diagnose_run` is a rule-based check encoding known RL-trading failure modes and returning
recommendations:

| Failure mode | What it flags |
|--------------|---------------|
| Under-performs Random | Policy has learned nothing useful |
| Over-/under-trading | Activity floor breach or fee-churn |
| Fee drag | Edge eaten by costs |
| Large drawdown | DQ-gate risk |
| Negative Sharpe | No risk-adjusted edge |

The heavy iteration trail (64 runs, 2,134 checkpoints, 23 finalized models) is the tell of
real research, not a single lucky run. Evaluation rigor is shared with `quant-analyst`
([[Simulated Market]]).

## Training orchestration

Driven by [[MCP Server]] tools so workflows run the loop deterministically:

`start_training` (launches a **background subprocess**, returns a run id) → `training_status`
(progress + live metrics) → `evaluate_model` (held-out vs baselines) → `diagnose_run`
(failure-mode check) → iterate. `list_models` / `model_info` enumerate and describe finalized
models. A `/workflows` script can drive *train → evaluate → diagnose → retrain* until a model
clears the bar or is abandoned. Host question (a CPU-core-bound, env-stepping workload — not
GPU) is deferred to [[Remote Capabilities]].

## As-built (2026-06-09) — the loop + the exposure-overlay env

The train → evaluate → diagnose **loop is built and proven end-to-end** on real hardware; the
RL **env + trainer are built**, pending a desktop smoke run. Deliberately **simpler than the
ported TradeSim design** — start from the validated baseline, beat it first, add complexity
only if it earns its way in.

**The loop (autonomy Level B)** — `trader.train`: `config` (RL-extensible dicts + stable key),
`registry` (JSON experiment store with config→run→result **lineage**), `diagnose` (gates
below), `loop.run_iteration` (dispatch → fetch the published bundle from `data.alexlouis.dev` →
diagnose → record), `scripts/train_loop.py`. MCP read tools: `list_experiments` / `experiment`
/ `diagnose_run`. **Gates** (the post-mortem's discipline, encoded): drawdown DQ, positive
Sharpe, fee drag, **beats-baseline** (vs the token buy&hold / vol-tilt), ≥1-trade/day.
"Improve" = beat the baseline **OOS**, not training reward.

**The env** — `trader.train.env.PortfolioEnv` (plain numpy/pandas, torch-free so it's testable
on the laptop; gymnasium adapter `gym_env.GymPortfolioEnv` for sb3):
- **Action C (exposure overlay):** exposure ∈ [0,1] → `exposure/k` on each vol-top8 token
  (universe picked causally from the warmup window). Starts from the validated vol-tilt it
  can't underperform by construction; widens to full weights (B) later, eval/baseline
  unchanged. *(A cross-sectional allocator — not TradeSim's single-asset `Discrete(3)`; the
  discrete-beats-continuous finding was for single-asset entry timing, a different problem.)*
- **Reward** (the earned shape): **differential (online) Sharpe** increment − **quadratic
  drawdown-proximity penalty** ramping to the ~30% DQ. **AMM cost is netted into equity, NOT
  in the reward** — exactly the post-mortem's fee-blind fix. Intra-step equity path for honest
  drawdown; next-bar execution; no look-ahead.
- **Obs (first cut, 6-dim):** BTC trend (vs EMA), BTC recent return, drawdown, current
  exposure, last-step return, realized vol. *Deferred:* the 28-indicator
  `GroupedIndicatorExtractor` — expand only if the policy plateaus for lack of signal.

**The trainer** — `scripts/train_rl.py` (DESKTOP-only): time-split train/val/frozen-test, PPO
**MlpPolicy** on **SubprocVecEnv + VecNormalize** (`n_envs ≈ cores`), eval on held-out val →
Apentic bundle → self-publish, `progress.json` throughout for fire-and-poll
(`remote_train.submit_background` / `poll`). *Deferred:* RecurrentPPO/LSTM + the grouped
extractor (the converged TradeSim config) until the simple MlpPolicy is shown to beat — or
clearly can't beat — the baseline.

**Regime reality (corrected 2026-06-10 — the "~6-month bull sample" claim was wrong).** Two
regime signals, and they **diverge**:

| split | BTC (macro) | vol-top-k 8 (the **traded** universe) | Buy&Hold net |
|-------|-------------|----------------------------------------|--------------|
| train | **−31.1% (bear)** | **+26.1% (bull)** | +25.6% |
| val   | +9.2% (reversal) | +7.2% (flat) | +6.8% |
| test  | **−22.5% (bear)** | +5.5% (flat) | +5.1% |

BTC is **macro-bear** across the data (matching the real timeline: bear since Oct 2025, an
Apr–May reversal = val, renewed downtrend since). But the high-vol **alts the agent trades
decouple from BTC** — they pump on their own volume dynamics (train: BTC −31% while the basket
*+26%*). Consequences that correct the earlier plan: **(a)** we already have real BTC-bear data,
so synthetic crash injection (`trader.sim.crash`) is for **alt-specific** crash stress — the alts
never crash in this sample — **not** the primary bear source the old note assumed; **(b)**
Buy&Hold is **positive in every split**, so the agent cannot win by hiding in cash even in
BTC-bear windows — the edge must be *harvesting* the alt volatility ([[embrace-volatility-dont-dismiss]]),
not avoiding it; **(c)** the obs needs a **universe-breadth** regime feature, not just `btc_trend`,
since the two diverge; **(d)** tuning on val (the bull-reversal pocket) is the *least*
representative window for a likely-bearish live week — weight the per-regime gate toward the
BTC-bear train/test-like windows.

**Curriculum status:** the env samples random windows from the training split — a *real* sampler,
not cosmetic (the post-mortem's #1 lesson). The frozen-test split is reserved; tuning happens on
validation to avoid the loop meta-overfitting (but see (d) — val is the unrepresentative pocket).

### Substrate redesign (2026-06-10) — discrete actions, universe knob, risk-parity caps

After exp1→exp5 (continuous-action proxy-reward drift), three structural changes to `EventRungEnv`,
each defaulting OFF so the prior behavior is unchanged (225 tests green):

1. **Discrete action space** (`action_mode="discrete"`, `n_action_levels=4` → size/keep ∈ {0,⅓,⅔,1}).
   The TradeSim "Discrete(3) beat continuous decisively" lesson, scoped correctly: the failure is a
   **Gaussian head over a `Box` dead-gradienting to the boundary** — observed *twice* here (exp1b
   collapsed to 0 trades; the residual corner-solution). A categorical head structurally cannot
   corner. Keeps the semi-MDP event timing (a fixed-clock rebuild was a documented dead-end);
   only *what* the agent does at each event is discretized. Gym adapter exposes `spaces.Discrete`.

2. **Universe-volatility knob** (`universe_mode`: `voltopk` | `broad` | `lowvol`) — the curriculum's
   VOLATILITY axis. `voltopk` (default) = the k most volatile (max chaos, current); `lowvol` = the
   calmest k (S0: learn basics on tractable dynamics); `broad` = vol-stratified spread. Motivated by
   the universe being **bimodal**: a few monsters (HUMA ~1310% ann vol, 8.2× median; SIREN/SKYAI
   +3000-3950% total peaks) vs a calm tail (XRP/ADA/LINK/gold) the agent *never sees* because
   `vol-top-k` selects only the monsters. One-shot 40× events have no learnable structure — closer
   to noise than signal — so basics must be learned on calmer data first.

3. **Risk-parity per-token caps** (`vol_target>0` → per-token weight cap ∝ `vol_target/trailing_vol`,
   clipped `[cap_floor, max_entry_frac]`). **The decisive finding:** the current top-8-vol universe
   is **DQ'd by construction** — equal-weight buy&hold of it has maxDD **−31.1%**, over the 30% gate,
   before the agent acts. The alts are **near-uncorrelated** (avg pairwise +0.13; the monsters +0.035
   — idiosyncratic pumps), so inverse-vol weighting across a broadened universe cuts ann vol 1.96→0.32
   and maxDD to **−24.2%** (under the gate). High-vol tokens stay present (floor) for convex upside but
   can't blow the gate; calm tokens anchor at the ceiling. A hard guardrail *and* a training constraint
   (train how we trade). Reframes the agent's job: rung-0 + caps define a survivable risk envelope; the
   agent allocates *within* it to harvest the idiosyncratic vol. Tests: `tests/test_discrete_riskparity.py`.

**GATE-1 outcome (2026-06-10):** both variants (voltopk concentrated, broad k=12 risk-parity) FAIL
the per-regime DQ gate — but structurally, not as a policy bug (full table in [[Experiment Log]]).
The finding: **no static risk posture wins both regimes** — risk-parity caps *helped* on val (the RL
beat a DQ'd rung-0 that blew 31% DD) and *hurt* on test (missed the monster-pump rung-0 caught at +29%).
The block is two structural gaps: (1) **no regime signal in the obs** (`btc_trend` misleads — alts
decouple from BTC) → add a **universe-breadth** feature; (2) **no alt-crash in the data** (every split
has the alts rising/flat) → **synthetic alt-crash injection**, so de-risking can pay. Active build:
the crash scenario + the breadth feature, then gate a **regime-adaptive** policy.

**GATE-2 outcome (2026-06-10):** both built (breadth obs OBS_DIM 12→13; `sim/crash.py` inject_crash +
the `gate2` config: broad k=12 + risk-parity + 4 training crashes + a held-out crash regime). Result
(full table in [[Experiment Log]]): **the crash-survival mechanism WORKS** — 3/4 seeds de-risk on the
breadth collapse (s0/s1 hold 3–5% DD in an 82% crash; **s1 +5.8%, positive**), the first RL behavior
static strategies can't match. **But the policy learned defensive-*everywhere*, not regime-*adaptive*** —
uniformly cautious (4–13% DD in every regime), so it *loses the bull* (val −6.9% while the basket rose
+27%) and isn't robust (s3 DQ'd at 34.7%). Next levers: (1) rebalance the reward toward bull-harvest
(lower `dd_lambda`); (2) **RecurrentPPO** — breadth is a time series, now correctly sequenced with a
feedforward champion to A/B against.

### Post-GATE-2 plan (2026-06-10, `rl-ml-trainer`) — harvest obs, lever sequence, gate

The GATE-2 gap (defensive-everywhere) is a **reward/credit problem, not an information-starvation
one** — the exp1→exp5 arc proved adding obs features to a sample-starved decision set does not move
the gate; exp5's fix was structural (`--ungate`, ~960 decisions), not a feature. So features are
sequenced *after* a reward that can use them, not bolted onto one that taught the opposite.

**Harvest obs spec (OBS_DIM 13 → 17, append-only so saved VecNormalize stats degrade gracefully).**
Four token-relative slots (like `cush`/`surge`, describing the event token), all on `self._px`
(causal — ratios of past cumprod rows):
- **13 `r24`** = `px[bar]/px[bar−24]−1`, **14 `r3d`** = `/px[bar−72]`, **15 `r7d`** = `/px[bar−168]`,
  each `clip(±RET_CLIP)` then `tanh(3·x)` to squash fat alt tails into [−1,1].
- **16 `brk`** (breakout-distance) = `px[bar] / rolling_max(px, N=72)[bar−1] − 1` (the `bar−1` window
  is the leakage guard — the current bar can't be its own high), clipped `[−CUSHION_CLIP, +small]`.
  Takes whatever continuous breakout form `market-indicator-expert` finalizes.
- **`r30d` dropped** from the original ask: the [[Trading Strategies]] §intraday spec says the edge is
  short-window (the 30d/5d-high conditions don't occur in a downtrend), and r30d is collinear with
  `cush` + breadth. Re-add only if the subset probe shows incremental IC over `cush`.
- Leakage test to add: OBS_DIM==17 end-to-end through `GymEventRungEnv`; all slots finite/bounded; a
  future-price perturbation leaves the obs at `bar` unchanged.

**Why it (might) fix the gap + the DQ risk.** `breadth-high (slot 12) + fresh breakout (16) +
short-horizon momentum (13–14) → size up` is the harvest half of the regime-adaptive pair. The
breakout is the **nonlinearity** ([[Trading Strategies]]): linear trailing-24h return is *negatively*
correlated with forward return (the universe mean-reverts), but the breakout condition selects the
momentum-continuation sub-population — a feedforward MLP can represent the interaction. **Biggest
risk:** a harvest feature is a *size-up* trigger, and the obvious "ramp up in bulls" lever (cutting
`dd_lambda`) removes the only brake preventing the GATE-1/SIREN-corpse concentration that DQ'd GATE-1
rung-0 (31%) and GATE-2 s3 (34.7%). Harvest and de-risk pull opposite ways. What makes it
attemptable: **risk-parity per-token caps stay ON** (a high-vol breakout gets a tiny cap ∝
vol_target/vol → convex tail harvested *bounded*, can't blow the gate); prefer a **selective,
budgeted reward** (`residual_ranked` γ≈0.1, interior optimum — the targeted fix for β=0.8's DQ) over
a blanket `dd_lambda` cut; and always read the **worst-seed crash DD**, not the mean.

**Lever sequence (one variable per gate).**
1. **Reward-rebalance — FIRST, cheapest/highest-info.** GATE-2 config frozen (broad k=12,
   risk-parity, breadth obs OBS_DIM-13, 4 crashes, held-out crash), change **only** `dd_lambda 1.0 →
   0.5`. 4×1M, val/test/crash. Isolates "is the bull-loss a reward problem?" with zero new code
   surface. If it ramps the bull but blows the crash DD → the brake was load-bearing → switch the
   reward to `residual_ranked` (γ≈0.1) rather than cutting `dd_lambda` blindly.
2. **Harvest obs (13→17) — SECOND, gated by a probe before any compute.** No sweep until
   `scripts/probe_subset_ic.py` shows r24/r3d/brk carry **incremental-over-`cush` OOS IC on the
   ungated ~960-event pool**. No headroom ⇒ don't run (saves a day, the exp4 lesson). If headroom:
   lever-(1) champion + OBS_DIM 17, A/B'd vs that champion.
3. **RecurrentPPO — LAST.** GATE-2 says the gap is reward, not capacity; LSTM is the most expensive +
   most overfit-prone; only buy it once a feedforward champion + the feature show the feedforward
   ceiling. Breadth-as-time-series is the right use, but earned, not first.

**Net-of-cost validation (not gross IC).** The env nets the ~1% round-trip into the equity path
(`amm_cost_usd` on every entry/exit) so the reward sees post-cost equity; the gate runs B&H/Random/
rung-0 through the *same* broker (equal costs); the subset-IC is only a go/no-go for *running* a
sweep, never a success claim. Success = `honest_gate` PASS on held-out test+crash, seed-mean AND
worst-seed < 30% DD. The breakout edge (+0.77% gross < 1% cost, profit in the convex tail) is exactly
the case env-cost — not IC — adjudicates.

**Recommended next single experiment + gate.** Reward-rebalance `dd_lambda 1.0→0.5`, GATE-2 otherwise
frozen, 4×1M, val/test/crash. **PASS** = seed-mean beats Buy&Hold + Random + surviving rung-0 on
val(bull) AND test(pump) AND crash, worst-seed maxDD < 30% every regime, AND **retains crash survival**
(crash DD not regressing materially from GATE-2's 3–5%). Concretely: turn val from −6.9% toward basket
+27% *without* crash DD exceeding ~10% on any non-DQ'd seed. If it ramps the bull but blows the crash,
that's the signal to go selective (`residual_ranked`), then add the harvest features.

**Honest first question:** can the exposure-overlay PPO beat the vol-tilt baseline OOS? "No"
is a valid result the `beats_baseline` gate is built to surface.

## As-built (2026-06-10) - the rungs ladder + event-driven rung-1

After the exposure-overlay PPO failed OOS (val +83..156% -> frozen-test +11% / -1.8%, both
breaching the gate), RL-**from-scratch** was shelved and the work reframed as a **rungs ladder**:
encode the discipline as rules first, then let RL learn only the *discretion* those rules hard-code
- never the whole policy from zero.

- **Rung 0 - the rule (`trader.strategy.rung0`).** A per-token, **event-driven (intra-day)** state
  machine encoding the user's discretionary discipline: enter on a **volume ignition** (a sharp
  `vol_fast`-bar surge >= `vol_mult`x baseline while price rises **above a rising trend-EMA**), let
  winners run untrimmed, exit on the rollover (price `stop_k` off the peak OR below the trend-EMA),
  with a **dead-zone/cooldown** anti-churn guard (no FOMO re-entry below a prior runup's origin) and
  **loser-funded rotation** (fund a fresh ignition by closing the weakest holding, only if it's
  weaker than the candidate). On the frozen-test split it **beats both vol-top8 baselines on return
  AND drawdown** (+29% / 17% DD). Detail + the trade-logic forensics that built it: [[Build Log]],
  [[Trading Strategies]].

- **Rung 1 - RL learns rung-0's discretion.** Two ways to "train RL with the rung-0 rules" were
  tried; the first revealed *why* the env architecture, not the signal, was the constraint:

  - **Option A - signals as features (shelved).** Fed rung-0's per-token signals (ignite /
    volume-surge / price-EMA cushion) into the **daily-rebalance** `PortfolioEnv` as observations
    (`--rung0-obs`). It trains and the obs are causal, but the env **acts once per day** - so every
    trade lands at the same hour (07:00 UTC), the exact rigid clock the discretionary thesis
    rejects. And on **val (a melt-up)** the policy's +137% merely **matched plain-hold** (+137%):
    full allocation wins in an up-only regime, so "perfectly timed" was the regime, not skill.
    Verdict: features-on-the-daily-env can give the policy rung-0's *information* but never rung-0's
    *intra-day execution*. Dead end for the goal.

  - **Option D - event-driven rung-1 (`trader.train.event_env.EventRungEnv`, the pivot).** A
    **semi-MDP** that steps at rung-0's **events**, not a clock: the agent acts only when a volume
    ignition fires (size it / skip it) or a held position trips its stop / EMA-break (cut it / hold
    through). Between events the env advances bar-by-bar - positions drift, no trades - so execution
    is **intra-day and event-timed, structurally unable to collapse to a clock** (smoke: decisions
    on 20 of 24 hours-of-day vs all-at-07:00 for Option A). **rung-0 supplies the edge** (ignition
    timing, exit triggers, dead-zone/cooldown, loser-funded rotation); **RL learns the discretion
    rung-0 hard-codes** - entry **sizing** (conviction, up to `max_entry_frac`) and whether to
    **override an exit** (hold a winner through its stop / re-arm it, or cut early). One scalar
    action in [0,1] interpreted by event type. **Reward** = the interval equity change since the
    last decision minus the same quadratic drawdown brake (semi-MDP credit assignment). Positions
    valued by price-index ratios; signals precomputed once (causal, scale-invariant) so the per-bar
    advance is cheap. Trainer `scripts/train_event.py` (eval/publish path torch-free + laptop-
    validated; PPO `learn()` desktop-only); seed sweep `scripts/run_eventrung_sweep.sh`; the
    **baseline is the rung-0 RULE itself** - the honest question is *does learned discretion beat
    the hand-coded version, with intra-day execution?* (4-seed x 1M-step sweep running 2026-06-10).

  v1 keeps rung-0's rotation rule fixed (learns sizing + override only); "which candidate to fund"
  is a later lever. Why this beats both prior attempts: not RL-from-scratch (no edge prior, failed
  OOS), not features-on-a-rigid-clock (inherits the rigidity) - RL constrained to **rung-0's
  event-driven skeleton**, learning only the discretion, with the rules' anti-churn discipline intact.

### Rung-1 experiment 1 — relative-to-rule reward (2026-06-10)

The first event-driven sweep (absolute reward) **under-traded**: 2–4 trades/seed, +9.7% test, the
agent riding 2 winners and skipping rung-0's ~30 ignitions. Diagnosis (`rl-ml-trainer`, grounded in
`event_env.py`): the **absolute interval-return reward makes passivity optimal** in a bull sample,
and it **never references the rule** — so skipping an ignition the rule would have taken costs the
agent nothing. Five compounding mechanisms all point at inaction (absolute reward, the one-sided
drawdown penalty acting as a hidden position-count tax, the Gaussian-on-[0,1] boundary attractor,
sparse semi-MDP credit, the melt-up-biased sample).

**The fix — reward relative to the rung-0 rule.** Each interval, subtract the rung-0 RULE's return
over the same bars: `reward = (agent interval-return − rule interval-return) − dd_lambda·penalty`.
Now *matching* the rule = 0; the **only** way to score positive is to **beat** it (size a winner
bigger, skip a loser the rule took, hold through a stop it cut). Passivity and melt-up beta net ~0,
so they stop paying. Implemented as a **shadow rung-0 equity curve precomputed in-env**
(`EventRungEnv._rule_equity_curve`, a faithful mirror of `run_rung0` on the precomputed signals),
**parity-verified VAL 0.0pt / TEST 0.3pt** before trusting any reward (the guard the plan requires).
Paired with a relaxed drawdown penalty (`dd_lambda` 0.5, `dd_soft` 0.20), the post-mortem's
exploration config (`ent_coef` 0.2, `lr` 3e-4→3e-5 anneal), and 2-week episodes.

**The boundary-collapse detour (exp 1b).** The 100k smoke collapsed completely — **action mean
0.000, 0 trades**: a Gaussian policy on a `Box[0,1]` drifts its mean to the lower bound, every
sub-0 sample clips to the same no-trade outcome, and the dead gradient traps it before the relative
reward can teach it to act. Fix: **reparameterize the action to `[−1,1]`** (`m = (a+1)/2`), so the
network's neutral init (a≈0 → m=0.5) lands in the **interior and trades** — collapsing to never-trade
now means actively driving to −1 against exploration *and* a reward that punishes idleness. The
smoke then traded actively (action mean 0.649, full range). (Beta-policy head held in reserve if a
future config re-pins.)

**Result (frozen TEST, 4 seeds): +8.6% avg (±3.7%), maxDD 15.7%, ~18 trades/seed** — the
**under-trading is solved** (16–22 trades vs 0–4), every seed positive and gate-safe, the first RL
config that behaves like a real active agent across seeds. It does **not yet beat the rule** (~+18%
causal) — return ≈ the absolute version but now *with* participation, i.e. it learned to **act like**
the rule, not yet to **out-discriminate** it. Standings table → [[Experiment Log]].

### Rung-1 experiment 2 — per-decision (residual) reward (2026-06-10)

A 2nd `rl-ml-trainer` consult + a **deviation-alpha diagnostic** redirected the next step from
"capacity (LSTM)" to "reward": correlating each executed entry's over-size-vs-rule with its
forward-24h return gave **corr = −0.027** (`scripts/diag_deviation_alpha.py`) — the agent's bigger
bets land on up- and down-moves **indiscriminately**, and it never sizes *below* the rule. So it's
**reward-bound, not capacity-bound**: the flat "copy-the-rule" basin, where the whole-portfolio
relative reward smears the marginal decision into base-divergence noise. *Don't buy an LSTM to escape
a flat-gradient basin — fix the gradient.*

**Experiment 2 — `reward_mode="residual"`:** reward the agent's **weight deviations from the rule**
dotted with token returns, `Σ_tok (agent_w − rule_w)·ret_tok`, over the interval since the last
decision. Shared positions (`agent_w == rule_w`) cancel, so **only the agent's active bets vs the
rule earn/lose** — oversizing a winner pays, oversizing a loser hurts. The shadow book now also
tracks the rule's **per-token weights** (`_rule_equity_curve` returns `(eq, w)`); the rule's exposure
is added to the obs (O1, OBS_DIM 11→12); `norm_reward=True` for the small zero-centered reward.
**Verified locally:** a rule-mimic agent nets **~0** residual (+0.013), a max-size agent **+0.538**
(deviations score) — the gradient the −0.027 says is missing. Sweep: `... test residual` →
`ppo-event-res-test-s<seed>`. **Gate: seed-mean test > +18%, worst-DD < 25%.** LSTM + regime obs stay
**deferred** — earned only if a clean reward still can't beat the rule.

## Rung-1b — rule-default discretion (SPEC, 2026-06-10)

Motivated by the g2b trade forensics (`scripts/diag_token_events.py`, [[Experiment Log]]): the
trained discretion **vetoes the rule for free** — it skipped every strong SIREN/BANANAS31 ignition
(~80 prompts), bought only the weakest ones (surge ≈2.6×), and answered every exit prompt with a
partial keep, producing a geometric dust-tail of losing sells (plus an invisible sub-$1 gas grind).
The env's neutral action is "do nothing"; rung-0's behavior must be actively learned. Rung-1b
inverts that: **the default action EXECUTES rung-0's decision; deviations must be earned.**

### Sequencing — probe before build (the exp4 lesson)

**Gate A (FIRST, zero new env code): the skeleton oracle ceiling.** A hindsight-greedy scripted
agent through the *current* env (g2b config: broad k=12, risk-parity caps, real costs): at each
entry prompt take max size iff the token's fwd-24h return is positive, else skip; at each exit
prompt hold iff fwd is up, else cut. Decompose entry-only / exit-only / both. This bounds what ANY
learned discretion can extract from rung-0's event set. **Kill criterion: if the oracle's val
return < Buy&Hold (+27.5%), no reward/policy inside this skeleton can pass the honest gate →
pivot the substrate (long-default basket overlay) instead of building rung-1b.**
Script: `scripts/preflight_skeleton_ceiling.py`.

### Mechanics (built only if Gate A clears)

1. **`rule_default=True` entry semantics** — discrete levels become multipliers of the RULE's
   sizing (`ef=0.20·eq`): **idx 0 = 1× (THE RULE, the biasable default)**, idx 1 = ½×, idx 2 =
   skip, idx 3 = 2× — still clipped by the risk-parity cap and cash; rotation unchanged. (Index 0
   is the default for BOTH event types — exit idx 0 = the rule's cut — so one logit bias makes the
   untrained policy ≈ rung-0.)
2. **Exit semantics + the two defect fixes** — levels stay keep ∈ {**0 (DEFAULT — the rule's
   cut)**, ⅓, ⅔, 1} but: (a) **no peak re-anchor** on trim/override (removes the stop-ratchet that
   rode s2 into the 63.7% crash DD); (b) **`exit_commit=12`**: after any non-cut decision the
   position is not re-prompted for 12 bars (a trim/hold is one committed decision, not a per-bar
   liquidation drip); (c) **`dust_usd=10`**: a partial keep whose remainder is below $10 forces a
   full close (no sub-$1 gas-bleeding tail).
3. **Rule-prior init** — +2.0 logit bias on the default level at policy init (entry idx 2, exit
   idx 0), so the untrained policy ≈ rung-0 and PPO must learn *against the prior* to deviate.
4. **Reward unchanged from g2b** (relative, `dd_lambda` 0.5) — the substrate semantics are the one
   variable under test.

All new flags default OFF (prior behavior byte-identical; regression suite must stay green).

### Gates after build (in order, all laptop-side before any desktop compute)

- **B — parity:** the all-default scripted policy through the env ≈ the rule mirror (report the
  capped-vs-uncapped-`ef` gap separately — the risk-parity caps mean "1× rule" on a monster is
  capped; that gap is the guardrail working, not an error).
- **C — in-env landscape (exp5-style):** rule-mimic ≥ both corners (skip-all, all-max); a fwd-fit
  selector is the unique argmax, all through the real env on total reward.
- **D — the honest training gate (unchanged):** 4×1M seeds, seed-mean beats Buy&Hold + Random +
  surviving rung-0 on val AND test AND crash; worst-seed maxDD < 30% everywhere. Run-ids
  sha-stamped (`ppo-event-rd-<sha>-s<seed>`).

**STATUS (2026-06-10): built, 258 tests green, gates A/B/C ALL PASSED** — ceiling val +74.6%
(exits carry it: entries-only +7.3%), parity ≈0 uncapped (±3.6pt = the caps), landscape oracle
unique argmax (+0.744/+0.405 margins), oracle-through-rd-env +77.3%/+44.4% at ~12% DD. Results +
the launch command: [[Experiment Log]] §"Decided next — rung-1b". Awaiting desktop go.

## As-built (2026-06-11) — the rd ladder: 8 sweeps, the named equilibria, → RecurrentPPO

The rung-1b substrate swept through 8 configs in ~24h (standings: [[Experiment Log]]). What each
lever taught, in the order it was learned:

| lever | sweep | the read |
|-------|-------|----------|
| rule-default actions | rd | veto pathology fixed (takes the rule's trades); s1 first-ever full regime-gate passes (test AND crash) |
| voltop8 + tp_rungs | rd8/rd8tp | calm tokens cut; **sell-into-strength added** (the env structurally lacked it; oracle ceiling → +95.5% val @ 7.1% DD) |
| harvest obs | rd8h | 3rd obs lever with no discrimination effect — the obs-hypothesis family is exhausted on this arch |
| `dd_lambda 0` | rd8h0 | the **diet-rule equilibrium** named + half-confirmed: a dd penalty on top of a relative reward pays a non-discriminating agent to under-size the rule; and the substrate alone held worst-seed DD to 26.9% with NO reward brake — **structural DQ-safety proven** |
| `crash_train 4→1` + loss-floor 0.2 | rd8h0c1 | the inflated crash prior was real; the Q disaster path (override down a crash) closed; **first positive val of the project (+4.7% mean, all seeds beat the rule)** |
| det-blacklist 672 | rd9 | probe-confirmed (post-detonation ignitions fwd48 −8/−24%, expiring ~4wk) — the Q pump/dump class deleted for agent AND rule mirror |
| **5M steps** | rd9 | REGRESSION — converged to rule-hugging flat (val −0.7/test +2.0/crash −7.7). The exploration anneal collapses at scale; more steps ≠ discrimination |
| **RecurrentPPO LSTM-256** | rdL (running) | the user's failure classes — re-buying post-pump bleed, not holding winners to the top, "no sign of learned experience," seed incoherence — are **sequence skills a stateless MLP cannot represent**. The deferred-until-earned condition (a demonstrated feedforward ceiling) is met |

**Where the honest gate stands:** best learned config = rd8h0c1 (val +4.7%, 11/12 cells positive,
worst DD 15%); the rung-0 rule on the prepad windows is the towering bar (test +89.3%, crash +62%
surviving). The residual gap is **per-decision discrimination + cross-time coherence** — the LSTM
is the first lever aimed at the second half. rdL verdict is judged on a behavioral checklist
(bleed re-entries down? winners held longer? SIREN Mar 22 taken? seed coherence up?), not returns
alone. Trainer support: `--recurrent --lstm-size N` (sb3-contrib MlpLstmPolicy), stateful eval
threading (fresh LSTM state per split episode), recurrent-aware smoke timeout in the MCP launch
tier ([[MCP Server]]).

## Is RL worth it here? (candid)

A single 7-day live ranking is a hostile setting for a learned policy. Both sides honestly:

**For RL**
- The pipeline already exists and is proven *as engineering* — low marginal cost to try.
- Learned signal-weighting (attention extractor) can adapt across regimes in ways fixed rules
  cannot, and the drawdown-penalized reward is well-aligned with the DQ gate.
- If the [[Simulated Market]] is honest, a policy that survives many scenarios is a real edge.

**Against RL**
- **One week is high-variance** — a strong backtest can still rank poorly on a 7-day sample;
  much of any edge may be noise ([[Simulated Market]] open question on bootstrap CIs).
- **Limited live on-chain data.** TradeSim trained on years of *CEX* candles; BSC on-chain
  history on 149 thin tokens is thinner and noisier — overfitting risk is real.
- **Regime risk.** The live week's regime is unknown; a policy tuned on past windows can
  meet an out-of-distribution market and behave unpredictably.
- **The CEX→on-chain gap** ([[Simulated Market]]): if the AMM/pool-depth cost model is
  miscalibrated, the policy optimizes against a fiction.
- A **simpler robust strategy** (Buy & Hold on the eligible list, or a conservative rule with
  the same hard guardrails) is lower-variance, fully inspectable, and may be the safer bet for
  *staying inside the DQ gate* — which is what scores.

**Working stance:** develop RL offline in parallel because it is cheap given the ported
stack, but **only ship a learned policy if it beats the baselines convincingly on held-out
data and the edge survives a one-week resample**. Otherwise prefer the simpler strategy. The
decision-core interface lets us decide this late.

## Open questions

- **Does any backtested RL edge survive a 7-day sample?** Block-resample the live window to
  size the confidence interval before trusting a model over a baseline. Unverified.
- **Is there enough on-chain history** to train without overfitting the thin eligible tokens,
  or should features lean on CEX proxies where they exist? Coordinate with
  `market-indicator-expert`.
- **Reward vs the hourly ≤$1 → 0% rule.** The current reward penalizes drawdown but is not
  yet wired to the dust-out scoring quirk ([[Simulated Market]]) — confirm whether the reward
  needs a term for it.
- **Generalization to on-chain execution.** A policy trained against simulated AMM costs must
  still behave under real Amber/Rango fills — the sim-to-live gap is the biggest unknown.
- **Training host.** Where the offline runs execute — the **desktop**, chosen for CPU cores +
  RAM (this workload is env-stepping-bound; torch CPU-only). Parallelize via vectorized envs
  (`n_envs ≈ physical cores`), not GPU → [[Remote Capabilities]].

## As-built (2026-06-12) — the autonomous loop arc + the knowledge-expansion era

**The loop ran the lab.** Six autonomous iterations in ~24h (driver: [[MCP Server]] 4B/4C):
rdL (LSTM muted) -> rdLq (Q-tail guards, +3.2pp) -> **rdLe4 (ent 0.4 - BREAKTHROUGH: val +13.6 /
test +14.7 / crash +13.2, worst-DD 10.5%, two individual val gate passes)** -> rdLe6/rdL2m/rdLp1
(entropy/steps/prior all REGRESS) -> **drift-alarm self-halt**. rdLe4 = family champion, its
neighborhood mapped on four sides. Two desktop crashes absorbed (WSL-close mid-sweep; partial-death
detection added). Verdicts auto-logged, leaderboard auto-published each iteration.

**The knowledge era (post-plateau, user direction: expand what the agent KNOWS).**
- `scripts/trade_postmortem.py` — round-trip grader (entry/exit/alloc/freq/risk). Findings: seed
  variance IS craft variance (champion seed: entries +12% off the low, MAE -3%, sizes winners
  +0.19; laggard: chases +21%, MAE -13.5%, sizes losers -0.28); **TP-rung exits are perfect
  (100% capture); trailing exits give back 9-19%** — the missing skill is mid-trade exhaustion
  recognition. Quant-corrected rubric: vs-baselines GO/NO-GO panel FIRST (anti-proxy-drift),
  causal metrics only are scored, skip-quality panel, cross-seed coherence axis.
- **Probe scoreboard** (probe-before-build, 5 theories this era): cross-sectional rank REFUTED;
  linear cycle-memory REFUTED; **spent-move flag VALIDATED both splits -> `cycle_obs` built
  (rdLc sweeping)**; token-personality kernel real (pooled persistence rho +0.256, sign ~2/3
  stable) but **entry-payoff REFUTED** (-0.065 OOS IC) -> no build, exit-style variant parked;
  liquidity/flow DATA-GATED (static sim liquidity) -> the wallet-attributed logger parked in
  [[Trading Strategies]] as the post-competition mechanism.

## Checkpoints + curriculum — the next direction (2026-06-14)

After the hyperparameter neighborhood plateaued (the rdL arc), the user reframed the work:
**a training run is not the deliverable — a reproducible CHECKPOINT is**, to **warm-start** further /
**curriculum** training. Two corrections lock this in:

- **Curriculum + checkpoint warm-start are first-class levers, not "cheats."** The earlier reflex
  against phased/curriculum training was a category error: curriculum governs **optimization** (how a
  policy reaches a good basin when the obs/decision space is too rich to learn end-to-end); the honest
  gate governs **evaluation**. They are orthogonal — run an aggressive curriculum, still judge the
  OUTPUT on the gate. (The 1-in-12 good-val basin rate is itself the symptom of a landscape too hard to
  navigate cold — exactly what curriculum addresses. Cf. a Jan-2026 locomotive-control RL project where
  curriculum was the breakthrough after many failed one-shot runs.) `rule_default` is already implicit
  curriculum; making it explicit + checkpoint-seeded is the natural next step.
- **Checkpoints reproduce bit-identically.** rdLe4/s0 re-ran to 17 decimals three times → training is
  deterministic on the box, so any seed's policy is recapturable on demand. Capture workflow + the
  save-enabled-sha gotcha → [[Build Log]] (2026-06-13/14); persisted artifact in S3.

**Diagnose before designing (the simulator-first rule).** Curriculum is designed against MEASURED gaps,
not guesses. The cross-timeframe replay of s0 (`scripts/simulate.py`, [[Simulated Market]];
table in [[Experiment Log]]) gives the target list — outside its memorized val pocket s0:
  - **(a) doesn't ride bull upside** — its discretion DESTROYS value vs holding the same risk-parity
    basket (6mo −1.2% vs B&H +127%); it sells winners. Likely cause: 336-bar (2wk) episodes never
    taught long-horizon holding → a **horizon/episode-length curriculum** is the prime candidate.
  - **(b) loses to its own rung-0 rule OOS** in every window — the "beat the rule" objective didn't
    generalize past the val window it overfit.
  - **(c) churns in chop** (1wk −8.7% on 18 trades) — needs to learn to stand down in flat regimes.
The one virtue (bear capital preservation) is what the val pocket rewarded. **A curriculum warm-started
from the checkpoint targets (a)/(b)/(c); validate its OUTPUT on the per-regime honest gate.** Design
starts on the user's go.

## The train/deploy STRUCTURE mismatch — the fork (2026-06-14)

The weekly simulator ([[Simulated Market]], [[Experiment Log]] §2026-06-14) surfaced a methodology error
that reframes the next phase. **We trained and evaluated the agent in structures that do not match how
it will actually run:**
  - **Trained on:** random 2-week (336-bar) episodes from the train split.
  - **Evaluated on (flattering):** ONE continuous multi-week episode — where s0 looked like a star
    (ZEC +$2,747, traded with precision).
  - **Deployed / honestly tested as:** a COLD ~1-week session, fresh $10k, **no cross-week holds** (the
    competition is a single week; the weekly sim samples 28 such sessions).

**The conceptual point (so this is never confused again):** a trained policy IS a generalizable
obs→action function; it does NOT need to have been "running continuously." Every training episode also
started cold (zero LSTM state), so a cold weekly start is in-distribution — the agent is *trained* to
start cold. So the cold start is NOT the failure. The failure is that the SAME ignition setup is traded
in the continuous eval and **skipped** in the cold weekly session — the fingerprint of a model overfit
to windows rather than generalized (already proven OOS: it loses to B&H and its own rule). A genuinely
general policy would trade ZEC's ignition regardless of date or portfolio path. This one doesn't.

**Requirements for the next training phase:**
1. **Evaluate in the deployment structure.** Score on cold ~1-week sessions (the competition shape), not
   a generous continuous run. The continuous eval flattered s0 and hid its fragility — that eval must
   change before any config is judged "good."
2. **Train in (or curriculum toward) that structure** so train ≈ test ≈ deploy, removing the mismatch
   rather than hoping the policy bridges it. (A weekly-horizon episode is the natural unit.)
3. **Honest gate across unseen regimes stays the bar** — beat B&H + rung-0 OOS; only a model that passes
   is deployable, and by construction its logic WILL apply broadly (the user's correct expectation of
   what a trained agent is).
4. The curriculum targets (ride winners / beat rung-0 OOS / stand down in chop) feed into this; the
   warm-start from the s0 checkpoint remains the seed. See [[curriculum-and-checkpoints-are-legitimate]].

**Bottom line:** s0 is a reproducible checkpoint and a sharp diagnostic, not the product. The route to a
deployable agent is to train AND evaluate in the real (weekly/cold) structure, to the honest gate. That
is the work the loop returns to.

## As-built (2026-06-14) — the cold-weekly bar, the random-week gate, and the long-default OVERLAY

Returning to training (user `/orient`). Two decisions: **train FROM SCRATCH** (no warm-start — s0 is a
proven-overfit val-pocket policy and no `--warm-start` path exists), and **decide the substrate AFTER a
diagnostic**. A third reframe from the user mid-session: **drop the BTC/regime focus** — the universe was
selected for *low BTC correlation*, so a BTC-anchored regime signal is near-noise — and judge configs on
the **distribution over randomly-selected cold weeks**, not curated bull/bear/crash buckets.

**1. The deployment-honest BAR (`src/trader/train/weekly_eval.py`, `scripts/eval_weekly_baselines.py`).**
A torch-free cold-weekly grader (Mon-00:00-UTC weeks, fresh $10k, 168h warmup prepad, per-week causal
vol-top-8, no cross-week holds — `simulate_weekly`'s slicing) grading rung-0 + risk-parity B&H per week.
Result (full table → [[Experiment Log]] §2026-06-14): **OOS rung-0 +7.8%/wk vs B&H +15.0%, BULL-GAP
+13.2%**, one DQ week, ≥1-trade/day missed every week. The bull-gap *quantifies the skeleton ceiling in
the deployment structure*: event-only can't express "just stay long," so it bleeds vs holding in up-weeks;
its one positive OOS number is carried by a single +92% week. Same verdict whether framed by regime or as
a raw random-week distribution (skeleton beats holding in only ~36% of weeks, loses big when it loses).

**2. The random-week distribution GATE (`weekly_eval.weekly_gate` + `bootstrap_mean_ci`).** Judges a
config on its weekly-return *distribution* via the bootstrap-CI **lower bound** (so one lucky +92% week —
the s0 flatterer — can't crown it): pass iff worst-week DD < 30% AND CI-low beats B&H's weekly mean AND
beats rung-0's AND respects the activity floor; binding constraint named. (Activity ≥1-trade/day is a
*universal* daily requirement — B&H misses it too — so deployment needs a forced minimal daily rebalance,
a guardrail, not a strategy discriminator.) **Wired into `train_event` as `--eval-mode weekly`** (grades
the policy on cold weekly sessions vs rung-0 + B&H, the `--no-btc-obs` flag neutralizes the `btc_trend`
slot). The gate uses a **PAIRED** bootstrap — `policy_week − baseline_week`, CI-low > 0 — because policy
and baseline see the SAME weeks, so the difference cancels the common (huge) market variance an unpaired
test is swamped by. A subtle artifact was caught and fixed: measure the policy's return from the $10k
**deposit**, not the post-entry-cost `eq[0]`, else a basket policy fakes a +0.48%/wk edge; after the fix a
hold-everything overlay == B&H *exactly* and correctly FAILS `beats_buyhold` (it must add tilt value).

**3. The long-default basket OVERLAY (`EventRungEnv.basket_default`, built + validated).** The substrate
fix the bar demanded (user picked it after the diagnostic): `reset()` buys the full risk-parity vol-top-8
basket (= B&H, cost baked in), and the exit/profit action tables **invert** — idx 0 = HOLD the basket,
deviations trim — so the default action holds and *doing nothing ≈ B&H*. The relative-reward benchmark
becomes the **held-basket B&H curve** (`_basket_equity_curve`), so a do-nothing agent nets ~0 and *only
correct tilts score* — a well-posed gradient that directly targets the +13pp. rung-0's ignition/exit
discretion becomes a **tilt** on top (its real value is the exit/risk side — TradeSim's #1 lesson — now
additive instead of the whole policy). Flag defaults OFF (378 tests green, prior behavior byte-identical);
new flag builds on `rule_default` (discrete 4-level), with `rule_prior` making the untrained policy ≈ B&H.
Validated on real data: hold-everything == B&H to 5 decimals over 6 cold weeks. Plumbed through the gym
adapter, `train_event` (`--basket-default` + provenance), and `simulate` provenance.

**Next:** finish the random-week gate wiring (drop btc_trend obs), then the **from-scratch random-week
sweep** on the overlay to the distribution gate (desktop — shared-box gotchas apply). The honest question:
can learned tilts beat holding the basket OOS across random weeks? "No, just hold" is now a valid,
gate-safe answer the substrate makes reachable. See [[curriculum-and-checkpoints-are-legitimate]],
[[Simulated Market]], [[Experiment Log]] §2026-06-14.
