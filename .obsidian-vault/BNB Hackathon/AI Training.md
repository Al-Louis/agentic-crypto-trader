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

**Curriculum status:** the env samples random windows from the training split; **synthetic
crash injection** (`trader.sim.crash`) to carry the bear/crash regime the ~6-month bull sample
lacks is the next step — a *real* sampler, not cosmetic (the post-mortem's #1 lesson). The
frozen-test split is reserved; tuning happens on validation to avoid the loop meta-overfitting.

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
