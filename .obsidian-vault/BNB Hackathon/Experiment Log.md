# Experiment Log

The rigid, permanent record of every training iteration — **exact config → performance** — so a
tweak that degrades the agent never strands us without a way back to the best known formula.

> **Why this exists (the TradeSim lesson).** TradeSim's biggest process failure was inconsistent,
> manual performance tracking: when a change degraded the model, it was painfully hard to recover
> the *previous best formula*. We do not repeat that. Capture is **automatic and version-controlled**,
> not a note someone remembers to write.

## How it works

1. **Provenance is baked into every run.** `scripts/train_rl.py` writes a `provenance` block into
   each published `metrics.json` — git commit + every hyperparameter (reward mode, rich-obs,
   timesteps, seed, lambdas, ent_coef, lr, step_bars, split…). Every bundle is self-describing and
   reproducible.
2. **The ledger is rebuilt from the bundles.** `scripts/build_ledger.py` reads every published
   bundle and writes:
   - `experiments/ledger.jsonl` — one append-only line per run (config + all metrics + verdict).
   - `experiments/champion.json` — the current best formula + the exact command to reproduce it.
   Both are **committed to git**, so the full history of what-produced-what is permanent and
   diffable. The bundles are the source of truth; the ledger is a deterministic aggregation.
3. **Reproduce anything** by reading its `provenance.git_commit` + config and re-running.

## The evaluation bar

A run is judged on three axes, in order — never on raw return alone:

1. **Survives the DQ gate** — max drawdown **< ~30%**. Breaching this is disqualification *before
   live PnL is even counted*. This is the hard constraint.
2. **Beats the baseline** — the validated vol-tilt(trend50) overlay on the *same* window.
3. **Return** — maximize, given 1 and 2.

Judge on the **seed mean** (single-seed RL is unstable), and always read the **worst seed's
drawdown** — a mean under the gate with a worst seed over it is *not* yet competition-safe.

## Current champion

**None.** No config has passed frozen-test OOS (the rule: split=test, beats its test baseline, AND
worst-seed maxDD under the gate). The val champions (`ppo2-real` +83%, `ppo2-real-give` +156%) both
**collapsed out-of-sample** — see the OOS verdict below. `champion.json` records `null`, honestly.
The val numbers were regime/era overfitting, confirmed by the held-out test.

## Standings — Reward-Shaping Sweep #1

- **Config (identical across all):** action `weights`, `--rich-obs`, 100k timesteps, eval split
  `val`, seeds {0,1,2}, n_envs 6, ent_coef 0.2, lr 3e-4, step_bars 24, episode_steps 30,
  lambdas gb=10 / turn=0.5 / realized=10. (Pre-provenance-stamping; sweep code ≈ commit `d8c5817`.)
- **Baseline:** vol-tilt(trend50) on the same `val` window = **+78.7%**.

| mode | mean return | mean maxDD | worst-seed DD | Sharpe | PF | legal (mean)? |
|------|-------------|------------|---------------|--------|-----|------|
| realized | **+198.2%** | 34.6% | 41.5% | 4.75 | 1.73 | ❌ |
| sharpe (control) | +151.6% | 31.8% | 42.8% | 4.39 | 1.70 | ❌ |
| **turnover** ⭐ | +126.5% | **29.6%** | 41.1% | 4.24 | 1.56 | ✅ |
| giveback | +103.1% | **28.7%** | 40.5% | 3.85 | 1.21 | ✅ |

### What it means

- **RL now decisively beats the baseline** — *all four* modes (103–198%) clear the +78.7% vol-tilt
  overlay. At 20k steps without rich observations, RL *lost* to this baseline; rich obs (per-token
  unrealized gain + distance-below-recent-high) + more steps flipped it. The exploration is working.
  See [[AI Training]].
- **The frontier is return-vs-DQ.** The aggressive vol-harvesters (`realized` +198%, `sharpe`
  +152%) win on return but breach the 30% gate. The brakes (`turnover`, `giveback`) stay under the
  mean gate but give up return. This is the central tradeoff to engineer.
- **Robustness is the real gap.** *Every* mode's worst seed sits at ~40–43% DD. No config yet
  *reliably* survives the gate across seeds. For a one-shot live run we need comfortable margin,
  not a mean grazing the line.

## Standings — Reward-Shaping Sweep #2 (1M-step composite frontier)

- **Config:** `composite` reward (realized engine + brakes by lambda), `weights`, `--rich-obs`,
  **1M timesteps**, `val`, seeds {0,1,2}. Provenance-stamped (≈ commit `2c92a2f`).
- **Baseline (same window):** +78.7%.

| config | brake | mean ret | mean DD | worst-seed DD | Sharpe | worst-seed legal? |
|--------|-------|----------|---------|---------------|--------|------|
| real-give | giveback 15 | **+156.5%** | 27.9% | 37.8% | 4.79 | ❌ tail-risky |
| real-dd5 | dd-penalty 5 | +94.1% | 26.5% | 30.1% | 5.17 | ❌ (barely) |
| real-combo | all three | +83.4% | 26.5% | 32.1% | 5.15 | ❌ |
| **real** ⭐ | none | +83.1% | 25.3% | **26.6%** | 5.12 | ✅ |
| real-turn3 | turnover 3 | +66.1% | 25.6% | **25.8%** | 4.74 | ✅ |
| real-turn1 | turnover 1 | +66.0% | 25.5% | **25.5%** | 4.74 | ✅ |

### Headline finding — more training regularizes the engine (the +198% was froth)

`ppo-realized`@100k vs `ppo2-real`@1M are the **identical reward** — only training length differs:

| | return | worst-seed DD | Sharpe |
|--|--------|---------------|--------|
| realized @ 100k | +198.2% | 41.5% | 4.75 |
| **real @ 1M** | +83.1% | **26.6%** | **5.12** |

More training **halved the return but slashed the drawdown and raised the Sharpe.** The +198% was
an **undertrained, high-variance** policy making wild concentrated bets that landed on this window —
froth that looks like genius and dies in production. At convergence the same reward yields a calmer,
**gate-safe, higher-Sharpe** policy. The ledger caught this on its first comparison; without it we'd
have chased the mirage.

- **Best deployment pick:** `ppo2-real` (+83.1%, all seeds <30% DD). The engine at full training
  needs no extra brake.
- **`real-give`** is the interesting lever — the giveback brake *raised* mean return over the bare
  engine (+156% vs +83%) at low mean DD, but its worst seed breaches (37.8%): buys return with tail
  risk. Tunable, not yet one-shot-safe.
- **Sobering:** at convergence the gate-safe configs (+66–94%) sit ~*at* the +78.7% baseline; only
  tail-risky `real-give` clearly beats it. And this is still **val** (the tuning window). The earlier
  "RL decisively beats baseline" softens. **OOS is now the decisive question.**

### Fee/turnover consistency audit (resolved)

Sweep-#2 fees came in far lower than Sweep #1 at *similar trade counts* — verified **consistent, not
a bug**: **fees track dollar turnover, not trade count.** The `fee/turnover` rate is ~constant at
**0.4–0.6%** (the AMM cost rate) across every run. The 1M policies make similar-count but **much
smaller** trades (fee/trade $12 → $3; turnover $440k → $195k), so fees fall proportionally. Same
convergence fingerprint as the drawdown drop — the well-trained policy is calmer *and cheaper* to
run (smaller trades also cut slippage: effective rate 0.6% → 0.4%).

## Standings — Frozen-Test OOS Verdict (the honest result)

The two configs that beat the val baseline, run on the **never-touched test split** (model trains on
`train` as always; only the eval window changed). Test is a calmer regime — **vol-tilt baseline
+25.7%, Sharpe 2.76, maxDD 22.0% (gate-safe)**.

| config | val (tuning) | **test (OOS)** | mean DD | worst DD | vs test baseline |
|--------|-------------|----------------|---------|----------|------------------|
| `real` | +83.1% | **+11.1%** (+13.0/+18.6/+1.6) | 33.3% | 34.4% | **−15 pts** |
| `real-give` | +156.5% | **−1.8%** (−11.0/+7.1/−1.4) | 42.9% | 49.5% | **−27 pts** |

### Verdict — the edge did not generalize

- **Both collapsed OOS** — they underperform the simple vol-tilt baseline *and* breach the drawdown
  gate (which they respected on val). The +83–156% was **regime/era overfitting**, confirmed.
- **The dumb baseline beat the RL agent OOS, on both, while staying gate-safe.** The hand-tuned
  heuristic generalizes; the learned policy does not. That is the single most important clue.
- This is **not** a premature write-off — it's the *earned* conclusion of the full pipeline (rich
  obs → 1M convergence → multi-seed → clean frozen window). RL-learns-allocation-from-scratch, as
  built, has **no generalizable edge**. The frozen test caught it before any capital.

### Decided next — the generalization redesign

The direction is set by the clue (baseline generalizes, RL overfits):
1. **Train across regimes, not one bull window** — walk-forward / multiple windows incl. bear/chop.
   The biggest lever.
2. **Dynamic universe re-ranking** (`rerank_every`, built [[Build Log]]) — A/B 0 vs 1 *on test*: does
   a more stationary task close the OOS gap? Cheap, decisive.
3. **Regularize hard** — smaller net, higher entropy, shorter training (the 1M froth = too much
   capacity for the data).
4. **Reframe RL as a tuner on top of the baseline**, not a from-scratch allocator — the baseline is
   what generalizes. See [[Strategy Logic]], [[Market Conditions]], [[Trading Strategies]].

## Standings — Event-Driven Rung-1 (RL learns the discretion, 2026-06-10)

Acting on "Decided next" #4 (reframe RL as a tuner on the baseline). **Rung 0** = a hand-coded,
event-driven volume-ignition rule (the baseline that generalizes; **+29% frozen-test**, full-window
universe). **Rung 1** = RL learns only the *discretion* rung-0 hard-codes (entry sizing, exit
override) on rung-0's event skeleton — intra-day, no daily clock. Env `trader.train.event_env`,
trainer `scripts/train_event.py`, 4 seeds × 1M, frozen **TEST**. Baseline per run = the rung-0 RULE
on the **same causal-at-start universe** (**~+18%** test; the +29% used hindsight universe selection).

**Absolute-reward sweep (first event-driven run).** Test **+9.7%** avg, maxDD 9.8% — but only
**2–4 trades/seed**, 3 of 4 seeds byte-identical. The agent **under-trades**: skips rung-0's ~30
ignitions, rides 2 winners. Diagnosis (`rl-ml-trainer`): the **absolute** equity-change reward makes
passivity optimal in a bull sample and never references the rule, so skipping costs nothing.

**Experiment 1 — relative-to-rule reward.** `reward = agent interval-return − the rung-0 RULE's on
the same bars` (shadow book in-env, parity-verified **VAL 0.0pt / TEST 0.3pt** vs `run_rung0`), so
matching the rule = 0 and only **beating** it scores. + relaxed drawdown (`dd_lambda` 0.5, soft
0.20), post-mortem exploration (`ent_coef` 0.2, `lr` 3e-4→3e-5), 2-week episodes. The 100k **smoke
collapsed** (action mean 0.000, **0 trades**) — a Gaussian-on-[0,1] dead-gradient at the skip
boundary. **Fix 1b:** reparameterize the action to **[−1,1]** (neutral a=0 → trades from init); the
smoke then traded actively (action mean 0.649, full range).

| seed | **test (OOS)** | maxDD | Sharpe | trades |
|------|----------------|-------|--------|--------|
| s0 | +4.1% | 12.8% | 0.99 | 16 |
| s1 | +14.3% | 21.3% | 1.97 | 20 |
| s2 | +9.1% | 12.2% | 1.60 | 22 |
| s3 | +6.9% | 16.4% | 1.39 | 16 |
| **avg** | **+8.6%** (±3.7%) | **15.7%** | ~1.5 | **~18** |

### What it means — the under-trading is solved; the alpha gap is next
- **Behavior fixed (robustly).** 16–22 trades/seed (vs 0–4), all positive, all gate-safe, tight
  ±3.7% spread — a **stable, active, learned** policy, not a collapsed one. The relative reward +
  the [−1,1] reparam did exactly what the diagnosis predicted. This is the first RL config that
  *behaves* like a real agent across seeds.
- **Doesn't beat the rule yet** (+8.6% vs the causal ~+18%). Return ≈ the absolute-reward version,
  but now *with* participation — it learned to **act like** the rule, not yet to **out-discriminate** it.

### Deviation-alpha diagnostic — it's REWARD-bound, not capacity-bound (2026-06-10)
Before spending an LSTM, the `rl-ml-trainer` (2nd consult) called the gap reward-bound and proposed a
cheap check on the exp1 bundles: correlate each executed entry's **over/under-size vs the rule (0.20)**
with that token's **forward-24h return**. If oversizing doesn't predict the move, the agent is
deviating *without skill* → the **reward** isn't teaching discrimination. (`scripts/diag_deviation_alpha.py`.)
- **Result: corr = −0.027** (37 entries, 4 seeds) — **flat zero.** And every entry was sized **≥ 0.20**
  (0.20–0.34): the agent learned a crude "always size big," never "size by conviction," and the
  within-range sizing is **pure noise**. The flat "copy-the-rule" basin made visible.
- **Verdict:** the binding constraint is the objective's signal-to-noise on *beat the rule*, NOT the
  policy's representational power (the regime obs `btc_trend` is already present; the agent already
  out-sizes the rule and still doesn't win). **Don't buy capacity (LSTM) to escape a flat-gradient
  basin — fix the gradient first.** Caveat: tests entry-sizing only (not exits/skips), but random
  sizing is strong reward-bound evidence.

### Decided next — experiment 2: per-decision (residual) reward
Fix the credit assignment so the gradient points at *beating the rule per decision*, on the cheap MLP:
1. **Per-decision / residual credit** (highest-leverage) — reward the agent's **weight deviation from
   the rule** dotted with token returns, `Σ(agent_w − rule_w)·ret`, so the SHARED positions cancel and
   only the agent's *active bets* earn/lose. Oversizing a winner pays; oversizing a loser hurts — the
   signal the −0.027 says is missing.
2. **Rule-context obs** (the rule's current exposure) + **`norm_reward=True`** (the reward is now
   small & zero-centered).
LSTM + regime obs stay **deferred** — earned only if a clean reward still can't beat the rule.
Gate: seed-mean test **> +18%**, worst-DD **< 25%**. Mechanics → [[AI Training]].

### Experiment 2 smoke → the minimal-deviation basin → exp2b (residual + R4)
The exp2 residual 100k smoke was **alive** (action mean 0.727, 163 trades) but **under-sized the
rule** — every entry 0.03–0.12, *below* 0.20 (exp1 was always *above*). 3rd `rl-ml-trainer` consult:
this is the **minimal-deviation basin**. The dd brake (`−λ·ddpen`, still in the residual reward) is
**one-sided** — over-sizing raises variance → raises dd-penalty → negative EV, while under-sizing
lowers it. So for a skill-less agent the expected-reward *maximum* is to size *below* the rule. The
residual punishes *wrong* big bets but doesn't *require* right ones → necessary, not sufficient.

**Discrimination-headroom probe** (`scripts/probe_obs_alpha.py`, no training): do the obs features at
each ignition predict the token's forward-24h return OOS within train? **Yes — OOS IC = +0.246**,
top-vs-bottom-predicted spread **+3.26pt**; the driver is **`cush = −0.423`** (stretched ignitions
revert — size by *inverse* cushion). So the alpha **is in the obs**; the agent just isn't taught to
use it. **Reward-bound confirmed; LSTM stays deferred.**

**exp2b = residual + R4 (foregone-opportunity).** `R4 = −β·Σ max(0, rule_w − agent_w)·max(0, ret)`:
when the agent sizes *below* the rule on a token that *rose*, charge β× the surrendered upside.
One-sided (no charge for under-sizing a loser; no new over-size incentive); E[max(0,ret)]>0 always, so
it's a **strictly-negative expected penalty on under-sizing** → closes the basin. Now both deviation
directions cost a skill-less agent, so hugging the rule nets ~0 and the *only* path to positive is
**deviating correctly** (which the +0.246 IC says is achievable). **Verified:** R4 (β=0.4) drives a
min-size agent −0.155 → −0.544 while the rule-mimic stays ≈0. Sweep `... test residual` (now β=0.4) →
`ppo-event-res-test`. **Gate: seed-mean > +18%, worst-DD < 25%, AND deviation-alpha corr > 0.**

### exp2b verdict → the corner-solution finding (the real result)
β-tuned on smokes (β=0.4 → entries all *under* 0.20; β=0.8 → action mean up, return positive), then
4 seeds × 1M, frozen TEST at **β=0.8**:

| seed | test | maxDD | trades |
|------|------|-------|--------|
| s0 | +13.6% | **31.8% (DQ breach)** | 20 |
| s1 | +15.2% | 20.9% | 22 |
| s2 | +18.6% | 18.9% | 23 |
| s3 | +13.6% | 13.7% | 23 |
| **avg** | **+15.2%** (±2.0%) | 21.3% | ~22 |

corr = **+0.013** (43 entries), sizes **0.13–0.35 (mostly maxed)**. Return improved (+8.6% → +15.2%)
but **fails the gate**: avg < +18%, s0 breaches the 30% DQ, and corr ≈ 0 — the +15.2% is **beta, not
skill** (over-size everything → more return, more drawdown).

**The corner-solution finding (the lesson across all four reward variants):**

| reward | sizing | corr |
|--------|--------|------|
| exp1 relative | oversize-all (0.20–0.34) | −0.027 |
| exp2 residual β=0 | undersize-all | ~0 |
| exp2b R4 β=0.4 | undersize-all | +0.008 |
| exp2b R4 β=0.8 | **oversize-all** (beta+DD) | +0.013 |

Every reward so far penalizes/rewards sizing **magnitude**, and the agent responds by going to a
**corner** (all-small or all-big by β) — never to the `cush`-conditional sizing the probe proved is
there (**IC +0.246**). Tuning β just slides between corners; it can't manufacture *conditional*
behavior. The alpha is sitting untouched because **no reward yet pays for *rank-correct* sizing**
(size up the low-cush winners, down the high-cush losers) — only for sizing direction in aggregate.

### exp3 — demeaned-ranked residual (the corner is a functional-form problem)
4th [[rl-ml-trainer]] consult, sharpened: a reward **linear in `dev`** has a per-decision gradient
that's a *constant direction* → SGD rails every entry to a bound; β only slides between bounds. The
cure isn't another β — it's making the reward depend on the **interaction of `dev` with the obs**:

`R = Σ dev·(ret − ret_bar) − res_gamma·Σ dev²`  (`reward_mode="residual_ranked"`)

- **Demean** by the interval's cross-sectional mean → for a skill-less agent `E[ret−ret_bar]=0`, so the
  constant-drift gradient *vanishes*; the only thing left to earn is the **obs-predictable** part
  (`cush`, IC +0.246) → conditional sizing is the *only* way to score.
- **Quadratic budget** → interior optimum `dev* ∝ (ret−ret_bar)/2γ` (rank-correct), so neither corner
  is optimal. (Retires R4 — centering removes the drift-corner R4 was patching.) dd brake softened
  (`dd_lambda` 2.0→1.0); the budget caps per-name tilt → caps drawdown (the targeted fix for β=0.8's DQ).

**Preflight (`scripts/preflight_residual.py`, the check we never ran before any prior reward):** score
scripted agents on the reward landscape over real train ignitions; require the **correct-discriminator
(`dev ∝ −cush`) to be the unique argmax** with both corners ≤ 0 and an IC-hacker losing. **PASSES** —
the demean *alone* (γ=0) collapses all-big to exactly 0; the budget makes corners strictly negative;
correct-disc wins (+2.69 at **γ=0.1**, corr(dev,ret) +0.239). The corner is provably gone in the
reward *form*, before any compute. Sweep `... test residual_ranked` → `ppo-event-rank-test`. **Gate:
mean > +18%, worst-DD < 25%, and corr > +0.10** (corr is now a *success gate*, not a diagnostic).
LSTM still deferred — the alpha is in the obs; this reward makes the agent use it. → [[AI Training]].

### exp3 verdict + exp4 (entry-forward) — reward==metric, and the preflight-fidelity wall
**exp3 (residual_ranked γ=0.1), 4-seed test: +18.2% avg, all DD<25% (best yet, gate-safe) — but
corr −0.068, sizes 0.32–0.34 (oversize-all). Cornered again.** Verified the cause in-env: the env's
per-interval **universe** demean leaves the ignition-beta (all-big = +0.40 at γ=0, correct-disc ≈0) —
the **preflight (global demean) gave a false PASS**, modeling a different objective than the env.

**Root cause (5th consult):** we *grade* on `deviation_alpha` corr (`dev=size−0.20` vs the token's
**fwd-24h return**) but *trained* on held-interval-vs-universe-mean — **different objects.** exp4
fixes it: `reward_mode="entry_forward"`, `R = dev·(fwd_ret − mu_base) − γ·dev²`, credited per ENTRY,
semi-MDP-delayed `H=24` bars (causal), demeaned by the **typical-ignition** return (`_ignition_base_rate`,
not the universe). `dev·fwd_ret` **is** what corr measures → objective == metric. Meta-fix: one shared
`event_reward.entry_forward_reward()` imported by both env and `scripts/preflight_entry_forward.py`.

**But the preflight-fidelity wall is deeper than the shared function.** Faithful preflight **PASSES**
(correct-disc +4.06, unique argmax), yet the **in-env check still has all-big winning** (+0.13 vs
correct-disc −0.08). Why: **the preflight scores all ignitions with free sizing; the env scores the
agent's *realized* entries** — a funding/selection/event-dependent subset, sizing constrained by cash
+ rotation. Different `(dev, fwd)` distributions. **So the preflight can't be made faithful by sharing
the reward fn — the only faithful preflight is running the env, and that's the in-env check, which is
not clearly passing.** exp4 reward is built + correctly aligned (14 env tests pass), but **gating on
the in-env landscape (not the preflight)** is the lesson. Decision pending: sweep (definitive but
likely another beta-corner) vs redesign the realized-entry dynamics (6th consult). → [[AI Training]].

### exp4 blocker resolved — the entry-sizing lever has NO headroom (the make-or-break probe)
6th `rl-ml-trainer` consult. Before sweeping exp4, ran the decisive check the whole arc skipped:
restrict the headroom probe to the subset the agent **actually sizes** (`scripts/probe_subset_ic.py`,
nested L0→L2 through the env's real event engine, OOS temporal holdout). The result kills entry-sizing
as the lever and explains every prior corner:

| level | what | n events (train, ~128d) | OOS combined IC | univariate cush-IC (in-test) |
|-------|------|------|------|------|
| L0 all ignitions | what `probe_obs_alpha` scored | 2176 | **+0.246** | −0.285 |
| L1 +in vol-top-k universe | the candidate pool | 960 | **+0.103** | −0.250 |
| L2gate +cooled & reclaimed | the entry-gate set (cash-decoupled) | **39** | noise (+0.07..+0.59, spread flips sign by horizon) | −0.39..−0.75 |
| L2real +cash-throttled | **realized entries** | **14** | — (too few to measure) | — |

**Three findings, in order of force:**
1. **The decision set is nearly empty.** The rung-0 entry gate fires **39 times in ~128 days** (≈ one
   fundable decision every 3.3 days); only **14** survive the cash/rotation throttle. Entry-sizing
   discretion operates on ~1–2 dozen decisions — **far too few for PPO to learn a conditional `cush→size`
   map.** This is why every reward cornered: with ~20 gradient-bearing entries per episode the policy can
   only learn a *scalar* (all-big / all-small), never a *function* of cush. The corner was never a reward
   bug — it's a **sample-starvation** bug the reward shape can't fix.
2. **L1 alone halves the IC** (+0.246→+0.103): restricting to the vol-top-k universe removes most of the
   discriminative signal `probe_obs_alpha` advertised.
3. **The alpha *sign* survives the gate** (cush stays −0.4..−0.75 in-test) but the **OOS combined IC at the
   gate is noise** — n_test = 12–20, spread flips sign across horizons {12,24,48} and holdouts {0.3,0.5}.
   There is no measurable, stable headroom to discriminate *among* rung-0's selected entries. rung-0's
   `cush>0 & rising & ema_up` gate **already harvested the entry-cush alpha** — exactly the structural
   worry in the hypothesis, now confirmed empirically.

**This rediscovers TradeSim's #1 hard lesson** ([[AI Training]]): *entry timing never clearly beat random;
exits/risk-management carried performance.* The arc spent exp1→exp4 trying to make RL out-discriminate the
rule on **entry sizing** — the one lever that lesson says has no edge.

**The corrected process gate (replaces the free-ignition preflight, which false-PASSed exp3 AND exp4):**
no sweep until an **in-env landscape** check — scripted agents (rule-mimic / all-big / all-small /
correct-disc) scored through the actual env on **total reward** — shows correct-disc the **unique argmax**
with both corners ≤ rule-mimic. The preflight's fatal flaw was structural: it scores all ignitions with
free sizing; the env scores the funding/selection-constrained realized subset (a different `(dev,fwd)`
distribution). Only running the env is faithful.

**The exit lever has the SAME starvation** (checked, not assumed). Under realistic play (rule cuts each
stop), exit prompts = **exactly 39** — one per position, because you can't exit more positions than you
open. (A naive replay showed 5564 "exit events," but that was the degenerate all-override path re-prompting
the same un-closed positions every bar — an artifact, not decisions.) And the override-value IC on the
exit subset is also flat: `corr(giveback, post-exit fwd-24h) = −0.058`, post-exit move 51% up / +1.11% mean
— a coin flip. **Both event-level discretion levers (entry-size, exit-override) are capped at ~39 sparse
decisions with no measurable obs-conditional alpha.** The event skeleton itself is the ceiling.

### Decided next — exp5: the bottleneck is rung-0's sparsity; loosen the SELECTION, don't tune discretion
The arc's framing — "RL learns the *discretion* rung-0 hard-codes" — is the trap: rung-0's strict gate
(`surge≥2.5 & rising & cush>0 & ema_up`, +cooldown +dead-zone +loser-rotation) leaves only **39 decisions
in 128 days**, far too few to learn *any* conditional map, and its `cush>0` filter already spent the
entry-alpha. The lever must move to where decisions are **plentiful** and alpha is **unspent**:
1. **RL as SELECTION / sizing over the candidate POOL (primary).** L0's +0.246 cush-IC lives in the
   ignitions rung-0 *discards*. Reframe the action from per-isolated-entry sizing to a **cross-sectional
   ranker**: at each bar, over all in-universe ignition candidates, allocate by a learned `f(cush, surge,
   …)` (loosen rung-0's hard gate into a soft, learned score). This (a) multiplies the decision count by
   un-gating, (b) targets the IC that survives OOS (cush, robustly negative), and (c) is a *rank* objective
   — structurally immune to the magnitude-corner that broke exp1–4. The honest baseline stays rung-0.
2. **Exit-override is retired** alongside entry-sizing — same 39-decision ceiling, flat IC.

**Process gate (the structural fix):** no sweep until an **in-env landscape** check — scripted agents
scored *through the env* on total reward — shows the correct discriminator the **unique argmax**, both
corners ≤ rule-mimic. The free-pool preflight is abandoned (it false-PASSed exp3 AND exp4). Gate unchanged:
seed-mean test **> ~+18%**, worst-seed maxDD **< 25%**, selection-IC corr **> +0.10**, gated in-env.
Probe: `scripts/probe_subset_ic.py`. Full design in [[AI Training]].

## Thesis (the lens for reading all of the above)

This is volatile shitcoin/vaporware trading, **not the S&P 500**. **Realized-volatility capture is
the intended edge** — the agent should lean into the swings. The job is to harvest that volatility
**while staying under the ~30% drawdown DQ gate**, scored on live PnL (June 22–28). High returns
are not suspect; getting DQ'd is the failure. See [[Market Conditions]].

> **Data-realism audit (resolved).** The +100–200% returns were stress-tested: per-token PnL
> reconciles to the equity curve (not a frontend bug); SIREN's violent path is *real* data (the
> −81% bar traded ~900× median volume — a genuine liquidation event, CMC #72, vetted at $1.1M/24h);
> and the AMM friction (~0.36% on a $5k trade vs a $9.2M pool) is defensible constant-product math.
> The returns are real within a mostly-sound simulation. Residual realism gaps to tighten later:
> static liquidity that doesn't collapse under stress, and concentration (one token can dominate
> the portfolio). Details in [[Build Log]].
