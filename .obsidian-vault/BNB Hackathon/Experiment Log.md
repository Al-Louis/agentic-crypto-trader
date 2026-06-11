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

### exp5 in-env landscape gate — **PASS** (the first gate the arc has cleared)
Built `scripts/preflight_selector.py`: the structural fix in code. It OLS-fits a forward-return
predictor on the **early** in-universe ignitions (causal, 60% temporal holdout), then runs four
scripted agents — rule-mimic / all-big / all-small / **correct-selector** (sizes by the OOS
prediction's rank) — **through the real `entry_forward` env** with `ungate=True`, summing the *exact*
reward PPO maximizes. No proxy: the env scores realized, funding-constrained entries.

In-universe ignitions: **960** (vs rung-0's 39). OOS coef `[1,cush,surge,btcT] = [−0.008, −0.336,
+0.008, +0.977]` — cush stays robustly negative; the predictor has real OOS signal. Total in-env
reward across γ (deviation-budget weight):

| γ | rule-mimic | all-big | all-small | correct-selector | gate |
|---|-----------|---------|-----------|------------------|------|
| 0.05 | 0.0000 | **+0.0039** | −0.7599 | +0.1390 | FAIL (all-big > rule by 0.004) |
| **0.10** | 0.0000 | **−0.1231** | −1.1599 | **+0.1066** | **PASS** |
| 0.20 | 0.0000 | −0.3772 | −1.9599 | +0.0417 | PASS (selector edge shrinking) |
| 0.40 | 0.0000 | −0.8853 | −3.5599 | −0.0881 | FAIL (budget over-taxes the selector) |

**This is the first time in the entire arc the in-env gate has passed.** At **γ=0.10**, correct-selector
is the **unique in-env argmax** (+0.107) and **both corners are strictly penalized** (all-big −0.12,
all-small −1.16). Contrast exp3/exp4, where all-big *dominated* (+0.13..+0.40) and the skilled agent
was ~0 — the magnitude-corner that broke every prior reward. **The fix was structural, not a reward
tweak:** un-gating moves the decision from rung-0's 39 cash-decoupled entries (where L0's +0.246 cush-IC
had collapsed to noise — exp4's sample-starvation finding) onto the **960-event in-universe pool where
the +0.103 IC survives**, and the γ=0.10 quadratic budget taxes the residual all-big selection-beta below
rule-parity. The reward landscape now has a single optimum, and it's the skilled cross-sectional selector.

γ=0.40 over-taxes (selector goes negative) → the budget is genuinely sizing the deviation, not just
killing corners. **exp5 sweep config:** `--reward-mode entry_forward --ungate --fwd-horizon 24
--res-gamma 0.1` (`REWARD_MODE=selector` in `run_eventrung_sweep.sh`, run-id `ppo-event-sel`). The gate
is the green light the arc earned the hard way; **PPO now has a landscape where the env itself rewards
discrimination.** Sweep next (val first, then frozen test). Gate for the verdict unchanged: seed-mean
test **> ~+18%**, worst-seed maxDD **< 25%**, selection-IC **> +0.10**.

### exp5 val sweep — **first sweep to beat the rule, off-corner, on every seed**
4 × 1M timesteps, sequenced, val split, `REWARD_MODE=selector` (run-ids `ppo-event-sel-s{0..3}`,
each confirmed 1.00M steps complete). HEAD `36ceb00`.

| seed | policy | Sharpe | maxDD | vs rung-0 rule −9.4% | eval action (mean / range) |
|------|--------|--------|-------|----------------------|----------------------------|
| s0 | −6.1% | −2.62 | 10.2% | BEATS | −0.103 / [−1, +1] |
| s1 | −3.8% | −1.06 | 13.1% | BEATS | −0.008 / [−1, +1] |
| s2 | −5.0% | −2.17 | 9.3% | BEATS | −0.125 / [−1, +1] |
| s3 | −3.8% | −1.06 | 13.1% | BEATS | −0.008 / [−1, +1] |
| **mean** | **−4.7%** | | **all < 25%** | **+4.7pp over rule** | full-range, ~neutral mean |

**Three firsts for the arc, simultaneously:** (1) the in-env landscape gate passed (γ=0.10);
(2) the trained policy stayed **off the corner** — full [−1,+1] action range with a near-neutral
mean on every seed, i.e. it *discriminates* (sizes some entries big, some small) instead of railing
to all-big/all-small as exp1–4 did; (3) it **beat the rung-0 baseline on all 4 seeds**, drawdowns
roughly half the 25% gate. The ungate + entry_forward + γ=0.10 landscape produced a policy that
learns selection rather than a scalar.

**Honest caveat — val is a down regime** (rule −9.4%, all returns negative). The selector wins
**defensively**: it down-sizes/skips the worst ignitions and loses ~half as much, not by profiting.
Robust relative edge (4/4), but "loses less," not "makes money." Absolute verdict awaits frozen test.

**The smoke's events=3172 vs the sweep's ~115 is resolved:** the smoke (40k, undertrained) sat in the
exit-reprompt degeneracy (un-closed positions re-prompted every bar — the artifact the L0→L2 analysis
flagged); the 1M-trained policies exit cleanly, so events collapse to the real val ignition count
(~100 entries + clean exits). The trained policy escaping that loop is itself a positive signal.

Next: (a) deviation-alpha diagnostic on a trained seed to confirm the sizing genuinely correlates with
forward returns (mechanism, not just regime luck — `diag_deviation_alpha` / `rl_diagnose`); (b) the
frozen **test** sweep — the one-shot final verdict (meta-overfit guard: test reserved until now).

### GATE 1 — discrete + risk-parity + honest DQ gate: the static-risk-posture wall
After the exp1→exp5 reward-proxy drift, the substrate was rebuilt (discrete actions, universe knob,
risk-parity per-token caps; the honest gate made structural in code — judge vs rung-0 **and** Buy&Hold
**and** Random, **per regime**, under a hard 30% DQ). GATE 1's question: can a discrete policy + the
simplest honest PnL reward (`relative`) beat the baselines on **both** held-out regimes (val + test)?

Two variants, 4×1M seeds each, seed-mean (single-seed RL is unstable — the mean is the read):

| | val | test | note |
|---|-----|------|------|
| **voltopk RL** (concentrated top-8) | +2.6% | **+9.9%** | s0 +18.8% was a lucky outlier; 3/4 seeds negative on val |
| **broad RL** (k=12 risk-parity, stables+monsters) | −2.8% | −0.2% | diversified *away* from the pump → return-drag |
| rung-0 (canonical voltopk-8) | −9.4% **(DQ'd, 31% DD)** | **+29%** (17% DD) | blew the gate on val; won on test |
| Buy&Hold (risk-parity, agent's universe) | +8.7% | +15.2% | the bar both RLs fail |

**Both variants FAIL the gate** — but the failure is structural, not a policy bug, and it carries the
real finding: **no static risk posture wins both regimes.**
- **On val**, concentrated rung-0 **blew a 31% drawdown (DQ'd) and lost −9.4%**; the risk-managed RL
  *beat* it (positive, single-digit DD). **Risk management HELPED — survived where the rule got DQ'd.**
- **On test**, the monsters pumped, surviving rung-0 made **+29%**, and the risk-parity caps that protect
  val *missed the upside* → RL underperformed. **Risk management HURT.**

Concentration blows the DQ gate when wrong; diversification misses the upside when concentration is
right. The data measures this tradeoff cleanly. **This is exactly what regime-awareness is for** —
modulate risk by regime (concentrate to harvest pumps when crash-risk is low, de-risk when high). A
*static* policy can't; a *regime-adaptive* one can. GATE 1 is not "RL can't work" — it's "a fixed risk
posture is the wrong target."

**Two structural gaps block measuring the RL's value, both now the active build:**
1. **No regime signal in the obs** — the policy can't see whether it's a pump or a crash setup. `btc_trend`
   misleads (the alts decouple from BTC). → add a **universe-breadth** feature.
2. **No alt-crash in the data** — every split has the alts rising/flat (only BTC fell); the gate
   structurally rewards concentration because nothing crashes. We're grading a survival strategy with
   nothing to survive. → **synthetic alt-crash injection**, so de-risking can finally *pay*.

Next: build the crash scenario + the regime-breadth obs feature, then gate a regime-adaptive policy —
the config where "embrace the volatility, survive the drawdown" becomes measurable. → [[AI Training]].

### GATE 2 — regime-adaptive (breadth obs + crash): the mechanism works, the adaptation doesn't (yet)
The two structural gaps GATE-1 exposed, built and gated (full build in [[AI Training]]): a
**regime-breadth** obs feature (fraction of the traded basket above its EMA — the alts' own regime,
decoupled from BTC) and **synthetic alt-crash injection** (`sim/crash.py`: a SIREN-scale liquidation,
correlations spike to 1). GATE-2 config: discrete + broad k=12 + risk-parity + breadth obs + **4
training crashes** + a held-out **crash regime** (a crash spliced into test, where Buy&Hold loses −82%).
4×1M seeds, graded on **three regimes** (val bull / test pump / crash):

| regime | s0 | s1 | s2 | s3 | mean | vs Buy&Hold |
|--------|----|----|----|----|------|-------------|
| **val** (bull) | −5.9% | −5.4% | −7.5% | −8.8% | **−6.9%** | B&H ≈ +27% → loses badly |
| **test** (pump) | −1.8% | +2.5% | −5.6% | −1.1% | −1.5% | ~flat |
| **crash** | −1.6% (DD 5.5%) | **+5.8%** (Sh 3.97, DD 3.3%) | −13.0% (DD 13.6%) | −28.3% (**DD 34.7% DQ**) | survives 3/4 | B&H −82% → **+70..88pp** |

**The win — crash-survival is real, and it's the first RL behavior no static strategy can match.**
3 of 4 seeds read the breadth collapse and **de-risked**: s0/s1 held drawdown to **3–5% in an 82%
crash** that nearly DQ'd even rung-0 (28% DD), and **s1 made +5.8% — positive in the crash**. The
breadth feature + crash training produce learnable regime-aware de-risking. Thesis validated in
principle.

**The gap — the policy learned defensive-EVERYWHERE, not regime-ADAPTIVE.** DDs are 4–13% in *every*
regime: it's uniformly cautious. That caution saves the crash but **kills the bull** — val −6.9% while
the basket rose +27% (it *lost money in a rising market* by holding cash). It uses breadth to stay
safe, not to *ramp up* when breadth is high. And it's not robust (s3 DQ'd at 34.7%; high seed variance).

**Both the over-defensiveness and the inconsistency say the same thing: the policy must *condition*
risk on breadth more sharply.** Two levers: (1) **rebalance the reward toward bull-harvest** — the
heavy `dd_lambda=1.0` + 4 crashes taught blanket caution; lower the drawdown fear so it sizes up in
high-breadth bulls (one cheap sweep); (2) **recurrence (RecurrentPPO)** — breadth is a *time series*
(sustained-high = harvest, deteriorating = de-risk early), which a feedforward snapshot can't track
but an LSTM can; now correctly sequenced (a feedforward champion exists to A/B against). → [[AI Training]].

### ▶ NEXT EXPERIMENT (decided 2026-06-10) — reward-rebalance: the reward-vs-features decider
*Relayed from the signal-research branch (the intraday breakout-reversal handoff — [[Trading Strategies]]).
Two owners split on GATE-2's bull-loss: `rl-ml-trainer` says it's a **reward** problem (the `dd_lambda=1.0`
taught blanket caution), `market-indicator-expert` says it's an **information** problem (no harvest signal
to size up on). One cheap run adjudicates — run it before any feature code.*

**The run:** GATE-2 config **frozen** (discrete, broad k=12, risk-parity caps, breadth obs OBS_DIM-13, 4
training crashes, held-out crash regime) — change **only `dd_lambda 1.0 → 0.5`**. 4×1M seeds, graded
val(bull) / test(pump) / crash. **Zero new code.** (`REWARD_MODE` stays the GATE-2 reward; just the lambda.)

**PASS** = seed-mean beats Buy&Hold + Random + surviving rung-0 on **all three** regimes; worst-seed maxDD
**< 30% everywhere**; **crash survival retained** (crash DD not regressing from GATE-2's 3–5%). Concretely:
val moves from −6.9% toward the basket's +27% *without* crash DD > ~10% on any non-DQ'd seed.

**It reads the cause:**
- ramps the bull cleanly → **reward** problem (rl-ml right; harvest features may be unnecessary).
- ramps the bull but **blows the crash** → the brake was load-bearing → switch to a *budgeted* reward
  (`residual_ranked` γ≈0.1), do **not** cut `dd_lambda` blindly.
- stays defensive → **information** problem (market-indicator right) → add harvest obs (13→17: r24/r3d/r7d +
  breakout-distance), but only after `scripts/probe_subset_ic.py` shows incremental-over-`cush` OOS IC.

**Safety (both agents, non-negotiable):** keep **risk-parity caps ON**; judge on **worst-seed crash DD**,
never the mean — concentration is what DQ'd GATE-1 rung-0 (31%) and GATE-2 s3 (34.7%). Full plan +
lever sequence: [[AI Training]] §"Post-GATE-2 plan". Then features (gated) → RecurrentPPO (last).

**RESULT (2026-06-10, `ppo-event-g2b`, 4×1M):** it's an **INFORMATION problem** — cutting `dd_lambda`
did **not** ramp the bull.

| regime | GATE-2 (λ=1.0) | g2b (λ=0.5) | read |
|--------|---------------|-------------|------|
| val (bull) | −6.9% | **−6.7%** | **unchanged — stays defensive even with the brake halved** |
| test (pump) | −1.5% | +3.1% | improved |
| crash | 3/4 survive (s3 DQ 34.7%) | **4/4 survive** (worst DD 14.8%, s1 +7.0% / s3 +12.5%) | held + more robust |

The bull-loss is **not** a reward/brake problem (`rl-ml-trainer`'s hypothesis) — halving the drawdown
penalty left val flat at −6.7% while crash survival actually *improved* (no DQ). The policy lacks the
**harvest signal** to size up in bulls (`market-indicator-expert`'s hypothesis confirmed). **Lever 2
next: the harvest obs (13→17: r24/r3d/r7d + breakout-distance), but GATED — no sweep until
`scripts/probe_subset_ic.py` shows incremental-over-`cush` OOS IC on the ungated ~960-event pool.**
Signal spec: [[Trading Strategies]] §"Intraday breakout-reversal".

**LEVER-2 GATE RESULT (2026-06-10, `scripts/probe_harvest_ic.py`) — PASS.** On the ungated in-universe
pool (960 events, k=8; baseline `[cush,surge,btcT]` OOS IC **+0.103**, matching exp5 exactly): adding
the harvest features lifts combined OOS IC to **+0.167 (incremental +0.063)**, well over the +0.02 bar.
The breakout bucket (`r3d>0 & r7d<0`) is a real momentum-continuation sub-population: **+0.38% mean fwd
vs the pool's −2.02%** (+2.4pp). **Decomposition refines the spec:** the *linear* lift comes from the
**momentum** features (`r24/r3d/r7d`: +0.103→+0.176), NOT the breakout-distance (`bkout_s/m` alone
*degrades* to +0.074). Each harvest feature is individually *negative* (universe mean-reverts) but
helps in combination — the nonlinearity the spec predicted. The breakout-distance features are a
**nonlinear hypothesis only the trained MLP can test** (a linear probe can't adjudicate them). → Build
lever-2 obs and A/B vs the GATE-2 champion; let the **net-of-cost** gate (not IC) decide success — the
breakout edge (+0.38% < ~1% round-trip) lives in the convex tail the risk-parity caps must harvest.

**LEVER-2 A/B RESULT (2026-06-10, `ppo-event-l2`, momentum obs 13→16, 4×1M) — clean NEGATIVE.**
Treatment (g2b + r24/r3d/r7d) vs the on-disk g2b control:

| regime | g2b control | lever-2 | read |
|--------|-------------|---------|------|
| val (bull) | −6.7% | **−8.0%** | NO lift — slightly worse |
| test (pump) | +3.1% | +6.5% | better |
| crash | +0.9%, **4/4 survive** (worst DD 14.8%) | −11.1%, **s3 DQ'd 41.1%** | worse |

The harvest momentum obs **failed the A/B** — no bull lift on the seed-mean AND degraded crash
survival. (The smoke's +13.4% val was froth: one undertrained seed — the "+198% was froth" lesson.)
It adjudicates the agent split: **momentum-only is too BLUNT** — "size up when recent returns are
positive" fires *before crashes too*, so it ramped into the crash (s3 DQ) without reliably ramping the
bull. That's `market-indicator`'s critique (→ the *selective* breakout-distance, lever-2b), but it also
vindicates `rl-ml`'s cost-wall worry: the bucket edge (+0.38%) is **below the ~1% round-trip**, so
net-of-cost there may be no harvestable bull edge to capture. **Open fork:** (a) lever-2b — breakout-
distance + fresh-flag (the *selective* signal, ramps only on fresh breakouts, less crash-correlated);
(b) accept the cost-wall — the bull may not be beatable net-of-cost by event-driven trading, and g2b's
**crash-survival (4/4, the thing no static posture does)** is the real asset (RL-as-crash-insurance).
**Persistent pattern across GATE-2 / g2b / lever-2: the policy survives crashes but cannot beat
Buy&Hold in the bull.** Decision pending (loop both agents with this result).

> ## ⚠ ALL RESULTS ABOVE (GATE-1, GATE-2, g2b, lever-2) ARE INVALID — env exit bug (fixed `8ccad69`)
> Found 2026-06-10 by inspecting the agent's trades: `EventRungEnv._scan_bar` stopped off `ref_px`
> (the **entry** price), not the trailing `peak_px` — so a winner gave back its **whole run** before
> exiting ("sell the bottom"). Proof: env rule-mimic returned **−27.1%** on val where the canonical
> tested `run_rung0` returned **−9.4%** (18pp gap); after the fix, matched-causal-universe **parity**
> (−5.1% vs −4.6%). Also fixed: `rung0_baseline` used `select_vol_tokens` (full-window std = LOOKAHEAD),
> inflating the bar (test +29% lookahead → +18% causal). **Every model above trained in the broken env
> — the "survives crashes, can't beat Buy&Hold in the bull" plateau is very plausibly this bug giving
> back every winner.** TradeSim's #1 lesson (exits carried performance) — and our exit was the broken
> one. Re-run everything on the fixed env. Regression: `test_trailing_stop_fires_off_peak_not_entry`.
> See [[env-exit-stop-bug-fixed]]. The ENTRY side (agent rides rung-0's momentum, can't buy dips) is the
> next frontier — the user wants the agent to own entry/exit TIMING (buy low, sell high), not just sizing.

## Standings — g2b re-run on the FIXED env (2026-06-10, `ppo-event-g2b` @ `e466f0e`)

The first valid result post-`8ccad69`. Config identical to g2b (discrete, broad k=12, risk-parity,
breadth obs OBS_DIM-13, 4 training crashes, `dd_lambda` 0.5), 4×1M, published over the old run-ids
(desktop HEAD `e466f0e` predates the `ec1e487` naming fix — provenance commit distinguishes them).
Baselines are now causal (the lookahead universe-selection fix): rule val **−4.6%**, test **+18.0%**.

| regime | s0 | s1 | s2 | s3 | mean | rule | B&H |
|--------|----|----|----|----|------|------|-----|
| val (bull) | −8.0% | −7.1% | −9.8% | −5.3% | **−7.6%** | −4.6% | **+27.5%** |
| test (pump) | −3.3% | −3.2% | −0.9% | +6.2% | **−0.3%** | **+18.0%** | +1.5% |
| crash | −1.9% (DD 5.4%) | −3.5% (DD 5.6%) | **−59.5% (DD 63.7% DQ)** | −14.0% (DD 25.8%) | survives **2.5/4** | ~−6% | −82% |

### Verdict — the env bug was NOT the plateau's cause, and crash survival was partly an artifact

1. **The bull-loss persists on the fixed env** (−6.7% invalid → −7.6% fixed; statistically unchanged).
   The invalidation note's hope ("the plateau is very plausibly this bug") is now tested and **falsified**.
2. **Crash survival REGRESSED** (4/4 → 2/4 comfortable; s2 catastrophically DQ'd at 63.7% DD, s3 grazing
   at 25.8%). The pre-fix uniform defensiveness was partly trained *by the broken env* (sell-the-bottom
   exits made aggression unsurvivable). On the honest env the policies are more aggressive and less
   reliably safe. "RL-as-crash-insurance" is weaker than the invalid data suggested.
3. **The structural read (the load-bearing fact): the RULE ITSELF loses the bull** — rung-0 makes −4.6%
   on val while the basket B&H makes +27.5%. The event skeleton can only be long via ignition entries
   and exits on trailing stops; in a grind-up bull it structurally bleeds vs holding. The agent inherits
   that ceiling: **no reward/obs tweak inside the ignition-gated skeleton can reach a gate that demands
   beating B&H in the bull**, because the action space cannot express "just stay long." This single
   constraint explains the whole GATE-1→lever-2 plateau ("survives crashes, can't beat B&H in the bull").

### SIREN trade forensics (g2b-s3, `scripts/diag_token_events.py`) — the discretion VETOES the rule
The user inspected s3's SIREN chart and called the behavior non-rung-0. Confirmed, bar by bar:
the ignition fired **19 consecutive hours on Mar 22** (surge to 41×; the rule entered 05:00, exited
19:00 — almost exactly the user's discretionary read), again Mar 25 17:00, Apr 4 15:00–Apr 5 02:00,
and Apr 16 15:00–Apr 17 07:00 (surge to 17.6×). **The agent was prompted ~55 times and skipped every
strong ignition; its single buy was the WEAKEST prompt on the board** (Apr 15 14:00, surge 2.6× —
barely over the 2.5 threshold). It then **overrode the exit prompts through the Apr 17 crash** (each
override re-anchors the trailing stop lower) and bled out via geometric 1/3-trims from Apr 20,
selling below entry after riding a >+100% gain round-trip. So both discretion levers RL owns —
skip-entry and override-exit — are used to do the *opposite* of what made rung-0 work: it skips the
rule's winners and un-cuts the rule's losers. "86/86 buys on ignition bars" (the gating proof) and
"the policy doesn't trade like rung-0" are both true: the skeleton bounds what it CAN do; the learned
discretion vetoes what it SHOULD do. This is GATE-2's "defensive-everywhere" at single-trade
resolution, and it makes the next design constraint concrete: **the neutral/default action must
EXECUTE the rule's decision** (skip/deviate must be earned per-decision), not veto it for free.

### Decided next — rung-1b rule-default discretion (gates A/B/C PASSED, ready for the desktop)
The forensics' design conclusion, built and gated 2026-06-10 (spec: [[AI Training]] §Rung-1b).
**Gate A — skeleton oracle ceiling (`preflight_skeleton_ceiling.py`, run FIRST, zero env code):**
hindsight-greedy discretion through the g2b env scores **val +74.6% / test +45.7%** (vs B&H +27.5%
/ +1.5%, rule −4.6% / +18%) — the skeleton's ceiling clears the honest gate ~3×, so the plateau is
the *discretion*, not the event set. Decomposition: entries-only +7.3% val — **the exits carry the
ceiling** (TradeSim's lesson, measured). Skip-all = exactly 0% (the g2b policy's de-facto floor).
**The build:** `rule_default` (discrete idx 0 EXECUTES rung-0 — entry at rule sizing, exit full
cut; deviations ½×/skip/2× and trim/hold are explicit), **no peak re-anchor** on override/trim
(kills the s2 stop-ratchet), `exit_commit=12` (an exit decision commits — no per-bar liquidation
drip), `dust_usd=10` (no sub-$1 gas-bleed tail), `--rule-prior 2.0` (+logit bias on idx 0 at init
so the untrained policy ≈ the rule). Reward/config otherwise g2b-frozen (one variable). 258 tests.
**Gate B — parity:** all-default through the env tracks the uncapped mirror within −0.5/+0.8pt;
the capped gap (−3.7pt val / **+3.6pt test**) is the risk-parity caps working. **Gate C — in-env
reward landscape:** PASS both splits — oracle-24h is the unique argmax by +0.744 (val) / +0.405
(test); both corners far below; all-max is hammered (−40 reward, test). Oracle through the rd env:
**+77.3% val / +44.4% test at ~12% maxDD** — the mechanics don't cost the ceiling.
**Sweep (pending go):** `nohup bash scripts/run_eventrung_sweep.sh 1000000 "0 1 2 3" val
ruledefault > runs-rl/ruledefault.log 2>&1 < /dev/null &` → run-ids `ppo-event-rd-<sha>-s{0..3}`.
Verdict gate unchanged: seed-mean beats B&H + Random + surviving rung-0 on val AND test AND crash,
worst-seed maxDD < 30% everywhere.

## Standings — Rung-1b rule-default sweep (`ppo-event-rd-df943bf`, 4×1M, 2026-06-10)

| seed | val (bull) | test (pump) | crash | trades |
|------|-----------|-------------|-------|--------|
| s0 | −7.5% (DD 9.9%) | −1.7% (DD 2.0%) | −3.0% (DD 8.6%) | 30 |
| s1 | −6.4% (DD 17.3%) | **+28.4% (DD 18.0%) — PASSES the full test gate** | **+28.8% (DD 17.7%) — PASSES** | 41 |
| s2 | −9.9% (DD 21.6%) | +1.4% (DD 7.0%) | −5.3% (DD 12.8%) | 61 |
| s3 | −6.6% (DD 6.7%) | +4.7% (DD 11.1%) | +8.8% (DD 10.2%) — PASSES | 48 |
| **mean** | **−7.6%** | **+8.2%** | **+7.3%, 4/4 survive (worst DD 17.7%)** | |
| bars | rule −4.6% / B&H +27.5% | rule +18.0% / B&H +1.5% | rule ~−6% / B&H −82% | |

### Verdict — the veto pathology is FIXED; the val bull-wall is now isolated and precise

- **The behavior the substrate was built for, delivered.** s1 bought SIREN **Mar 22 05:00** (the
  exact bar the user called, the rule's entry), rode it to the **Mar 22 19:00** rollover (+59% in
  14h, the rule's exact exit), took the Mar 25 17:00 re-ignition, cut its loser cleanly. 2 buys /
  2 sells — no skip-everything, no dust tail, no weakest-prompt-only buys. Forensics on the record.
- **Crash robustness is structural now: 4/4 survive, worst DD 17.7%** (g2b-fixed: s2 DQ'd at
  63.7%). The no-ratchet + exit-commit fixes eliminated the blowup mechanism, and s1 made +28.8%
  IN the crash regime. Test mean −0.3% → **+8.2%**; **s1 is the first seed in the project to PASS
  a full per-regime gate — and it passed two (test AND crash).**
- **val is UNCHANGED (−7.6%, third consecutive config at −7±1%) — and that is now informative,
  not mysterious.** With the veto gone the policy trades ≈ the rule, and the rule nets −4.6% on
  val: its wins (SIREN +59%) are bled away by its other entries while B&H just holds the +27.5%
  basket. The oracle (+74.6% val) closes that gap only by *per-decision discrimination* — skip the
  rule's losing ignitions, hold winners longer. Gate C proved the reward now pays for exactly
  that; PPO didn't find it from the current 13-obs at 1M steps. **The bottleneck has moved from
  the substrate (fixed) to the per-prompt signal.** Overall honest gate: FAIL (val B&H binds).
- Next-lever fork (pending decision): (a) **harvest/breakout obs on the rd substrate** — the
  lever-2 A/B was invalid (broken env) AND bolted onto a veto-happy policy; its probe gate
  (`probe_harvest_ic`, +0.063 incremental OOS IC) already PASSED, and rd is the first substrate
  where "take this ignition bigger" is expressible; (b) more capacity/steps (only after obs);
  (c) seed variance (s1 vs s2) remains the deployment risk either way — champion is the seed-mean.

### rd8 / rd8tp launched (2026-06-10, @ `a4132cc`) — the user's forensic feedback, built and gated
User review of rd-s3 drove two changes: **(1) voltop8** — the calm half of the broad-12 universe
(XRP/LINK/LTC/BabyDoge/SFP…) bleeds in chop and shouldn't be traded → `universe_mode=voltopk k=8`;
**(2) take-profit prompts** (`tp_rungs 0.25/0.5/1/2`) — the env structurally could NOT sell into
strength (exit prompts fire on weakness only); now a position crossing an unrealized-gain rung
prompts once, default idx0 = let-it-run (rung-0 preserved), selling is the learned deviation.
**Gates B+C PASS for both configs; the tp prompts RAISE the oracle ceiling and HALVE its DD:**
val +74.8% (DD 12.1%) → **+95.5% (DD 7.1%)**, test +65.3% → **+79.5% (DD 6.1%)** — the user's
"take profit on the way up" read confirmed at the ceiling. Sequenced A/B chain on the desktop:
`ppo-event-rd8-a4132cc-s{0..3}` then `ppo-event-rd8tp-a4132cc-s{0..3}`, vs `ppo-event-rd` (broad
k12) as control. Known eval artifact (NOT a live limitation): the first 168 bars of every eval
window are signal warmup — entries there (B Mar 16, early BANANAS31) are unreachable in eval but
live trading always has full history; an `--eval-prepad` (seed signals from the prior split's
tail) is queued if we want the eval window fully tradeable. Sub-hourly execution: real but
deferred (hourly pipeline end-to-end; minute data exists locally; revisit post-PoC).

## Standings — rd8h (`ppo-event-rd8h-53409b4`, voltop8 + tp + harvest obs + eval-prepad, 4×1M)

**⚠ New windows:** `--eval-prepad` makes every eval window tradeable from bar 0 AND moves the causal
universe pick to the last pre-window bar — so the bars are NOT comparable to any prior round. On the
prepad windows the rung-0 rule is very strong: **val −0.6% / test +89.3% / crash +62–64% (surviving)**;
B&H val +17.1% / test +58.6% / crash ~−74%.

| seed | val | test | crash | trades |
|------|-----|------|-------|--------|
| s0 | −6.5% (DD 13.2%) | +30.0% (DD 4.4%) | +12.1% (DD 9.4%) | 42 |
| s1 | −4.3% (DD 13.8%) | +25.4% (DD 6.4%) | +13.0% (DD 11.6%) | 33 |
| s2 | −4.1% (DD 8.8%) | +19.8% (DD 12.2%) | +10.7% (DD 16.2%) | 51 |
| s3 | −9.5% (DD 20.7%) | +21.8% (DD 6.8%) | +25.8% (DD 4.1%) | 38 |
| **mean** | **−6.1%** | **+24.3%** | **+15.4%, 4/4 survive** | |

### Verdict — FAIL (every seed, every regime), and the failure mode is now named: the DIET-RULE equilibrium
All seeds positive on test/crash with low DDs (4–16%) — but capturing only ~a quarter of the rule's
return everywhere. The policy is a *scaled-down* rule: right direction, fractional magnitude. The
mechanism is visible in the reward arithmetic: with a **relative** reward, matching the rule = 0, and
the **`dd_lambda=0.5` penalty still charges the agent for the rule-sized drawdown path** — so
under-sizing trades a small relative-return loss for a larger dd-penalty saving. Under-sizing is the
reward-optimal policy for a non-discriminating agent: **the dd penalty double-counts risk that the
substrate already bounds** (rule-default + risk-parity caps + trailing stops held worst-seed DD to
20.7% this round, far under the 30% DQ). The harvest obs did not produce discrimination (3rd
no-result for obs levers). **Next single-variable lever: `dd_lambda 0` on this exact config** — let
the relative reward alone drive (parity=0, beat-the-rule positive), keep the DQ enforcement in the
gate where it belongs, and watch worst-seed DD for the regression signal.

## Standings — rd8h0 (`ppo-event-rd8h0-a12ba18`, rd8h with dd_lambda 0, 4×1M)

| seed | val | test | crash | trades |
|------|-----|------|-------|--------|
| s0 | +2.3% (DD 26.9%) | +15.8% (DD 17.5%) | +11.8% (DD 19.7%) | 63 |
| s1 | −20.3% (DD 25.3%) | +11.9% (DD 17.5%) | +12.3% (DD 12.7%) | 61 |
| s2 | −6.9% (DD 10.4%) | +33.0% (DD 4.1%) | +22.2% (DD 11.0%) | 38 |
| s3 | −9.9% (DD 23.5%) | **+51.0% (DD 8.4%)** | **+52.9% (DD 4.2%)** | 49 |
| **mean** | **−8.7%** | **+27.9%** | **+24.8%, 4/4 survive** | |
| bars | rule −0.6% / B&H +17.1% | rule +89.3% / B&H +58.6% | rule +62–64% / B&H ~−74% | |

### Verdict — FAIL, but the dd-penalty hypothesis was HALF confirmed, and the residual is the crash curriculum
- **Confirmed:** removing `dd_lambda` un-shrank the policy — crash mean +15.4%→+24.8% (s3 +52.9% at
  4.2% DD, nearly rule-parity), test up, sizes/DDs grew — **and the substrate held the DQ bound with
  NO reward brake at all (worst seed anywhere 26.9% < 30%)**. The structural-safety claim is proven.
- **Not confirmed:** no discrimination appeared — variance exploded (test +11.9%..+51.0%; val s0
  +2.3% vs s1 −20.3%) and the seed-mean still trails the rule everywhere. val got *worse* (−8.7%).
- **The remaining suspect, carried since GATE-2: `crash_train=4`** — four synthetic crashes per
  ~128-day train window ≈ one per month, vs ZERO alt-crashes in the real sample. The policy's
  learned crash prior is massively inflated, so deviating DOWN from the rule pays *on the training
  distribution* (the rule rides ignitions into injected crashes) and transfers as blanket caution to
  real windows. Every "defensive-everywhere" reading since GATE-2 is consistent with this.
- **Next single variable: `crash_train 4 → 1`** on the rd8h0 config. Expect: means move toward the
  rule on val/test; the regression signals are crash-split worst-seed DD (does survival skill
  survive a thinner curriculum, with breadth obs still present?) and worst-seed DD anywhere.

### The Q false-flag probe (2026-06-10, `scripts/probe_false_flag.py`) — hypothesis REFUTED at the population level
Q on Mar 28 00:00: ignition fired on surge 15× with rising only +11.5% (vs SIREN's 7.6×/+55.9%) and
collapsed −27% within 13h; rd8h0-s1 bought 2×-rule size, overrode the exits, bled ~−$1.1k (its val
−20.3%). Hypothesis: low-rising-on-high-surge = distribution → filter with `rising ≥ 15%`. **The
probe says the filter would cut the WRONG bucket.** On 960 train / 359 val in-universe ignitions:
the KEPT bucket (rising ≥15%) has *worse* forward returns (train fwd48 −5.83% vs −2.16% killed; val
Q4-rising fwd48 **−22.4%**), and the explicit false-flag corner (surge≥8× & rising<15%, n=61/54) is
*positive* on both splits (val +2.05% fwd24, 63% win). The universe mean-reverts — extended movers
are the poison, exactly the standing `cush`-negative finding; Q was a real disaster but an outlier
inside a healthy bucket. **The damage mechanism wasn't the entry — the rule exited Q in 2h with a
small loss; the agent's EXIT OVERRIDE riding −45% down at 2× size did the damage.** Surge and
rising are both already in the agent's obs (surge slot + harvest r24). Decided: no entry filter;
the Q-class protection is a substrate guardrail on the override — a **disaster floor** (a position
below entry−20% cannot be overridden; forced cut), closing the one remaining unbounded loss path.

## Standings — rd8h0c1 (`ppo-event-rd8h0c1-d4b155e`, crash_train 1 + loss-floor 0.2, 4×1M)

| seed | val | test | crash | trades |
|------|-----|------|-------|--------|
| s0 | +5.6% (DD 12.7%) | +30.9% (DD 7.9%) | +17.2% (DD 11.4%) | 40 |
| s1 | −2.3% (DD 9.9%) | +4.8% (DD 9.1%) | +7.9% (DD 12.8%) | 42 |
| s2 | **+15.2% (DD 6.7%)** | +8.2% (DD 15.0%) | +6.6% (DD 13.6%) | 28 |
| s3 | +0.2% (DD 9.5%) | +2.0% (DD 14.8%) | +9.9% (DD 8.3%) | 34 |
| **mean** | **+4.7%** | +11.5% | +10.4%, 4/4 survive | |
| bars | rule −0.6% / B&H +17.1% | rule +89.3% / B&H +58.6% | rule +62–64% | |

### Verdict — FIRST POSITIVE VAL in the project; the floor+crash-prior fix traded tails for safety
- **val seed-mean +4.7% — positive for the first time across the entire arc** (every config since
  GATE-1: −5..−9%). All 4 seeds beat the rule on val; s2 +15.2% at 6.7% DD came within 2pts of
  Buy&Hold. The crash-prior cut (4→1) + the disaster floor removed the Q-class bleed (s1-style
  −20% val is gone). The defensive-everywhere era appears OVER on val.
- **But test/crash regressed** (test +27.9%→+11.5%, crash +24.8%→+10.4%): the loss floor force-cuts
  −20% dips that V-recover in this universe (monster runners routinely draw down 20%+ mid-run), and
  4-seed noise is large. The levers now trade one regime against another — the cheap single-flag
  iteration space is showing diminishing returns.
- **Risk profile is now exceptional:** worst DD anywhere across 12 seed-regimes = 15.0%; every cell
  positive except one (s1 val −2.3%). As a DQ-gated competition profile this is genuinely strong.
- **Still FAIL everywhere** vs the honest gate — on prepad windows the rung-0 rule is a monster
  (test +89.3%, crash +62%), and 6 sweeps of substrate/reward/obs surgery have not produced the
  per-decision discrimination needed to match it. **Deployment-leader fact (plainly): rung-0 + caps
  + floor currently beats every learned policy on these windows — RL has not yet added value over
  the rule it rides.** Remaining untried standard lever: **training scale** (all sweeps today were
  1M steps; TradeSim's converged config was ~5M) — one 4×5M run (~100 min) is the last cheap test
  before concluding the discrimination isn't learnable on this obs set.

### The detonation blacklist (2026-06-11, `probe_detonation.py` → built @ `84ee6a0`) + rd9 overnight
User 2nd Q observation: all 4 rd8h0c1 seeds skipped Q's spikes but ALL FOUR bought the Apr 22–23
post-detonation chop (entries 0.0113–0.0115, exits 0.0086–0.0099, ~−$600–800 each ≈ 6–8% of equity
per seed — s2's val would have been ~+21%). Proposal: after extreme untradeable volatility, ignore
the token. **Probe (population, both splits): post-detonation ignitions are robustly toxic —
det-2–4wk bucket fwd48 −8.4% train (win 10%) / −24.3% val (win 8%), n=121 — and the poison EXPIRES
(>4wk ≈ clean baseline).** Built `det_blacklist`: detonation = surge≥8× while rising≤−15% → the
token's ignitions zeroed for 672 bars (4wk, probe-calibrated). Applied in the env's ignite
precompute, so agent prompts AND the rule mirror share it (parity); the canonical rung-0 gate bar
stays unfiltered (honest). 265 tests. **rd9 = rd8h0c1 + det-blacklist, swept overnight at 5M steps**
(the last standard lever — every prior sweep was 1M; TradeSim converged ~5M):
`ppo-event-rd9-84ee6a0-s{0..3}`.

## Standings — rd9 (`ppo-event-rd9-84ee6a0`, rd8h0c1 + det-blacklist, **5M steps**, overnight 2026-06-11)

| seed | val | test | crash | trades |
|------|-----|------|-------|--------|
| s0 | +2.9% (DD 5.8%) | +10.4% (DD 11.9%) | −14.4% (DD 28.6%) | 34 |
| s1 | +0.6% (DD 5.4%) | −1.3% (DD 18.9%) | −3.4% (DD 20.6%) | 40 |
| s2 | −5.5% (DD 7.7%) | −1.3% (DD 11.4%) | −3.7% (DD 13.6%) | 39 |
| s3 | −0.9% (DD 7.2%) | +0.2% (DD 14.1%) | −9.1% (DD 22.1%) | 36 |
| **mean** | **−0.7%** | **+2.0%** | **−7.7%** | |

### Verdict — REGRESSION vs rd8h0c1@1M (val +4.7→−0.7, test +11.5→+2.0, crash +10.4→−7.7)
5× training + the det-blacklist made everything worse, despite the blacklist removing the Q chop
that cost rd8h0c1 ~6–8pp/seed — so the 5M convergence itself is strongly implicated: low-variance,
low-return, near-flat policies (val DDs 5–8%, returns ±3%) — converged to rule-hugging mediocrity,
not discrimination. (Two variables confounded — blacklist alone at 1M was not run — but neither
direction rescued it.) **The scale lever was the last standard one; it failed. Conclusion of the
phase: rd8h0c1@1M stands as the best learned config (val +4.7%, all-positive ex one seed, worst DD
15%); the DEPLOYMENT LEADER remains rung-0 on this substrate (caps + floor + blacklist); the
stable-baseline goal for the automation loop is met. Next: the MCP iteration loop + Phase 2 (June
16 live TWAK trade) — further RL levers (discrimination obs, recurrence) run through the automated
loop, not hand-driven sessions.**

### rdL launched (2026-06-11, @ `a27e469`) — RecurrentPPO: the memory the forensics demand
The user's full per-token rd9 review (B/BANANAS31/HUMA/Q/SIREN/TAG/UB/ZEC, all seeds) distilled to
three recurring failure classes — **re-buying the post-pump bleed churn** (B, BANANAS31, UB-s2,
ZEC-s1: "trade the structured pump then WALK AWAY"), **failure to hold a winner to the top**
(ZEC-s2 sold 5× instead of holding; TAG-s0 no profit-take at the high), **missed obvious early
runs** (SIREN Mar 22 again at 5M) — plus the meta-observation: *no sign of learned experience,
seeds incoherent*. **All of these are SEQUENCE skills, structurally inexpressible for the stateless
feedforward MLP** (each decision sees only the current 16-dim snapshot; "this token already gave
its move" requires memory). RecurrentPPO was roadmapped since GATE-2, sequenced LAST pending a
demonstrated feedforward ceiling — 7 configs later that ceiling is demonstrated, and the user
called for it independently. Built: `--recurrent --lstm-size 256` (sb3-contrib MlpLstmPolicy,
TradeSim's converged size), stateful eval threading (fresh LSTM state per split episode), rule-prior
bias unchanged. Q note: the det-blacklist behaved exactly as scoped (zero Q trades, signal-level,
per-token — no general-logic impact possible). **Verdict checklist (behavioral, not just returns):
(1) bleed-churn re-entries down? (2) winners held longer / TP rungs used at tops? (3) SIREN Mar 22
participation? (4) cross-seed coherence up?** Sweep: `ppo-event-rdL-a27e469-s{0..3}`, 4×1M, val.

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
