# AI Training

The learned-policy candidate for the decision core — an RL pipeline ported from [[TradeSim]]
that trains a trading agent against the [[Simulated Market]], with reward and evaluation tied
to the competition's risk gate. **RL is one option, not a mandate** — weighed here against
simpler robust strategies ([[Trading Strategies]]) for a single, high-variance live week.
Owned by `rl-ml-trainer`. Regime/scenario context: [[Market Conditions]]; training-host
question: [[Remote Capabilities]].

> ## ⚠ CURRENT DIRECTION (2026-06-16) — read before acting on any "as-built" section below
> The substrate is the **SELECTIVE event-driven ignition** model (the rd/rdL lineage): the agent
> enters ONLY on real volume-ignition setups, the trap guards (`det_blacklist`, loss-floor) are
> active, and RL learns the *discretion* (sizing, exit timing) on top. **The bar is: beat the
> rung-0 RULE out-of-sample + survive the ~30% DD DQ + ≥1 trade/day, graded on the cold-weekly
> eval.** Buy&Hold AND Random are REPORTED references, **never binding** — the gate fix is now
> COMPLETE across all sites (commit `503b784`; see §"The honest-gate fix finished").
>
> Where the arc stands (2026-06-16):
> 1. **The gate fix is done.** The 2026-06-15 reset demoted B&H only in `weekly_gate`; this session
>    finished it everywhere (`train_event.honest_gate`, `compare_seeds`/`regime_verdict`,
>    `champion._honest_gate`, the loop's north-star metric `margin_vs_buyhold → margin_vs_rung0`,
>    the `rl_diagnose` note, `contract.SUCCESS_METRIC`, [[Agent Communication Contract]]). 398 tests
>    pass; adversarially reviewed clean.
> 2. **Both curricula are now REFUTED.** The horizon curriculum already failed its kill criterion on
>    the (now-shelved) overlay; the new **universe-regime curriculum** (`curriculum_universe`,
>    lowvol→broad→voltopk) now failed its kill criterion on the selective substrate — it cost ~16pts
>    vs the curriculum-OFF control. So the plateau is **NOT** a capacity/exposure problem.
> 3. **The selective substrate PASSES the corrected gate on VAL.** The curriculum-OFF control
>    `ppo-event-rdLe4-wk` makes **+13.66%/wk** mean (all 4 seeds beat the rung-0 rule, worst-seed
>    maxDD 10.1% < 30%) — so the long-documented "rd/rdL plateaus vs the rule" was substantially an
>    artifact of the old continuous eval + the B&H gate, **not** the policy. CAVEATS, stated plainly:
>    val is the possible overfit pocket (frozen TEST is UNSPENT), and the +13.66% mean is inflated by
>    s0 (+35.3%, the historical val-pocket seed) — the robust majority beats the rule by a more
>    modest +2.5–11pts.
> 4. **Diagnosis: the control is REWARD-BOUND, not capacity-bound.** `deviation_alpha` on the control
>    = **+0.001** (entry over/under-sizing vs the rule does not predict forward returns) — the
>    `relative` portfolio-reward isn't teaching entry-sizing discrimination, so the +14pt val edge is
>    not skill-driven entry sizing (exit-timing and/or s0 luck) and is unlikely to generalize to test.
> 5. **Active lever = reward shaping (`entry_forward`), LAUNCHED, no verdict yet.** A per-entry reward
>    that pays for correct sizing (`dev × (fwd_ret − typical-ignition)` — literally the quantity
>    `deviation_alpha` measures). The residual preflight confirms a strong learnable signal
>    (corr(cush, fwd-24h) = −0.423; correct-discriminator is the unique argmax at res_gamma 0.0). The
>    `ppo-event-rdLe4-ef` sweep is **TRAINING NOW**; success = `deviation_alpha` goes clearly POSITIVE
>    **and** it beats the rung-0 rule on val + survives DQ → ONLY then spend the frozen test.
>
> The `basket_default` **OVERLAY stays SHELVED** (flag default-OFF, parked) — it was a buy-everything
> detour the "beat B&H" gate drove us into; see §"DRIFT POST-MORTEM". The horizon + universe curricula
> stay parked (both refuted, flags default-OFF). Full record: [[Experiment Log]] §2026-06-16.

## 2026-06-18 — Regime-conditional entry_forward baseline (design)

**Status: BUILD-READY SPEC, not launched.** Design + cheap preflight/probe plan only. No desktop
training in this session. Owner `rl-ml-trainer`.

### The bar (restate — this is what we're held to)
RL trading agent for the BNB hackathon, **scored on live PnL over one held-out week (June 22–28),
hard 30% max-drawdown DQ gate, ≥1 trade/day**. The deploy week is **forecast flat/bear**. The honest
gate (`src/trader/train/weekly_eval.py`) over **val+test COLD weeks** (fresh $10k Mon-00:00, universe
re-picked causally): (1) worst single-week maxDD < 30%, (2) **beats the rung-0 RULE** on the paired
edge (bootstrap CI lower bound > 0). Judge on the **seed mean**; always read the **worst seed's DD**.
B&H + Random are reported, never binding. Substrate ceiling today = `wkw` (rdLe4 + wick_reject 0.25),
+5.1pts vs rung-0; deploy pick = `ef-s2`. *(As of 2026-06-18. SUPERSEDED — the deploy pick / champion is
now `sbq-s1`; see §2026-06-21.)*

### Why this lever (the motivating finding — [[Experiment Log]] §2026-06-18)
The `_ignite` trigger is **regime-conditional**: on the bear/chop-heavy TRAIN era, ignitions are
**negative-EV on 19/20 tokens** (dead-cat bounces); the positive payoffs live only in the val/test
bull windows. Per-regime cold-week split (`scripts/probe_regime_split.py`): the real OOS axis is
**flat/chop 64% · bull 29% · bear 7%**; **CASH (0%) beats deployment in 71% of weeks** — in flat/chop
the rule bleeds −3.9% (DQ once at 35.5%), B&H −1.8%, cash 0%. Deployment only wins in the rare bull.
The highest-value regime job is **"recognize chop → sit in cash"** (USDT = the env's existing flat
state), NOT harvest-the-bull (rare, held-out, deploy-misaligned). The agent already SEES the regime
(`breadth` obs) — so this is **NOT an obs gap, it is a REWARD gap.**

### The exact defect in the current reward
`entry_forward` (env `event_reward.py:entry_forward_reward`, matured by `_mature_entries`) credits
`dev·(fwd − mu_base) − γ·dev²`, where `mu_base = _ignition_base_rate()` is a **single panel-wide
scalar** = mean forward-`fwd_horizon` return over **ALL** ignitions in the panel (regime-blind). The
chop-ignitions are the 64% majority and lose to cost, but because the null is one global number, a
chop-ignition's `fwd − mu_base ≈ 0` → it scores "near average / size me at the rule" instead of
"skip me." The reward never tells the policy that a chop-regime ignition is a SKIP. A
**regime-conditional baseline** — demean each ignition against the typical ignition **in the same
regime** — lifts chop-ignitions' demeaned outcome NEGATIVE (so `dev*<0` → skip → stay in cash) while
preserving the size-up signal where bull-regime ignitions beat their own bucket's mean.

### Mechanism (single-variable, byte-identical when OFF)
New flag `regime_base: int = 0` (default 0 = OFF = today's behavior exactly, scalar `_mu_base`).
When `regime_base = R > 0` (R = number of breadth buckets, e.g. 3), `_mu_base` becomes a **lookup**
`mu_base_by_regime[r]`, and each ignition is demeaned against the bucket its own (causal) breadth
falls into. The interior optimum `dev* = (fwd − mu_base[r]) / 2γ` then becomes regime-correct:
negative in chop (→ skip → cash), positive in bull (→ size up).

**1. Regime label at an ignition bar (CAUSAL — past data only).** Use **`breadth`** = fraction of
the *episode universe* above its EMA at the ignition bar, the obs slot the env already computes:
`breadth = mean(self._cush[bar, self._uni_ix] > 0)`. `_cush = px/ema − 1` uses only `ewm(adjust=False)`
of past closes — strictly causal, no future bar. Bucket by **fixed breadth thresholds** (not
per-episode quantiles, which would leak the episode's distribution): default 3 buckets
`chop/mid/bull` at breadth `< 0.33 / 0.33–0.66 / ≥ 0.66` (chop = few names above EMA = the bleed
regime). Thresholds are constants, identical train/val/test/deploy → no leakage, no per-episode
fitting. (Rationale for breadth over BTC-trend: the universe is selected for LOW BTC correlation —
`no_btc_obs` already neutralizes the BTC slot; breadth is the alts' own regime, the documented signal
that "earns its keep.")

**2. The per-bucket baseline statistic.** A **fixed panel lookup**, same "panel-statistic null" role
as today's scalar `_mu_base` (NOT episode-causal — it's a constant of the panel, computed once in
`__init__`, mirrored verbatim in the preflight). New private method `_ignition_base_rate_by_regime()`:
for every panel ignition `(bar, j)` with `_px[bar,j] > 0` and `bar + H < n_bars`, compute its
**universe-EW breadth label at `bar`** over the SAME causal panel-wide breadth proxy used in the
preflight (see leakage note), bucket it, and accumulate its forward return `fwd = _px[bar+H,j]/_px[bar,j] − 1`
into that bucket. Return `mu[r] = mean(bucket r)`; **empty/thin buckets (n < `regime_base_floor`,
default 200 ignitions) FALL BACK to the global scalar** `_ignition_base_rate()` so a sparse bucket
can never produce a degenerate (one-sample) null. Store `self._mu_base_vec` (length R) alongside the
scalar `self._mu_base` (kept for the fallback + OFF path).

> **Subtlety to resolve in code:** `_ignition_base_rate()` ranges over the **whole panel** (all
> tokens, not the episode's k), but `breadth` in `_obs` is over the **episode `_uni_ix`** (the k
> picked tokens). For the *panel statistic* there is no episode yet. Spec the panel-breadth proxy as
> **the fraction of ALL panel tokens above their EMA at `bar`** (`mean(_cush[bar, :] > 0)`), a single
> causal panel array `self._breadth_panel = (self._cush > 0).mean(axis=1)`. At *maturation time* in
> `_mature_entries`, label the entry by **the same panel-breadth proxy at the ENTRY bar `eb`**
> (`self._breadth_panel[eb]`), NOT the episode breadth — so the bucket used to credit an entry is the
> SAME function the panel statistic was built from (objective == null, the exp3 lesson). The episode
> `breadth` obs is what the POLICY sees; the panel-breadth proxy is what the REWARD nulls against.
> They need only be consistent within the reward, which this makes them.

**3. `_mature_entries` selection.** Change the credit line from
`entry_forward_reward(dev, fwd, self._mu_base, ...)` to look up the entry's bucket:
`r = self._regime_bucket(self._breadth_panel[eb]); mu = self._mu_base_vec[r]` then
`entry_forward_reward(dev, fwd, mu, self.res_gamma)`. `eb` is already stored in `_pending_entries`.
When `regime_base == 0`, `_mu_base_vec` is unset and the line uses `self._mu_base` (identical bytes
to today).

### Leakage audit (state it explicitly)
- **The regime LABEL at bar `b` is causal**: `_breadth_panel[b] = mean(_cush[b,:] > 0)`, and `_cush`
  is `px/ema − 1` with `ema = px.ewm(span, adjust=False)` — an exponential mean of closes **at or
  before b**. No `b+1` term enters the label. Thresholds are fixed constants. So no future bar can
  flip a bucket.
- **The per-bucket baseline is a fixed PANEL statistic**, exactly the same epistemic status as
  today's `_mu_base` (which already ranges over `bar + H` forward returns of every panel ignition).
  It is a constant of the dataset, not episode-conditional; the env has always demeaned against a
  panel-forward statistic and the preflight mirrors it. We are not adding lookahead — we are slicing
  the SAME panel statistic by a causal label.
- **Distinct from `week_regime` (`weekly_eval.py:114`)** which is open→close (forward) and therefore
  NON-causal — that one is descriptive/grading-only and must NOT be used for the in-env label. The
  in-env label is breadth-at-bar, past-only.

### Preflight mirror (REQUIRED — or the in-env gate is meaningless)
`scripts/preflight_selector.py` builds its scoring world from the SAME env (`EventRungEnv` with
`reward_mode="entry_forward"`). Because the baseline now lives entirely inside the env
(`_ignition_base_rate_by_regime` + `_mature_entries` lookup), the preflight inherits it for free **as
long as it constructs the env with the same `regime_base`/`regime_base_floor` kwargs**. Touch-points:
- Add `--regime-base` / `--regime-base-floor` args to `preflight_selector.py`, thread into the `kw`
  dict (line ~41). The scripted agents already run THROUGH the env, so the reward they sum is the
  regime-conditional one automatically — no separate statistic to recompute. This is the whole point
  of the "one definition" design (`event_reward.py` docstring): keep the baseline inside the env and
  the preflight cannot drift from training.
- **Do NOT** recompute `mu` in the preflight's `predict()`/lstsq block — that block fits a *selector*
  (cush/surge/btcT → fwd), orthogonal to the baseline; leave it untouched.

### Probe-before-build gate (CHEAP, panel-only, no sweep) — `scripts/probe_regime_base.py`
The reward-bound finding is `corr(deviation, fwd-return) ≈ 0` overall, and specifically the chop
bucket is where over-sizing is mis-rewarded. The probe must show the regime-conditional baseline
**makes the chop bucket's demeaned signal correctly negative** while keeping bull positive — i.e. it
fixes the discriminator the global baseline smears. Panel-only (no PPO, runs on the laptop):
1. Build the env on `train_r` with `reward_mode="entry_forward"`, `ungate=True` (as the preflight),
   `regime_base=3`. Pull `_breadth_panel`, the ignition set, each ignition's `fwd = _px[b+H,j]/_px[b,j]−1`,
   and its bucket.
2. Report, per bucket: `n`, `mean(fwd)` (= the bucket baseline), and **`mean(fwd) − global_mu_base`**
   (how far the bucket sits from today's single null).
3. **The discriminator test.** Under the GLOBAL null, the "demeaned outcome" of a typical chop
   ignition is `mean(fwd_chop) − global_mu`. Under the REGIME null it is `0` by construction — but
   the relevant quantity is the **sign the reward assigns to sizing a chop ignition at the rule**:
   with global null a chop ignition that returns `mean(fwd_chop)` scores `dev·(mean(fwd_chop) − global_mu)`
   — and because `mean(fwd_chop) − global_mu` is NEAR ZERO (the smear), the reward gives chop sizing
   no clear skip pressure. Compute and PASS if:
   - **(a) Separation**: `mean(fwd_chop) < mean(fwd_bull)` with a non-trivial gap
     **`mean(fwd_bull) − mean(fwd_chop) ≥ 0.01`** (1pt of forward-H return), AND
   - **(b) Chop is a skip under the regime null relative to global**: `mean(fwd_chop) − global_mu < −0.003`
     (the chop bucket sits clearly BELOW the global null, so regime-demeaning pushes chop dev
     negative where global-demeaning left it ~0), AND
   - **(c) Bucket health**: every bucket used (not falling back) has `n ≥ regime_base_floor`.
   If (a)–(c) hold, the regime baseline measurably sharpens the chop=skip / bull=size signal the
   global null blurs → BUILD the sweep. **If the chop bucket's `mean(fwd)` is NOT meaningfully below
   global (gap < 0.003), the finding does not translate into a reward fix and we DO NOT sweep** —
   the bleed is then a selection/exit problem, not a baseline problem, and this lever is refuted
   cheaply. (Stretch, if quick: re-run with the OOS selector and confirm `corr(predicted_dev, fwd)`
   computed WITHIN the chop bucket is ≥ 0 under the regime null vs ≈0 under global — the direct
   "discriminator improves" read; gate on the panel separation above if the in-env corr is noisy at
   chop's n.)

### Sweep config + pre-registered kill criterion (only if the probe PASSes)
- **Single variable**: `regime_base=3` ON, everything else = the named control. **Control = `ef`**
  (the `entry_forward` config, `regime_base=0`) — the cleanest 1-var comparison (same reward family,
  same `res_gamma`, only the baseline changes). Run the SAME seed set as `ef` (≥4 seeds; single-seed
  RL is noise). Pattern: `scripts/run_reward_sweep.sh` style, one config, sequenced seeds.
- **Success** = seed-MEAN beats `ef`'s cold-weekly edge vs the rung-0 rule (bootstrap CI lower bound
  > the control's), **measured with the chop/flat + bear weeks weighted as they'll appear live** (do
  NOT let a bull-week outlier crown it — read the per-regime breakdown), at a **worst-seed maxDD
  survivably under 30%** (target ≤ the control's worst-seed DD; the whole thesis is *less* bleed, so
  DD should improve or hold, never worsen).
- **KILL** (pre-registered, any one triggers): (i) seed-mean cold-weekly edge vs rung-0 ≤ `ef`'s
  (bootstrap CI lower bound not strictly above the control) → no improvement, refute; (ii) worst-seed
  maxDD > `ef`'s worst-seed DD by ≥ 2pts → it traded MORE risk, not less, opposite the thesis;
  (iii) the chop/flat-week mean return does NOT improve vs `ef` (the bleed regime is the target — if
  flat weeks don't get better, the lever missed its own thesis even if bull weeks flatter the mean).
  On any kill: park the flag default-OFF, log to [[Experiment Log]], move on. Do NOT re-tune buckets
  to rescue a bull number (see eval caveat).

### Eval caveat (do NOT design past it)
Train is bear/chop-heavy; val/test catch the bull; **deploy is flat/bear**. The win condition is
**"stop the chop bleed / sit in cash,"** measured on the flat+bear regimes the live week will
actually be — NOT harvesting the held-out bull. **Over-fit risk to watch:** because val/test contain
the only bull weeks, the seed-mean can be inflated by bull performance while flat/bear (what we
deploy into) is unchanged or worse. Mitigation baked into the kill criterion (iii): the flat-week
mean must improve, independently of the bull. If a candidate's edge comes ENTIRELY from bull weeks,
treat it as a fail for our purpose even if the overall mean rises.

### Desktop handoff (clean, when/if the probe passes)
1. Implement `regime_base` + `regime_base_floor` + `_breadth_panel` + `_ignition_base_rate_by_regime`
   + `_regime_bucket` + the `_mature_entries` lookup in `event_env.py`; default-OFF, **byte-identical
   when off** (add a test asserting `regime_base=0` reward stream == pre-change). Mirror the kwargs in
   `preflight_selector.py`. Land + push; record the sha.
2. Run `scripts/probe_regime_base.py` on the laptop (panel-only) → PASS/FAIL per the thresholds above.
   Only on PASS proceed.
3. On the desktop, follow the [[Remote Capabilities]] runbook EXACTLY (PowerShell-ssh, tiny output,
   launch-once-wait-60–90s, `mkdir -p runs-rl`, sync+preflight to the pushed sha). Sweep `ef` vs
   `ef+regime_base=3`, same seeds, **sequenced never parallel**. Aggregate via `compare_seeds.py`.
4. Apply the kill criterion on the seed-mean + per-regime breakdown + worst-seed DD. Do NOT spend the
   **frozen TEST** unless the val gate + DD pass; test is the human's one-shot OOS certification.

### Code touch-points (summary)
- `src/trader/train/event_env.py`: `__init__` (new kwargs + `_breadth_panel` + `_mu_base_vec`),
  `_ignition_base_rate_by_regime()`, `_regime_bucket()`, `_mature_entries()` (bucket lookup).
- `scripts/preflight_selector.py`: thread `--regime-base`/`--regime-base-floor` into `kw`.
- `src/trader/train/event_reward.py`: **UNCHANGED** (the baseline is an env input, not a new reward
  shape — keep the one definition).
- New: `scripts/probe_regime_base.py` (the cheap gate).

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
iterations and 64 runs. The seed is single-asset BTC with a heavy technical-indicator stack — a
different market and a more complicated training problem — so what carries over is the
**engineering discipline**, not its strategy conclusions. Non-negotiable discipline takeaways:

- **Make a baseline (Buy&Hold / cross-sectional momentum) behind an honest gate the first
  validated thing** — prove any edge out-of-sample before trusting it. (Entry timing + sizing
  ARE the edge in this selective-ignition strategy; the honest gate is how we prove it, not
  assume it.)
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

## Participation/lateness lever closure (2026-06-17, rl-ml-trainer)

The "~89% non-participation" is mostly the ENV NOT PROMPTING, not the agent refusing: cooldown(48)
+ reclaim + held-skip cut prompts ~960→~39/episode. On the ~39 it sees, the agent acts on a high
fraction (rule_default idx0 = execute-rule, warm-started by `rule_prior 2.0`). So loosening prompts
funds the UNFUNDED stream — which P-CAPACITY showed is −EV and pure beta.

Levers, all closed bar one:
- **cooldown / reclaim / entry_frac** — P-CAPACITY swept {48,24,12,0}/{on,off}/{0.20,0.125,0.10} on
  the real cold-weekly DQ object → NULL (rotation-rejected=0; loosening funds worse ignitions; sizing
  down = pure beta give-up; rung-0 itself breaches 35.5% DQ uncapped). REFUTED.
- **scale_in** — REFUTED (wsi val −2.75%, −2.17pts vs rung-0, DD 12.2%). Do not re-propose.
- **ungate** — the exp5 drift; funds the full −EV stream. REFUTED.
- **rule_prior toward more action** — rdLp1 (prior 2.0→1.0) collapsed val +13.6→+1.8 (prior is
  load-bearing; DRIFT ALARM). Raising it pushes blanket rule-execution → the buy-everything basin
  ([[benchmark-driven-drift]]). REFUTED both directions.
- **exit_commit 48→? ** — the ONE untested single-variable lever. Held fixed at 12 across all 10
  rule_default configs. It is a LATENESS-ON-RE-ENTRY knob (trimmed positions lock capital + go
  off-prompt for N bars), NOT a participation-count knob → alpha (timing on the existing selective
  book), not beta (no new concurrent −EV positions). Recommended next: `exit_commit 12→4` vs wkw.

DRIFT FLAG (per contract): wkw val +4.53% still LOSES Buy&Hold (+17.07%) by ~12.5pts. wkw beats
rung-0/Random and is DQ-protective (7.84% worst-week, fixes rung-0's own 35.5% breach), which is the
agreed config-selection gate — but no participation lever recovers the B&H gap; that gap is the
substrate ceiling, not a knob.

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
discretion becomes a **tilt** on top (additive instead of the whole policy). Flag defaults OFF (378 tests green, prior behavior byte-identical);
new flag builds on `rule_default` (discrete 4-level), with `rule_prior` making the untrained policy ≈ B&H.
Validated on real data: hold-everything == B&H to 5 decimals over 6 cold weeks. Plumbed through the gym
adapter, `train_event` (`--basket-default` + provenance), and `simulate` provenance.

**Next:** finish the random-week gate wiring (drop btc_trend obs), then the **from-scratch random-week
sweep** on the overlay to the distribution gate (desktop — shared-box gotchas apply). The honest question:
can learned tilts beat holding the basket OOS across random weeks? "No, just hold" is now a valid,
gate-safe answer the substrate makes reachable. See [[curriculum-and-checkpoints-are-legitimate]],
[[Simulated Market]], [[Experiment Log]] §2026-06-14.

## DRIFT POST-MORTEM (2026-06-15) — how the "beat B&H" gate drove us to buy-everything

A full session drifted from the core thesis. Recording the MECHANISM so it never recurs (this is the
same class as the exp1→exp5 reward-proxy drift — optimizing a metric that diverged from the goal).

**The drift, step by step:**
1. The 2026-06-14 fork's honest gate required "beat **Buy&Hold** + rung-0 OOS." Reasonable on its face.
2. The cold-weekly diagnostic showed the SELECTIVE ignition skeleton **structurally cannot out-return
   B&H in a bull** — it sits in cash between ignitions while the basket runs up (the "+13.2% bull-gap").
3. Pressure to satisfy the *B&H* bar → proposed widening the substrate to a long-default **basket
   overlay** (`basket_default`): default = **hold the WHOLE risk-parity basket** = literally Buy&Hold.
   This "beat" the gate by *becoming the benchmark*.
4. That is the **buy-everything** behavior the user rejected from the start. It abandoned selective
   ignition entry AND made every trap guard inert (`det_blacklist` only blocks ignition *entries*; the
   overlay never enters via ignition). Result: all 4 overlay seeds bought the **Q detonation trap** and
   ate ~−$1k trades — the exact failure the selective rules exist to prevent.
5. Two more sweeps (OVERLAY-1/2 + a horizon curriculum) were spent optimizing *inside* this wrong
   substrate before the user caught it.

**The tell we missed:** the rl-loop **drift-alarm was firing on "no PnL-vs-Buy&Hold improvement in 8
experiments."** That was the system saying the *B&H comparison* was the binding, unproductive constraint.
The correct response was to **question the METRIC**; instead the response was to **engineer a substrate to
satisfy it**. When a gate halts progress, ask whether the gate measures the agent you actually want.

**The durable lesson (the rule going forward):**
- **Do not let a benchmark define the agent.** B&H is what passive competitors do — a *reference*, not a
  *design target*. Requiring "beat B&H" rewards holding-everything, which is the opposite of a selective
  trader. (Operationalized: `weekly_gate` `require_buyhold` defaults False; the bar is beat-the-RULE.)
- **The north star is a SELECTIVE RL agent** that earns each position through ignition logic + the trap
  rules, and **beats the rung-0 RULE** via better learned discretion. The rule — not B&H — is the bar.
- **When optimizing a metric drives the agent AWAY from the thesis, the metric is wrong, not the thesis.**
  Same failure class as reward-proxy drift; the fix is to correct the objective, not the agent.

The overlay/curriculum code stays parked (flags default-OFF, byte-identical when unused). Return to the
selective substrate; full record in [[Experiment Log]] §2026-06-15 DIRECTION RESET.

## As-built (2026-06-16) — gate fix finished, both curricula refuted, the substrate passes on VAL, reward-bound diagnosis, entry_forward launched

The session that turned the 2026-06-15 reset into measured results. The arc, in the order it happened:

### The honest-gate fix finished (commit `503b784`, on `origin/main`)

The 2026-06-15 reset (`6eda1d5`) demoted Buy&Hold from a binding gate to a *reference* — but **only in
`weekly_gate`**. Every other gate site still bound on B&H, so the codebase disagreed with itself about what
"pass" meant. This session finished the demotion **everywhere**:

- `train_event.honest_gate` — binds on {survive DQ, beat the rung-0 RULE}; B&H/Random accepted, computed,
  reported, never binding.
- `diagnostics.compare_seeds` + `regime_verdict` — same.
- `champion._honest_gate` — same. **Consequence for the champion contract:** a config that beats the rule
  + survives DQ but **LOSES to B&H can now be champion.** This is the intended behavior for a *selective*
  agent (it structurally sits in cash between ignitions and can't out-return B&H in a bull), but it is a
  real change to what "best formula" means — stated here so it is never mistaken for a bug.
- `loop_control` — its north-star metric switched `margin_vs_buyhold → margin_vs_rung0`; the **drift alarm
  now fires on no edge-vs-the-rung-0-RULE improvement** (the exact signal that, under the old metric, fired
  on "no B&H improvement" and drove the buy-everything drift — see §DRIFT POST-MORTEM).
- The `rl_diagnose` note, `contract.SUCCESS_METRIC`, and [[Agent Communication Contract]] — all aligned.

The honest gate now binds ONLY on **{survive the ~30% max-drawdown DQ, beat the rung-0 RULE
out-of-sample}**; **Buy&Hold and Random are REPORTED references, never binding.** 398 tests pass, 1
skipped; adversarially reviewed under two lenses, both clean. A companion fix (`7458aa8`) added
`GymEventRungEnv.set_universe_mode` passthrough — without it the new universe-curriculum callback crashed
the `SubprocVecEnv` worker.

### The universe-regime curriculum — BUILT then REFUTED (kill criterion MET)

A new `curriculum_universe` flag stages the **training universe** `lowvol → broad → voltopk` over training
progress — the volatility-axis analog of the (already-refuted) horizon curriculum. The intent: learn basics
on tractable low-vol dynamics, then ramp into the chaotic deploy distribution.

Result on the selective rdLe4 substrate, cold-weekly VAL:

| config | mean/wk | per-seed | vs rung-0 rule | seeds ≥0 | verdict |
|--------|---------|----------|----------------|----------|---------|
| `ppo-event-rdLe4-curu` (curriculum ON) | **−2.65%** | −4.4 / −4.6 / +4.95 / −6.6% | **−2.1pts** | 1/4 | **FAIL** (binding rung-0) |
| `ppo-event-rdLe4-wk` (CONTROL, curriculum OFF) | **+13.66%** | +35.3 / +6.6 / +1.9 / +10.8% | **+14.2pts** | 4/4 | PASS |

The curriculum **cost ~16pts** → its pre-registered KILL CRITERION was met. Likely cause: the schedule
trains ~65% OFF the `voltopk` deploy distribution (in `lowvol`+`broad`), starving deploy-regime experience —
the agent spends most of training on a universe it will never trade.

**The durable read across BOTH curricula.** The horizon curriculum earlier failed its kill criterion (on
the now-shelved overlay); the universe curriculum now fails on the selective substrate. **Both curricula
refuted** → the plateau is **NOT** a capacity/exposure problem. (This does not retract
[[curriculum-and-checkpoints-are-legitimate]] — curriculum remains orthogonal to the honest gate and a
legitimate optimization lever; it simply was not the operative lever *here*, on these two axes, measured.)

### The control PASSES the corrected gate on VAL — the "plateau vs the rule" was largely an eval artifact

The curriculum-OFF control `ppo-event-rdLe4-wk` on cold-weekly val: mean **+13.66%/wk**, per-seed +35.3 /
+6.6 / +1.9 / +10.8% (**all 4 beat the rung-0 rule**, which is −0.58%), worst-seed maxDD **10.1%** (< 30%
DQ). It **loses to B&H** (+17.07%) — now a reference, not a gate. So the SELECTIVE rd/rdL substrate, judged
CORRECTLY (beat-the-rule + DQ, on the deployment-shaped cold-weekly eval), **WORKS on val.** The
long-documented "rd/rdL plateaus vs the rule" was substantially an **artifact of the old continuous eval +
the B&H gate, not the policy itself.**

**Caveats, stated plainly (this is val, not a verdict):**
- **VAL is the possible overfit pocket** — the frozen TEST is UNSPENT. A val pass is necessary, not
  sufficient; the project's history (the `ppo2-real` +83% val → +11% test collapse) is the reason TEST is
  reserved until a config earns the spend.
- **The +13.66% mean is inflated by s0** (+35.3%, the historical val-pocket overfit seed). The robust
  majority (s1/s2/s3) beats the rule by a more modest **+2.5–11pts** — still a pass, but read the majority,
  not the headline.
- **Determinism check:** `rdLe4-wk-s0` is **bit-identical** to the prior `rdLe4r-68b268f-s0` (same
  deterministic policy) — confirms training determinism on the box AND that the curriculum code is
  byte-identical when the flag is OFF (the no-regression guarantee, verified empirically).

### Diagnosis — the control is REWARD-BOUND, not capacity-bound

`deviation_alpha` on the control (the same diagnostic that redirected exp1→exp2 from "buy an LSTM" to "fix
the reward"): corr(entry over/under-size vs the rule's 0.20, fwd-24h return) = **+0.001**. Oversized entries
returned +3.29% fwd (n=19); undersized +3.81% fwd (n=53) — **bigger bets land on NO bigger moves.** The
`relative` (portfolio-level) reward is not teaching entry-sizing discrimination, so the control's +14pt val
edge is **NOT skill-driven entry sizing** — it is exit-timing and/or s0 luck → unlikely to generalize to
test. (The curriculum was worse and INVERSE: corr −0.198, over-sized losers.) This **quantifies** the
long-standing "defensive / won't discriminate" plateau as REWARD-BOUND — the same finding as the exp1→exp4
arc, now confirmed on the cold-weekly-passing control. The fix is the reward, not capacity (the LSTM is
already in this lineage; it is not the missing piece for *entry-sizing* discrimination).

### Active lever — reward shaping (`entry_forward`), LAUNCHED, NO verdict yet

The residual preflight (`scripts/preflight_residual.py --horizon 24`) confirms a strong learnable signal:
**corr(cush, fwd-24h ret) = −0.423** over 2176 train ignitions; the correct-discriminator (size ∝ −cush) is
the **unique argmax at res_gamma 0.0** (both corners ≤ 0; an IC-hacker loses). So a per-entry reward —
`entry_forward = dev × (fwd_ret − typical-ignition)`, which is **literally the quantity `deviation_alpha`
measures** — should teach skillful sizing (objective == metric, the exp4 lesson learned the hard way).

LAUNCHED `ppo-event-rdLe4-ef` — a **single-variable swap** vs the control: reward_mode `relative →
entry_forward`, fwd_horizon 24, res_gamma 0.0, on val cold-weekly. **TRAINING NOW (loop iteration 3, NO
verdict yet — do not present any number for it).**

- **PRE-REGISTERED SUCCESS** = `deviation_alpha` corr goes clearly POSITIVE **and** it beats the rung-0
  rule on val + survives DQ → ONLY THEN spend the frozen test (on a skill-driven config, not the
  reward-bound control).
- **KILL** = corr stays ~0 or it loses to the rule.

The point of `entry_forward` is to convert the control's *unexplained* val edge into a *skill-grounded* one
before betting the frozen test on it — so that a test pass, if it comes, generalizes rather than repeats the
s0 val-pocket pattern. Full record + tables: [[Experiment Log]] §2026-06-16.

## Probe methodology & scope — the diagnostic instrument (2026-06-16/17)

The probe-before-build discipline that drove the exit-logic work: before coding any entry/exit rule,
**characterize the ignition dynamics on real data and let the numbers kill bad ideas cheaply.** This session
it refuted six plausible-but-losing rules (green-bar / rising-volume / 2-bar-rising entry filters; blanket &
climactic wick-exits; gentler & aggressive tp ladders) and located the real lever (the **exit**) — at ~30 s
per probe vs ~2 h per training run. **What a probe covers — and does NOT — is recorded here so future reads
aren't over-trusted.**

### What a probe measures
Each probe instantiates `EventRungEnv` (torch-free) on a returns split and reads two precomputed arrays:
- `env._ignite[bar, token]` — every bar where the **rung-0 signal** fires (`surge ≥ vol_mult & rising>0 &
  cushion>0 & ema_up`; env defaults: vol_mult 2.5, vol_spk 24 = the 1-day rising window, vol_base 168,
  vol_fast 4 = the rolling-volume window, ema_span 72).
- `env._px[bar, token]` — the env's price index (`cumprod(1+r_alt)`), the SAME basis the agent's PnL uses
  (NOT the raw candle prices).
It enumerates every ignition for the `env.universe` tokens and measures forward outcomes (run-up to local
high, drawdown, captured-by-some-exit) over a fixed forward window.

### Scope — what a probe is NOT (read before trusting a finding)
1. **Not the agent's trades — it's the OPPORTUNITY set.** A probe sees *every* ignition the rung-0 RULE
   could take, not the RL agent's selective entries/exits/sizing. The numbers are closest to "what the rung-0
   rule sees." A well-selecting agent can **beat the probe averages** (they average over all ignitions,
   including junk a good agent would skip) — so a mechanical-exit "breakeven" is a floor, not the agent's
   ceiling.
2. **Only the vol-top-8 universe, fixed at the split start.** Not all ~14 pool tokens, and NOT the
   **weekly-rotating** universe the deployed strategy re-picks each Monday. A token that turns volatile
   mid-split is invisible to the probe but tradeable live.
3. **Train + val only, each as one continuous window.** NOT the frozen TEST (reserved), and NOT the
   cold-weekly session structure ($10k Monday resets) the strategy actually deploys in.
4. **Forward windows, not real holds.** "Run-up to local high" is the max over a fixed 24/48/72-bar window
   from entry — the *opportunity*, not however long the agent holds. Captured-return sims apply a fixed exit
   rule (trailing stop / tp ladder), again not the agent's learned exit.
5. **`_px` carries the known r_alt-vs-candle drift** (≈ a few %); fine for ratios/forward returns, but
   absolute levels are the index, not the candle.

### How to run one (reproducible, laptop, torch-free)
`.venv\Scripts\python.exe -c "..."` with:
```
sys.path.insert(0,"scripts"); sys.path.insert(0,"src")
from train_rl import load_data, time_split, build_volume_panel   # + _load_token_ohlcv for candles
from trader.train.event_env import EventRungEnv
returns,btc,_,liq = load_data(); train_r,val_r,test_r = time_split(returns)
vol = build_volume_panel(list(returns.columns), returns.index)
env = EventRungEnv(r, btc, liq, volume=vol, episode_bars=len(r)-WARMUP-1, k=8,
                   warmup=168, universe_mode="voltopk", seed=0)
env.reset(start=WARMUP)            # no stepping needed — _px/_ignite are built in __init__
px, ig, uni = env._px, env._ignite, env.universe   # iterate uni x bars where ig[b,j]
```
Pattern reference: `scripts/probe_wick.py`. Forward-return basis matches the trainer's own
`evaluate_event_policy` (same `_px`/`_ignite`), so probe numbers are consistent with training-eval on the
same window.

### The complementary probe (the realized view — NOT yet run)
To see the AGENT's *realized* trades vs this opportunity baseline: fetch a published bundle's
`tk_<slug>_trades.json` + `token_pnl.json` (the env's exact ledger, post the 2026-06-16 export fix) and fold
into round-trips → realized capture / win-rate / per-token PnL on the **deployed weekly-rotating universe** —
the piece these opportunity-set probes don't cover.

Findings from the session's probes (run-up profile, "entries fine / exit is the alpha", the refuted levers):
[[Experiment Log]] §"PROBE SESSION".

## 2026-06-19 — rung-0 demoted to a reference floor; the substrate-error lesson; the EMA-break leak + sideways suppression (the new open lever)

The session that reset the gate's DIRECTION and located a real exit leak by probing the **policy**, not
the rule. Restate the bar first: an RL trading agent scored on live PnL over one held-out week (June
22–28), HARD ~30% max-drawdown DQ gate, ≥1 trade/day; the deploy week is forecast flat/bear. The honest
grader is the **cold-weekly** sim (fresh $10k Mon-00:00, causal vol-top-k re-picked each Monday, no
cross-week holds) over val+test weeks.

### The gate's DIRECTION RESET — rung-0 demoted to a reference floor (commit `8009973`)

The same demotion Buy&Hold got 2026-06-15 (see §DRIFT POST-MORTEM) now applies to **rung-0**: it is
demoted from the BINDING gate to a **REFERENCE floor**. The corrected gate: a config earns a version iff
it (1) **survives the DQ gate** (worst single-week maxDD < ~30%, still HARD) AND (2) **improves on the
PREVIOUS best iteration (the champion)** on the honest cold-weekly metric. rung-0, Buy&Hold and Random
are all computed and reported references — **none binding.** The loop north-star becomes
**margin-vs-prior-champion**; the drift alarm fires on **no-improvement-over-champion**. Deploy on the
best single seed; select configs on the seed-mean (see [[seed-mean-is-iteration-not-deployment]]).

This is already written into the [[Agent Communication Contract]]. **CODE TODO (not yet shipped):**
`weekly_gate` / `honest_gate` / `loop_control.decide` / `rl_north_star` still enforce `beats_rung0` —
the code still binds on the rule and must be migrated to the margin-vs-champion gate.

### The substrate-error lesson — probe the POLICY, not the rule, for any exit / profit-taking question

A methodological flaw corrected (the user was right). Earlier EMA-break probes all ran on the **rung-0
RULE**, whose only exits are sell-on-weakness (trailing stop / EMA-cross), so it structurally gives back
pumps — a rule-substrate probe can never reveal how a POLICY (which has the `tp_rungs` profit-take ladder
+ learned hold/override) actually behaves. `scripts/probe_policy_exits.py` on `ef2-s0` over OOS weeks:
exits are **74% EMA_BREAK / 15% PROFIT_TAKE / 7% ROTATION_OUT / ~5% stops.** LESSON: **probe the
deployable substrate (the policy), not the rule, for any exit / profit-taking question.**

### How the policy actually captures pumps — it HOLDS clean rips to the close

The P&L engine is **not** profit-taking — it is the few pumps the policy **HOLDS** to the week close. In
the `probe_policy_exits` decomposition: **4 held-to-end positions made $6235** (best TAG $5432) vs **66
exited trades making $4782 combined.** TAG (+170% rip) was captured by HOLDING (no exit fired,
force-closed near the high), not by profit-taking. The policy's OWN EMA-break sells **give back +3.5%
mean fwd-48h** (55% recover) — i.e. the EMA-break exiting small dips in sideways is a real leak, while
the upside comes from holding clean rips through to the close plus the 15% of exits that profit-take.

### The EMA-break leak — sideways shake-out before the pump (the FF case)

The exit uses a 72-bar EMA (~3 days hourly), confirmed identical in rung0 and event_env; the EMA-break
is the weakness-exit (sell when price < ema). It **fires on shallow noise-dips during tight sideways
consolidation**, shaking the agent out before the pump. The anecdote: in the eff-s1 sim, FF was sold
Apr 9 12:00 with reason **EMA_BREAK** on a −0.1% cushion NOISE break at the tight EMA during the
pre-rip consolidation; the 48h cooldown then locked re-entry, and FF's real rip-ignition (Apr 10 18:00,
surge 10 = the peak bar) was missed flat — the agent gave up the +106%/+151% rip entirely.

EMA-break was conditioned FOUR ways, ALL ON THE RULE (the flawed substrate above), and all refuted:
(1) P&L gate (the prior P-EMA-COND); (2) consolidation/low-vol suppression — the consol-and-shallow cell
(the FF pattern) was the WORST (terminal fwd-48h −5.4%, 8% win rate; FF was the 8% anecdote, most
consolidation breaks keep falling); (3) deep-dip INVERSE — refuted, the break-depth forward-return trend
is MONOTONICALLY NEGATIVE (deeper = worse), and the earlier +13.9% high-vol reading was a thin
single-dimension artifact that vanished with proper conditioning; (4) EMA-PERIOD sweep (50–240, global
and exit-only) — no net win, a longer EMA gives no robust val gain, worse test return, and MORE DQ weeks
(it holds losers/givebacks longer). The 2 cold weeks that breach the 30% DD gate are Mar 23 (a pure
loser: peaked 1.04x, ended −25%) and Apr 13 (caught SIREN's +163% pump to a +23% peak, then GAVE IT ALL
BACK and ended −14%); a longer EMA makes both worse.

### THE NEW OPEN LEVER — sideways EMA-break suppression (BUILT `abf089b`, RETRAIN PENDING)

The fix the policy-substrate finding demanded: when a break is **SHALLOW** (cushion > −`shallow_break_max`)
AND the token is **QUIET** (24h realized vol < `consol_vol_max`), **do NOT fire the EMA-break.** The
loss_floor (−20%) and the trailing stop stay fully active, so real breakdowns still cut (bounded
downside) while the position survives the noise to catch the pump — **asymmetric** (bounded downside via
the floor, large upside via pump capture). The user chose the "shallow + quiet" definition as the most
surgical (a deep break or a high-vol break still cuts). Both knobs 0 ⇒ OFF, byte-identical (30 env tests
pass). Wired through `REWARD_KEYS` (`shallow_break_max`, `consol_vol_max`) + `train_event` flags +
provenance + the `simulate` loader, so it is sweepable via `rl_loop` and graded honestly.

Mechanical check on the rule (the FF fixed-13 week): OFF sells FF twice (Apr 7 + Apr 9 EMA_BREAK, FF PnL
−$115); ON suppresses both and holds FF through the chop. **STATUS: committed `abf089b`, NOT yet
retrained.** Planned next experiment: retrain `ef2` + `shallow_break_max=0.02` + `consol_vol_max=0.015`
(the FF-validated thresholds), 4 seeds, graded honest cold-weekly vs `ef2`, pending the user's
green-light (shared desktop — [[Remote Capabilities]] runbook applies). **Open co-factor:** ROTATION_OUT
can still swap a held-but-flat token out before its pump (a second shake-out mechanism) — the next thread
if suppression helps but rotation caps it.

### Fixed-13 universe — a CLOSED BRANCH (loses to causal vol-top-k)

The proposal (drop the 7 most-BTC-correlated / lowest-vol tokens — ADA, SFP, XRP, BabyDoge, LINK, LTC,
XAUt — to a FIXED 13-token set so high-vol spikes like FF are never selected out mid-week) was retrained
in-distribution (`ppo-event-rdLe4-eff-2345fd6`, `universe_mode=fixed` on the 13, vol_mult 2.0, 4 seeds):
honest cold-weekly seed-mean **+5.9%/wk** (best seed s1 +7.7%, DQ-safe). vs `ef2` re-graded at the
correct vol_mult 2.0: seed-mean **+8.5%/wk** (best seed s0 +10.0%). **ef2 BEATS fixed-13 on BOTH config
seed-mean (8.5 vs 5.9) and best-seed (10.0 vs 7.7) → fixed-13 is a CLOSED BRANCH; causal vol-top-k stays
the substrate.** The `universe_mode="fixed"` mechanism remains as tooling. (The +51% val vs 25%/6mo
discrepancy that opened the session was a grader mismatch — continuous eval picks the universe ONCE at
val open and compounds; the cold-weekly sim re-picks causally every Monday and is the deployment-honest
grader. A `vol_mult` provenance bug — never recorded, so the sim defaulted to 2.5 while ef2 trained at
2.0 — was found and fixed; all 4 ef2 seeds were re-published at the correct 2.0. Detail: [[Experiment Log]]
§2026-06-19, [[Simulated Market]].)

## 2026-06-19 — the ≥1-trade/day floor is a DEPLOY GUARDRAIL, now implemented (does NOT touch the honest gate)

The competition's **≥1-trade/day rule (Rule-1, a hard DQ axis)** has always been treated here as a
**deploy guardrail — a forced daily rebalance — NOT a strategy discriminator** (see the §2026-06-18
note: "In AI Training terms, the activity floor was always meant to be a deploy guardrail ... not a
strategy discriminator"). The event champion is **selective** (idle between ignitions), so several cold
weeks legitimately miss the floor; that is correct policy behavior, not a training failure to chase.

It is now **implemented as the BNB↔USDT compliance overlay** (a separate sleeve), so the floor is
satisfied without re-shaping the policy or the reward:

- **The overlay is OFF the honest grader.** It runs as a **separate sleeve** in both the live runner
  (`src/trader/agent/event_runner.py`, `d936101`) and the sim replay (`scripts/simulate_weekly.py`,
  `b43d0e2`): each UTC day BUY 3% of equity into BNB at 01:00 and SELL back to USDT at 23:00 (two
  recorded trades/day, flat overnight). It is **NOT** added to `recon_pnl` / the week return / DD / the
  cold-weekly `weekly_score` — the strategy env stays at exactly $10k for fill/obs-parity, so **the
  leaderboard rank is UNCHANGED (no silent re-grade)**. The sleeve's realized PnL is tracked separately
  (`compliance_pnl_usd` in the equity ledger row; `weeks[].compliance_pnl` in the bundle).
- **It is a guardrail, not a signal.** Pure module `src/trader/agent/compliance.py`
  (`COMPLIANCE_TOKEN=BNB`, `BUY_HOUR=1`, `SELL_HOUR=23`, `DEFAULT_FRAC=0.03`); routed through the same
  trader.risk guardrails; idempotent by **bar-day** so a restart/re-tick never double-trades. It is
  **not** part of the decision core and must never be confused with a learned entry — the selective
  thesis (sit in cash through chop) is intact.
- **Honest caveat (training-relevant):** the 01:00→23:00 hold is a **22-hour daily long-BNB exposure**,
  so it is **directional** — it DRAGS in a down/bear week (a sample week realized −$74 = −0.74% of the
  $10k book) and gains in an up week. Given the bearish live-week thesis it will tend to drag; that drag
  is the price of Rule-1 and is tunable (BUY_HOUR / SELL_HOUR / DEFAULT_FRAC). It sits in the **separate
  sleeve**, so it does NOT bias the policy's honest cold-weekly evaluation.

**STATUS:** paper/sim logic only, committed + pushed (`d936101` + `b43d0e2`; now on `main`
after the 2026-06-21 branch reconciliation, §below). LIVE on-chain execution of these trades on June 22 still needs the **TWAK
signing path** (separate, not built — this fixed BNB↔USDT swap is the ideal first live trade). The
end-to-end dashboard render of the compliance asset is NOT yet verified on the desktop (pending a
`simulate_weekly` re-run after the sbq sweep). Execution detail (runner overlay, schedule, sleeve
PnL, dashboard schema fields) lives in the Live Forward-Run Harness note + [[Simulated Market]] — this
note owns training, not execution.

## 2026-06-21 — sbq-s1 is the CHAMPION + deploy pick (frozen-TEST CERTIFIED, now live); two env knobs swept

The session that turned the §2026-06-19 sideways-suppression lever into a deployed champion and spent the
frozen test on it. **The deployed/champion model is now `sbq-s1`, NOT `ef-s2`/`ef2`** — every "deploy pick
= ef-s2" / "champion = ef2" reference in the older dated sections above is superseded by this entry. The
Simulation leaderboard's PRIMARY rank metric also changed from `weekly_score` (OOS per-week mean) to
**`cumulative_score`** (6-month cumulative return = `windows.overall.ret_sum`); `weekly_score` is still
computed + displayed. Board now #1 `sbq-s1` (+125% 6-mo) / #2 `eff-s1` (+104%) / #3 `fxsbqc-s0` (+86%).
Detail + the export-bug fixes + branch reconciliation → [[Experiment Log]] §2026-06-21, [[Simulated Market]],
[[Dashboard Leaderboard]].

### The champion — `sbq-s1` (suppression substrate, certified)

`ppo-event-rdLe4-sbq-3c84b4a-s1` — `universe_mode=voltopk` k=10, `vol_mult=2.0`, **sideways EMA-break
suppression ON** (`shallow_break_max=0.02`, `consol_vol_max=0.015` — the FF-validated thresholds the
§2026-06-19 "new open lever" built at `abf089b`), `reward_mode=entry_forward`, RecurrentPPO LSTM-256. The
suppression knob is the **substrate**; the two knobs below were swept on top of it this session.

**FROZEN-TEST CERTIFICATION (now CONSUMED).** `sbq-s1` was selected on VALIDATION (the loop gates on val;
best-seed by val return), so the TEST split was genuinely held out. Held-out TEST (5 cold weeks, fresh
$10k each): **+58.6% sum / +11.7%/wk mean / 5-of-5 winning weeks / worst-week DD 8.8% / DQ-safe** — HELD
UP vs validation (+7.1%/wk, 67% win), no overfitting collapse → **PASS.** Caveat (stated plainly):
**pump-concentrated** — W24 +37.6% is a single token ~3×; the other 4 weeks +0.3% to +16.9% (big in
volatile weeks, near-flat-but-positive / capital-preserving in quiet ones). Per the meta-overfit guard,
**NO further tuning to the `sbq` config now that test is spent** (the `ppo2-real` +83%→+11% collapse is
why test is reserved).

**Now deployed.** `sbq-s1` replaced `ef-s2` on the EC2 paper harness (surgical 2-file inference update,
weights pulled via boto3; matched train/serve env — `sbq` was TRAINED with suppression so it SHIPS with
the suppression env, no skew, unlike `ef-s2` which was served on the frozen pre-suppression env). The
live harness owns that detail; this note owns training. See the Live Forward-Run Harness note.

### The two env knobs swept this session — both config-gated, default-OFF (byte-identical when OFF)

Both were graded honest cold-weekly against `fxsbq` (`ppo-event-rdLe4-fxsbq-62800ff`, fixed-13 + suppression,
the FF-thesis vehicle; its val cold-weekly seed-mean **0.543** is the bar). The probe-before-build discipline
held: each had a cheap offline probe whose read was honored in the verdict.

| knob | run | lever | val seed-mean vs `fxsbq` (0.543) | worst-week DD | verdict |
|------|-----|-------|----------------------------------|---------------|---------|
| `rotate_pump_block` | `fxsbqr` (`71bdfc9`) | in loser-funded rotation, do NOT liquidate a holding to fund a candidate that already ran up > `rotate_pump_block`(0.15) over the prior `rotate_pump_win`(24h) bars | **0.505** (return wash / hair below, no DD benefit) | — | **NO-GO — REFUTED** |
| `candle_exit` | `fxsbqc` (`d0f926e`) | if HOLDING an in-profit position and the bar is an INVERTED HAMMER (`candle_uw_min`0.5 / `candle_lw_max`0.25) or a DOJI (`candle_doji_max`0.10), PROMPT an exit — DISCRETIONARY (rule-default sells, agent can hold), reason `CANDLE_EXIT`, precedence BELOW trailing stop + EMA-break | **0.540** (return WASH) | **14.0% vs 20.0% = DD-BETTER** | RETURN-NEUTRAL, RISK-REDUCING (DQ-protective) |

- **`rotate_pump_block` (anti-chase rotation brake) — REFUTED.** Motivation: `fxsbq-s1` Week-21 the FF→ZEC
  rotation at Apr-10 01:00 SOLD FF to buy ZEC's SECOND pump leg (entry ~5h after a local top, +44% over
  ZEC's cycle); both legs lost (FF −1.2%, ZEC#2 −1.6%). A thin offline calibration on the published-s1
  realized trades had warned the run-up penalty was suggestive only above ~15% with n~5 / no test-split
  coverage — and the sweep confirmed it: return wash, no DD benefit. **The anti-chase / rotation lever is
  refuted as a net win; `rotate_pump_block` stays default-OFF.**
- **`candle_exit` (candlestick exit) — RETURN-NEUTRAL, DD-BETTER.** Motivation: Q-token traps — W19 an
  inverted hammer one bar after entry preceded a dump (−20% floor); W16 a doji (Mar 4 21:00). The offline
  probe (`probe_candle_exit.py`) found the signal ~flat/noise on held-in-profit positions (forwards ~0%,
  inconsistent across splits); built per the user's direction anyway with the gate as arbiter. Verdict:
  return-neutral but **worst-week DD 14.0% vs 20.0% = DD-protective** — the agent learned to honor the
  prompt SELECTIVELY (it did NOT dump winners). A risk-reducing, return-neutral add. Note `fxsbqc-s0` sits
  #3 on the cumulative-return leaderboard (+86%).

Both knobs default 0 / OFF (byte-identical to the prior reward stream when off), consistent with every prior
env lever (the §2026-06-19 suppression knobs, `regime_base`, `basket_default`, `rule_default`).
