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
