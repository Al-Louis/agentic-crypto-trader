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

**`turnover`** — best mean return among configs whose **mean** drawdown clears the gate.
Provisional: its *worst* seed still breaches (see standings). Reproduce: `champion.json`.

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

### Decided next

Make the reward terms **composable** (currently mutually exclusive) and run Sweep #2 to hunt for
**max return subject to worst-seed DD < 30%** — likely `realized`'s engine + a drawdown/turnover
brake. Then confirm on the **frozen test split**. See [[Strategy Logic]].

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
