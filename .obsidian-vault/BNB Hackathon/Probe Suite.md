# Probe Suite — the alpha-hunt program

Designed **2026-06-17** by an 8-agent quant workflow (recon → per-facet design → adversarial verify → synthesis), grounded in [[AI Training]] §"Probe methodology & scope" and the [[Experiment Log]] findings. Every probe is bound to the **honest gate** (beat the rung-0 RULE OOS + survive the ~30% max-drawdown DQ on the cold-weekly eval; Buy&Hold / Random are *reported*, never the bar). **Thesis guardrail:** the goal is the LEARNED agent, not a new hand rule — every surviving probe must inform an **obs feature** the LSTM conditions on or a **reward shape**, never a fixed gate (fixed gates are the refuted ledger).

## The reframe — the binding constraint is the EXIT REWARD, not information

Findings 1 + 3 (the exit is the alpha; entry selection adds no edge yet) establish that the obs-hypothesis family is largely *exhausted* on this architecture. The single highest-value arm is therefore a **giveback-from-peak reward shape** — validated NOT by a clairvoyant oracle but by a **causal surrogate-exit test**: can an obs-only exit (driven solely by state at bar *b* — giveback, unreal, surge, cush, held_frac) close a *positive OOS-val fraction* of the +12–15% available run-up gap? The oracle "gap closed" number is a **ceiling only**; letting it become a success bar is the exp1→exp5 drift, and the drift alarm is armed against it (and against any re-bind on B&H).

## Trade-scenario taxonomy (the history-labeling library)

Causal definitions; "clean-runner"/"fakeout" labels that use forward bars are for **retrospective labeling only**, never live features.

| Scenario | Causal definition (sketch) | Why it matters |
|---|---|---|
| **clean-runner** | ignition → monotone advance to a 72h high, run-up ≥ +10%, never breaches the 25% trailing stop before peak | where the +12–15% uncaptured gap lives — HOLD + SCALE these |
| **fakeout-reversal** | pop (<+5%) then roll-over breaching loss_floor/stop in 24–48h; fwd-48h strongly negative | the **modal** ignition (38% fwd-48h win rate) — EXIT FAST / SIZE DOWN; green/rising-vol ignitions land here (finding 2) |
| **multi-leg runner** (ZEC Apr-9 class) | held + in-profit token fires a FRESH ignition mid-move (39% of ignition-bars) | the **premise of `scale_in`** — validated by P-REIGNITE |
| **climax top / blow-off** | late-in-move bar, stretched run-up, surge *collapsing* from a within-move peak (read UNCLIPPED vrec/vbase — `_surge` clips at 10) | the exhaustion regime to BANK near peak — distinct from the refuted blanket-wick / wick+surge≥8 sellers (those fired on bar shape regardless of move-state) |
| **slow grind** | steady (non-climactic) volume, surge flat across the move, positive run-up | contrarian-consistent "good" continuation — HOLD/SCALE |
| **chop / no-follow-through** | bounded oscillation, run-up <+5%, drawdown shallow, never resolves | fee+gas drag regime (the ≥1-trade/day floor forces participation) — minimal sizing; ensure costs hit baselines equally |
| **detonation trap** | surge≥8 & rising≤−0.15 (already masked by `rising>0` + det_blacklist) | the Q-detonation; must be EXCLUDED from the surge-sizing probe so upper surge bins are the valid-ignition continuum |
| **spent-move re-buy** (the bleed) | flat cooled token re-igniting after a prior ignition that already paid (ret_since >~10%) | the `cycle_obs` population — sign known (already-paid fwd-24h −6..−7% vs fresh −1..−2%), but bound by the refuted rdLc sweep |

## Prioritized probe suite

Status: ☐ pending · ▶ running · ✅ done · ✗ refuted · ~ inconclusive

| # | Probe | Facet | What it answers | Effort | Decision value | Status |
|---|---|---|---|---|---|---|
| 1 | **P-EXIT-REWARD** | exit reward | does a giveback-from-peak reward close a +OOS fraction of the gap via a **causal surrogate** (oracle = ceiling only)? | med | **HIGHEST** — the only arm directly on the alpha | ⚠ 2026-06-17 — **NO-GO** (gate failed): surrogate OOS gap +6.4%, CI straddles 0, single-token, no incremental over giveback, only 20 val trades. Run-up REAL but not learnable on current obs/data |
| 2 | **P-REIGNITE** | continuation | do held+in-profit re-ignitions (bucket B) beat a matched single-leg control (A) on fwd run-up? | cheap | **VERY HIGH, bidirectional** — gates the live `scale_in` sweep | ✅ 2026-06-17 — **SELECTION refuted** (well-powered null), capacity OPEN |
| 3 | **P-DECEL** | continuation/exit | does acceleration roll-over (signed v_fast−v_slow) DISCRIMINATE runners from reversals *among bars at the same giveback*? | med | HIGH — a new exit/scale obs slot | ☐ |
| 4 | **P-SURGE-SHAPE** | entry sizing | is P(+10%)/mean-fwd-48h *non-monotone / decreasing* in surge magnitude among valid ignitions? | cheap | HIGH — reads `env._surge` directly, zero drift | ☐ |
| 5 | **P-PULLBACK** | entry sizing | do pre-ignition pullback depth + coil (short/long realized-vol ratio) carry continuous fwd IC the obs lacks? | med | MED-HIGH — the contrarian setup state variable | ☐ |
| 6 | **P-CLUSTER** | entry sizing | is an ISOLATED ignition alpha vs a BROAD co-firing cluster beta froth — surviving a btc_trend partial-effect control? | med | MEDIUM — alpha-vs-beta, on the low-BTC-corr thesis | ☐ |
| 7 | **P-CYCLE-CI** | entry | which single cycle component (bars- vs ret-since prior ignition) survives CI+FDR? ship one slot or none | cheap | MED-LOW — a re-pass of `probe_knowledge.py`, NOT new | ☐ |
| 8 | **P-MOVEAGE** | continuation | does causal within-move maturity (÷ token-typical run length) beat the raw `held_frac` obs on Spearman IC? | heavy | LOW-MED — highest leakage; expected INCONCLUSIVE | ☐ |
| 9 | **P-VOLREGIME** | regime / entry+exit | do two GENUINELY-NEW features — a vol-regime ratio (`σ_fast/σ_slow`) + a robust return-impulse (`\|r\|/median\|r\|`) — carry *incremental* IC over the existing obs (which has NO realized-vol term)? | cheap | MED — the one obs the hunt-map left open (NEW info, not a `surge`/`cush` re-pack); skeptical prior (vol-redundancy null 5×) | ✗ 2026-06-17 — **NULL on the gate (6th vol-redundancy null); EXIT arm INCONCLUSIVE (data-starved).** Neither feature clears incremental-IC CI-excludes-0 + sign-stable + FDR. Wire nothing |

**Methodological refinements (what makes each honest — folded in by the skeptic pass):**
- **Continuous IC on a temporal holdout is the PRIMARY deliverable**; bucket grids are human-readable illustration only (thin +10% cells, ~13% base rate, ~321 val ignitions).
- **Facet-level BH-FDR** (not just within-probe), one pre-registered headline per probe, baselines reported (run_up-so-far-only / held_frac-only / btc_trend-only) so *incremental* value is visible.
- **Move-clustered / 168h-block / token-paired bootstrap** for autocorrelation; collapse consecutive-bar ignitions on one token to one obs per (token, move).
- **Pre-registered n-floors → INCONCLUSIVE (not refute, not pass)** below them.
- P-DECEL: drop the v_fast/v_slow *ratio* (unstable), use the **signed difference**; replace the tautological lead-over-the-stop metric with a **giveback-matched discrimination** test; sign-agnostic (the agent learns the sign).
- P-EXIT-REWARD: the **oracle dev\*** number is labelled UPPER BOUND; the **causal surrogate OOS gap** is the real metric; per-trade DD augmented with a cold-weekly PORTFOLIO-DD sim (the actual DQ object).
- P-MOVEAGE: HARD-ASSERT `b0+72 ≤ b` in code + a synthetic unit test; cross-token pooled trailing-median fallback (per-token starves early in a split).

## Execution order

1. **WHILE THE SWEEP TRAINS (cheap, torch-free, no desktop contention):** P-REIGNITE first (gates `scale_in`; a refute prevents a wasted sweep), then P-SURGE-SHAPE (cheapest, reads `_surge` directly), then P-CYCLE-CI (resolves one-slot-or-none + hardens the rdLc caveat).
2. **THE PRIORITY ARM:** P-EXIT-REWARD — stand up the giveback-from-peak reward via the shared `event_reward.py` pure-function pattern; run the offline landscape with BOTH the oracle upper-bound AND the causal surrogate gap. **Do not launch an exit-reward sweep until the surrogate (obs-only) gap is shown > 0 OOS on val.**
3. **Medium continuation/entry features in decision-value order:** P-DECEL → P-PULLBACK → P-CLUSTER. Each that survives facet-level FDR becomes a **single-variable** obs add on the `wkw` base (never bundled — the discipline that kept `wkw` clean).
4. **LAST, only if a continuation feature survives:** P-MOVEAGE (heavy, highest leakage; gated behind Spearman-beats-`held_frac` + the causality assert).

**Sequencing rules:** launch sweeps ONE AT A TIME, single-variable, on the `wkw` base via `scripts/rl_loop.py` (NOT the MCP `rl_loop_*` tools — SSH goes stale); the desktop is a SHARED box (check idle/ask before launch); publish static-JSON from the laptop. The frozen TEST split is spent ONCE, only on the single final champion candidate after the cold-weekly VAL gate passes with paired-bootstrap CI-low > 0 vs rung-0.

## RL integration plan

Priority = the exit. The agent already has the exit head (`_do_exit`/`_do_profit`) and the `giveback` obs slot; what it lacks is (a) a reward crediting banking near the peak and (b) state features that discriminate a runner from a reversal at the exit moment.

1. **Reward shape** — if P-EXIT-REWARD's causal surrogate closes a +OOS-val gap, add a giveback-from-peak / peak-capture term via a NEW pure function in `event_reward.py` (the ONE shared definition imported by both env and preflight — the exp3 false-PASS guard). Sweep `kappa` as a Pareto curve; keep the quadratic `dd_penalty` (`dd_soft` 0.15 → `dd_gate` 0.30) intact so the DQ binding is unchanged. The reward, not a new obs, is the binding constraint.
2. **Exit/scale obs features** — each continuation feature surviving facet-level FDR with a sign-stable IC gets exactly ONE appended obs slot, added as a SINGLE-VARIABLE flag on `wkw`: acceleration/curvature (P-DECEL) → exit+scale; surge-decay (P-SURGE-SHAPE/P3) → hold-vs-bank; pullback_depth + coil (P-PULLBACK) and cluster_frac partial-effect (P-CLUSTER) → entry sizing. The LSTM learns the (possibly non-monotone, contrarian) MAP.
3. **Scale-in** — `scale_in` (commit d68c824) launches as `wkw + scale_in` (single variable) ONLY if P-REIGNITE passes; the blended `cost_px` + per-token cap already fence the floor and prevent pyramiding.
4. **Cycle** — at most ONE `cycle_obs` slot (the P-CYCLE-CI survivor) or none; explicitly NOT a re-run of the refuted rdLc sweep.

**Guardrails on every addition:** tested against the honest gate (paired-bootstrap CI-low vs rung-0), default-OFF/byte-identical until proven, never converted into a fixed gate. If optimizing any added term drives the agent toward buy-everything, the metric is wrong, not the thesis — sound the drift alarm ([[benchmark-driven-drift]]).

## Open risks (the guards)

- **Thin val cells** — +10% events rare (~13% base rate, ~321 val ignitions); several probes will honestly land INCONCLUSIVE. Continuous IC is primary; buckets illustrate.
- **Multiple comparisons / fishing** — facet-level BH-FDR; one headline per probe; baselines reported.
- **Autocorrelation pseudo-replication** — move-clustered / 168h-block / token-paired bootstrap; collapse consecutive-bar ignitions.
- **Oracle leakage** (P-EXIT-REWARD) — the dev\*-exit is clairvoyant; label UPPER BOUND, the surrogate OOS gap is the metric.
- **Forward-window leakage** (P-MOVEAGE) — bars-to-local-peak peeks; hard-assert `b0+72 ≤ b` + unit test, else KILL.
- **Universe selection** (verified clean) — `_pick_universe` reads trailing `_std[at-1]`, NOT the full-window vol-rank that once peeked at late pumpers; all probes inherit this; read features as RATIOS within one token (the `_px` r_alt-vs-candle drift).
- **Regime / scope** — probes use vol-top-8 fixed at split start over ONE continuous window with forward WINDOWS, NOT the cold-weekly $10k-reset rotating-universe deploy structure. A probe edge is necessary but NOT sufficient — every surviving feature must re-clear the honest gate on the cold-weekly eval.
- **One-week live variance** — the June 22–28 window is a single draw; `wkw`'s edge leans on one seed (s3 +16.5%; s0 −2.9%). Frame DQ-protective worst-week DD as a first-class objective; report paired CI-low, never crown on one lucky week.
- **Cost realism** — STATIC liquidity caps `amm_cost_usd` realism; in chop, fee+gas drag can masquerade as a deficit vs B&H — apply costs equally to all baselines.
- **Thesis / metric drift** — `wkw` LAGS B&H (+17.1% vs +20.6% bull); B&H stays reported, never the bar (the buy-everything-overlay lesson).

## Status log

- **2026-06-17** — suite designed (8-agent quant workflow). **P-REIGNITE launched** (torch-free, laptop-local) to validate the just-shipped `scale_in` feature against forward data while the `ppo-event-rdLe4-wsi` (wkw + scale_in) sweep trains. Next cheap probes queued: P-SURGE-SHAPE, P-CYCLE-CI. The priority arm is P-EXIT-REWARD (the exit-reward reframe).
- **2026-06-17** — **P-REIGNITE DONE (`scripts/probe_reignite.py`, uncommitted): SELECTION premise REFUTED, CAPACITY premise open.** Held + in-profit re-ignitions (bucket B; VAL n=169, TRAIN n=228 — well above the n≥30 floor) carry strong *absolute* run-up (VAL fwd48 +15.7% mean / +11.1% median, 92% win) — adding to a held winner is NOT chasing duds — but **B−A vs fresh single-leg ignitions is a well-powered NULL**: small and every 95% CI straddles 0 on both splits (VAL fwd48 B−A −1.3% [−17.5,+14.5]; surge-matched −3.7%..+4.0%). Held re-ignitions are *not better entries* than fresh → `scale_in` buys **no selection edge**. **Narrative correction:** the ZEC Apr-9 +16.2% poster-child lands in bucket **A** — under the *rule* the prior leg was already stopped out (rule FLAT → fresh entry); the "missed re-ignition" was the *trained s3 agent's* sliver-hold, a different book. **Survives:** the *capacity* premise — the rule funds only ~6–12% of flat ignition candidates (39/637 train, 20/163 val; usually capacity-constrained), so `scale_in` could still help by deploying more capital into the equally-good held-winner stream — a sizing/portfolio question only the `wsi` sweep's equity curves vs the honest gate can judge, NOT a run-up probe. Verdict = **STOP-AND-RECONSIDER on selection**; let the `wsi` sweep finish as the capacity test but temper expectations (a null / slight-negative is consistent); the clearer next big arm is **P-EXIT-REWARD**.
- **2026-06-17** — **P-EXIT-REWARD DONE (`scripts/probe_exit_reward.py`, uncommitted): NO-GO — do not spend a desktop slot.** Build → 3 adversarial skeptics (leakage SOUND, incremental FLAWED, stats INCONCLUSIVE) → decision NO. The oracle confirms the run-up is **REAL** (+9.3%/trade train, +14.2%/trade val above the rule's ~breakeven). But the CAUSAL surrogate (obs-only, fit-train/eval-val) closes only **+6.4% (H24) / +13.5% (H48)** of the gap; the move/token-clustered **95% CI straddles 0** [−10%,+20.4%]; the entire edge is **one token (Q — leave-it-out → −1.5%, below the rule)**; it wins **<50% of trades** (4/20); and it shows **no significant incremental over the giveback obs the policy already carries** (giveback-dominated logit, fair giveback-logit floor → incremental +12.1% [−23.7%,+70.4%], inside noise). Leakage-clean (independently re-audited, reproduces to the digit; the train-argmax threshold 0.95 ≠ val-argmax 0.75 = the no-peek signature), drift alarm held (oracle = upper-bound only, never the bar). **THE BINDING LIMIT IS DATA/POWER:** the rule funds only **39 train / 20 val closed trades** (~8 token-clusters) of ~952/321 ignitions — >95% of ignitions never become trades an exit can act on. The exit-is-alpha thesis is **unresolved-by-data, NOT refuted**.

- **2026-06-17** — **P-VOLREGIME specced** (from the `mlmodelpoly` external-repo review; see
  [[reviewed-mlmodelpoly-repo]]). The two transferable ideas from that repo (vol-regime ratio + robust
  impulse) became a single gated obs-probe targeting the ONE opening the hunt-map left — a *genuinely
  new* obs axis (verified: no realized-vol term in `_obs`), gated on **incremental IC over the existing
  obs**. Skeptical prior (vol-redundancy null 5×); cheap/torch-free; does not block the frozen-`wkw`-TEST
  call. Script to build: `scripts/probe_volregime.py`. ☐ pending.

- **2026-06-17** — **P-VOLREGIME DONE (`scripts/probe_volregime.py`, uncommitted): NULL on the gate —
  the 6th vol-redundancy null; EXIT arm INCONCLUSIVE.** Causality self-test PASS (recompute from
  `returns[:bar+1]` matches the array to 1e-9). **ENTRY** (well-powered: 109 train / 38 val ignitions,
  move-collapsed): `volreg` is the only sign-stable feature — partial-IC vs H48 run-up **+0.082 train /
  +0.135 val** — but **both CIs straddle 0** (p=0.23/0.21) and it FAILS facet BH-FDR; `rimp` weakly
  negative & non-robust (TRAIN H48 CI just excludes 0 p=0.044 but fails FDR; VAL straddles, p=0.79). So
  the only NEW-info candidate the hunt-map left open is null-leaning-INCONCLUSIVE on entry — fails on
  power (thin val cells), not a believed edge (drift alarm held). **EXIT** (rule discretionary stop/ema
  exits): **37 train / 17 val — val BELOW the n≥30 floor → INCONCLUSIVE**; the lone nominal hit (`rimp`
  exit H24 train −0.349, CI excludes 0) doesn't replicate at H48 and has no powered val counterpart
  (noise). The thin exit book directly re-confirms the CAPACITY meta-finding (the rule trades so little
  there isn't enough exit book to test an exit feature offline — the same 20-trade wall P-EXIT-REWARD
  hit). **Verdict: wire nothing** (CI-excludes-0 + sign-stable + FDR all required; none met). The
  `mlmodelpoly` import added no capturable alpha — consistent with the review's prior. Does NOT affect
  the frozen-`wkw`-TEST call.

## META-FINDING (2026-06-17) — the binding constraint may be CAPACITY / PARTICIPATION

P-REIGNITE and P-EXIT-REWARD independently hit the **same wall**: the rule **trades very little** — it funds only ~5–12% of flat ignition candidates (**39 train / 20 val** closed trades), capacity/rotation-gated (risk-parity caps + 48-bar cooldown + reclaimed gate + swap-weak-for-strong rotation). Consequences: (a) the exit-timing problem is **data-starved** offline (20 val trades); (b) the agent leaves **~95% of the +EV ignition stream untouched**. Both the entry-selection edge (refuted) and the exit-reward edge (not learnable on this book) sit **downstream of a more basic limit: how much of the +EV ignition stream can the agent safely participate in?** `scale_in` (the live `wsi` sweep) is one capacity lever; the 51% flat-skip/cooled bucket and the cooldown/reclaimed/rotation gates are the others. **NEXT research direction: a CAPACITY / PARTICIPATION probe** — are the ~95% unfunded ignitions +EV, and what is the *portfolio-DD* cost of participating in more of them (the real DQ object, not per-trade)? Two caveats on power: the *agent* trades more than the *rule* (its training sees more exit-decisions than the rule's 20-trade book), so the offline exit-probe's power limit is partly an artifact of measuring the rule; and the desktop's highest-*confidence* use right now is the **frozen-TEST spend on `wkw`** (the real OOS check of the best-known, still unspent — the human's one-time, irreversible call).

## EXIT-TRIGGER forensic (2026-06-17) — the EMA-break is the winner-killer (→ P-EMABREAK)

A forensic of `wsi`-s3's ZEC trade (CDN bundle + torch-free env replication) **localized** the exit-alpha to a specific trigger: the rung-0 **EMA-break exit** (`ema_hit = cushion<0`, `_scan_bar` L396), re-fired every `exit_commit`=12 bars, cut ZEC only **−0.9% / −1.9%** from blended cost (~8% off peak, a hair below the EMA — an ordinary breather) right before a **+60%-on-cost** run-up; at the full-exit bar **surge was RISING** (the ignition was not dead). Ruled out (hard evidence): tp (never hit +25%), rotation (sole holding → 100% cash, a *deliberate* exit), floor (px cost×0.99). **The binding exit trigger is `ema_hit`, NOT the trailing stop / tp / scale-in.** This refines the exit facet: the issue is **selling too EARLY** (the EMA-break hair-trigger on shallow below-EMA pullbacks — which the contrarian finding says OUTPERFORM), *distinct from* P-EXIT-REWARD's harder *peak-timing* problem and likely more tractable.

**NEW PROBE CANDIDATE — P-EMABREAK** (population-level, the one-seed caveat demands it): across all rule/agent trades (train+val, test frozen), how often does an EMA-break exit cut a position that then **resumes to a new high** within H bars — and does a *shallow-giveback + surge-still-alive* discriminator at the EMA-break bar separate the resumers from the real trend-breaks? If yes → a **giveback-penalty reward** and/or a **shallow-dip-vs-trend-break exit obs** (the surge-at-exit discriminator), as a single-variable add on the `wkw` base, validated cold-weekly. If the EMA-break premature-cut is NOT a broad pattern (just this ZEC anecdote), the lever is weaker than it looks — verify before any substrate/reward change.

## CAPACITY probe verdict (2026-06-17) — capacity is NOT the constraint (NULL)

The capacity/participation arm (workflow: build → 3 skeptics, **selection FLAWED** → decision NO) **refutes capacity as a lever.** (1) The rule's rotation NEVER fails for cash/slots (`capacity-no-rot = rotation-rejected = 0`) — there is no ceiling to relieve; the binding gates are cooldown / not-reclaimed / already-held. (2) On REALIZABLE **terminal** return (the build's max-run-up was positive-by-construction — **methodology lesson: use terminal/realizable return for selection contrasts, never max-run-up**), loosening the gates funds **WORSE** ignitions, not equal ones (INCONCLUSIVE-on-selection, NOT "money on the table"). (3) More participation WORSENED the return-vs-DD frontier. The build's headline "smaller-`entry_frac` Pareto win" (DD 35.5%→19.9%, +1.70%/wk) was caught by the selection skeptic as **BETA GIVE-UP** — Sharpe-like mean/std AND Calmar are *invariant* to `entry_frac` (no frontier bend); the edge is **−0.48 correlated with the market and loses in every bull week** — DQ *insurance*, not capturable alpha. The adversarial verify prevented a false-positive sizing experiment. **DRIFT ALARM (production-confirmed): the rung-0 RULE breaches the 30% DQ at default knobs (35.53% worst cold week 2026-04-13); `wkw` (the TRAINED agent) is DQ-protective at 7.84% — it FIXES the rule's DQ hole**, strengthening its champion case. TRIANGULATION: every mechanical lever (entry filters / tp / trailing-stop / scale_in / capacity-sizing) caps near breakeven and the rule loses the bull → the frontier is **LEARNED EXIT TIMING** (P-EMABREAK / the giveback-aware exit reward), NOT capacity. **Do NOT re-propose a capacity/gate-loosening experiment.**

## P-EMABREAK verdict (2026-06-17) — NO-GO; the alpha-hunt is comprehensively mapped

P-EMABREAK (workflow, 3 skeptics SOUND → NO): the EMA-break premature-exit is **regime beta, not a pathology** — ignition EMA-breaks resume NO MORE than random below-EMA dips (lift −17..−22% TRAIN); the surge-alive discriminator adds **0 OOS over the giveback obs the agent already carries** (incremental AUC +0.007, CI straddles 0; *inverts* within the shallow-giveback bucket); the realizable hold-through edge is ~0 (you can't identify resumers OOS, so you must hold the trend-breaks too); the ZEC case is an anecdote (its +60% is ~216 bars out, oracle-only). The exit arm is CLOSED.

**THE HUNT MAP — all five lever-classes closed this session:** entry-selection (P-REIGNITE, refuted) · `scale_in` (`wsi`, refuted on the gate) · exit-reward/peak-timing (P-EXIT-REWARD, NO-GO) · capacity/sizing (capacity probe, NULL) · EMA-break/surge-at-break (P-EMABREAK, NO-GO). The selective-ignition substrate + the current obs {surge, cush, giveback, unreal, surge_decay, accel, held_frac} is **comprehensively mined**; the run-up that exists is regime beta NOT capturable by a state-conditioned exit on the obs available (the agent already sees the only separating features, giveback + cush → a training-dynamics / data-ceiling limit, NOT an obs gap). **`wkw` (rdLe4 + wick_reject 0.25) is the substrate ceiling**: +5.1pts vs rung-0, DQ-protective 7.84%, fixes the rule's own 35.53% DQ breach. **NEXT (decision-critical): the frozen-TEST validation of `wkw`** — the real OOS check of the deployable, still unspent, before the live window (Jun 22–28); the human's one-time, irreversible call. A genuinely NEW obs/feature (higher-timeframe, order-flow) could reopen the exit hunt — the *current-obs* family is what's exhausted.

## P-VOLREGIME — spec (2026-06-17, from the [[reviewed-mlmodelpoly-repo|mlmodelpoly]] review)

The hunt-map closer states the *current* obs family {surge, cush, giveback, unreal, surge_decay, accel,
held_frac} is exhausted, **but explicitly names "a genuinely NEW obs/feature (higher-timeframe,
order-flow)" as the one thing that could reopen the exit hunt.** Reviewing the external `mlmodelpoly`
repo surfaced two cheap candidates that are *verified absent* from our 13-dim obs vector
(`[is_exit, cush, surge, unreal, held_frac, giveback, cash/eq, exposure, n_pos/k, dd, btc_trend,
rule_expo, breadth]`, `event_env._obs` L888) — critically **there is NO realized-volatility term in the
obs** (`_std` drives `_pick_universe` + risk-parity `_token_caps` only, never the policy). So these are
NEW *information*, not a re-pack of `surge` (a *volume* ratio) or `cush` (price-vs-EMA). This is the
disciplined test of exactly that escape hatch.

> **Honest prior — skeptical.** Vol-derived functionals have hit the **redundancy null five times**
> (last: the depth-normalized-turnover de-risking arm, dominated by what the risk-parity caps already
> encode). The bar is therefore **incremental IC over the existing obs**, not raw IC. Expect a real
> chance of INCONCLUSIVE / refute; the value is that it's the *one* obs question the session left open,
> and it's cheap + torch-free.

### The two candidate features (causal, leakage-free)

Mapped from mlmodelpoly's `volatility.py` (the RVOL-tilted fast/slow σ blend) and `features.py`
(robust median-normalized impulse). We expose the **state**, not their hand-tuned sizing heuristic — the
LSTM learns the (possibly non-monotone, contrarian) response, per the thesis guardrail.

1. **`volreg` — vol-regime ratio.** `σ_fast(tok,bar) / σ_slow(tok,bar)`, where σ is the trailing realized
   stdev of `returns` (the same series `_std` uses): `σ_fast` over `VOL_FAST=24` bars, `σ_slow` over
   `VOL_SLOW=168` bars, both right-anchored at `bar` (causal — no shift needed; rolling std at `bar` uses
   only `…bar`). Squash `log(volreg)` with `tanh` so contractions/expansions land in [−1,1]. Reads "is
   this token's vol expanding *relative to its own baseline* right now" — orthogonal to the *level* of vol
   (which the obs lacks anyway) and to `surge` (volume). Do **not** add mlmodelpoly's RVOL-weighted blend
   *weight* as a feature — that's a sizing rule; the ratio is the observable, the policy is the blender.
2. **`rimp` — robust return-impulse.** `|r_1bar(tok,bar)| / median(|r|, win=168)` over the trailing
   window (median/MAD-style, robust to the heavy alt tails — the same robust-z discipline already used in
   the turnover-alert ops spec). Sign-free magnitude anomaly: "how big is this bar's move vs this token's
   typical bar." `tanh`-squash. Distinct from `cush` (cumulative distance) and the harvest r24/r3d/r7d
   (multi-bar returns) — none normalize the *single-bar* move by the token's typical bar.

Precompute `_volreg[bar,j]`, `_rimp[bar,j]` alongside `_std`/`_surge` in `__init__` (vectorized
rolling ops; ~free). Probe reads them off a constructed env over the panel — **no torch, laptop-local, no
desktop contention** (the P-SURGE-SHAPE pattern: `scripts/probe_volregime.py`).

### The gate (what makes it honest)

- **PRIMARY = incremental continuous IC.** For each feature, residualize forward return (H∈{24,48}h)
  on the existing-obs baseline {surge, cush, giveback, unreal, held_frac, btc_trend, breadth} (OLS), then
  Spearman-IC the feature against the residual. Report ΔIC vs the baseline-only fit with a
  **move-clustered / 168h-block / token-paired bootstrap CI**. Buckets illustrate only (thin +10% cells).
- **TWO contexts, one headline each** (facet-level BH-FDR across both + any future vol probe):
  - **ENTRY (ignition bars):** does the feature predict fwd run-up / a size-up signal among valid
    ignitions (det_blacklist excluded, the P-SURGE-SHAPE population)?
  - **EXIT (giveback-matched bars):** does it discriminate **resumers from trend-breaks** *among bars at
    the same giveback* — the exact discrimination P-DECEL/P-EMABREAK ran. This is the reopening test: the
    hunt-map says giveback+cush can't separate them; the question is whether a NEW vol-regime/impulse axis
    can where the current obs provably can't (incremental AUC over the giveback-logit floor, CI must
    exclude 0).
- **n-floor → INCONCLUSIVE** below ~30 per cell / the bootstrap CI bands (not refute, not pass).
- **Causality asserts:** `volreg`/`rimp` read only `…bar`; unit-test that swapping in future bars changes
  the value (no accidental shift-forward), mirroring the P-MOVEAGE `b0+72 ≤ b` guard.

### Decision rule (survival → integration)

Each feature surviving facet-level FDR with a **sign-stable incremental IC** earns exactly **ONE appended
obs slot**, added as a **single-variable** flag on the `wkw` base (`vol_regime_obs` / `impulse_obs`,
default-OFF/byte-identical), then validated on the **cold-weekly gate** (paired-bootstrap CI-low > rung-0)
*before* any TEST spend — never bundled, never a fixed gate (the discipline that kept `wkw` clean). If
both survive, they go in as two separate sweeps, not one. If neither shows incremental IC, **STOP** — log
the sixth vol-redundancy null and do not wire anything (the dilution risk the rdLc/turnover precedents
warn against). This probe does **not** gate or delay the frozen-`wkw`-TEST decision, which remains the
session's highest-confidence move and runs independently.

## EMA-break re-probed on the POLICY (2026-06-19) — the SUBSTRATE CORRECTION

The 2026-06-17 EMA-break verdicts (P-EMABREAK NO-GO, conditional-EMA-break REFUTED) **all ran on the
rung-0 RULE.** The user flagged this as a methodological flaw: the rule's *only* exits are
sell-on-weakness (trailing stop / EMA-cross), so it structurally gives back pumps — it is the wrong
substrate for any exit / profit-taking question. The **deployable POLICY** has the `tp_rungs`
profit-take ladder + a learned hold/override, so it captures pumps differently. This session re-ran the
exit forensic on the policy (`probe_policy_exits`); the four EMA-break conditionings (P&L / consolidation /
deep-dip inverse / EMA-period) were all done on the **rule** — the flawed substrate. **Meta-lesson: probe
the DEPLOYABLE substrate (the policy), not the rule, for any exit / profit-taking question.**

### `probe_policy_exits` — the substrate correction (the headline)
Ran on **ef2-s0** over the OOS cold weeks (`scripts/probe_policy_exits.py`). Exit-trigger mix:
**74% EMA_BREAK / 15% PROFIT_TAKE / 7% ROTATION_OUT / ~5% stops.** The policy's *own* EMA-break sells
**give back +3.5% mean fwd-48h** (55% recover) — so the EMA-break IS a real leak on the policy too, but
**the P&L engine is the few pumps the policy HOLDS to the week close, not profit-taking:** 4
held-to-end positions made **$6235** (best TAG **$5432**) vs 66 exited trades making **$4782** combined.
**TAG's +170% rip was captured by HOLDING** (no exit fired, force-closed near the high), NOT by the
profit-take ladder. This re-frames the exit facet: the agent's alpha comes from *not selling* clean
rips, and the EMA-break shaking it out in sideways is the leak to plug.

### EMA-break conditioned FOUR ways (all on the rule) — re-confirmations + one new cell
1. **P&L gate** = the prior conditional-EMA-break / P-EMA-COND — already refuted (see below).
2. **Consolidation / low-vol suppression (NEW worst cell):** the **consol-and-shallow** cell — a shallow
   EMA-break during tight low-vol consolidation, the **FF pattern** — was the **WORST** (terminal
   fwd-48h **−5.4%, 8% win rate**). FF was the 8% anecdote; most consolidation breaks keep falling.
3. **Deep-dip INVERSE (do deep + high-vol breaks bounce?):** REFUTED — the break-depth forward-return
   trend is **MONOTONICALLY NEGATIVE** (deeper = worse); the deep-and-high-vol cell is **n=4 with ZERO
   val events**, and the earlier +13.9% high-vol reading was a **thin single-dimension artifact** that
   vanished with proper conditioning + train data.
4. **EMA-PERIOD sweep (global + exit-only, spans 50–240):** **no net win** — a longer EMA gives no robust
   val gain, worse test return, and MORE DQ weeks (it holds losers/givebacks longer).

### DQ-week anatomy (the 2 cold weeks that breach the 30% DD gate)
- **Mar 23** — a pure loser: peaked 1.04x, ended −25%.
- **Apr 13** — caught SIREN's +163% pump to a +23% peak, then **GAVE IT ALL BACK and ended −14%** — a
  giveback that ends NEGATIVE, not a profitable-but-DQ'd false alarm.
A longer EMA makes **both** worse.

### The fix that came out of it — SIDEWAYS EMA-BREAK SUPPRESSION (built, retrain pending)
Mechanism of the leak: the EMA-break fires on shallow noise-dips during tight sideways consolidation,
shaking the agent out before the pump (the FF Apr-9 case: a −0.1% cushion NOISE break, then a 48h
cooldown locked re-entry through FF's real +106%/+151% rip). **Fix:** when a break is **SHALLOW**
(`cushion > -shallow_break_max`) AND the token is **QUIET** (24h realized vol < `consol_vol_max`), do
NOT fire the EMA-break; the loss_floor (−20%) and trailing stop stay fully active. Asymmetric: bounded
downside via the floor, large upside via pump capture. The user chose the "shallow + quiet" definition
(most surgical — a deep break or a high-vol break still cuts). Both knobs 0 ⇒ OFF, byte-identical (30
env tests pass). Committed `abf089b`, wired through REWARD_KEYS + train_event + provenance + simulate.
**Planned experiment:** retrain ef2 + `shallow_break_max=0.02` + `consol_vol_max=0.015` (FF-validated
thresholds), 4 seeds, graded honest cold-weekly vs ef2 (pending the user's green-light — shared
desktop). Open co-factor: ROTATION_OUT can still swap a held-but-flat token out before its pump (a
second shakeout mechanism) — the next thread if suppression helps but rotation caps it. (See
[[Trading Strategies]] for the rotation lever and [[Market Conditions]] for the live-week read.)

### Two meta-lessons from this session
- **Probe the policy, not the rule.** The rule's structural sell-on-weakness exits make it the wrong
  substrate for any exit/profit-taking question; the policy's `tp_rungs` + learned hold change the
  answer (held-to-end pumps are the P&L engine, invisible on the rule).
- **Explore conditional sub-regimes instead of blanket-refuting a lever.** The EMA-break is not
  uniformly beta: the consol+shallow cell is the worst (−5.4%, 8% win), distinct from deep/high-vol
  breaks — grounding every cell (terminal return, val/test split, n-floor, DQ-aware) caught a
  single-dimension +13.9% artifact and located the real, surgical fix.

## Conditional EMA-break (user hypothesis, 2026-06-17) — REFUTED (beta)

The user's intuition: the EMA-break is too rigid → only honor it when the position is >T below the buy price (`unreal < -T`), making it a stop-loss instead of a profit-giveback exit. Tested as a faithful rule counterfactual (`scripts/probe_ema_cond.py`, T=baseline bit-identical to production `run_rung0`) on the cold-weekly PORTFOLIO grader; workflow (3 skeptics: causality SOUND, alpha-vs-beta FLAWED, dd-realism FLAWED). **REFUTED — beta, not alpha.** No T beats the rule (every paired CI straddles 0; T=0.05 +0.24% [−1.48,+2.30]); OLS market-neutral alpha −0.087%/wk (t=−0.08); the edge **flips sign with the market** (+1.79% bull / −0.31% flat-down; 6 bull weeks = ~199% of the summed edge); the DQ is NOT relieved (35.63% at T=0.05, slightly worse; DD rises in 20/23 weeks). **The ZEC anchor self-refutes:** T=0.05 confirms in the cited Apr-06 week (+57) but is −2463 WORSE summed over all 8 ZEC cold weeks (the Dec-08 week flips a +195 win to −2009 by holding through a shallow EMA-break into the deep trailing stop). This makes the hunt **six lever-classes closed**, and the EXIT specifically closed via BOTH a fixed rule (this + P-EMABREAK) AND a reward on the current obs (P-EXIT-REWARD). The run-up is regime beta; `wkw` is the ceiling; the frozen-TEST validation of `wkw` is the move.
