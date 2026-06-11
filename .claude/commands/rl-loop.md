---
description: One tick of the autonomous RL iterate loop — step the driver, and when it asks, analyze the verdict and propose the next single-variable config. Drive with /loop (self-paced) or cron.
---

You are executing ONE wake of the autonomous RL training loop for agentic-crypto-trader.
The mechanical state machine lives in the driver (`scripts/rl_loop.py` / the `rl_loop_*` MCP
tools); your job is the judgment it deliberately leaves out. The user's standing direction
([[rl-is-the-core-thesis]]): the RL agent IS the project — a failed sweep means diagnose and
tune again, never "ship the rule" or deadline steering.

## The tick

1. Run `python scripts/rl_loop.py step` (or the `rl_loop_step` MCP tool).
2. Branch on `phase`:
   - **`running`** — report one line (sweep, n_published, load) and END THE TURN. If driving via
     /loop, schedule the next wake ~20–30 min out (LSTM seeds ≈ 25 min each, MLP ≈ 5).
   - **`launched`** — confirm verify/smoke look clean (the driver already guarded); end the turn.
   - **`verdict`** — the core wake. Do ALL of:
     a. Append the standings + verdict to **[[Experiment Log]]** (the table format used all
        arc: per-seed × val/test/crash, the bars, the read) and commit the vault edit.
     b. If `needs_proposal=true`: ANALYZE before proposing — what was the binding regime/baseline?
        Run `rl_forensics` on 1–2 suspicious tokens if behavior (not magnitude) looks wrong.
        Consult the prior levers in [[Experiment Log]] + [[AI Training]] §rd-ladder so no refuted
        idea is re-proposed (e.g. the low-rising filter is REFUTED; obs levers are exhausted on
        the MLP). Then `rl_loop_propose` with ONE config changing ONE variable vs the best-known
        (`experiment_champion` / the history's best margin), `note` = the hypothesis. Then `step`
        again to launch it, verify, and end the turn.
   - **`idle` + `needs_proposal`** — same as 2c-b.
   - **`halted`** — STOP THE LOOP (do not reschedule). Notify the user with the reason
     (PushNotification if available): drift alarm / budget / refused launch / dead sweep /
     PROMOTE candidate. A drift alarm means N configs without a new best margin-vs-Buy&Hold —
     present the history and the open hypotheses; the human decides the escalation.

## Hard rules (the contract, enforced in code but restated)

- **val only.** Never propose `split=test`; the frozen test is spent by the human, once, via
  `rl_train(..., final_verdict=True)` when a config has earned it.
- **One variable per config.** The note must say what it tests and why the last verdict
  motivates it.
- **Never launch around the driver** (no ad-hoc ssh sweeps) — the driver's guards (launch-once,
  preflight, smoke, sha-stamping) are the runbook.
- **The desktop is shared** — if `rl_train` refuses because a sweep is running that the loop
  didn't start, halt and notify, don't kill it.
- Judge on the per-regime seed-mean and worst-seed DD; the north star is the WORST regime's
  margin-vs-Buy&Hold. Behavioral checks (forensics) outrank single-number reads.
