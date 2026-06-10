#!/usr/bin/env bash
# Event-driven rung-1 seed sweep (desktop, overnight). Trains ONE config — PPO on the event-driven
# env (EventRungEnv: the agent learns rung-0's entry-sizing + exit-override discretion on top of
# rung-0's event timing) — across N seeds, each a different random draw of WEEKLY episode windows so
# the policy is exposed to different start/end dates and can't memorize a timeframe. Single-seed RL
# is unstable, so the read is the per-seed AVERAGE return vs the rung-0 RULE baseline (does learned
# discretion beat the hand-coded version?). One bundle published per seed; intra-day markers.
#
# Mirrors scripts/run_reward_sweep.sh (the documented deploy pattern). Usage (detached, survives logout):
#   nohup bash scripts/run_eventrung_sweep.sh [TIMESTEPS] ["SEEDS"] [EVAL_SPLIT] [REWARD_MODE] > runs-rl/eventrung.log 2>&1 < /dev/null &
#   # default:  TIMESTEPS=1000000  SEEDS="0 1 2 3"  EVAL_SPLIT=val  REWARD_MODE=absolute
#   # EXPERIMENT 1:  ... 1000000 "0 1 2 3" test relative   (reward vs the rung-0 rule; only beating it scores)
# Aggregate:  python scripts/compare_seeds.py --prefix ppo-event[-rel][-test] --seeds "0 1 2 3"
set -uo pipefail
cd "$(dirname "$0")/.."

TIMESTEPS="${1:-1000000}"
SEEDS="${2:-0 1 2 3}"
EVAL_SPLIT="${3:-val}"
REWARD_MODE="${4:-absolute}"
SFX=""; [ "$EVAL_SPLIT" = "test" ] && SFX="-test"
if [ "$REWARD_MODE" = "rd8h0" ]; then                 # rd8h with dd_lambda 0: kill the diet-rule equilibrium —
  PFX="ppo-event-rd8h0"                                # the relative reward alone drives; DQ enforced by the
  EXTRA="--reward-mode relative --rule-default --exit-commit 12 --dust-usd 10 --rule-prior 2.0 --tp-rungs 0.25,0.5,1.0,2.0 --harvest-obs --eval-prepad --action-mode discrete --n-action-levels 4 --universe-mode voltopk --k 8 --vol-target 0.005 --cap-floor 0.02 --crash-train 4 --crash-eval --norm-reward --dd-lambda 0.0 --dd-soft 0.15 --ent-coef 0.2 --lr 3e-4 --lr-end 3e-5 --episode-bars 336"
elif [ "$REWARD_MODE" = "rd8h" ]; then                # rd8tp + harvest obs (r24/r3d/r7d, the probe-passed
  PFX="ppo-event-rd8h"                                 # signal lever) + eval-prepad (window tradeable bar 0)
  EXTRA="--reward-mode relative --rule-default --exit-commit 12 --dust-usd 10 --rule-prior 2.0 --tp-rungs 0.25,0.5,1.0,2.0 --harvest-obs --eval-prepad --action-mode discrete --n-action-levels 4 --universe-mode voltopk --k 8 --vol-target 0.005 --cap-floor 0.02 --crash-train 4 --crash-eval --norm-reward --dd-lambda 0.5 --dd-soft 0.15 --ent-coef 0.2 --lr 3e-4 --lr-end 3e-5 --episode-bars 336"
elif [ "$REWARD_MODE" = "rd8" ]; then                 # Rung-1b on VOLTOP8 (user: cut the calm tokens —
  PFX="ppo-event-rd8"                                  # they bleed; concentrate on the monsters)
  EXTRA="--reward-mode relative --rule-default --exit-commit 12 --dust-usd 10 --rule-prior 2.0 --action-mode discrete --n-action-levels 4 --universe-mode voltopk --k 8 --vol-target 0.005 --cap-floor 0.02 --crash-train 4 --crash-eval --norm-reward --dd-lambda 0.5 --dd-soft 0.15 --ent-coef 0.2 --lr 3e-4 --lr-end 3e-5 --episode-bars 336"
elif [ "$REWARD_MODE" = "rd8tp" ]; then               # rd8 + profit-take prompts (sell into strength)
  PFX="ppo-event-rd8tp"
  EXTRA="--reward-mode relative --rule-default --exit-commit 12 --dust-usd 10 --rule-prior 2.0 --tp-rungs 0.25,0.5,1.0,2.0 --action-mode discrete --n-action-levels 4 --universe-mode voltopk --k 8 --vol-target 0.005 --cap-floor 0.02 --crash-train 4 --crash-eval --norm-reward --dd-lambda 0.5 --dd-soft 0.15 --ent-coef 0.2 --lr 3e-4 --lr-end 3e-5 --episode-bars 336"
elif [ "$REWARD_MODE" = "ruledefault" ]; then         # Rung-1b: g2b frozen + rule-default discretion —
  PFX="ppo-event-rd"                                   # action idx0 EXECUTES rung-0; exit-commit, dust
  EXTRA="--reward-mode relative --rule-default --exit-commit 12 --dust-usd 10 --rule-prior 2.0 --action-mode discrete --n-action-levels 4 --universe-mode broad --k 12 --vol-target 0.005 --cap-floor 0.02 --crash-train 4 --crash-eval --norm-reward --dd-lambda 0.5 --dd-soft 0.15 --ent-coef 0.2 --lr 3e-4 --lr-end 3e-5 --episode-bars 336"
elif [ "$REWARD_MODE" = "lever2" ]; then              # Lever 2: g2b + harvest momentum obs (13->16).
  PFX="ppo-event-l2"                                   # A/B vs the on-disk g2b control (one variable: the obs)
  EXTRA="--reward-mode relative --harvest-obs --action-mode discrete --n-action-levels 4 --universe-mode broad --k 12 --vol-target 0.005 --cap-floor 0.02 --crash-train 4 --crash-eval --norm-reward --dd-lambda 0.5 --dd-soft 0.15 --ent-coef 0.2 --lr 3e-4 --lr-end 3e-5 --episode-bars 336"
elif [ "$REWARD_MODE" = "gate2b" ]; then              # Reward-rebalance decider: GATE-2 FROZEN, only
  PFX="ppo-event-g2b"                                  # dd_lambda 1.0 -> 0.5 (does it ramp the bull?)
  EXTRA="--reward-mode relative --action-mode discrete --n-action-levels 4 --universe-mode broad --k 12 --vol-target 0.005 --cap-floor 0.02 --crash-train 4 --crash-eval --norm-reward --dd-lambda 0.5 --dd-soft 0.15 --ent-coef 0.2 --lr 3e-4 --lr-end 3e-5 --episode-bars 336"
elif [ "$REWARD_MODE" = "gate2" ]; then               # GATE 2: regime-adaptive — breadth obs (auto) +
  PFX="ppo-event-g2"                                   # crash-augmented training + a held-out CRASH regime
  EXTRA="--reward-mode relative --action-mode discrete --n-action-levels 4 --universe-mode broad --k 12 --vol-target 0.005 --cap-floor 0.02 --crash-train 4 --crash-eval --norm-reward --dd-lambda 1.0 --dd-soft 0.15 --ent-coef 0.2 --lr 3e-4 --lr-end 3e-5 --episode-bars 336"
elif [ "$REWARD_MODE" = "gate1b" ]; then              # GATE 1 (broad): discrete + risk-parity on a
  PFX="ppo-event-g1b"                                  # 12-token universe (stables + monsters), DQ gate
  EXTRA="--reward-mode relative --action-mode discrete --n-action-levels 4 --universe-mode broad --k 12 --vol-target 0.005 --cap-floor 0.02 --norm-reward --dd-lambda 1.0 --dd-soft 0.15 --ent-coef 0.2 --lr 3e-4 --lr-end 3e-5 --episode-bars 336"
elif [ "$REWARD_MODE" = "gate1" ]; then               # GATE 1 (isolation): all on voltopk-8
  PFX="ppo-event-g1"
  EXTRA="--reward-mode relative --action-mode discrete --n-action-levels 4 --universe-mode voltopk --vol-target 0.005 --cap-floor 0.02 --norm-reward --dd-lambda 1.0 --dd-soft 0.15 --ent-coef 0.2 --lr 3e-4 --lr-end 3e-5 --episode-bars 336"
elif [ "$REWARD_MODE" = "selector" ]; then            # Experiment 5: ungated cross-sectional selector
  PFX="ppo-event-sel"                                  # (in-env landscape gate PASSED at gamma=0.1)
  EXTRA="--reward-mode entry_forward --ungate --fwd-horizon 24 --res-gamma 0.1 --norm-reward --dd-lambda 1.0 --dd-soft 0.15 --ent-coef 0.2 --lr 3e-4 --lr-end 3e-5 --episode-bars 336"
elif [ "$REWARD_MODE" = "entry_forward" ]; then       # Experiment 4: entry-forward residual (reward == metric)
  PFX="ppo-event-efwd"
  EXTRA="--reward-mode entry_forward --fwd-horizon 24 --res-gamma 0.05 --norm-reward --dd-lambda 1.0 --dd-soft 0.15 --ent-coef 0.2 --lr 3e-4 --lr-end 3e-5 --episode-bars 336"
elif [ "$REWARD_MODE" = "residual_ranked" ]; then     # Experiment 3: demeaned-ranked residual + budget
  PFX="ppo-event-rank"
  EXTRA="--reward-mode residual_ranked --res-gamma 0.1 --norm-reward --dd-lambda 1.0 --dd-soft 0.15 --ent-coef 0.2 --lr 3e-4 --lr-end 3e-5 --episode-bars 336"
elif [ "$REWARD_MODE" = "residual" ]; then            # Experiment 2b: per-decision (residual) reward + R4
  PFX="ppo-event-res"
  EXTRA="--reward-mode residual --r4-beta 0.8 --norm-reward --dd-lambda 0.5 --dd-soft 0.20 --ent-coef 0.2 --lr 3e-4 --lr-end 3e-5 --episode-bars 336"
elif [ "$REWARD_MODE" = "relative" ]; then            # Experiment 1: relative-to-rule reward + dense exploration
  PFX="ppo-event-rel"
  EXTRA="--reward-mode relative --dd-lambda 0.5 --dd-soft 0.20 --ent-coef 0.2 --lr 3e-4 --lr-end 3e-5 --episode-bars 336"
else
  PFX="ppo-event"
  EXTRA="--episode-bars 168"
fi
PY=.venv/bin/python
SHA=$(git rev-parse --short HEAD 2>/dev/null || echo nogit)   # stamp the code version into every run-id
LOGDIR=runs-rl/eventrung-logs                                 # -> re-runs NEVER overwrite/alias an old name
mkdir -p "$LOGDIR"

echo "[eventrung] START $(date -u +%FT%TZ)  timesteps=$TIMESTEPS  seeds='$SEEDS'  split=$EVAL_SPLIT  reward=$REWARD_MODE"
n=0
for seed in $SEEDS; do
  rid="${PFX}-${SHA}${SFX}-s${seed}"                          # e.g. ppo-event-g2b-8ccad69-test-s3
  n=$((n+1))
  echo "[eventrung] === ($n) $rid  $(date -u +%FT%TZ) ==="
  # shellcheck disable=SC2086
  $PY scripts/train_event.py --timesteps "$TIMESTEPS" --n-envs 8 $EXTRA \
      --eval-split "$EVAL_SPLIT" --seed "$seed" --run-id "$rid" > "$LOGDIR/${rid}.log" 2>&1
  if [ $? -eq 0 ]; then tail -1 "$LOGDIR/${rid}.log"; else echo "[eventrung] $rid FAILED — see $LOGDIR/${rid}.log"; fi
done
echo "[eventrung] DONE $(date -u +%FT%TZ)  ($n runs)"
