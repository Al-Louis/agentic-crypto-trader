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
if [ "$REWARD_MODE" = "residual_ranked" ]; then       # Experiment 3: demeaned-ranked residual + budget
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
LOGDIR=runs-rl/eventrung-logs
mkdir -p "$LOGDIR"

echo "[eventrung] START $(date -u +%FT%TZ)  timesteps=$TIMESTEPS  seeds='$SEEDS'  split=$EVAL_SPLIT  reward=$REWARD_MODE"
n=0
for seed in $SEEDS; do
  rid="${PFX}${SFX}-s${seed}"
  n=$((n+1))
  echo "[eventrung] === ($n) $rid  $(date -u +%FT%TZ) ==="
  # shellcheck disable=SC2086
  $PY scripts/train_event.py --timesteps "$TIMESTEPS" --n-envs 8 $EXTRA \
      --eval-split "$EVAL_SPLIT" --seed "$seed" --run-id "$rid" > "$LOGDIR/${rid}.log" 2>&1
  if [ $? -eq 0 ]; then tail -1 "$LOGDIR/${rid}.log"; else echo "[eventrung] $rid FAILED — see $LOGDIR/${rid}.log"; fi
done
echo "[eventrung] DONE $(date -u +%FT%TZ)  ($n runs)"
