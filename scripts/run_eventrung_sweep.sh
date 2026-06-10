#!/usr/bin/env bash
# Event-driven rung-1 seed sweep (desktop, overnight). Trains ONE config — PPO on the event-driven
# env (EventRungEnv: the agent learns rung-0's entry-sizing + exit-override discretion on top of
# rung-0's event timing) — across N seeds, each a different random draw of WEEKLY episode windows so
# the policy is exposed to different start/end dates and can't memorize a timeframe. Single-seed RL
# is unstable, so the read is the per-seed AVERAGE return vs the rung-0 RULE baseline (does learned
# discretion beat the hand-coded version?). One bundle published per seed; intra-day markers.
#
# Mirrors scripts/run_reward_sweep.sh (the documented deploy pattern). Usage (detached, survives logout):
#   nohup bash scripts/run_eventrung_sweep.sh [TIMESTEPS] ["SEEDS"] > runs-rl/eventrung.log 2>&1 < /dev/null &
#   # default:  TIMESTEPS=1000000  SEEDS="0 1 2 3"
# Aggregate:  python scripts/compare_seeds.py --prefix ppo-event --seeds "0 1 2 3"
set -uo pipefail
cd "$(dirname "$0")/.."

TIMESTEPS="${1:-1000000}"
SEEDS="${2:-0 1 2 3}"
PY=.venv/bin/python
LOGDIR=runs-rl/eventrung-logs
mkdir -p "$LOGDIR"

echo "[eventrung] START $(date -u +%FT%TZ)  timesteps=$TIMESTEPS  seeds='$SEEDS'  (event-driven, weekly episodes)"
n=0
for seed in $SEEDS; do
  rid="ppo-event-s${seed}"
  n=$((n+1))
  echo "[eventrung] === ($n) $rid  $(date -u +%FT%TZ) ==="
  $PY scripts/train_event.py --timesteps "$TIMESTEPS" --n-envs 8 --episode-bars 168 \
      --eval-split val --seed "$seed" --run-id "$rid" > "$LOGDIR/${rid}.log" 2>&1
  if [ $? -eq 0 ]; then tail -1 "$LOGDIR/${rid}.log"; else echo "[eventrung] $rid FAILED — see $LOGDIR/${rid}.log"; fi
done
echo "[eventrung] DONE $(date -u +%FT%TZ)  ($n runs)"
