#!/usr/bin/env bash
# rung-0-features seed sweep (desktop, overnight). Trains ONE config — PPO weights mode with rung-0's
# own signals as observations (--rung0-obs: per-token ignite / volume-surge / price-EMA cushion) —
# across N seeds, each a different random draw of WEEKLY episode windows (--episode-steps 7) so the
# policy is exposed to different start/end dates and can't memorize a fixed timeframe. The rung-0
# parameters are FIXED; only the window seed varies. Single-seed RL is unstable, so the read is the
# per-seed AVERAGE return (see scripts/compare_seeds.py). One bundle published per seed.
#
# Mirrors scripts/run_reward_sweep.sh (the documented deploy pattern). Usage (detached, survives logout):
#   nohup bash scripts/run_rung0feat_sweep.sh [TIMESTEPS] ["SEEDS"] > runs-rl/rung0feat.log 2>&1 < /dev/null &
#   # default:  TIMESTEPS=1000000  SEEDS="0 1 2 3"
set -uo pipefail
cd "$(dirname "$0")/.."

TIMESTEPS="${1:-1000000}"
SEEDS="${2:-0 1 2 3}"
PY=.venv/bin/python
LOGDIR=runs-rl/rung0feat-logs
mkdir -p "$LOGDIR"

echo "[rung0feat] START $(date -u +%FT%TZ)  timesteps=$TIMESTEPS  seeds='$SEEDS'  (weekly episodes, rung-0 obs)"
n=0
for seed in $SEEDS; do
  rid="ppo-rung0feat-s${seed}"
  n=$((n+1))
  echo "[rung0feat] === ($n) $rid  $(date -u +%FT%TZ) ==="
  $PY scripts/train_rl.py --action-mode weights --rung0-obs --reward-mode sharpe \
      --episode-steps 7 --eval-split val --seed "$seed" --timesteps "$TIMESTEPS" --n-envs 8 \
      --run-id "$rid" > "$LOGDIR/${rid}.log" 2>&1
  if [ $? -eq 0 ]; then tail -1 "$LOGDIR/${rid}.log"; else echo "[rung0feat] $rid FAILED — see $LOGDIR/${rid}.log"; fi
done
echo "[rung0feat] DONE $(date -u +%FT%TZ)  ($n runs)"
