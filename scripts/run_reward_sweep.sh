#!/usr/bin/env bash
# Overnight reward-shaping sweep (desktop). Trains each reward mode — sharpe (control), giveback,
# realized, turnover — on IDENTICAL obs (rich), action mode, split, and seed, so the *reward* is
# the only variable. Publishes one bundle per (mode, seed) to the Apentic frontend for comparison.
#
# Multi-seed because single-seed RL is unstable (frozen-test: -5.8/+10/+15.6 across seeds) — we
# compare the per-mode *average*, not one lucky run.
#
# Usage (detached, survives logout):
#   nohup bash scripts/run_reward_sweep.sh [TIMESTEPS] ["SEEDS"] > runs-rl/sweep.log 2>&1 &
#   # e.g.  nohup bash scripts/run_reward_sweep.sh 500000 "0 1 2" > runs-rl/sweep.log 2>&1 &
set -uo pipefail
cd "$(dirname "$0")/.."

TIMESTEPS="${1:-500000}"
SEEDS="${2:-0 1 2}"
MODES="sharpe giveback realized turnover"
PY=.venv/bin/python
LOGDIR=runs-rl/sweep-logs
mkdir -p "$LOGDIR"

echo "[sweep] START $(date -u +%FT%TZ)  timesteps=$TIMESTEPS  seeds='$SEEDS'  modes='$MODES'"
n=0
for mode in $MODES; do
  for seed in $SEEDS; do
    rid="ppo-${mode}-s${seed}"
    n=$((n+1))
    echo "[sweep] === ($n) $rid  $(date -u +%FT%TZ) ==="
    $PY scripts/train_rl.py --action-mode weights --reward-mode "$mode" --rich-obs \
        --eval-split val --seed "$seed" --timesteps "$TIMESTEPS" --n-envs 6 \
        --run-id "$rid" > "$LOGDIR/${rid}.log" 2>&1
    if [ $? -eq 0 ]; then tail -1 "$LOGDIR/${rid}.log"; else echo "[sweep] $rid FAILED — see $LOGDIR/${rid}.log"; fi
  done
done
echo "[sweep] DONE $(date -u +%FT%TZ)  ($n runs)"
