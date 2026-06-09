#!/usr/bin/env bash
# OOS validation on the FROZEN TEST split — the honest verdict.
#
# The reward configs were SELECTED by looking at val performance. This runs the two that beat the
# val baseline — `real` (the worst-seed-safe champion, +83% val) and `real-give` (highest val
# return, +156%, tail-risky) — on the never-touched TEST window. The model trains on `train` as
# always; ONLY the eval window changes (val -> test), so val never leaked into selection of *these*
# numbers. 1M steps (the converged regime; 100k was undertrained froth). Multi-seed for the
# mean + worst-seed read.
#
#   nohup bash scripts/run_oos_test.sh [TIMESTEPS] ["SEEDS"] > runs-rl/oos.log 2>&1 < /dev/null &
set -uo pipefail
cd "$(dirname "$0")/.."

TIMESTEPS="${1:-1000000}"
SEEDS="${2:-0 1 2}"
PY=.venv/bin/python
LOGDIR=runs-rl/oos-logs
mkdir -p "$LOGDIR"

CONFIGS=(
  "real|--realized-lambda 10 --turn-lambda 0 --gb-lambda 0  --dd-lambda 2.0"
  "real-give|--realized-lambda 10 --turn-lambda 0 --gb-lambda 15 --dd-lambda 2.0"
)

echo "[oos] START $(date -u +%FT%TZ)  timesteps=$TIMESTEPS  seeds='$SEEDS'  split=TEST (frozen)"
n=0
for entry in "${CONFIGS[@]}"; do
  label="${entry%%|*}"
  rargs="${entry#*|}"
  for seed in $SEEDS; do
    rid="ppo-oos-${label}-s${seed}"
    n=$((n+1))
    if grep -q "^\[train_rl\]" "$LOGDIR/${rid}.log" 2>/dev/null; then
      echo "[oos] ($n) skip $rid — already complete"; continue   # resumable after interruption
    fi
    echo "[oos] === ($n) $rid  $(date -u +%FT%TZ) ==="
    # shellcheck disable=SC2086
    $PY scripts/train_rl.py --action-mode weights --reward-mode composite --rich-obs \
        --eval-split test --seed "$seed" --timesteps "$TIMESTEPS" --n-envs 6 \
        $rargs --run-id "$rid" > "$LOGDIR/${rid}.log" 2>&1
    if [ $? -eq 0 ]; then tail -1 "$LOGDIR/${rid}.log"; else echo "[oos] $rid FAILED — see $LOGDIR/${rid}.log"; fi
  done
done
echo "[oos] DONE $(date -u +%FT%TZ)  ($n runs)"
