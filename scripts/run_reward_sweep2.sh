#!/usr/bin/env bash
# Reward-shaping sweep #2 (desktop, overnight). Sweep #1 found realized's engine has the most
# return (+198%) but breaches the 30% DD gate, and EVERY mode's worst seed hits ~40-43% DD. So #2
# takes realized's profit engine and bolts on drawdown brakes of varying strength (turnover penalty,
# giveback penalty, and the drawdown-proximity penalty dd_lambda), hunting for the highest return
# whose WORST seed stays under the gate.
#
# composite reward = DSR - dd_lambda*dd_penalty - gb_lambda*giveback + realized_lambda*realized
#                          - turn_lambda*turnover     (set any lambda to 0 to disable that term)
#
# Usage (detached, survives logout):
#   nohup bash scripts/run_reward_sweep2.sh [TIMESTEPS] ["SEEDS"] > runs-rl/sweep2.log 2>&1 < /dev/null &
set -uo pipefail
cd "$(dirname "$0")/.."

TIMESTEPS="${1:-1000000}"
SEEDS="${2:-0 1 2}"
PY=.venv/bin/python
LOGDIR=runs-rl/sweep2-logs
mkdir -p "$LOGDIR"

# label | reward-config args (all composite, realized engine on, varying the brake)
CONFIGS=(
  "real|--realized-lambda 10 --turn-lambda 0   --gb-lambda 0  --dd-lambda 2.0"   # engine alone (expect +198/breach)
  "real-turn1|--realized-lambda 10 --turn-lambda 1.0 --gb-lambda 0  --dd-lambda 2.0"
  "real-turn3|--realized-lambda 10 --turn-lambda 3.0 --gb-lambda 0  --dd-lambda 2.0"
  "real-give|--realized-lambda 10 --turn-lambda 0   --gb-lambda 15 --dd-lambda 2.0"
  "real-dd5|--realized-lambda 10 --turn-lambda 0   --gb-lambda 0  --dd-lambda 5.0"   # stronger gate brake
  "real-combo|--realized-lambda 10 --turn-lambda 1.5 --gb-lambda 10 --dd-lambda 4.0"  # everything braked
)

echo "[sweep2] START $(date -u +%FT%TZ)  timesteps=$TIMESTEPS  seeds='$SEEDS'  configs=${#CONFIGS[@]}"
n=0
for entry in "${CONFIGS[@]}"; do
  label="${entry%%|*}"
  rargs="${entry#*|}"
  for seed in $SEEDS; do
    rid="ppo2-${label}-s${seed}"
    n=$((n+1))
    echo "[sweep2] === ($n) $rid  $(date -u +%FT%TZ) ==="
    # shellcheck disable=SC2086
    $PY scripts/train_rl.py --action-mode weights --reward-mode composite --rich-obs \
        --eval-split val --seed "$seed" --timesteps "$TIMESTEPS" --n-envs 6 \
        $rargs --run-id "$rid" > "$LOGDIR/${rid}.log" 2>&1
    if [ $? -eq 0 ]; then tail -1 "$LOGDIR/${rid}.log"; else echo "[sweep2] $rid FAILED — see $LOGDIR/${rid}.log"; fi
  done
done
echo "[sweep2] DONE $(date -u +%FT%TZ)  ($n runs)"
