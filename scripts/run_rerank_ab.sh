#!/usr/bin/env bash
# Re-ranking A/B on the FROZEN TEST split — the fast, decisive generalization probe.
#
# Same reward config (`real`: realized engine, dd brake 2.0 — the config that LOST OOS at +11% vs a
# +25.7% baseline). The ONLY variable is universe selection:
#   static  (rerank_every=0) — pick vol-top8 once at window start, hold it (the OOS-failing setup)
#   rerank  (rerank_every=1) — re-pick the vol-top8 daily, tracking the current vol leaders
# Question: does a more stationary "manage today's hot names" task close the OOS gap? Both arms on
# TEST, 3 seeds, 1M steps. Resumable (skips completed runs after an interruption).
#
#   nohup bash scripts/run_rerank_ab.sh [TIMESTEPS] ["SEEDS"] > runs-rl/ab.log 2>&1 < /dev/null &
set -uo pipefail
cd "$(dirname "$0")/.."

TIMESTEPS="${1:-1000000}"
SEEDS="${2:-0 1 2}"
PY=.venv/bin/python
LOGDIR=runs-rl/ab-logs
mkdir -p "$LOGDIR"

ARMS=(
  "static|--rerank-every 0"
  "rerank|--rerank-every 1"
)

echo "[ab] START $(date -u +%FT%TZ)  timesteps=$TIMESTEPS  seeds='$SEEDS'  split=TEST (frozen)"
n=0
for entry in "${ARMS[@]}"; do
  arm="${entry%%|*}"
  rargs="${entry#*|}"
  for seed in $SEEDS; do
    rid="ppo-ab-${arm}-s${seed}"
    n=$((n+1))
    if grep -q "^\[train_rl\]" "$LOGDIR/${rid}.log" 2>/dev/null; then
      echo "[ab] ($n) skip $rid — already complete"; continue
    fi
    echo "[ab] === ($n) $rid  $(date -u +%FT%TZ) ==="
    # shellcheck disable=SC2086
    $PY scripts/train_rl.py --action-mode weights --reward-mode composite --rich-obs \
        --eval-split test --seed "$seed" --timesteps "$TIMESTEPS" --n-envs 6 \
        --realized-lambda 10 --turn-lambda 0 --gb-lambda 0 --dd-lambda 2.0 \
        $rargs --run-id "$rid" > "$LOGDIR/${rid}.log" 2>&1
    if [ $? -eq 0 ]; then tail -1 "$LOGDIR/${rid}.log"; else echo "[ab] $rid FAILED — see $LOGDIR/${rid}.log"; fi
  done
done
echo "[ab] DONE $(date -u +%FT%TZ)  ($n runs)"
