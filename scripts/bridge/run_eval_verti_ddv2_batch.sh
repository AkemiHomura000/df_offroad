#!/usr/bin/env bash
# Headless batch Verti-Bench eval with fine-tuned DDV2 (plan server + chrono sim).
# Saves per-scene scene.json for replay; aggregates results.csv / summary.json.
#
# Examples:
#   bash scripts/bridge/run_eval_verti_ddv2_batch.sh
#   WORLDS=low TASKS=2 bash scripts/bridge/run_eval_verti_ddv2_batch.sh
#   WORLDS=1,5,9 MAX_SCENES=5 bash scripts/bridge/run_eval_verti_ddv2_batch.sh
#   SCENES_FILE=log/verti_ddv2_eval_v3_density/scenes_manifest.json \
#     OUT=log/verti_ddv2_eval_replay bash scripts/bridge/run_eval_verti_ddv2_batch.sh
set -eo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
CKPT="${CKPT:-$ROOT/log/verti_il_finetune_elevation/ckpts/best.ckpt}"
if [[ ! -f "$CKPT" ]]; then
  CKPT="$ROOT/log/verti_il_finetune_v3_density/ckpts/best.ckpt"
fi
PLAN_ANCHOR="${PLAN_ANCHOR:-$ROOT/DiffusionDriveV2/kmeans_verti_traj_20.npy}"
HOST="${DDV2_HOST:-127.0.0.1}"
PORT="${DDV2_PORT:-8765}"
OUT="${OUT:-$ROOT/log/verti_ddv2_eval_elevation}"
WORLDS="${WORLDS:-low}"
TASKS="${TASKS:-3}"
GOAL_MODE="${GOAL_MODE:-yaml}"
SEED="${SEED:-0}"
VEHICLE="${VEHICLE:-hmmwv}"
SPEED="${SPEED:-4.0}"
MAX_TIME="${MAX_TIME:-60}"
MAX_SCENES="${MAX_SCENES:-0}"
SCENES_FILE="${SCENES_FILE:-}"

mkdir -p "$OUT" "$ROOT/log"

cleanup() {
  if [[ -n "${SERVER_PID:-}" ]] && kill -0 "$SERVER_PID" 2>/dev/null; then
    kill "$SERVER_PID" 2>/dev/null || true
    wait "$SERVER_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT

echo "========== DDV2 plan server =========="
# shellcheck disable=SC1091
source "$ROOT/ddv2_env.sh"
python "$ROOT/scripts/bridge/ddv2_plan_server.py" \
  --ckpt "$CKPT" \
  --plan_anchor_path "$PLAN_ANCHOR" \
  --host "$HOST" \
  --port "$PORT" \
  >"$ROOT/log/ddv2_plan_server_eval.log" 2>&1 &
SERVER_PID=$!
echo "server pid=$SERVER_PID  log=$ROOT/log/ddv2_plan_server_eval.log  ckpt=$CKPT"

for i in $(seq 1 90); do
  if python -c "import socket; s=socket.create_connection(('$HOST',$PORT),1); s.close()" 2>/dev/null; then
    echo "server ready"
    break
  fi
  if ! kill -0 "$SERVER_PID" 2>/dev/null; then
    echo "server died; see $ROOT/log/ddv2_plan_server_eval.log"
    tail -40 "$ROOT/log/ddv2_plan_server_eval.log" || true
    exit 1
  fi
  sleep 1
done

echo "========== batch eval (headless) =========="
# shellcheck disable=SC1091
source "$ROOT/verti_bench_env.sh"
cd "$ROOT"

EXTRA=()
if [[ -n "$SCENES_FILE" ]]; then
  EXTRA+=(--scenes_file "$SCENES_FILE")
fi
if [[ "${SKIP_EXISTING:-0}" == "1" ]]; then
  EXTRA+=(--skip_existing)
fi

python "$ROOT/scripts/bridge/eval_verti_ddv2_batch.py" \
  --out "$OUT" \
  --ckpt "$CKPT" \
  --plan_anchor_path "$PLAN_ANCHOR" \
  --worlds "$WORLDS" \
  --tasks_per_world "$TASKS" \
  --goal_mode "$GOAL_MODE" \
  --seed "$SEED" \
  --vehicle "$VEHICLE" \
  --speed "$SPEED" \
  --max_time "$MAX_TIME" \
  --host "$HOST" \
  --port "$PORT" \
  --max_scenes "$MAX_SCENES" \
  "${EXTRA[@]}" \
  "$@"

echo "done -> $OUT/summary.json"
