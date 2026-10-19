#!/usr/bin/env bash
# Verti-Bench with DDV2 replacing A* planning.
# Starts plan server (ddv2 env) then sim (chrono9). Ctrl+C stops both.
set -eo pipefail
ROOT=/home/rm/Desktop/df_offroad
CKPT="${CKPT:-$ROOT/log/verti_il_finetune_v3_density/ckpts/best.ckpt}"
HOST="${DDV2_HOST:-127.0.0.1}"
PORT="${DDV2_PORT:-8765}"
WORLD="${WORLD:-5}"
VEHICLE="${VEHICLE:-hmmwv}"
SPEED="${SPEED:-4.0}"
MAX_TIME="${MAX_TIME:-60}"
RENDER="${RENDER:-true}"
DUMP="${DUMP:-$ROOT/log/verti_ddv2}"

mkdir -p "$DUMP" "$ROOT/log"

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
python "$ROOT/scripts/bridge/ddv2_plan_server.py" --ckpt "$CKPT" --host "$HOST" --port "$PORT" \
  > "$ROOT/log/ddv2_plan_server.log" 2>&1 &
SERVER_PID=$!
echo "server pid=$SERVER_PID  log=$ROOT/log/ddv2_plan_server.log"

# wait until port is open
for i in $(seq 1 60); do
  if python -c "import socket; s=socket.create_connection(('$HOST',$PORT),1); s.close()" 2>/dev/null; then
    echo "server ready"
    break
  fi
  if ! kill -0 "$SERVER_PID" 2>/dev/null; then
    echo "server died; see $ROOT/log/ddv2_plan_server.log"
    tail -40 "$ROOT/log/ddv2_plan_server.log" || true
    exit 1
  fi
  sleep 1
done

echo "========== Verti-Bench system=ddv2 =========="
# shellcheck disable=SC1091
source "$ROOT/verti_bench_env.sh"
cd "$ROOT/verti_bench"
LIVE_VIZ="${LIVE_VIZ:-true}"
DRAW_WORLD="${DRAW_WORLD:-true}"

python setup.py \
  system=ddv2 \
  vehicle="$VEHICLE" \
  world_id="$WORLD" \
  speed="$SPEED" \
  max_time="$MAX_TIME" \
  render="$RENDER" \
  ddv2_host="$HOST" \
  ddv2_port="$PORT" \
  ddv2_dump_dir="$DUMP" \
  ddv2_live_viz="$LIVE_VIZ" \
  ddv2_draw_world="$DRAW_WORLD" \
  "$@"
