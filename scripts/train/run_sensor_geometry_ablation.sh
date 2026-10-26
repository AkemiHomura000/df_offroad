#!/usr/bin/env bash
# Scheme D: Chrono attitude re-render → open-loop eval → figures.
# Env: OUT NUM_FRAMES SEED MIN_ATT_DEG CKPT MODES
set -eo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

OUT="${OUT:-$ROOT/log/sensor_geom_ablation}"
NUM_FRAMES="${NUM_FRAMES:-24}"
SEED="${SEED:-0}"
MIN_ATT_DEG="${MIN_ATT_DEG:-8}"
CKPT="${CKPT:-$ROOT/log/verti_il_scratch_elevation_goal_attitude/ckpts/best.ckpt}"
DATA_DIR="${DATA_DIR:-$ROOT/datasets/verti_ddv2_elevation}"
EXPERT_DIR="${EXPERT_DIR:-$ROOT/log/verti_expert}"
PLAN_ANCHOR="${PLAN_ANCHOR:-$ROOT/DiffusionDriveV2/kmeans_verti_traj_20.npy}"

mkdir -p "$OUT"

echo "========== [1/3] Chrono re-render =========="
# shellcheck disable=SC1091
source "$ROOT/verti_bench_env.sh"
cd "$ROOT"
python "$ROOT/scripts/bridge/rerender_attitude_views.py" \
  --data_dir "$DATA_DIR" \
  --expert_dir "$EXPERT_DIR" \
  --out "$OUT" \
  --num_frames "$NUM_FRAMES" \
  --seed "$SEED" \
  --min_att_deg "$MIN_ATT_DEG" \
  --balance

echo "========== [2/3] Open-loop eval (ddv2) =========="
# shellcheck disable=SC1091
source "$ROOT/ddv2_env.sh"
cd "$ROOT"
EXTRA=()
if [[ "${MODES:-0}" == "1" || "${MODES:-}" == "true" ]]; then
  EXTRA+=(--modes)
fi
python "$ROOT/scripts/train/eval_sensor_geometry_ablation.py" \
  --render_dir "$OUT" \
  --ckpt "$CKPT" \
  --out_dir "$OUT/eval" \
  --plan_anchor_path "$PLAN_ANCHOR" \
  "${EXTRA[@]}"

echo "========== [3/3] Figures =========="
python "$ROOT/scripts/train/plot_sensor_geometry_ablation.py" \
  --render_dir "$OUT" \
  --eval_dir "$OUT/eval"

echo "done: $OUT/eval/geometry_ablation.md"
echo "figures: $OUT/eval/figures/"
