#!/usr/bin/env bash
# Status-channel ablation (camera+lidar fixed). Env overrides:
#   OUT CKPT DATA_DIR SPLIT PLAN_ANCHOR MAX_SAMPLES MODES NOISE_DEG SEED BATCH
set -eo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
# shellcheck disable=SC1091
source "$ROOT/ddv2_env.sh"
cd "$ROOT"  # ddv2_env.sh may cd into DiffusionDriveV2
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1

OUT="${OUT:-$ROOT/log/status_ablation_attitude}"
mkdir -p "$OUT"

CKPT="${CKPT:-$ROOT/log/verti_il_scratch_elevation_goal_attitude/ckpts/best.ckpt}"
DATA_DIR="${DATA_DIR:-$ROOT/datasets/verti_ddv2_elevation}"
PLAN_ANCHOR="${PLAN_ANCHOR:-$ROOT/DiffusionDriveV2/kmeans_verti_traj_20.npy}"
SPLIT="${SPLIT:-val}"
MAX_SAMPLES="${MAX_SAMPLES:-0}"
NOISE_DEG="${NOISE_DEG:-5,15}"
SEED="${SEED:-0}"
BATCH="${BATCH:-8}"

EXTRA=()
if [[ "${MODES:-0}" == "1" || "${MODES:-}" == "true" ]]; then
  EXTRA+=(--modes)
fi
if [[ "$MAX_SAMPLES" != "0" ]]; then
  EXTRA+=(--max_samples "$MAX_SAMPLES")
fi

echo "========== Status ablation =========="
echo "out=$OUT"
echo "ckpt=$CKPT"
echo "data=$DATA_DIR split=$SPLIT max_samples=$MAX_SAMPLES modes=${MODES:-0}"

python "$ROOT/scripts/train/eval_status_ablation.py" \
  --ckpt "$CKPT" \
  --data_dir "$DATA_DIR" \
  --out_dir "$OUT" \
  --split "$SPLIT" \
  --plan_anchor_path "$PLAN_ANCHOR" \
  --noise_deg "$NOISE_DEG" \
  --seed "$SEED" \
  --batch_size "$BATCH" \
  "${EXTRA[@]}" \
  "$@"

echo "summary: $OUT/ablation.md"
echo "json:    $OUT/ablation_summary.json"
