#!/usr/bin/env bash
# Verti GRPO on top of Verti IL ckpt; rewards from VertiPDM.
# CLOSED_LOOP=1 → Chrono-map kinematic closed-loop rollout before score.
# Env overrides (defaults match train_verti_grpo.py):
#   OUT DATA_DIR CACHE_DIR CKPT PLAN_ANCHOR
#   EPOCHS BATCH LR NUM_GROUPS STEP_NUM ETA NUM_WORKERS MAX_STEPS DEVICE
#   CLOSED_LOOP CL_V_REF
set -eo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
# shellcheck disable=SC1091
source "$ROOT/ddv2_env.sh"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

OUT="${OUT:-$ROOT/log/verti_grpo_v1}"
mkdir -p "$OUT"

DATA_DIR="${DATA_DIR:-$ROOT/datasets/verti_ddv2_density}"
CACHE_DIR="${CACHE_DIR:-$ROOT/datasets/verti_metric_cache_density}"
CKPT="${CKPT:-$ROOT/log/verti_il_finetune_v2_density/ckpts/best.ckpt}"
PLAN_ANCHOR="${PLAN_ANCHOR:-$ROOT/DiffusionDriveV2/kmeans_verti_traj_20.npy}"

EXTRA=()
if [[ -n "${MAX_STEPS:-}" && "${MAX_STEPS}" != "0" ]]; then
  EXTRA+=(--max_steps "$MAX_STEPS")
fi
if [[ "${CLOSED_LOOP:-0}" == "1" || "${CLOSED_LOOP:-}" == "true" ]]; then
  EXTRA+=(--closed_loop --cl_v_ref "${CL_V_REF:-4.0}")
fi

echo "========== Verti GRPO =========="
echo "out=$OUT"
echo "data=$DATA_DIR  cache=$CACHE_DIR"
echo "ckpt=$CKPT  anchors=$PLAN_ANCHOR"
echo "epochs=${EPOCHS:-5} batch=${BATCH:-2} lr=${LR:-1e-5}"
echo "groups=${NUM_GROUPS:-2} step_num=${STEP_NUM:-5} eta=${ETA:-1.0}"
echo "closed_loop=${CLOSED_LOOP:-0} cl_v_ref=${CL_V_REF:-4.0}"
echo "curves=$OUT/curves/  (per-epoch)"

python "$ROOT/scripts/train/train_verti_grpo.py" \
  --data_dir "$DATA_DIR" \
  --cache_dir "$CACHE_DIR" \
  --ckpt "$CKPT" \
  --plan_anchor_path "$PLAN_ANCHOR" \
  --out_dir "$OUT" \
  --epochs "${EPOCHS:-5}" \
  --batch_size "${BATCH:-2}" \
  --lr "${LR:-1e-5}" \
  --num_groups "${NUM_GROUPS:-2}" \
  --step_num "${STEP_NUM:-5}" \
  --eta "${ETA:-1.0}" \
  --num_workers "${NUM_WORKERS:-2}" \
  --device "${DEVICE:-cuda}" \
  "${EXTRA[@]}" \
  "$@"
