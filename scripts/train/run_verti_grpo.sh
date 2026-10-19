#!/usr/bin/env bash
# Verti GRPO on top of Verti IL ckpt; rewards from VertiPDM (open-loop).
# Env overrides (defaults match train_verti_grpo.py):
#   OUT DATA_DIR CACHE_DIR CKPT PLAN_ANCHOR
#   EPOCHS BATCH LR NUM_GROUPS STEP_NUM ETA NUM_WORKERS MAX_STEPS DEVICE
set -eo pipefail
ROOT=/home/rm/Desktop/df_offroad
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

echo "========== Verti GRPO =========="
echo "out=$OUT"
echo "data=$DATA_DIR  cache=$CACHE_DIR"
echo "ckpt=$CKPT  anchors=$PLAN_ANCHOR"
echo "epochs=${EPOCHS:-5} batch=${BATCH:-2} lr=${LR:-1e-5}"
echo "groups=${NUM_GROUPS:-2} step_num=${STEP_NUM:-5} eta=${ETA:-1.0}"
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
