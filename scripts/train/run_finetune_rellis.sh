#!/usr/bin/env bash
# Official-style DiffusionDrive IL fine-tune on RELLIS.
# Matches TransfuserAgent recipe: WarmupCosLR, AdamW (backbone lr×0.5),
# trajectory_weight * LossComputer; optional anchor_keep for unused modes.
set -eo pipefail
ROOT=/home/rm/Desktop/df_offroad
# shellcheck disable=SC1091
source "$ROOT/ddv2_env.sh"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

OUT="${OUT:-$ROOT/log/rellis_il_finetune_v3}"
mkdir -p "$OUT"

# Resume: RESUME=last EPOCHS=40 bash run_finetune_rellis.sh
#   --epochs = total target epoch (not extra). last.ckpt at ep20 + EPOCHS=40 → train 21..40.
RESUME_ARGS=()
if [[ -n "${RESUME:-}" ]]; then
  RESUME_ARGS=(--resume "$RESUME")
fi

PLAN_ANCHOR="${PLAN_ANCHOR:-$ROOT/DiffusionDriveV2/kmeans_navsim_traj_20.npy}"
DATA_DIR="${DATA_DIR:-$ROOT/datasets/ddv2_rellis_full}"
INIT_CKPT="${INIT_CKPT:-$ROOT/DiffusionDriveV2/ckpts/diffusiondrive_navsim_88p1_PDMS}"

python "$ROOT/scripts/train/train_rellis_il.py" \
  --data_dir "$DATA_DIR" \
  --ckpt "$INIT_CKPT" \
  --out_dir "$OUT" \
  --epochs "${EPOCHS:-20}" \
  --batch_size "${BATCH:-8}" \
  --lr "${LR:-1e-4}" \
  --warmup_epochs "${WARMUP:-2}" \
  --anchor_keep "${ANCHOR_KEEP:-1.0}" \
  --plan_anchor_path "$PLAN_ANCHOR" \
  --amp \
  --val_every 1 \
  --viz_every 2 \
  --val_batches "${VAL_BATCHES:-20}" \
  "${RESUME_ARGS[@]}" \
  "$@"

echo "TensorBoard: tensorboard --logdir $OUT/tb --port 6006"
echo "curves: $OUT/curves.png"
echo "viz: $OUT/viz/"