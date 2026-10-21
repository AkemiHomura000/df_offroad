#!/usr/bin/env bash
# IL fine-tune on RELLIS (default: RELLIS anchors, hold-out val).
# Env overrides: OUT DATA_DIR INIT_CKPT PLAN_ANCHOR EPOCHS BATCH LR WARMUP ANCHOR_KEEP RESUME
set -eo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$ROOT"
# shellcheck disable=SC1091
source "$ROOT/ddv2_env.sh"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1

if ! nvidia-smi &>/dev/null; then
  echo "[warn] nvidia-smi failed; trying nvidia-modprobe ..."
  sudo nvidia-modprobe -u -c=0 || true
  sleep 1
fi
if ! nvidia-smi &>/dev/null; then
  echo "[error] GPU unavailable"
  exit 1
fi
nvidia-smi --query-gpu=name,memory.free --format=csv,noheader

OUT="${OUT:-$ROOT/log/rellis_il_finetune}"
mkdir -p "$OUT"
export OUT \
  DATA_DIR="${DATA_DIR:-$ROOT/datasets/ddv2_rellis_full}" \
  INIT_CKPT="${INIT_CKPT:-$ROOT/DiffusionDriveV2/ckpts/diffusiondrive_navsim_88p1_PDMS}" \
  PLAN_ANCHOR="${PLAN_ANCHOR:-$ROOT/DiffusionDriveV2/kmeans_rellis_traj_20.npy}" \
  EPOCHS="${EPOCHS:-10}" \
  BATCH="${BATCH:-8}" \
  LR="${LR:-1e-4}" \
  WARMUP="${WARMUP:-2}" \
  ANCHOR_KEEP="${ANCHOR_KEEP:-1.0}"
nohup bash "$ROOT/scripts/train/run_finetune_rellis.sh" >"$OUT/train.stdout" 2>&1 &
echo "PID=$!  log=$OUT/train.stdout"
echo "data=$DATA_DIR  anchors=$PLAN_ANCHOR  epochs=$EPOCHS  out=$OUT"
