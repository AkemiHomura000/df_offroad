#!/usr/bin/env bash
# Launch Verti GRPO (IL dens ckpt → VertiPDM rewards).
# Env overrides: OUT DATA_DIR CACHE_DIR CKPT PLAN_ANCHOR EPOCHS BATCH LR NUM_GROUPS …
#   EPOCHS=5 BATCH=2 NUM_GROUPS=2 bash scripts/train/start_verti_grpo.sh
set -eo pipefail
ROOT=/home/rm/Desktop/df_offroad
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

OUT="${OUT:-$ROOT/log/verti_grpo}"
mkdir -p "$OUT"
export OUT \
  DATA_DIR="${DATA_DIR:-$ROOT/datasets/verti_ddv2_density}" \
  CACHE_DIR="${CACHE_DIR:-$ROOT/datasets/verti_metric_cache_density}" \
  CKPT="${CKPT:-$ROOT/log/verti_il_finetune_v3_density/ckpts/best.ckpt}" \
  PLAN_ANCHOR="${PLAN_ANCHOR:-$ROOT/DiffusionDriveV2/kmeans_verti_traj_20.npy}" \
  EPOCHS="${EPOCHS:-5}" \
  BATCH="${BATCH:-2}" \
  LR="${LR:-1e-5}" \
  NUM_GROUPS="${NUM_GROUPS:-2}" \
  STEP_NUM="${STEP_NUM:-5}" \
  ETA="${ETA:-1.0}" \
  NUM_WORKERS="${NUM_WORKERS:-2}" \
  DEVICE="${DEVICE:-cuda}" \
  MAX_STEPS="${MAX_STEPS:-0}"

nohup bash "$ROOT/scripts/train/run_verti_grpo.sh" >"$OUT/train.stdout" 2>&1 &
echo "PID=$!  log=$OUT/train.stdout  curves=$OUT/curves/"
echo "ckpt=$CKPT  cache=$CACHE_DIR  epochs=$EPOCHS  groups=$NUM_GROUPS"
