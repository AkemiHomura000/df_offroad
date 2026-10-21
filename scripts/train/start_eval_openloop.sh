#!/usr/bin/env bash
# Open-loop ADE/FDE eval (RELLIS or Verti dataset) via eval_rellis_il.py.
# Optional baseline compare. Env: OUT CKPT BASELINE DATA_DIR SPLIT VIZ_K PLAN_ANCHOR MAX_SAMPLES
set -eo pipefail
ROOT=/home/rm/Desktop/df_offroad
cd "$ROOT"
# shellcheck disable=SC1091
source "$ROOT/ddv2_env.sh"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1

OUT="${OUT:-$ROOT/log/openloop_eval_elevation}"
mkdir -p "$OUT"
export OUT \
  CKPT="${CKPT:-$ROOT/log/verti_il_finetune_elevation/ckpts/best.ckpt}" \
  DATA_DIR="${DATA_DIR:-$ROOT/datasets/verti_ddv2_elevation}" \
  BASELINE="${BASELINE:-$ROOT/DiffusionDriveV2/ckpts/diffusiondrive_navsim_88p1_PDMS}" \
  SPLIT="${SPLIT:-val}" \
  VIZ_K="${VIZ_K:-4}"

MAX_SAMPLES="${MAX_SAMPLES:-100}"
EXTRA=()
if [[ -n "${PLAN_ANCHOR:-}" ]]; then
  EXTRA+=(--plan_anchor_path "$PLAN_ANCHOR")
fi
if [[ "$MAX_SAMPLES" != "0" ]]; then
  EXTRA+=(--max_samples "$MAX_SAMPLES")
fi

nohup bash "$ROOT/scripts/train/run_eval_rellis.sh" "${EXTRA[@]}" \
  >"$OUT/eval.stdout" 2>&1 &
echo "PID=$!  log=$OUT/eval.stdout"
echo "ckpt=$CKPT  baseline=$BASELINE  data=$DATA_DIR  max_samples=$MAX_SAMPLES"
