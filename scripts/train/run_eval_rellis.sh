#!/usr/bin/env bash
# Evaluate RELLIS IL checkpoint (default: best finetuned).
set -eo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
# shellcheck disable=SC1091
source "$ROOT/ddv2_env.sh"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

CKPT="${CKPT:-$ROOT/log/rellis_il_finetune/ckpts/best.ckpt}"
OUT="${OUT:-$ROOT/log/rellis_il_eval}"
BASELINE="${BASELINE:-$ROOT/DiffusionDriveV2/ckpts/diffusiondrive_navsim_88p1_PDMS}"
DATA_DIR="${DATA_DIR:-$ROOT/datasets/ddv2_rellis_full}"
SPLIT="${SPLIT:-val}"
VIZ_K="${VIZ_K:-4}"

ANCHOR_ARGS=()
if [[ -n "${PLAN_ANCHOR:-}" ]]; then
  ANCHOR_ARGS=(--plan_anchor_path "$PLAN_ANCHOR")
fi
BASE_ARGS=()
if [[ -n "${BASELINE}" && "${BASELINE}" != "none" ]]; then
  BASE_ARGS=(--baseline_ckpt "$BASELINE")
fi

python "$ROOT/scripts/train/eval_rellis_il.py" \
  --ckpt "$CKPT" \
  --data_dir "$DATA_DIR" \
  --split "$SPLIT" \
  --out_dir "$OUT" \
  --viz_k "$VIZ_K" \
  "${BASE_ARGS[@]}" \
  "${ANCHOR_ARGS[@]}" \
  "$@"

echo "summary: $OUT/summary.json"
echo "hists/csv/viz under $OUT/"
