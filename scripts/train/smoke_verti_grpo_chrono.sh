#!/usr/bin/env bash
# Smoke: Chrono-map closed-loop VertiPDM + 1 GRPO step.
# SKIP_TRAIN=1 → score-only.
set -eo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
# shellcheck disable=SC1091
source "$ROOT/ddv2_env.sh"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

OUT="${OUT:-$ROOT/log/verti_grpo_chrono_smoke}"
CACHE_DIR="${CACHE_DIR:-$ROOT/datasets/verti_metric_cache_density}"
DATA_DIR="${DATA_DIR:-$ROOT/datasets/verti_ddv2_density}"
CKPT="${CKPT:-$ROOT/log/verti_il_finetune_v2_density/ckpts/best.ckpt}"

EXTRA=()
if [[ "${SKIP_TRAIN:-0}" == "1" ]]; then
  EXTRA+=(--skip_train)
fi

python "$ROOT/scripts/train/smoke_verti_grpo_chrono.py" \
  --cache_dir "$CACHE_DIR" \
  --data_dir "$DATA_DIR" \
  --ckpt "$CKPT" \
  --out_dir "$OUT" \
  --n "${N:-8}" \
  "${EXTRA[@]}" \
  "$@"
