#!/usr/bin/env bash
# Collect Verti-Bench expert trajectories. Ctrl+C stops safely (data already on disk).
# Extra args pass through, e.g.:
#   bash run_collect_expert.sh --systems mppi6,pid --worlds 5,9,11 --goal_mode random
#   bash run_collect_expert.sh --worlds 5 --goal_mode fixed --start 40,0 --goal -40,10
#   bash run_collect_expert.sh --num_workers 4 --cuda_devices 0 --systems mppi6,pid --worlds low --episodes 200
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
# shellcheck disable=SC1091
source "$ROOT/verti_bench_env.sh"
cd "$ROOT/verti_bench"
exec python "$ROOT/scripts/bridge/collect_expert_verti.py" "$@"
