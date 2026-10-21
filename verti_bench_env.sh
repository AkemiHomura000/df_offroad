# Activate Verti-Bench / PyChrono env (chrono9).
# Usage (from anywhere):  source /path/to/df_offroad/verti_bench_env.sh
# Or:                    cd ~/df_offroad && source ./verti_bench_env.sh

# Repo root = directory containing this script (works when sourced).
_DF_OFFROAD_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Find conda.sh: CONDA_EXE / common install locations / `conda` on PATH.
_CONDA_SH=""
if [[ -n "${CONDA_EXE:-}" ]]; then
  _CONDA_SH="$(dirname "$(dirname "$CONDA_EXE")")/etc/profile.d/conda.sh"
fi
if [[ ! -f "$_CONDA_SH" ]]; then
  for d in \
    "$HOME/miniconda3" \
    "$HOME/anaconda3" \
    "$HOME/mambaforge" \
    "$HOME/miniforge3" \
    /opt/conda \
    /usr/local/miniconda3
  do
    if [[ -f "$d/etc/profile.d/conda.sh" ]]; then
      _CONDA_SH="$d/etc/profile.d/conda.sh"
      break
    fi
  done
fi
if [[ ! -f "$_CONDA_SH" ]] && command -v conda >/dev/null 2>&1; then
  _CONDA_SH="$(conda info --base 2>/dev/null)/etc/profile.d/conda.sh"
fi
if [[ ! -f "$_CONDA_SH" ]]; then
  echo "[verti_bench_env] conda.sh not found. Install Miniconda or set CONDA_EXE." >&2
  return 1 2>/dev/null || exit 1
fi

# shellcheck disable=SC1090
source "$_CONDA_SH"
conda activate chrono9

export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export CHRONO_DATA_DIR="${_DF_OFFROAD_ROOT}/verti_bench/envs/data/"
export PYTHONPATH="${_DF_OFFROAD_ROOT}${PYTHONPATH:+:$PYTHONPATH}"

# ROS2 Humble if present
[[ -f /opt/ros/humble/setup.bash ]] && source /opt/ros/humble/setup.bash

unset _CONDA_SH _DF_OFFROAD_ROOT
