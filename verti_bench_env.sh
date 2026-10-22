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

# Offline ROS msg stubs for MPPI/TAL/TNT when grid_map_msgs is not installed.
# Real ROS (humble/foxy below) still wins if those packages import successfully.
_ROS_STUBS="${_DF_OFFROAD_ROOT}/verti_bench/third_party/ros_msg_stubs"
if [[ -d "$_ROS_STUBS" ]]; then
  if ! python -c "import grid_map_msgs" >/dev/null 2>&1; then
    export PYTHONPATH="${_ROS_STUBS}${PYTHONPATH:+:$PYTHONPATH}"
    echo "[verti_bench_env] ROS msg stubs enabled (${_ROS_STUBS})"
  fi
fi

# Source-built PyChrono (Ubuntu 20.04): prefer repo-local downloads/ over conda/pip.
# Layout: downloads/chrono (src) + downloads/chrono_build (cmake build tree).
# Python modules usually live in chrono_build/bin/pychrono/; some builds put *.so in bin/.
_CHRONO_BUILD="${CHRONO_BUILD_DIR:-${_DF_OFFROAD_ROOT}/downloads/chrono_build}"
_CHRONO_SRC="${CHRONO_SRC_DIR:-${_DF_OFFROAD_ROOT}/downloads/chrono}"
_CHRONO_PY_OK=0
if [[ -d "${_CHRONO_BUILD}/bin/pychrono" ]]; then
  _CHRONO_PY_OK=1
elif [[ -f "${_CHRONO_BUILD}/bin/_core.so" || -f "${_CHRONO_BUILD}/bin/__init__.py" ]]; then
  _CHRONO_PY_OK=1
elif compgen -G "${_CHRONO_BUILD}/bin/_core*.so" >/dev/null 2>&1; then
  _CHRONO_PY_OK=1
fi
# ROS first (may rewrite PYTHONPATH); Chrono paths are prepended AFTER so they win.
[[ -f /opt/ros/humble/setup.bash ]] && source /opt/ros/humble/setup.bash
[[ -f /opt/ros/foxy/setup.bash ]] && source /opt/ros/foxy/setup.bash

if [[ "${_CHRONO_PY_OK}" -eq 1 ]]; then
  export PYTHONPATH="${_CHRONO_BUILD}/bin:${_DF_OFFROAD_ROOT}${PYTHONPATH:+:$PYTHONPATH}"
  export LD_LIBRARY_PATH="${_CHRONO_BUILD}/lib:${_CHRONO_BUILD}/bin${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
  if [[ -d "${_CHRONO_SRC}/libraries/urdf/lib" ]]; then
    export LD_LIBRARY_PATH="${_CHRONO_SRC}/libraries/urdf/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
  fi
  echo "[verti_bench_env] PyChrono PYTHONPATH+=${_CHRONO_BUILD}/bin"
else
  echo "[verti_bench_env] warn: no PyChrono under ${_CHRONO_BUILD}/bin (run: ls ${_CHRONO_BUILD}/bin)" >&2
fi

unset _CONDA_SH _DF_OFFROAD_ROOT _CHRONO_BUILD _CHRONO_SRC _CHRONO_PY_OK
