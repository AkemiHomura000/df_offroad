# Activate DiffusionDriveV2 env (ddv2).
# Usage:  source /path/to/df_offroad/ddv2_env.sh

_DF_OFFROAD_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

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
  echo "[ddv2_env] conda.sh not found. Install Miniconda or set CONDA_EXE." >&2
  return 1 2>/dev/null || exit 1
fi

# shellcheck disable=SC1090
source "$_CONDA_SH"
conda activate ddv2

export DDV2_ROOT="${_DF_OFFROAD_ROOT}/DiffusionDriveV2"
export PYTHONPATH="${DDV2_ROOT}${PYTHONPATH:+:$PYTHONPATH}"
cd "$DDV2_ROOT"

unset _CONDA_SH _DF_OFFROAD_ROOT
