#!/usr/bin/env bash
# Per-sequence: extract → build npz → delete raw (save disk). Merged zips stay on /tmp.
set -eo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
BASE=$ROOT/downloads/RELLIS3D
OUT_RAW=$BASE/extracted/Rellis-3D
POSE_SRC=$BASE/extracted/poses/Rellis-3D
NPZ_OUT=$ROOT/datasets/ddv2_rellis_full
LOG=$ROOT/log/rellis_full_process.log
# Resume from argv seqs, else remaining after what's already in meta
if [ "$#" -gt 0 ]; then
  SEQS=("$@")
else
  SEQS=(00000 00001 00002 00003 00004)
fi

# unbuffered-ish logging
exec > >(stdbuf -oL -eL tee -a "$LOG") 2>&1

echo "===== $(date) start full RELLIS process seqs=${SEQS[*]} ====="
df -h /home /tmp | grep -v '^文件系统' || df -h /home /tmp

# unzip returns 1 for warnings (e.g. skipped older files) — treat as success
safe_unzip() {
  set +e
  unzip "$@"
  local rc=$?
  set -e
  if [ "$rc" -le 1 ]; then
    return 0
  fi
  echo "[error] unzip failed rc=$rc: $*" >&2
  return "$rc"
}

ensure_merged_zips() {
  if [ ! -f /tmp/rellis_cam_full.zip ]; then
    echo "[join] camera multipart..."
    (cd "$BASE/Rellis_3D_pylon_camera_node" && zip -s0 Rellis_3D_pylon_camera_node.zip --out /tmp/rellis_cam_full.zip)
  fi
  if [ ! -f /tmp/rellis_lidar_full.zip ]; then
    echo "[join] lidar multipart..."
    (cd "$BASE/Rellis_3D_os1_cloud_node_kitti_bin" && zip -s0 Rellis_3D_os1_cloud_node_kitti_bin.zip --out /tmp/rellis_lidar_full.zip)
  fi
}

extract_seq() {
  local s=$1
  mkdir -p "$OUT_RAW/$s"
  [ -f "$POSE_SRC/$s/poses.txt" ] && cp -f "$POSE_SRC/$s/poses.txt" "$OUT_RAW/$s/"
  [ -f "$POSE_SRC/$s/calib.txt" ] && cp -f "$POSE_SRC/$s/calib.txt" "$OUT_RAW/$s/"

  count_files() {
    local d=$1
    if [ -d "$d" ]; then
      ls "$d" | wc -l
    else
      echo 0
    fi
  }

  local ncam nlid
  ncam=$(count_files "$OUT_RAW/$s/pylon_camera_node")
  nlid=$(count_files "$OUT_RAW/$s/os1_cloud_node_kitti_bin")

  if [ "$ncam" -lt 100 ]; then
    echo "[extract cam] $s"
    rm -rf /tmp/rellis_x_cam_$s
    mkdir -p /tmp/rellis_x_cam_$s
    safe_unzip -o /tmp/rellis_cam_full.zip "Rellis_3D_pylon_camera_node/Rellis-3D/${s}/*" -d /tmp/rellis_x_cam_$s >/tmp/rellis_x_cam_$s.log
    src=/tmp/rellis_x_cam_$s/Rellis_3D_pylon_camera_node/Rellis-3D/$s/pylon_camera_node
    rm -rf "$OUT_RAW/$s/pylon_camera_node"
    mv "$src" "$OUT_RAW/$s/"
    rm -rf /tmp/rellis_x_cam_$s
  else
    echo "[cam] $s already ok ($ncam)"
  fi

  nlid=$(count_files "$OUT_RAW/$s/os1_cloud_node_kitti_bin")
  if [ "$nlid" -lt 100 ]; then
    echo "[extract lidar] $s"
    rm -rf /tmp/rellis_x_lidar_$s
    mkdir -p /tmp/rellis_x_lidar_$s
    safe_unzip -o /tmp/rellis_lidar_full.zip "Rellis_3D_os1_cloud_node_kitti_bin/Rellis-3D/${s}/*" -d /tmp/rellis_x_lidar_$s >/tmp/rellis_x_lidar_$s.log
    src=/tmp/rellis_x_lidar_$s/Rellis_3D_os1_cloud_node_kitti_bin/Rellis-3D/$s/os1_cloud_node_kitti_bin
    rm -rf "$OUT_RAW/$s/os1_cloud_node_kitti_bin"
    mv "$src" "$OUT_RAW/$s/"
    rm -rf /tmp/rellis_x_lidar_$s
  else
    echo "[lidar] $s already ok ($nlid)"
  fi

  ncam=$(count_files "$OUT_RAW/$s/pylon_camera_node")
  nlid=$(count_files "$OUT_RAW/$s/os1_cloud_node_kitti_bin")
  npose=$(wc -l < "$OUT_RAW/$s/poses.txt")
  if [ "$ncam" -lt 100 ] || [ "$nlid" -lt 100 ]; then
    echo "[error] $s incomplete after extract cam=$ncam lidar=$nlid" >&2
    return 1
  fi
  echo "[ready] $s cam=$ncam lidar=$nlid poses=$npose"
  df -h /home | tail -1
}

build_seq() {
  local s=$1
  local append_flag=()
  if [ -f "$NPZ_OUT/meta.json" ]; then
    append_flag=(--append)
  fi
  echo "[build] $s → $NPZ_OUT"
  # shellcheck disable=SC1091
  source "$ROOT/ddv2_env.sh"
  python "$ROOT/scripts/rellis/build_rellis_ddv2_dataset.py" \
    --root_dir "$OUT_RAW" \
    --seqs "$s" \
    --out_dir "$NPZ_OUT" \
    --stride 1 \
    --max_samples 0 \
    "${append_flag[@]}"
}

delete_raw_media() {
  local s=$1
  echo "[cleanup] remove raw cam/lidar for $s (keep poses)"
  rm -rf "$OUT_RAW/$s/pylon_camera_node" "$OUT_RAW/$s/os1_cloud_node_kitti_bin"
  df -h /home | tail -1
}

ensure_merged_zips
mkdir -p "$NPZ_OUT"

for s in "${SEQS[@]}"; do
  echo "########## sequence $s $(date) ##########"
  extract_seq "$s"
  build_seq "$s"
  delete_raw_media "$s"
done

echo "===== $(date) ALL DONE ====="
python - <<PY
import json
from pathlib import Path
p=Path("$NPZ_OUT/meta.json")
m=json.loads(p.read_text())
print("sequences:", m.get("sequences"))
print("num_samples:", m.get("num_samples"), "train:", m.get("num_train"), "val:", m.get("num_val"))
print("out:", p.parent)
PY
du -sh "$NPZ_OUT"
df -h /home /tmp | tail -5
