#!/usr/bin/env bash
# Extract RELLIS-3D camera + Ouster lidar for selected sequences into extracted/Rellis-3D/
set -eo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
BASE="$ROOT/downloads/RELLIS3D"
OUT=$BASE/extracted/Rellis-3D
POSE_SRC=$BASE/extracted/poses/Rellis-3D
SEQS="${SEQS:-00001,00002,00003,00004}"
mkdir -p "$OUT" /tmp/rellis_extract

echo "[1/4] join multipart camera zip (if needed)"
if [ ! -f /tmp/rellis_cam_full.zip ]; then
  cd "$BASE/Rellis_3D_pylon_camera_node"
  zip -s0 Rellis_3D_pylon_camera_node.zip --out /tmp/rellis_cam_full.zip
fi

echo "[2/4] join multipart lidar zip (if needed)"
if [ ! -f /tmp/rellis_lidar_full.zip ]; then
  cd "$BASE/Rellis_3D_os1_cloud_node_kitti_bin"
  zip -s0 Rellis_3D_os1_cloud_node_kitti_bin.zip --out /tmp/rellis_lidar_full.zip
fi

IFS=',' read -ra SEQ_ARR <<< "$SEQS"
for s in "${SEQ_ARR[@]}"; do
  s=$(echo "$s" | tr -d ' ')
  echo "======== sequence $s ========"
  mkdir -p "$OUT/$s"

  # poses / calib
  if [ -f "$POSE_SRC/$s/poses.txt" ]; then
    cp -f "$POSE_SRC/$s/poses.txt" "$OUT/$s/"
    [ -f "$POSE_SRC/$s/calib.txt" ] && cp -f "$POSE_SRC/$s/calib.txt" "$OUT/$s/"
  else
    echo "WARN: missing poses for $s"
  fi

  if [ -d "$OUT/$s/pylon_camera_node" ] && [ "$(ls -A "$OUT/$s/pylon_camera_node" 2>/dev/null | wc -l)" -gt 100 ]; then
    echo "[cam] $s already present, skip"
  else
    echo "[cam] extracting $s ..."
    rm -rf /tmp/rellis_extract/cam_$s
    mkdir -p /tmp/rellis_extract/cam_$s
    unzip -o /tmp/rellis_cam_full.zip \
      "Rellis_3D_pylon_camera_node/Rellis-3D/${s}/*" \
      -d /tmp/rellis_extract/cam_$s >/tmp/rellis_extract/cam_${s}.log
    src=/tmp/rellis_extract/cam_$s/Rellis_3D_pylon_camera_node/Rellis-3D/$s/pylon_camera_node
    if [ -d "$src" ]; then
      rm -rf "$OUT/$s/pylon_camera_node"
      mv "$src" "$OUT/$s/"
    else
      echo "ERROR: cam extract failed for $s"; exit 1
    fi
    rm -rf /tmp/rellis_extract/cam_$s
  fi

  if [ -d "$OUT/$s/os1_cloud_node_kitti_bin" ] && [ "$(ls -A "$OUT/$s/os1_cloud_node_kitti_bin" 2>/dev/null | wc -l)" -gt 100 ]; then
    echo "[lidar] $s already present, skip"
  else
    echo "[lidar] extracting $s ..."
    rm -rf /tmp/rellis_extract/lidar_$s
    mkdir -p /tmp/rellis_extract/lidar_$s
    unzip -o /tmp/rellis_lidar_full.zip \
      "Rellis_3D_os1_cloud_node_kitti_bin/Rellis-3D/${s}/*" \
      -d /tmp/rellis_extract/lidar_$s >/tmp/rellis_extract/lidar_${s}.log
    src=/tmp/rellis_extract/lidar_$s/Rellis_3D_os1_cloud_node_kitti_bin/Rellis-3D/$s/os1_cloud_node_kitti_bin
    if [ -d "$src" ]; then
      rm -rf "$OUT/$s/os1_cloud_node_kitti_bin"
      mv "$src" "$OUT/$s/"
    else
      echo "ERROR: lidar extract failed for $s"; exit 1
    fi
    rm -rf /tmp/rellis_extract/lidar_$s
  fi

  echo "[ok] $s cam=$(ls "$OUT/$s/pylon_camera_node" | wc -l) lidar=$(ls "$OUT/$s/os1_cloud_node_kitti_bin" | wc -l) poses=$(wc -l < "$OUT/$s/poses.txt")"
  df -h /home | tail -1
done

echo "[3/4] ensure 00000 poses present"
if [ -d "$OUT/00000" ] && [ ! -f "$OUT/00000/poses.txt" ] && [ -f "$POSE_SRC/00000/poses.txt" ]; then
  cp -f "$POSE_SRC/00000/poses.txt" "$OUT/00000/"
  [ -f "$POSE_SRC/00000/calib.txt" ] && cp -f "$POSE_SRC/00000/calib.txt" "$OUT/00000/"
fi

echo "[4/4] summary"
du -sh "$OUT"/*
df -h /home | tail -1
echo "DONE extract"
