#!/usr/bin/env python3
"""Re-splat lidar_feature from raw Ouster KITTI bins into NAVSIM BEV, and
refresh trajectory/status. Idempotent; uses side-dir atomic writes.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_rellis_ddv2_dataset import (  # noqa: E402
    absolute_to_relative,
    load_lidar_bin,
    parse_pose_line,
    splat_lidar_bev,
    status_from_poses,
)

SAMPLE_RE = re.compile(r"^sample_\d{6}\.npz$")


def atomic_savez(path: Path, **kwargs) -> None:
    tmp_dir = path.parent.parent / ".npz_write_tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    tmp = tmp_dir / path.name
    np.savez_compressed(tmp, **kwargs)
    os.replace(tmp, path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_dir", type=Path, required=True)
    ap.add_argument(
        "--lidar_root",
        type=Path,
        required=True,
        help="Dir with 00000/os1_cloud_node_kitti_bin/*.bin",
    )
    ap.add_argument(
        "--poses_root",
        type=Path,
        default=(ROOT / "downloads/RELLIS3D/extracted/poses/Rellis-3D"),
    )
    ap.add_argument("--seqs", type=str, default="", help="comma list; empty=all")
    args = ap.parse_args()

    out = args.dataset_dir
    meta_path = out / "meta.json"
    meta = json.loads(meta_path.read_text())
    lidar_hz = float(meta.get("lidar_hz", 10.0))
    horizon_s = float(meta.get("horizon_s", 4.0))
    interval_s = float(meta.get("interval_s", 0.5))
    step = max(1, int(round(interval_s * lidar_hz)))
    future_steps = int(round(horizon_s / interval_s))
    want = {s.strip() for s in args.seqs.split(",") if s.strip()}

    pose_cache: dict[str, list] = {}
    files = []
    for split in ("train", "val"):
        d = out / split
        if d.is_dir():
            files.extend(sorted(p for p in d.iterdir() if SAMPLE_RE.match(p.name)))

    n_ok = n_skip = 0
    for i, path in enumerate(files):
        data = dict(np.load(path, allow_pickle=True))
        seq = str(np.asarray(data["seq"]).item())
        if want and seq not in want:
            n_skip += 1
            continue
        fi = int(np.asarray(data["frame_idx"]).item())
        lid_name = str(np.asarray(data["lidar"]).item())
        bin_path = args.lidar_root / seq / "os1_cloud_node_kitti_bin" / lid_name
        if not bin_path.is_file():
            # try flat / alternate layout from zip
            alt = (
                args.lidar_root
                / "Rellis_3D_os1_cloud_node_kitti_bin"
                / "Rellis-3D"
                / seq
                / "os1_cloud_node_kitti_bin"
                / lid_name
            )
            bin_path = alt if alt.is_file() else bin_path
        if not bin_path.is_file():
            raise SystemExit(f"missing bin for {path.name}: tried {bin_path}")

        if seq not in pose_cache:
            pp = args.poses_root / seq / "poses.txt"
            pose_cache[seq] = [
                parse_pose_line(l) for l in pp.read_text().strip().splitlines()
            ]
        poses = pose_cache[seq]
        pts = load_lidar_bin(bin_path)
        lid = splat_lidar_bev(pts, to_navsim=True)
        T0 = poses[fi]
        traj = np.stack(
            [
                absolute_to_relative(T0, poses[min(fi + m * step, len(poses) - 1)])
                for m in range(1, future_steps + 1)
            ],
            axis=0,
        ).astype(np.float32)
        status = status_from_poses(poses, fi, dt=1.0 / lidar_hz)
        data["lidar_feature"] = lid
        data["trajectory"] = traj
        data["status_feature"] = status
        data["lidar_frame"] = np.asarray("navsim_ddv2")
        atomic_savez(path, **data)
        n_ok += 1
        if n_ok % 200 == 0 or n_ok == 1:
            print(f"  [{n_ok}] {path.name} seq={seq} fi={fi} end={traj[-1]}")

    meta["ego_frame"] = "navsim_ddv2"
    meta["ego_frame_note"] = (
        "+x forward, +y left; RELLIS pose/lidar body (-x forward) remapped via [-x,-y,yaw]"
    )
    meta["lidar_frame"] = "navsim_ddv2"
    meta["lidar_frame_note"] = "re-splat from Ouster bins with to_navsim=True"
    meta["coord_fix"] = "resplat_rellis_lidar.py"
    meta_path.write_text(json.dumps(meta, indent=2))
    print(f"[done] resplat {n_ok} skipped={n_skip} → {out}")


if __name__ == "__main__":
    main()
