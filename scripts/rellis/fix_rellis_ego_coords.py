#!/usr/bin/env python3
"""Rewrite RELLIS DDV2 npz into NAVSIM ego frame (+x forward, +y left).

Fixes:
  - trajectory / status from KITTI poses via [-x,-y,yaw] (yaw kept)
  - optional lidar_feature 180° remap (Ouster KITTI +x rearward)

Atomic saves to a side directory (never globbed) to avoid BadZipFile / temp pickup.
Each sample stores lidar_frame=navsim_ddv2 after a successful lidar flip so reruns
are idempotent.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_rellis_ddv2_dataset import (  # noqa: E402
    absolute_to_relative,
    parse_pose_line,
    status_from_poses,
)

SAMPLE_RE = re.compile(r"^sample_\d{6}\.npz$")


def load_poses(poses_path: Path) -> list:
    return [parse_pose_line(l) for l in poses_path.read_text().strip().splitlines()]


def list_samples(out: Path) -> list[Path]:
    files = []
    for split in ("train", "val"):
        d = out / split
        if not d.is_dir():
            continue
        files.extend(sorted(p for p in d.iterdir() if SAMPLE_RE.match(p.name)))
    return files


def atomic_savez(path: Path, **kwargs) -> None:
    tmp_dir = path.parent.parent / ".npz_write_tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    tmp = tmp_dir / path.name
    np.savez_compressed(tmp, **kwargs)
    os.replace(tmp, path)


def rewrite_sample(
    npz_path: Path,
    poses: list,
    lidar_hz: float,
    horizon_s: float,
    interval_s: float,
    fix_lidar: bool,
) -> dict:
    data = dict(np.load(npz_path, allow_pickle=True))
    frame_idx = int(np.asarray(data["frame_idx"]).item())
    step = max(1, int(round(interval_s * lidar_hz)))
    future_steps = int(round(horizon_s / interval_s))

    T0 = poses[frame_idx]
    traj = np.stack(
        [
            absolute_to_relative(T0, poses[min(frame_idx + m * step, len(poses) - 1)])
            for m in range(1, future_steps + 1)
        ],
        axis=0,
    ).astype(np.float32)
    status = status_from_poses(poses, frame_idx, dt=1.0 / lidar_hz)

    old_traj = np.asarray(data["trajectory"], dtype=np.float32)
    data["trajectory"] = traj
    data["status_feature"] = status

    lidar_flipped = False
    cur_lid = str(np.asarray(data["lidar_frame"]).item()) if "lidar_frame" in data else ""
    if fix_lidar and "lidar_feature" in data and cur_lid != "navsim_ddv2":
        lid = np.asarray(data["lidar_feature"])
        data["lidar_feature"] = lid[..., ::-1, ::-1].copy()
        data["lidar_frame"] = np.asarray("navsim_ddv2")
        lidar_flipped = True
    elif cur_lid == "navsim_ddv2":
        data["lidar_frame"] = np.asarray("navsim_ddv2")

    atomic_savez(npz_path, **data)
    return {
        "file": str(npz_path),
        "old_x_mean": float(old_traj[:, 0].mean()),
        "new_x_mean": float(traj[:, 0].mean()),
        "old_yaw_end": float(old_traj[-1, 2]),
        "new_yaw_end": float(traj[-1, 2]),
        "lidar_flipped": lidar_flipped,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--dataset_dir",
        default="/home/rm/Desktop/df_offroad/datasets/ddv2_rellis_full",
    )
    ap.add_argument(
        "--poses_root",
        default="/home/rm/Desktop/df_offroad/downloads/RELLIS3D/extracted/poses/Rellis-3D",
    )
    ap.add_argument("--lidar_hz", type=float, default=0.0)
    ap.add_argument("--horizon_s", type=float, default=0.0)
    ap.add_argument("--interval_s", type=float, default=0.0)
    ap.add_argument("--fix_lidar", action="store_true", default=True)
    ap.add_argument("--no_fix_lidar", action="store_true")
    ap.add_argument(
        "--reset_lidar_frame_flag",
        action="store_true",
        help="Clear lidar_frame tags without flipping (for recovery)",
    )
    ap.add_argument(
        "--force_lidar_flip",
        action="store_true",
        help="Flip even if lidar_frame=navsim_ddv2 (dangerous)",
    )
    args = ap.parse_args()

    out = Path(args.dataset_dir)
    meta_path = out / "meta.json"
    meta = json.loads(meta_path.read_text()) if meta_path.is_file() else {}
    lidar_hz = args.lidar_hz or float(meta.get("lidar_hz", 10.0))
    horizon_s = args.horizon_s or float(meta.get("horizon_s", 4.0))
    interval_s = args.interval_s or float(meta.get("interval_s", 0.5))

    fix_lidar = (not args.no_fix_lidar) and args.fix_lidar
    poses_root = Path(args.poses_root)
    pose_cache: dict[str, list] = {}

    files = list_samples(out)
    if not files:
        raise SystemExit(f"no samples under {out}")

    # purge stray temps
    tmp_dir = out / ".npz_write_tmp"
    if tmp_dir.is_dir():
        for p in tmp_dir.glob("*.npz"):
            p.unlink()
    for split in ("train", "val"):
        d = out / split
        if d.is_dir():
            for p in d.glob("sample_*writetmp*.npz"):
                p.unlink()
            for p in d.glob("sample_*_writetmp.npz"):
                p.unlink()

    stats = {
        "n": 0,
        "pos_x": 0,
        "neg_x": 0,
        "sum_new_x": 0.0,
        "yaw_flipped": 0,
        "lid_flip": 0,
    }
    for i, path in enumerate(files):
        try:
            raw = np.load(path, allow_pickle=True)
            seq = str(np.asarray(raw["seq"]).item())
            if args.reset_lidar_frame_flag and "lidar_frame" in raw.files:
                data = dict(raw)
                data.pop("lidar_frame", None)
                atomic_savez(path, **data)
                raw = np.load(path, allow_pickle=True)
            if args.force_lidar_flip and "lidar_frame" in raw.files:
                data = dict(raw)
                data.pop("lidar_frame", None)
                atomic_savez(path, **data)
        except Exception as e:
            raise SystemExit(f"cannot read {path}: {e}") from e

        if seq not in pose_cache:
            pp = poses_root / seq / "poses.txt"
            if not pp.is_file():
                raise SystemExit(f"missing poses: {pp}")
            pose_cache[seq] = load_poses(pp)

        info = rewrite_sample(
            path,
            pose_cache[seq],
            lidar_hz=lidar_hz,
            horizon_s=horizon_s,
            interval_s=interval_s,
            fix_lidar=fix_lidar,
        )
        stats["n"] += 1
        if info["new_x_mean"] > 0.05:
            stats["pos_x"] += 1
        elif info["new_x_mean"] < -0.05:
            stats["neg_x"] += 1
        stats["sum_new_x"] += info["new_x_mean"]
        if info["lidar_flipped"]:
            stats["lid_flip"] += 1
        if abs(info["old_yaw_end"] + info["new_yaw_end"]) < 1e-4 and abs(
            info["new_yaw_end"]
        ) > 1e-3:
            stats["yaw_flipped"] += 1
        if (i + 1) % 500 == 0 or i == 0:
            print(
                f"  [{i+1}/{len(files)}] {path.name} "
                f"old_x={info['old_x_mean']:.3f} new_x={info['new_x_mean']:.3f} "
                f"yaw {info['old_yaw_end']:+.3f}→{info['new_yaw_end']:+.3f} "
                f"lid_flip={info['lidar_flipped']}"
            )

    meta["ego_frame"] = "navsim_ddv2"
    meta["ego_frame_note"] = (
        "+x forward, +y left; RELLIS pose/lidar body (-x forward) remapped via [-x,-y,yaw]"
    )
    if fix_lidar:
        meta["lidar_frame"] = "navsim_ddv2"
        meta["lidar_frame_note"] = (
            "BEV hist in NAVSIM; Ouster KITTI +x rearward → 180° flip to match camera"
        )
    meta["coord_fix"] = "fix_rellis_ego_coords.py"
    meta_path.write_text(json.dumps(meta, indent=2))
    mean_x = stats["sum_new_x"] / max(stats["n"], 1)
    print(
        f"[done] rewrote {stats['n']} samples → {out}\n"
        f"  traj mean-x>0: {stats['pos_x']}  mean-x<0: {stats['neg_x']}  "
        f"avg mean-x: {mean_x:.3f}  yaw_sign_fixed≈{stats['yaw_flipped']}  "
        f"lidar_flips={stats['lid_flip']}"
    )


if __name__ == "__main__":
    main()
