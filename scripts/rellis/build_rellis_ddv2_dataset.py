#!/usr/bin/env python3
"""
Build DDV2-style training dataset from RELLIS-3D sequences.

Outputs npz samples:
  camera_feature (3,256,1024)  lidar_feature (1,256,256)
  status_feature (8,)          trajectory (8,3)  # future 4s @ 0.5s in ego frame
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np


def parse_pose_line(line: str) -> np.ndarray:
    """KITTI 3x4 pose → 4x4."""
    v = np.fromstring(line.strip(), sep=" ", dtype=np.float64)
    if v.size != 12:
        raise ValueError(f"bad pose line: {line[:80]}")
    T = np.eye(4, dtype=np.float64)
    T[:3, :] = v.reshape(3, 4)
    return T


def T_to_xyyaw(T: np.ndarray) -> np.ndarray:
    """Extract x,y,yaw from a 4x4 pose (z-up, yaw from XY plane)."""
    x, y = float(T[0, 3]), float(T[1, 3])
    yaw = float(np.arctan2(T[1, 0], T[0, 0]))
    return np.array([x, y, yaw], dtype=np.float64)


def wrap_yaw(yaw: float) -> float:
    return float((yaw + np.pi) % (2 * np.pi) - np.pi)


def rellis_pose_to_navsim(xyyaw: np.ndarray) -> np.ndarray:
    """Map RELLIS pose-body xyyaw into DDV2/NAVSIM ego frame.

    RELLIS KITTI poses use a body frame whose +x points rearward
    (future motion is predominantly -x). DDV2 expects +x forward, +y left.
    A 180° rotation about z maps points as x'=-x, y'=-y. Relative yaw is
    invariant under that conjugation (R' = R_180 R R_180 = R), so keep yaw.
    """
    x, y, yaw = (float(xyyaw[0]), float(xyyaw[1]), float(xyyaw[2]))
    return np.array([-x, -y, wrap_yaw(yaw)], dtype=np.float64)


def invert_T(T: np.ndarray) -> np.ndarray:
    R = T[:3, :3]
    t = T[:3, 3]
    Ti = np.eye(4, dtype=np.float64)
    Ti[:3, :3] = R.T
    Ti[:3, 3] = -R.T @ t
    return Ti


def absolute_to_relative(T_w_t: np.ndarray, T_w_f: np.ndarray) -> np.ndarray:
    """Future pose in DDV2 ego frame (+x forward) at t."""
    T_t_f = invert_T(T_w_t) @ T_w_f
    return rellis_pose_to_navsim(T_to_xyyaw(T_t_f)).astype(np.float32)


def load_lidar_bin(path: Path) -> np.ndarray:
    pts = np.fromfile(path, dtype=np.float32).reshape(-1, 4)
    return pts[:, :3]


def splat_lidar_bev(
    pts: np.ndarray,
    min_xy: float = -32.0,
    max_xy: float = 32.0,
    ppm: float = 4.0,
    hist_max: int = 5,
    max_z: float = 100.0,
    to_navsim: bool = True,
) -> np.ndarray:
    """BEV occupancy in DDV2/NAVSIM ego frame (+x forward, +y left).

    RELLIS Ouster KITTI bins share the pose body frame whose +x points
    rearward (cam2lidar: camera +Z ≈ −X_lidar). Apply the same 180° yaw
    map as poses (−x, −y) so BEV aligns with camera-forward and trajectories.
    """
    pts = pts[np.isfinite(pts).all(axis=1)]
    pts = pts[pts[:, 2] < max_z]
    xy = pts[:, :2].copy()
    if to_navsim:
        xy[:, 0] *= -1.0
        xy[:, 1] *= -1.0
    n = int((max_xy - min_xy) * ppm) + 1
    bins = np.linspace(min_xy, max_xy, n)
    if xy.size == 0:
        return np.zeros((1, n - 1, n - 1), dtype=np.float32)
    hist = np.histogramdd(xy, bins=(bins, bins))[0]
    hist[hist > hist_max] = hist_max
    return (hist / hist_max).astype(np.float32)[None, ...]


def image_to_camera_feature(path: Path, out_w: int = 1024, out_h: int = 256) -> np.ndarray:
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise RuntimeError(f"failed to read {path}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    h, w = rgb.shape[:2]
    target_aspect = out_w / out_h
    cur_aspect = w / h
    if cur_aspect > target_aspect:
        new_w = w
        new_h = int(round(w / target_aspect))
        pad = (new_h - h) // 2
        canvas = np.zeros((new_h, new_w, 3), dtype=np.uint8)
        canvas[pad : pad + h] = rgb
        rgb = canvas
    else:
        new_h = h
        new_w = int(round(h * target_aspect))
        pad = (new_w - w) // 2
        canvas = np.zeros((new_h, new_w, 3), dtype=np.uint8)
        canvas[:, pad : pad + w] = rgb
        rgb = canvas
    rgb = cv2.resize(rgb, (out_w, out_h), interpolation=cv2.INTER_AREA)
    return np.transpose(rgb.astype(np.float32) / 255.0, (2, 0, 1))


def status_from_poses(poses: list, idx: int, dt: float = 0.1) -> np.ndarray:
    """Ego-frame status in DDV2 convention: cmd(4) + vel(2) + acc(2)."""
    if idx + 1 < len(poses) and idx - 1 >= 0:
        d_fwd = absolute_to_relative(poses[idx], poses[idx + 1])[:2]
        d_bwd = absolute_to_relative(poses[idx - 1], poses[idx])[:2]
        vel = ((d_fwd + d_bwd) / (2 * dt)).astype(np.float32)
    else:
        rel_f = absolute_to_relative(poses[idx], poses[min(idx + 1, len(poses) - 1)])
        vel = (rel_f[:2] / max(dt, 1e-6)).astype(np.float32)

    if idx + 2 < len(poses) and idx - 2 >= 0:
        vel_n = absolute_to_relative(poses[idx], poses[idx + 2])[:2] / (2 * dt)
        vel_p = absolute_to_relative(poses[idx - 2], poses[idx])[:2] / (2 * dt)
        acc = ((vel_n - vel_p) / (2 * dt)).astype(np.float32)
    else:
        acc = np.zeros(2, dtype=np.float32)

    fut = absolute_to_relative(poses[idx], poses[min(idx + 10, len(poses) - 1)])
    dyaw = wrap_yaw(float(fut[2]))
    cmd = np.zeros(4, dtype=np.float32)
    if dyaw > np.deg2rad(15):
        cmd[0] = 1.0  # left
    elif dyaw < -np.deg2rad(15):
        cmd[2] = 1.0  # right
    else:
        cmd[1] = 1.0  # straight
    return np.concatenate([cmd, vel, acc]).astype(np.float32)


def list_images(cam_dir: Path) -> list:
    files = sorted(cam_dir.glob("frame*.jpg")) + sorted(cam_dir.glob("frame*.png"))

    def key(p: Path):
        m = re.search(r"frame(\d+)", p.name)
        return int(m.group(1)) if m else p.name

    return sorted(files, key=key)


def list_bins(lidar_dir: Path) -> list:
    files = list(lidar_dir.glob("*.bin"))
    return sorted(files, key=lambda p: int(p.stem))


def discover_sequences(root: Path) -> List[Path]:
    seqs = sorted([p for p in root.iterdir() if p.is_dir() and re.fullmatch(r"\d{5}", p.name)])
    return seqs


def process_sequence(
    seq: Path,
    out_train: Path,
    out_val: Path,
    sample_id_start: int,
    stride: int,
    max_samples: Optional[int],
    lidar_hz: float,
    horizon_s: float,
    interval_s: float,
    val_ratio: float,
) -> tuple:
    cam_dir = seq / "pylon_camera_node"
    lidar_dir = seq / "os1_cloud_node_kitti_bin"
    poses_path = seq / "poses.txt"
    if not (cam_dir.is_dir() and lidar_dir.is_dir() and poses_path.is_file()):
        print(f"[skip] {seq.name}: missing cam/lidar/poses")
        return [], sample_id_start

    images = list_images(cam_dir)
    bins = list_bins(lidar_dir)
    poses = [parse_pose_line(l) for l in poses_path.read_text().strip().splitlines()]
    n = min(len(images), len(bins), len(poses))
    print(f"[seq {seq.name}] images={len(images)} bins={len(bins)} poses={len(poses)} use_n={n}")

    step = max(1, int(round(interval_s * lidar_hz)))
    future_steps = int(round(horizon_s / interval_s))
    need = future_steps * step
    indices = list(range(0, n - need, stride))
    if max_samples is not None and len(indices) > max_samples:
        sel = np.linspace(0, len(indices) - 1, max_samples).astype(int)
        indices = [indices[i] for i in sel]

    meta_samples = []
    sid = sample_id_start
    val_every = max(1, int(round(1.0 / val_ratio))) if val_ratio > 0 else 10**9

    for k, i in enumerate(indices):
        cam = image_to_camera_feature(images[i])
        pts = load_lidar_bin(bins[i])
        lid = splat_lidar_bev(pts)
        status = status_from_poses(poses, i, dt=1.0 / lidar_hz)

        traj = []
        T0 = poses[i]
        for m in range(1, future_steps + 1):
            traj.append(absolute_to_relative(T0, poses[i + m * step]))
        traj = np.stack(traj, axis=0).astype(np.float32)

        is_val = (k % val_every == 0)
        split_dir = out_val if is_val else out_train
        name = f"sample_{sid:06d}.npz"
        path = split_dir / name
        np.savez_compressed(
            path,
            camera_feature=cam,
            lidar_feature=lid,
            status_feature=status,
            trajectory=traj,
            seq=seq.name,
            frame_idx=i,
            image=str(images[i].name),
            lidar=str(bins[i].name),
        )
        meta_samples.append(
            {
                "file": f"{'val' if is_val else 'train'}/{name}",
                "seq": seq.name,
                "frame_idx": i,
                "image": images[i].name,
                "lidar": bins[i].name,
            }
        )
        sid += 1
        if (k + 1) % 100 == 0 or k == 0:
            print(f"  [{seq.name} {k+1}/{len(indices)}] {name}")

    return meta_samples, sid


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--root_dir",
        default="/home/rm/Desktop/df_offroad/downloads/RELLIS3D/extracted/Rellis-3D",
        help="Directory containing 00000/00001/... sequences",
    )
    ap.add_argument(
        "--seq_dir",
        default="",
        help="Optional single sequence dir (overrides --root_dir)",
    )
    ap.add_argument(
        "--out_dir",
        default="/home/rm/Desktop/df_offroad/datasets/ddv2_rellis_full",
    )
    ap.add_argument("--max_samples", type=int, default=0, help="0 = no cap (full)")
    ap.add_argument("--max_samples_per_seq", type=int, default=0, help="0 = no per-seq cap")
    ap.add_argument("--stride", type=int, default=1, help="sample every N frames")
    ap.add_argument("--lidar_hz", type=float, default=10.0)
    ap.add_argument("--horizon_s", type=float, default=4.0)
    ap.add_argument("--interval_s", type=float, default=0.5)
    ap.add_argument("--val_ratio", type=float, default=0.1)
    ap.add_argument("--seqs", type=str, default="", help="comma list e.g. 00000,00001")
    ap.add_argument("--append", action="store_true", help="append to existing out_dir instead of clearing")
    args = ap.parse_args()

    if args.seq_dir:
        sequences = [Path(args.seq_dir)]
    else:
        root = Path(args.root_dir)
        sequences = discover_sequences(root)
        if args.seqs:
            want = {s.strip() for s in args.seqs.split(",") if s.strip()}
            sequences = [s for s in sequences if s.name in want]

    if not sequences:
        raise SystemExit(f"no sequences found under {args.root_dir}")

    out = Path(args.out_dir)
    train_dir = out / "train"
    val_dir = out / "val"
    train_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)

    all_meta = []
    sid = 0
    if args.append and (out / "meta.json").is_file():
        prev = json.loads((out / "meta.json").read_text())
        all_meta = list(prev.get("samples", []))
        sid = prev.get("num_samples", len(all_meta))
        print(f"[append] resume from sample id {sid}, existing {len(all_meta)}")
    elif not args.append:
        for p in list(train_dir.glob("*.npz")) + list(val_dir.glob("*.npz")):
            p.unlink()

    global_cap = args.max_samples if args.max_samples > 0 else None
    per_seq_cap = args.max_samples_per_seq if args.max_samples_per_seq > 0 else None

    done_seqs = {m.get("seq") for m in all_meta} if args.append else set()
    for seq in sequences:
        if args.append and seq.name in done_seqs:
            print(f"[skip] {seq.name} already in meta")
            continue
        remaining = None
        if global_cap is not None:
            remaining = global_cap - len(all_meta)
            if remaining <= 0:
                break
        cap = per_seq_cap
        if remaining is not None:
            cap = remaining if cap is None else min(cap, remaining)

        meta, sid = process_sequence(
            seq=seq,
            out_train=train_dir,
            out_val=val_dir,
            sample_id_start=sid,
            stride=args.stride,
            max_samples=cap,
            lidar_hz=args.lidar_hz,
            horizon_s=args.horizon_s,
            interval_s=args.interval_s,
            val_ratio=args.val_ratio,
        )
        all_meta.extend(meta)

    seq_names = sorted({m.get("seq", "") for m in all_meta} | {s.name for s in sequences})
    meta = {
        "source": "RELLIS-3D",
        "ego_frame": "navsim_ddv2",
        "ego_frame_note": "+x forward, +y left; RELLIS pose body (-x forward) remapped via [-x,-y,yaw]",
        "sequences": [s for s in seq_names if s],
        "schema": {
            "camera_feature": [3, 256, 1024],
            "lidar_feature": [1, 256, 256],
            "status_feature": [8],
            "trajectory": [8, 3],
        },
        "lidar_hz": args.lidar_hz,
        "horizon_s": args.horizon_s,
        "interval_s": args.interval_s,
        "stride": args.stride,
        "num_samples": len(all_meta),
        "num_train": len(list(train_dir.glob("*.npz"))),
        "num_val": len(list(val_dir.glob("*.npz"))),
        "samples": all_meta,
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"[done] {len(all_meta)} samples → {out}")
    print(f"  train={meta['num_train']} val={meta['num_val']}")


if __name__ == "__main__":
    main()
