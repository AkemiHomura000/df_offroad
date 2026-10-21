"""Rewrite status_feature[0:4] in an existing Verti DDV2 dataset to ego-goal XY (scheme B).

Keeps camera/lidar/trajectory; only patches status nav slots using episode goal from
the source expert log referenced in meta.json (or --src).

Example:
  python scripts/bridge/rewrite_verti_status_goal.py \\
    --data_dir datasets/verti_ddv2_elevation \\
    --src log/verti_expert
"""
from __future__ import annotations

import argparse
import csv
import glob
import io
import json
import os
import sys
import zipfile
from collections import defaultdict

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from verti_bench.envs.utils.ddv2_status import GOAL_NORM_M, rewrite_status_nav_goal


def episode_goal_xy(meta, traj_path):
    g = meta.get("goal")
    if isinstance(g, (list, tuple)) and len(g) >= 2:
        return float(g[0]), float(g[1])
    with open(traj_path) as f:
        rows = list(csv.DictReader(f))
    return float(rows[0]["goal_x"]), float(rows[0]["goal_y"])


def load_episode_poses(ed: str) -> dict:
    """frame_idx → pose (4,) from expert obs npz."""
    poses = {}
    for op in glob.glob(os.path.join(ed, "obs", "obs_*.npz")):
        fidx = int(os.path.basename(op)[4:9])
        poses[fidx] = np.load(op)["pose"]
    return poses


def patch_status_in_npz(path: str, new_status: np.ndarray) -> None:
    """Replace only status_feature.npy inside an npz (zip) — avoid recompressing camera."""
    buf = io.BytesIO()
    np.save(buf, np.asarray(new_status, dtype=np.float32))
    status_bytes = buf.getvalue()
    tmp = path + ".tmp.npz"
    with zipfile.ZipFile(path, "r") as zin, zipfile.ZipFile(
        tmp, "w", compression=zipfile.ZIP_DEFLATED
    ) as zout:
        for info in zin.infolist():
            if info.filename == "status_feature.npy":
                zout.writestr(info, status_bytes)
            else:
                zout.writestr(info, zin.read(info.filename))
    os.replace(tmp, path)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data_dir", required=True, help="dataset root with train/val/*.npz + meta.json")
    ap.add_argument("--src", default="", help="expert log root (default: meta.source)")
    ap.add_argument("--goal_norm_m", type=float, default=GOAL_NORM_M)
    ap.add_argument("--dry_run", action="store_true")
    args = ap.parse_args()

    meta_path = os.path.join(args.data_dir, "meta.json")
    meta = json.load(open(meta_path))
    src = args.src or meta.get("source") or ""
    if not src or not os.path.isdir(src):
        raise SystemExit(f"expert src not found: {src!r} (pass --src)")

    # Group samples by episode so each ep's obs are loaded once.
    by_seq = defaultdict(list)
    for split in ("train", "val"):
        for path in sorted(glob.glob(os.path.join(args.data_dir, split, "sample_*.npz"))):
            # lightweight: only read seq/frame_idx arrays
            with np.load(path, allow_pickle=True) as d:
                seq = str(d["seq"])
                fidx = int(np.asarray(d["frame_idx"]).item())
                status = np.asarray(d["status_feature"], dtype=np.float32)
            by_seq[seq].append((path, fidx, status))

    n_ok = n_miss = 0
    n_seq = len(by_seq)
    for si, (seq, items) in enumerate(sorted(by_seq.items())):
        ed = os.path.join(src, seq)
        try:
            em = json.load(open(os.path.join(ed, "meta.json")))
            goal = episode_goal_xy(em, os.path.join(ed, "traj.csv"))
            poses = load_episode_poses(ed)
        except (OSError, ValueError, KeyError) as e:
            print(f"[skip] {seq}: {e}", flush=True)
            n_miss += len(items)
            continue
        for path, fidx, status in items:
            pose = poses.get(fidx)
            if pose is None:
                n_miss += 1
                continue
            new_status = rewrite_status_nav_goal(
                status, pose, goal, goal_norm_m=args.goal_norm_m
            )
            if args.dry_run:
                n_ok += 1
                continue
            patch_status_in_npz(path, new_status)
            n_ok += 1
        if (si + 1) % 50 == 0 or si + 1 == n_seq:
            print(f"[{si+1}/{n_seq}] ok={n_ok} miss={n_miss}", flush=True)

    meta["status_nav"] = "goal"
    meta["goal_norm_m"] = args.goal_norm_m
    if not args.dry_run:
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=1)
    print(f"done ok={n_ok} miss={n_miss} dry_run={args.dry_run} → {args.data_dir}")


if __name__ == "__main__":
    main()
