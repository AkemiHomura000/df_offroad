#!/usr/bin/env python3
"""Build VertiMetricCache index for Verti DDV2 datasets.

For each sample_*.npz in the dataset, looks up expert episode pose/goal and writes:
  OUT/index.jsonl          — one VertiMetricCache record per sample
  OUT/meta.json            — summary
  OUT/worlds/wXXX.npz      — optional dumped WorldMaps (for offline inspection)

Example:
  python build_verti_metric_cache.py \\
    --data_dir datasets/verti_ddv2_density \\
    --expert_dir log/verti_expert \\
    --out datasets/verti_metric_cache_density
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np

from verti_metric_cache import (
    VertiMetricCache,
    load_world_maps,
    save_index,
)

ROOT = Path(__file__).resolve().parents[2]


def _obs_path(expert_dir: Path, seq: str, frame_idx: int) -> Path:
    return expert_dir / seq / "obs" / f"obs_{frame_idx:05d}.npz"


def build_record(
    sample_meta: dict,
    data_dir: Path,
    expert_dir: Path,
) -> VertiMetricCache:
    rel = sample_meta["file"]
    npz = np.load(data_dir / rel)
    seq = str(npz["seq"]) if "seq" in npz.files else sample_meta["seq"]
    frame_idx = int(npz["frame_idx"]) if "frame_idx" in npz.files else int(sample_meta["frame_idx"])
    world_id = int(npz["world"]) if "world" in npz.files else int(sample_meta["world"])
    gt_traj = np.asarray(npz["trajectory"], dtype=np.float32)

    ep_meta = json.loads((expert_dir / seq / "meta.json").read_text())
    obs_p = _obs_path(expert_dir, seq, frame_idx)
    if not obs_p.is_file():
        # some collectors use 0-padded differently; try glob
        cand = sorted((expert_dir / seq / "obs").glob("obs_*.npz"))
        if not cand:
            raise FileNotFoundError(obs_p)
        # match by index in sorted list if exact missing
        if frame_idx < len(cand):
            obs_p = cand[frame_idx]
        else:
            raise FileNotFoundError(obs_p)
    obs = np.load(obs_p)
    pose = np.asarray(obs["pose"], dtype=np.float64)
    goal = np.asarray(ep_meta["goal"][:2], dtype=np.float64)
    token = rel.replace("/", "__").replace(".npz", "")

    return VertiMetricCache(
        token=token,
        world_id=world_id,
        difficulty=str(ep_meta.get("difficulty", "low")),
        scale_factor=float(ep_meta.get("scale_factor", 1.0)),
        pose=pose,
        goal_xy=goal,
        gt_traj=gt_traj,
        seq=seq,
        frame_idx=frame_idx,
        ep_status=str(ep_meta.get("status", "")),
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data_dir", type=Path, default=ROOT / "datasets/verti_ddv2_density")
    ap.add_argument("--expert_dir", type=Path, default=ROOT / "log/verti_expert")
    ap.add_argument("--out", type=Path, default=ROOT / "datasets/verti_metric_cache_density")
    ap.add_argument("--dump_worlds", action="store_true", help="also write worlds/wXXX.npz")
    ap.add_argument("--max_samples", type=int, default=0)
    args = ap.parse_args()

    meta = json.loads((args.data_dir / "meta.json").read_text())
    samples = meta["samples"]
    if args.max_samples:
        samples = samples[: args.max_samples]

    args.out.mkdir(parents=True, exist_ok=True)
    records = []
    errors = 0
    worlds_seen = {}

    for i, s in enumerate(samples):
        try:
            rec = build_record(s, args.data_dir, args.expert_dir)
        except Exception as ex:
            errors += 1
            if errors <= 5:
                print(f"[skip] {s.get('file')}: {ex!r}", flush=True)
            continue
        records.append(rec.to_dict())
        worlds_seen[(rec.world_id, rec.difficulty, rec.scale_factor)] = True
        if (i + 1) % 2000 == 0:
            print(f"… {i+1}/{len(samples)} ok={len(records)} err={errors}", flush=True)

    save_index(args.out / "index.jsonl", records)

    if args.dump_worlds:
        wdir = args.out / "worlds"
        wdir.mkdir(exist_ok=True)
        for wid, diff, scale in sorted(worlds_seen.keys()):
            w = load_world_maps(wid, scale, diff)
            np.savez_compressed(
                wdir / f"w{wid:03d}_{diff}.npz",
                height=w.height,
                obstacle=w.obstacle,
                terrain_length=w.terrain_length,
                terrain_width=w.terrain_width,
                scale_factor=w.scale_factor,
                difficulty=diff,
                world_id=wid,
            )

    summary = {
        "data_dir": str(args.data_dir.resolve()),
        "expert_dir": str(args.expert_dir.resolve()),
        "num_samples": len(records),
        "num_errors": errors,
        "num_worlds": len(worlds_seen),
        "worlds": [
            {"world_id": w, "difficulty": d, "scale_factor": s}
            for (w, d, s) in sorted(worlds_seen.keys())
        ],
    }
    (args.out / "meta.json").write_text(json.dumps(summary, indent=2))
    print(f"done: {len(records)} caches → {args.out}  errors={errors}")


if __name__ == "__main__":
    main()
