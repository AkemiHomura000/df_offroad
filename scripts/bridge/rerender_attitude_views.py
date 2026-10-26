#!/usr/bin/env python3
"""Scheme D: Chrono re-render at fixed (x,y,yaw) with forced chassis attitude.

For selected Verti DDV2 val frames, rebuild the world once per map and teleport a
sensor-bearing chassis to attitude variants, capturing RGB / elevation BEV.

Variants (roll,pitch relative to logged Cardan XYZ):
  as_logged   — traj.csv roll/pitch at obs time
  level       — roll=0, pitch=0
  pitch_p15   — pitch += 15°
  pitch_m15   — pitch -= 15°
  roll_p10    — roll  += 10°

Outputs under --out:
  frames/<stem>/<variant>.npz
  frames/<stem>/panel_rgb.png
  index.jsonl
  meta.json

Requires chrono9 (source verti_bench_env.sh).
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "bridge"))

from verti_bench.envs.utils.utils import SetChronoDataDirectories  # noqa: E402

VARIANT_DEFS = {
    "as_logged": (0.0, 0.0),
    "level": None,  # special
    "pitch_p15": (0.0, math.radians(15.0)),
    "pitch_m15": (0.0, -math.radians(15.0)),
    "roll_p10": (math.radians(10.0), 0.0),
}


def quat_cardan_xyz(roll: float, pitch: float, yaw: float):
    import pychrono as chrono

    return (
        chrono.QuatFromAngleZ(float(yaw))
        * chrono.QuatFromAngleY(float(pitch))
        * chrono.QuatFromAngleX(float(roll))
    )


def load_traj(path: Path):
    with open(path) as f:
        rows = list(csv.DictReader(f))
    t = np.array([float(r["t"]) for r in rows], dtype=np.float64)
    xy = np.array([[float(r["x"]), float(r["y"])] for r in rows], dtype=np.float64)
    z = np.array([float(r["z"]) for r in rows], dtype=np.float64)
    hd = np.unwrap(np.array([float(r["heading"]) for r in rows], dtype=np.float64))
    roll = np.array([float(r["roll"]) for r in rows], dtype=np.float64)
    pitch = np.array([float(r["pitch"]) for r in rows], dtype=np.float64)
    return t, xy, z, hd, roll, pitch


def interp_state(t, xy, z, hd, roll, pitch, t0):
    return (
        float(np.interp(t0, t, xy[:, 0])),
        float(np.interp(t0, t, xy[:, 1])),
        float(np.interp(t0, t, z)),
        float(np.interp(t0, t, hd)),
        float(np.interp(t0, t, roll)),
        float(np.interp(t0, t, pitch)),
    )


def select_frames(
    data_dir: Path,
    expert_dir: Path,
    split: str,
    n: int,
    seed: int,
    min_att_deg: float,
    balance: bool,
) -> List[dict]:
    meta = json.loads((data_dir / "meta.json").read_text())
    samples = [s for s in meta["samples"] if s["file"].startswith(f"{split}/")]
    scored = []
    for s in samples:
        ep = expert_dir / s["seq"]
        traj_p = ep / "traj.csv"
        obs_p = ep / "obs" / f"obs_{int(s['frame_idx']):05d}.npz"
        if not traj_p.is_file() or not obs_p.is_file():
            continue
        d = np.load(obs_p)
        t0 = float(d["t"])
        t, xy, z, hd, roll, pitch = load_traj(traj_p)
        x, y, zz, yaw, r, p = interp_state(t, xy, z, hd, roll, pitch, t0)
        mag = abs(r) + abs(p)
        scored.append(
            {
                **s,
                "t0": t0,
                "x": x,
                "y": y,
                "z": zz,
                "yaw": yaw,
                "roll": r,
                "pitch": p,
                "att_mag_deg": float(np.degrees(mag)),
                "obs_path": str(obs_p),
                "traj_path": str(traj_p),
                "ep_dir": str(ep),
            }
        )
    if not scored:
        raise FileNotFoundError("no linked val frames")
    rng = np.random.default_rng(seed)
    hi = [s for s in scored if s["att_mag_deg"] >= min_att_deg]
    lo = [s for s in scored if s["att_mag_deg"] < min_att_deg]
    pick: List[dict] = []
    if balance and hi and lo and n >= 2:
        n_hi = n // 2
        n_lo = n - n_hi
        rng.shuffle(hi)
        rng.shuffle(lo)
        pick = hi[:n_hi] + lo[:n_lo]
    else:
        scored.sort(key=lambda s: -s["att_mag_deg"])
        pick = scored[:n]
    seen = set()
    out = []
    for s in pick:
        if s["file"] in seen:
            continue
        seen.add(s["file"])
        out.append(s)
        if len(out) >= n:
            break
    return out


def make_sim(world_id: int, start, goal, scale: float = 1.0):
    """PIDSim with sensors; returns initialized sim."""
    from verti_bench.systems.PID.PID_sim import PIDSim

    config = dict(
        vehicle="hmmwv",
        speed=4.0,
        system="pid",
        world_id=int(world_id),
        max_time=30.0,
        scale_factor=float(scale),
        render=False,
        use_gui=False,
        ddv2_sensors=True,
    )
    sim = PIDSim(config)
    sim.initialize(start_pos=list(start), goal_pos=list(goal))
    if sim.ddv2_suite is None:
        raise RuntimeError("ddv2_suite not attached")
    return sim


def teleport(sim, x, y, z, roll, pitch, yaw):
    import pychrono as chrono

    body = sim.vehicle_manager.chassis_body
    body.SetFixed(True)
    body.SetPos(chrono.ChVector3d(float(x), float(y), float(z)))
    body.SetRot(quat_cardan_xyz(roll, pitch, yaw))
    body.SetPosDt(chrono.ChVector3d(0, 0, 0))
    body.SetAngVelLocal(chrono.ChVector3d(0, 0, 0))
    # settle sensors
    for _ in range(8):
        sim.system.DoStepDynamics(sim.step_size)
        sim.ddv2_suite.update()


def capture(sim) -> Optional[dict]:
    goal = sim.vehicle_manager.goal
    for _ in range(12):
        sim.ddv2_suite.update()
        obs = sim.ddv2_suite.get_observation(goal=goal)
        if obs is not None:
            return obs
        sim.system.DoStepDynamics(sim.step_size)
    return None


def variant_rpy(base_roll, base_pitch, name: str) -> Tuple[float, float]:
    if name == "level":
        return 0.0, 0.0
    if name == "as_logged":
        return float(base_roll), float(base_pitch)
    dr, dp = VARIANT_DEFS[name]
    return float(base_roll + dr), float(base_pitch + dp)


def rgb_panel(images: Dict[str, np.ndarray], title: str, out_path: Path):
    keys = [k for k in VARIANT_DEFS if k in images]
    if not keys:
        return
    hs = [images[k].shape[0] for k in keys]
    h = min(hs)
    parts = []
    for k in keys:
        im = images[k]
        if im.shape[0] != h:
            im = np.array(Image.fromarray(im).resize((im.shape[1], h), Image.BILINEAR))
        parts.append(im)
        parts.append(np.full((h, 4, 3), 30, dtype=np.uint8))
    collage = np.concatenate(parts[:-1], axis=1)
    img = Image.fromarray(collage)
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None
    x = 6
    for k in keys:
        draw.text((x, 4), k, fill=(255, 230, 40), font=font)
        x += images[k].shape[1] + 4
    draw.text((6, h - 14), title[:80], fill=(220, 220, 220), font=font)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data_dir", type=Path, default=ROOT / "datasets/verti_ddv2_elevation")
    ap.add_argument("--expert_dir", type=Path, default=ROOT / "log/verti_expert")
    ap.add_argument("--out", type=Path, default=ROOT / "log/sensor_geom_ablation")
    ap.add_argument("--split", default="val")
    ap.add_argument("--num_frames", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--min_att_deg", type=float, default=8.0)
    ap.add_argument("--balance", action="store_true", default=True)
    ap.add_argument("--no_balance", action="store_false", dest="balance")
    ap.add_argument("--warmup_frames", type=int, default=2, help="discard first captures after world init")
    ap.add_argument("--variants", default=",".join(VARIANT_DEFS.keys()))
    args = ap.parse_args()

    SetChronoDataDirectories()
    variants = [v.strip() for v in args.variants.split(",") if v.strip()]
    for v in variants:
        if v not in VARIANT_DEFS:
            raise SystemExit(f"unknown variant {v}")

    frames = select_frames(
        args.data_dir,
        args.expert_dir,
        args.split,
        args.num_frames,
        args.seed,
        args.min_att_deg,
        args.balance,
    )
    print(
        f"[select] n={len(frames)} att_deg "
        f"mean={np.mean([f['att_mag_deg'] for f in frames]):.1f} "
        f"max={np.max([f['att_mag_deg'] for f in frames]):.1f} "
        f"worlds={sorted({int(f['world']) for f in frames})}",
        flush=True,
    )

    by_world: Dict[int, List[dict]] = defaultdict(list)
    for f in frames:
        by_world[int(f["world"])].append(f)

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "frames").mkdir(exist_ok=True)
    index = []
    t_all = time.time()

    for wi, (world, flist) in enumerate(sorted(by_world.items())):
        # start/goal from first episode meta
        ep0 = Path(flist[0]["ep_dir"])
        em = json.loads((ep0 / "meta.json").read_text())
        start = em["start"]
        goal = em["goal"]
        scale = float(em.get("scale_factor", 1.0))
        print(f"\n[world {world}] frames={len(flist)} init Chrono…", flush=True)
        t0 = time.time()
        try:
            sim = make_sim(world, start, goal, scale=scale)
        except Exception as ex:
            print(f"[world {world}] INIT FAIL: {ex!r}", flush=True)
            continue
        print(f"[world {world}] ready in {time.time()-t0:.1f}s", flush=True)

        # warmup sensors at start pose
        for _ in range(max(args.warmup_frames, 1)):
            sim.ddv2_suite.update()
            sim.system.DoStepDynamics(sim.step_size)

        for fi, fr in enumerate(flist):
            stem = Path(fr["file"]).stem
            frame_dir = args.out / "frames" / stem
            frame_dir.mkdir(parents=True, exist_ok=True)
            # GT from dataset npz
            ds = np.load(args.data_dir / fr["file"])
            gt_traj = np.asarray(ds["trajectory"], dtype=np.float32)
            status_orig = np.asarray(ds["status_feature"], dtype=np.float32)
            cam_orig = None
            if "camera_feature" in ds.files:
                cam_o = np.asarray(ds["camera_feature"])
                if cam_o.ndim == 3 and cam_o.shape[0] == 3:
                    cam_orig = (np.transpose(cam_o, (1, 2, 0)) * 255.0).clip(0, 255).astype(np.uint8)

            rgb_for_panel = {}
            if cam_orig is not None:
                rgb_for_panel["dataset"] = cam_orig

            rec = {
                "file": fr["file"],
                "seq": fr["seq"],
                "frame_idx": int(fr["frame_idx"]),
                "world": int(fr["world"]),
                "att_mag_deg": fr["att_mag_deg"],
                "variants": {},
            }

            for vname in variants:
                roll, pitch = variant_rpy(fr["roll"], fr["pitch"], vname)
                teleport(sim, fr["x"], fr["y"], fr["z"], roll, pitch, fr["yaw"])
                obs = capture(sim)
                if obs is None:
                    print(f"  [{stem}] {vname}: capture failed", flush=True)
                    continue
                # status: keep dataset status (geometry-only ablation)
                # also store chassis attitude encoding for analysis
                from verti_bench.envs.utils.ddv2_status import encode_attitude

                att_n = encode_attitude(roll, pitch)
                status_chassis = status_orig.copy()
                status_chassis[2:4] = att_n

                out_npz = {
                    "camera_feature": np.asarray(obs["camera_feature"], dtype=np.float32),
                    "lidar_feature": np.asarray(obs["lidar_feature"], dtype=np.float32),
                    "status_feature": status_orig,  # FIXED original
                    "status_feature_chassis": status_chassis,
                    "trajectory": gt_traj,
                    "cam_front": np.asarray(obs["cam_front"], dtype=np.uint8),
                    "variant": vname,
                    "roll": np.float32(roll),
                    "pitch": np.float32(pitch),
                    "yaw": np.float32(fr["yaw"]),
                    "x": np.float32(fr["x"]),
                    "y": np.float32(fr["y"]),
                    "z": np.float32(fr["z"]),
                    "att_mag_deg_logged": np.float32(fr["att_mag_deg"]),
                    "seq": fr["seq"],
                    "frame_idx": np.int32(fr["frame_idx"]),
                    "world": np.int32(fr["world"]),
                    "source_file": fr["file"],
                }
                if "cam_depth" in obs:
                    out_npz["cam_depth"] = np.asarray(obs["cam_depth"], dtype=np.float32)
                path = frame_dir / f"{vname}.npz"
                np.savez_compressed(path, **out_npz)
                rgb_for_panel[vname] = out_npz["cam_front"]
                rec["variants"][vname] = str(path.relative_to(args.out))
                print(
                    f"  [{stem}] {vname}: roll={np.degrees(roll):+.1f}° pitch={np.degrees(pitch):+.1f}°",
                    flush=True,
                )

            rgb_panel(
                rgb_for_panel,
                f"{stem}  world={fr['world']}  logged |r|+|p|={fr['att_mag_deg']:.1f}°",
                frame_dir / "panel_rgb.png",
            )
            rec["panel"] = str((frame_dir / "panel_rgb.png").relative_to(args.out))
            index.append(rec)

        # cleanup
        try:
            del sim
        except Exception:
            pass

    index_path = args.out / "index.jsonl"
    with open(index_path, "w") as f:
        for r in index:
            f.write(json.dumps(r) + "\n")
    meta = {
        "data_dir": str(args.data_dir),
        "expert_dir": str(args.expert_dir),
        "split": args.split,
        "num_requested": args.num_frames,
        "num_rendered": len(index),
        "variants": variants,
        "status_policy": "fixed_to_dataset (geometry-only)",
        "seconds": time.time() - t_all,
        "min_att_deg": args.min_att_deg,
        "seed": args.seed,
    }
    (args.out / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"\n[done] rendered={len(index)} → {args.out}  ({meta['seconds']:.1f}s)", flush=True)


if __name__ == "__main__":
    main()
