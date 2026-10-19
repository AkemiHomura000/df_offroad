#!/usr/bin/env python3
"""Batch closed-loop Verti-Bench eval with DDV2 plan server (headless).

For each trial, writes a reproducible scene.json (world / start / goal / hyperparams)
plus result.json. Aggregate metrics go to results.csv and summary.json.

Requires a running plan server (see run_eval_verti_ddv2_batch.sh), e.g.:
  source ddv2_env.sh
  python scripts/bridge/ddv2_plan_server.py \\
    --ckpt log/verti_il_finetune_v3_density/ckpts/best.ckpt \\
    --plan_anchor_path DiffusionDriveV2/kmeans_verti_traj_20.npy

Then (chrono9 / verti_bench_env):
  python scripts/bridge/eval_verti_ddv2_batch.py \\
    --worlds low --tasks_per_world 3 --out log/verti_ddv2_eval_v3

Replay saved scenes:
  python scripts/bridge/eval_verti_ddv2_batch.py \\
    --scenes_file log/verti_ddv2_eval_v3/scenes_manifest.json --out log/verti_ddv2_eval_replay
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import random
import re
import socket
import sys
import time as wall
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import yaml

ROOT = Path("/home/rm/Desktop/df_offroad")
CFG_DIR = ROOT / "verti_bench/envs/data/BenchMaps/sampled_maps/Configs/Final"
sys.path.insert(0, str(ROOT))

from verti_bench.envs.utils.utils import SetChronoDataDirectories  # noqa: E402
from verti_bench.systems.DDV2.DDV2_sim import DDV2Sim  # noqa: E402


def parse_worlds(spec: str) -> List[int]:
    """'1,5,9' | '3-12' | 'low' | 'mid' | 'high' | 'all' (comma-combinable)."""
    worlds: List[int] = []
    for tok in spec.split(","):
        tok = tok.strip()
        if not tok:
            continue
        if tok in ("low", "mid", "high", "all"):
            pat = "*" if tok == "all" else tok
            for f in glob.glob(str(CFG_DIR / f"config*_{pat}.yaml")):
                m = re.search(r"config(\d+)_", os.path.basename(f))
                if m:
                    worlds.append(int(m.group(1)))
        elif "-" in tok:
            a, b = map(int, tok.split("-", 1))
            worlds.extend(range(a, b + 1))
        else:
            worlds.append(int(tok))
    return sorted(set(worlds))


def world_meta(world_id: int) -> Dict[str, Any]:
    matches = sorted(CFG_DIR.glob(f"config{world_id}_*.yaml"))
    if not matches:
        raise FileNotFoundError(f"no config for world {world_id}")
    with open(matches[0]) as f:
        cfg = yaml.safe_load(f)
    terr = cfg["terrain"]
    positions = cfg.get("positions") or []
    return {
        "world_id": int(world_id),
        "config_yaml": str(matches[0]),
        "difficulty": terr.get("difficulty"),
        "min_height": terr.get("min_height"),
        "max_height": terr.get("max_height"),
        "length": terr.get("length"),
        "width": terr.get("width"),
        "n_yaml_tasks": len(positions),
        "obstacle_density": cfg.get("obstacle_density"),
        "terrain_type": cfg.get("terrain_type"),
    }


def wait_server(host: str, port: int, timeout_s: float = 120.0) -> None:
    t0 = wall.time()
    while wall.time() - t0 < timeout_s:
        try:
            s = socket.create_connection((host, port), 1.0)
            s.close()
            return
        except OSError:
            wall.sleep(0.5)
    raise TimeoutError(f"plan server {host}:{port} not reachable within {timeout_s}s")


def pick_task(
    wm: Dict[str, Any],
    scale: float,
    goal_mode: str,
    task_idx: int,
    rng: np.random.Generator,
) -> Tuple[List[float], List[float], str, int]:
    """Return start, goal (xyz), source label, yaml index (-1 if N/A)."""
    z = 1.0 * scale
    with open(wm["config_yaml"]) as f:
        positions = yaml.safe_load(f).get("positions") or []
    if goal_mode == "yaml":
        if not positions:
            raise RuntimeError(f"world {wm['world_id']} has no yaml positions")
        i = int(task_idx) % len(positions)
        st = [float(v) * scale for v in positions[i]["start"]]
        gl = [float(v) * scale for v in positions[i]["goal"]]
        if len(st) < 3:
            st = st[:2] + [z]
        if len(gl) < 3:
            gl = gl[:2] + [z]
        return st, gl, f"yaml#{i}", i
    # random among yaml pairs (still reproducible via seed+task_idx)
    if not positions:
        raise RuntimeError(f"world {wm['world_id']} has no yaml positions for random mode")
    i = int(rng.integers(len(positions)))
    st = [float(v) * scale for v in positions[i]["start"]]
    gl = [float(v) * scale for v in positions[i]["goal"]]
    if len(st) < 3:
        st = st[:2] + [z]
    if len(gl) < 3:
        gl = gl[:2] + [z]
    return st, gl, f"random_yaml#{i}", i


def scene_from_args(
    world_id: int,
    task_idx: int,
    args: argparse.Namespace,
    rng: np.random.Generator,
) -> Dict[str, Any]:
    wm = world_meta(world_id)
    start, goal, source, yaml_i = pick_task(
        wm, args.scale_factor, args.goal_mode, task_idx, rng
    )
    return {
        "scene_id": f"w{world_id:03d}_task{task_idx:02d}",
        "world_id": world_id,
        "task_idx": task_idx,
        "yaml_pair_idx": yaml_i,
        "task_source": source,
        "difficulty": wm["difficulty"],
        "config_yaml": wm["config_yaml"],
        "min_height": wm["min_height"],
        "max_height": wm["max_height"],
        "obstacle_density": wm["obstacle_density"],
        "terrain_type": wm["terrain_type"],
        "vehicle": args.vehicle,
        "system": "ddv2",
        "speed": args.speed,
        "max_time": args.max_time,
        "scale_factor": args.scale_factor,
        "start": start,
        "goal": goal,
        "seed": args.seed,
        "ddv2_host": args.host,
        "ddv2_port": args.port,
        "ddv2_replan_interval": args.replan_interval,
        "ckpt": str(args.ckpt) if args.ckpt else None,
        "plan_anchor_path": str(args.plan_anchor_path) if args.plan_anchor_path else None,
        "render": False,
        "ddv2_live_viz": False,
        "ddv2_draw_world": False,
        "ddv2_verbose": False,
    }


def load_scenes_manifest(path: Path) -> List[Dict[str, Any]]:
    data = json.loads(path.read_text())
    if isinstance(data, dict) and "scenes" in data:
        scenes = data["scenes"]
    elif isinstance(data, list):
        scenes = data
    else:
        raise ValueError(f"unrecognized scenes file: {path}")
    out = []
    for s in scenes:
        if "scene_json" in s and Path(s["scene_json"]).is_file():
            out.append(json.loads(Path(s["scene_json"]).read_text()))
        else:
            out.append(s)
    return out


def run_one(scene: Dict[str, Any]) -> Dict[str, Any]:
    config = {
        "vehicle": scene["vehicle"],
        "speed": float(scene["speed"]),
        "system": "ddv2",
        "world_id": int(scene["world_id"]),
        "max_time": float(scene["max_time"]),
        "scale_factor": float(scene["scale_factor"]),
        "render": False,
        "use_gui": False,
        "ddv2_sensors": True,
        "ddv2_dump_dir": None,
        "ddv2_host": scene.get("ddv2_host", "127.0.0.1"),
        "ddv2_port": int(scene.get("ddv2_port", 8765)),
        "ddv2_replan_interval": float(scene.get("ddv2_replan_interval", 0.5)),
        "ddv2_live_viz": False,
        "ddv2_draw_world": False,
        "ddv2_verbose": bool(scene.get("ddv2_verbose", False)),
    }
    t0 = wall.time()
    sim = DDV2Sim(config)
    sim.initialize(start_pos=scene["start"], goal_pos=scene["goal"])
    ttg, success, avg_roll, avg_pitch = sim.run()
    wall_s = wall.time() - t0
    reason = sim.termination_reason or ("success" if success else "fail")
    # final distance
    try:
        pos = sim.vehicle_manager.get_position()
        goal = sim.vehicle_manager.goal
        dist = float(np.hypot(goal.x - pos.x, goal.y - pos.y))
        final_xy = [float(pos.x), float(pos.y), float(pos.z)]
    except Exception:
        dist = None
        final_xy = None
    infer = list(sim._infer_ms_hist) if sim._infer_ms_hist else []
    return {
        "success": bool(success),
        "termination_reason": reason,
        "time_to_goal": float(ttg) if success else None,
        "sim_time_s": float(ttg) if ttg is not None else None,
        "wall_time_s": wall_s,
        "avg_roll_deg": float(avg_roll),
        "avg_pitch_deg": float(avg_pitch),
        "final_dist_to_goal_m": dist,
        "final_xy": final_xy,
        "n_replans": len(infer),
        "mean_infer_ms": float(np.mean(infer)) if infer else None,
        "p95_infer_ms": float(np.percentile(infer, 95)) if infer else None,
    }


def summarize(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    n = len(rows)
    succ = [r for r in rows if r.get("success")]
    fail = [r for r in rows if not r.get("success")]
    reasons: Dict[str, int] = {}
    for r in rows:
        k = r.get("termination_reason") or "unknown"
        reasons[k] = reasons.get(k, 0) + 1

    def _mean(xs):
        xs = [x for x in xs if x is not None and np.isfinite(x)]
        return float(np.mean(xs)) if xs else None

    def _var(xs):
        xs = [x for x in xs if x is not None and np.isfinite(x)]
        return float(np.var(xs)) if xs else None

    by_diff: Dict[str, Any] = {}
    for r in rows:
        d = str(r.get("difficulty") or "unknown")
        by_diff.setdefault(d, []).append(r)
    per_diff = {}
    for d, rs in sorted(by_diff.items()):
        s = [x for x in rs if x.get("success")]
        per_diff[d] = {
            "n": len(rs),
            "success_rate": len(s) / max(len(rs), 1),
            "mean_ttg_s": _mean([x.get("time_to_goal") for x in s]),
            "mean_roll_deg": _mean([x.get("avg_roll_deg") for x in s]),
            "mean_pitch_deg": _mean([x.get("avg_pitch_deg") for x in s]),
        }

    return {
        "n_trials": n,
        "n_success": len(succ),
        "n_fail": len(fail),
        "success_rate": len(succ) / max(n, 1),
        "termination_counts": reasons,
        "mean_ttg_s": _mean([r.get("time_to_goal") for r in succ]),
        "mean_roll_deg": _mean([r.get("avg_roll_deg") for r in succ]),
        "var_roll_deg": _var([r.get("avg_roll_deg") for r in succ]),
        "mean_pitch_deg": _mean([r.get("avg_pitch_deg") for r in succ]),
        "var_pitch_deg": _var([r.get("avg_pitch_deg") for r in succ]),
        "mean_final_dist_fail_m": _mean([r.get("final_dist_to_goal_m") for r in fail]),
        "mean_infer_ms": _mean([r.get("mean_infer_ms") for r in rows]),
        "mean_wall_time_s": _mean([r.get("wall_time_s") for r in rows]),
        "per_difficulty": per_diff,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=ROOT / "log/verti_ddv2_eval_v3_density")
    ap.add_argument("--worlds", type=str, default="low", help="low|mid|high|all|1,5|3-10")
    ap.add_argument("--tasks_per_world", type=int, default=3)
    ap.add_argument("--goal_mode", choices=["yaml", "random_yaml"], default="yaml")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--vehicle", default="hmmwv")
    ap.add_argument("--speed", type=float, default=4.0)
    ap.add_argument("--max_time", type=float, default=60.0)
    ap.add_argument("--scale_factor", type=float, default=1.0)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--replan_interval", type=float, default=0.5)
    ap.add_argument(
        "--ckpt",
        type=Path,
        default=ROOT / "log/verti_il_finetune_v3_density/ckpts/best.ckpt",
        help="recorded in scene.json for reproducibility (server must load same ckpt)",
    )
    ap.add_argument(
        "--plan_anchor_path",
        type=Path,
        default=ROOT / "DiffusionDriveV2/kmeans_verti_traj_20.npy",
    )
    ap.add_argument(
        "--scenes_file",
        type=Path,
        default=None,
        help="replay scenes from a prior scenes_manifest.json / list of scene dicts",
    )
    ap.add_argument("--max_scenes", type=int, default=0, help="0 = all")
    ap.add_argument("--skip_existing", action="store_true", help="skip scene dirs that already have result.json")
    args = ap.parse_args()

    SetChronoDataDirectories()
    wait_server(args.host, args.port)

    args.out.mkdir(parents=True, exist_ok=True)
    scenes_dir = args.out / "scenes"
    scenes_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)
    if args.scenes_file:
        scenes = load_scenes_manifest(args.scenes_file)
        # force current server endpoint
        for s in scenes:
            s["ddv2_host"] = args.host
            s["ddv2_port"] = args.port
            s["ddv2_live_viz"] = False
            s["ddv2_draw_world"] = False
            s["render"] = False
            s["ddv2_verbose"] = False
    else:
        worlds = parse_worlds(args.worlds)
        if not worlds:
            raise SystemExit(f"no worlds matched: {args.worlds}")
        scenes = []
        for wid in worlds:
            for ti in range(args.tasks_per_world):
                scenes.append(scene_from_args(wid, ti, args, rng))

    if args.max_scenes > 0:
        scenes = scenes[: args.max_scenes]

    run_meta = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "out": str(args.out),
        "ckpt": str(args.ckpt),
        "plan_anchor_path": str(args.plan_anchor_path) if args.plan_anchor_path else None,
        "worlds_arg": args.worlds if not args.scenes_file else None,
        "scenes_file": str(args.scenes_file) if args.scenes_file else None,
        "n_scenes": len(scenes),
        "seed": args.seed,
        "vehicle": args.vehicle,
        "speed": args.speed,
        "max_time": args.max_time,
        "goal_mode": args.goal_mode if not args.scenes_file else "replay",
        "host": args.host,
        "port": args.port,
    }
    (args.out / "run_meta.json").write_text(json.dumps(run_meta, indent=2))

    rows: List[Dict[str, Any]] = []
    manifest: List[Dict[str, Any]] = []

    for i, scene in enumerate(scenes):
        sid = scene.get("scene_id") or f"scene_{i:04d}"
        scene["scene_id"] = sid
        sdir = scenes_dir / sid
        sdir.mkdir(parents=True, exist_ok=True)
        scene_path = sdir / "scene.json"
        result_path = sdir / "result.json"

        if args.skip_existing and result_path.is_file():
            result = json.loads(result_path.read_text())
            print(f"[{i+1}/{len(scenes)}] SKIP {sid}  success={result.get('success')}")
        else:
            scene_path.write_text(json.dumps(scene, indent=2))
            print(
                f"[{i+1}/{len(scenes)}] RUN {sid}  world={scene['world_id']} "
                f"diff={scene.get('difficulty')} src={scene.get('task_source')}",
                flush=True,
            )
            try:
                result = run_one(scene)
            except Exception as e:
                result = {
                    "success": False,
                    "termination_reason": "error",
                    "error": repr(e),
                    "time_to_goal": None,
                    "sim_time_s": None,
                    "wall_time_s": None,
                    "avg_roll_deg": None,
                    "avg_pitch_deg": None,
                    "final_dist_to_goal_m": None,
                    "final_xy": None,
                    "n_replans": 0,
                    "mean_infer_ms": None,
                    "p95_infer_ms": None,
                }
                print(f"  ERROR {e}", flush=True)
            result_path.write_text(json.dumps(result, indent=2))
            wt = result.get("wall_time_s")
            wt_s = f"{wt:.1f}s" if isinstance(wt, (int, float)) else "n/a"
            print(
                f"  -> {result.get('termination_reason')}  "
                f"ttg={result.get('time_to_goal')}  "
                f"dist={result.get('final_dist_to_goal_m')}  "
                f"wall={wt_s}",
                flush=True,
            )

        row = {
            "scene_id": sid,
            "world_id": scene.get("world_id"),
            "difficulty": scene.get("difficulty"),
            "task_idx": scene.get("task_idx"),
            "task_source": scene.get("task_source"),
            "yaml_pair_idx": scene.get("yaml_pair_idx"),
            **result,
        }
        rows.append(row)
        manifest.append(
            {
                "scene_id": sid,
                "scene_json": str(scene_path),
                "result_json": str(result_path),
                "success": row.get("success"),
                "termination_reason": row.get("termination_reason"),
            }
        )

    # CSV
    csv_path = args.out / "results.csv"
    fieldnames = [
        "scene_id",
        "world_id",
        "difficulty",
        "task_idx",
        "task_source",
        "yaml_pair_idx",
        "success",
        "termination_reason",
        "time_to_goal",
        "sim_time_s",
        "wall_time_s",
        "avg_roll_deg",
        "avg_pitch_deg",
        "final_dist_to_goal_m",
        "n_replans",
        "mean_infer_ms",
        "p95_infer_ms",
    ]
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)

    summary = summarize(rows)
    summary_path = args.out / "summary.json"
    summary_path.write_text(json.dumps({"run_meta": run_meta, "metrics": summary}, indent=2))
    (args.out / "scenes_manifest.json").write_text(json.dumps({"scenes": manifest}, indent=2))

    print("\n========== SUMMARY ==========")
    print(f"trials: {summary['n_trials']}  success: {summary['n_success']}  "
          f"rate: {summary['success_rate']*100:.1f}%")
    print(f"termination: {summary['termination_counts']}")
    if summary["mean_ttg_s"] is not None:
        print(f"mean TTG (success): {summary['mean_ttg_s']:.2f}s")
        print(f"mean roll/pitch (success): {summary['mean_roll_deg']:.2f} / "
              f"{summary['mean_pitch_deg']:.2f} deg")
    for d, m in summary["per_difficulty"].items():
        print(f"  [{d}] n={m['n']}  SR={m['success_rate']*100:.1f}%  "
              f"TTG={m['mean_ttg_s']}")
    print(f"wrote {csv_path}")
    print(f"wrote {summary_path}")
    print(f"scenes: {scenes_dir}")
    print("replay: python scripts/bridge/eval_verti_ddv2_batch.py "
          f"--scenes_file {args.out / 'scenes_manifest.json'} --out <new_out>")


if __name__ == "__main__":
    main()
