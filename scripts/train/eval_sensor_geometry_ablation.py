#!/usr/bin/env python3
"""Open-loop eval on Chrono attitude re-renders (Scheme D).

Loads frames/<stem>/<variant>.npz from rerender_attitude_views.py.
Keeps status_feature fixed (geometry-only) unless --status_from_chassis.

  python scripts/train/eval_sensor_geometry_ablation.py \\
    --render_dir log/sensor_geom_ablation \\
    --ckpt log/verti_il_scratch_elevation_goal_attitude/ckpts/best.ckpt \\
    --out_dir log/sensor_geom_ablation/eval
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
DDV2 = ROOT / "DiffusionDriveV2"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "bridge"))
sys.path.insert(0, str(ROOT / "scripts" / "rellis"))
sys.path.insert(0, str(ROOT / "scripts" / "train"))
sys.path.insert(0, str(DDV2))

from nuplan_min_stub import install as install_nuplan_stub  # noqa: E402

install_nuplan_stub()

import viz_infer_rellis as viz  # noqa: E402
from eval_rellis_il import load_model, modes_metrics, traj_metrics  # noqa: E402


def summarize(rows: List[dict]) -> dict:
    def mean(k):
        vals = [r[k] for r in rows if np.isfinite(r.get(k, np.nan))]
        return float(np.mean(vals)) if vals else float("nan")

    def pct(k, thr):
        vals = [r[k] for r in rows if np.isfinite(r.get(k, np.nan))]
        return float(np.mean([v > thr for v in vals])) if vals else float("nan")

    return {
        "num_samples": len(rows),
        "ADE@best": mean("ade"),
        "FDE@best": mean("fde"),
        "minADE@20": mean("min_ade20"),
        "minFDE@20": mean("min_fde20"),
        "miss_rate@2m": pct("fde", 2.0),
        "mean_entropy": mean("score_entropy"),
        "mean_delta_ade_vs_logged": mean("delta_ade_vs_logged"),
        "frac_top1_match_logged": mean("top1_match_logged"),
    }


@torch.no_grad()
def eval_variant(
    model,
    device,
    paths: List[Path],
    status_from_chassis: bool,
    compute_modes: bool,
) -> List[dict]:
    rows = []
    for p in paths:
        d = np.load(p)
        cam = torch.from_numpy(np.asarray(d["camera_feature"], dtype=np.float32))[None]
        lid = torch.from_numpy(np.asarray(d["lidar_feature"], dtype=np.float32))[None]
        if status_from_chassis and "status_feature_chassis" in d.files:
            st = np.asarray(d["status_feature_chassis"], dtype=np.float32)
        else:
            st = np.asarray(d["status_feature"], dtype=np.float32)
        status = torch.from_numpy(st)[None]
        gt = np.asarray(d["trajectory"], dtype=np.float32)
        feats = {
            "camera_feature": cam.to(device),
            "lidar_feature": lid.to(device),
            "status_feature": status.to(device),
        }
        pred = model(feats)["trajectory"][0].detach().cpu().numpy()
        ade, fde = traj_metrics(pred, gt)
        row = {
            "path": str(p),
            "stem": p.parent.name,
            "variant": str(d["variant"]) if "variant" in d.files else p.stem,
            "ade": ade,
            "fde": fde,
            "min_ade20": float("nan"),
            "min_fde20": float("nan"),
            "best_mode_by_ade": -1,
            "top1_mode": -1,
            "score_entropy": float("nan"),
            "att_mag_deg_logged": float(d["att_mag_deg_logged"]) if "att_mag_deg_logged" in d.files else float("nan"),
            "roll_deg": float(np.degrees(float(d["roll"]))) if "roll" in d.files else float("nan"),
            "pitch_deg": float(np.degrees(float(d["pitch"]))) if "pitch" in d.files else float("nan"),
            "delta_ade_vs_logged": float("nan"),
            "top1_match_logged": float("nan"),
            "world": int(d["world"]) if "world" in d.files else -1,
        }
        if compute_modes:
            sample = {
                "camera_feature": cam[0].numpy(),
                "lidar_feature": lid[0].numpy(),
                "status_feature": st,
                "trajectory": gt,
                "seq": str(d["seq"]) if "seq" in d.files else "",
                "frame_idx": int(d["frame_idx"]) if "frame_idx" in d.files else 0,
            }
            _, modes, scores = viz.infer_all_modes(model, sample, device)
            min_ade, min_fde, mi = modes_metrics(modes, gt)
            row["min_ade20"] = min_ade
            row["min_fde20"] = min_fde
            row["best_mode_by_ade"] = mi
            row["top1_mode"] = int(np.argmax(scores))
            row["score_entropy"] = float(-(scores * np.log(scores + 1e-12)).sum())
        rows.append(row)
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--render_dir", type=Path, required=True)
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--out_dir", type=Path, default=None)
    ap.add_argument("--plan_anchor_path", type=Path, default=DDV2 / "kmeans_verti_traj_20.npy")
    ap.add_argument("--modes", action="store_true")
    ap.add_argument("--status_from_chassis", action="store_true")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    out_dir = args.out_dir or (args.render_dir / "eval")
    out_dir.mkdir(parents=True, exist_ok=True)

    index = []
    with open(args.render_dir / "index.jsonl") as f:
        for line in f:
            if line.strip():
                index.append(json.loads(line))
    if not index:
        raise SystemExit(f"empty index: {args.render_dir / 'index.jsonl'}")

    # collect variant → paths
    var_paths: Dict[str, List[Path]] = defaultdict(list)
    for rec in index:
        for v, rel in rec.get("variants", {}).items():
            var_paths[v].append(args.render_dir / rel)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model = load_model(args.ckpt, device, plan_anchor_path=args.plan_anchor_path)

    rows_by_v = {}
    for v, paths in sorted(var_paths.items()):
        print(f"[eval] {v}: n={len(paths)}", flush=True)
        rows_by_v[v] = eval_variant(
            model, device, paths, args.status_from_chassis, args.modes
        )

    # deltas vs as_logged
    if "as_logged" in rows_by_v:
        id_by_stem = {r["stem"]: r for r in rows_by_v["as_logged"]}
        for v, rows in rows_by_v.items():
            for r in rows:
                base = id_by_stem.get(r["stem"])
                if base is None:
                    continue
                r["delta_ade_vs_logged"] = float(r["ade"] - base["ade"])
                if args.modes and base.get("top1_mode", -1) >= 0:
                    r["top1_match_logged"] = float(r["top1_mode"] == base["top1_mode"])
            if v == "as_logged":
                for r in rows:
                    r["delta_ade_vs_logged"] = 0.0
                    if args.modes:
                        r["top1_match_logged"] = 1.0

    summaries = {v: summarize(rows) for v, rows in rows_by_v.items()}
    base_ade = summaries.get("as_logged", {}).get("ADE@best", float("nan"))
    delta_table = {}
    for v, s in summaries.items():
        delta_table[v] = {
            "ADE@best": s["ADE@best"],
            "delta_ADE": s["ADE@best"] - base_ade if base_ade == base_ade else float("nan"),
            "FDE@best": s["FDE@best"],
            "minADE@20": s["minADE@20"],
            "frac_top1_match_logged": s.get("frac_top1_match_logged"),
        }

    report = {
        "meta": {
            "render_dir": str(args.render_dir),
            "ckpt": str(args.ckpt),
            "modes": args.modes,
            "status_from_chassis": args.status_from_chassis,
            "plan_anchor_path": str(args.plan_anchor_path),
            "n_frames": len(index),
        },
        "summaries": summaries,
        "delta_vs_as_logged": delta_table,
    }
    (out_dir / "geometry_ablation_summary.json").write_text(json.dumps(report, indent=2))

    all_rows = []
    for v, rows in rows_by_v.items():
        all_rows.extend(rows)
        keys = list(rows[0].keys()) if rows else []
        with open(out_dir / f"per_sample_{v}.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(rows)
    if all_rows:
        with open(out_dir / "per_sample_all.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
            w.writeheader()
            w.writerows(all_rows)

    lines = [
        "# Sensor geometry ablation (Chrono re-render)",
        "",
        f"- ckpt: `{args.ckpt}`",
        f"- renders: `{args.render_dir}` (n={len(index)})",
        f"- status: {'chassis-matched' if args.status_from_chassis else 'fixed dataset (geometry-only)'}",
        f"- modes: {args.modes}",
        "",
        "| variant | ADE@best | ΔADE | FDE@best | minADE@20 | top1≡logged |",
        "|---------|----------|------|----------|-----------|-------------|",
    ]
    for v in sorted(summaries.keys(), key=lambda x: (x != "as_logged", x)):
        s = summaries[v]
        d = delta_table[v]
        top1 = s.get("frac_top1_match_logged")
        top1_s = f"{top1*100:.1f}%" if top1 == top1 else "—"
        mina = s["minADE@20"]
        mina_s = f"{mina:.4f}" if mina == mina else "—"
        lines.append(
            f"| {v} | {s['ADE@best']:.4f} | {d['delta_ADE']:+.4f} | "
            f"{s['FDE@best']:.4f} | {mina_s} | {top1_s} |"
        )
    md = out_dir / "geometry_ablation.md"
    md.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"[wrote] {md}")


if __name__ == "__main__":
    main()
