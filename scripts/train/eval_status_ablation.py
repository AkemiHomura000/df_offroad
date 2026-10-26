#!/usr/bin/env python3
"""Status-channel ablation for Verti DDV2 open-loop eval.

Fixes camera_feature + lidar_feature; only mutates status_feature:
  identity     — as stored in the dataset
  zero_att     — status[2:4] = 0  (roll/pitch slots)
  flip_att     — status[2:4] *= -1
  noise_att_Xd — add N(0, X°) noise in rad, then /π encode + clip (seeded)
  zero_goal    — status[0:2] = 0  (control: goal slots, not attitude)

Compares ADE/FDE (+ optional 20-mode metrics) and mode/top-1 agreement vs identity.

Example:
  python scripts/train/eval_status_ablation.py \\
    --ckpt log/verti_il_scratch_elevation_goal_attitude/ckpts/best.ckpt \\
    --data_dir datasets/verti_ddv2_elevation \\
    --out_dir log/status_ablation_attitude \\
    --plan_anchor_path DiffusionDriveV2/kmeans_verti_traj_20.npy \\
    --max_samples 200 --modes
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

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
from eval_rellis_il import (  # noqa: E402
    load_model,
    modes_metrics,
    traj_metrics,
)
from train_rellis_il import RellisDDV2Dataset, collate  # noqa: E402
from verti_bench.envs.utils.ddv2_status import ATTITUDE_NORM_RAD  # noqa: E402

AblateFn = Callable[[np.ndarray, np.random.Generator], np.ndarray]


def _clip_att(st: np.ndarray) -> np.ndarray:
    out = st.copy()
    out[2:4] = np.clip(out[2:4], -1.0, 1.0)
    return out.astype(np.float32)


def make_variants(noise_deg: List[float]) -> Dict[str, AblateFn]:
    variants: Dict[str, AblateFn] = {
        "identity": lambda st, _rng: st.astype(np.float32, copy=True),
        "zero_att": lambda st, _rng: _clip_att(np.concatenate([st[:2], [0.0, 0.0], st[4:]])),
        "flip_att": lambda st, _rng: _clip_att(
            np.concatenate([st[:2], -st[2:4], st[4:]])
        ),
        "zero_goal": lambda st, _rng: np.concatenate(
            [[0.0, 0.0], st[2:]]
        ).astype(np.float32),
    }

    def _noise_fn(deg: float) -> AblateFn:
        sigma = float(np.radians(deg)) / float(ATTITUDE_NORM_RAD)

        def fn(st: np.ndarray, rng: np.random.Generator) -> np.ndarray:
            out = st.astype(np.float32, copy=True)
            out[2:4] = out[2:4] + rng.normal(0.0, sigma, size=2).astype(np.float32)
            return _clip_att(out)

        return fn

    for d in noise_deg:
        variants[f"noise_att_{int(d)}deg"] = _noise_fn(d)
    return variants


def attitude_mag_deg(status: np.ndarray) -> float:
    """|roll| + |pitch| in degrees from normalized status[2:4]."""
    roll = float(status[2]) * ATTITUDE_NORM_RAD
    pitch = float(status[3]) * ATTITUDE_NORM_RAD
    return float(np.degrees(abs(roll) + abs(pitch)))


def summarize_rows(rows: List[dict]) -> dict:
    def mean(key: str) -> float:
        vals = [r[key] for r in rows if np.isfinite(r.get(key, np.nan))]
        return float(np.mean(vals)) if vals else float("nan")

    def pct(key: str, thr: float) -> float:
        vals = [r[key] for r in rows if np.isfinite(r.get(key, np.nan))]
        if not vals:
            return float("nan")
        return float(np.mean([v > thr for v in vals]))

    out = {
        "num_samples": len(rows),
        "ADE@best": mean("ade"),
        "FDE@best": mean("fde"),
        "minADE@20": mean("min_ade20"),
        "minFDE@20": mean("min_fde20"),
        "miss_rate@2m": pct("fde", 2.0),
        "miss_rate@4m": pct("fde", 4.0),
        "mean_entropy": mean("score_entropy"),
        "mean_delta_ade_vs_id": mean("delta_ade_vs_id"),
        "frac_mode_match_id": mean("mode_match_id"),
        "frac_top1_match_id": mean("top1_match_id"),
        "mean_att_mag_deg": mean("att_mag_deg"),
    }
    return out


def bucket_summaries(rows: List[dict], edges: List[float]) -> dict:
    """Bucket by identity attitude magnitude (deg)."""
    buckets = {}
    labels = []
    for i, lo in enumerate(edges[:-1]):
        hi = edges[i + 1]
        labels.append((lo, hi, f"[{lo:g},{hi:g})"))
    labels.append((edges[-1], float("inf"), f"[{edges[-1]:g},inf)"))

    for lo, hi, name in labels:
        sub = [r for r in rows if lo <= r["att_mag_deg"] < hi]
        buckets[name] = summarize_rows(sub) if sub else {"num_samples": 0}
    return buckets


@torch.no_grad()
def run_ablation(
    model,
    loader: DataLoader,
    device: torch.device,
    variants: Dict[str, AblateFn],
    seed: int,
    max_samples: int,
    compute_modes: bool,
) -> Dict[str, List[dict]]:
    rng = np.random.default_rng(seed)
    out_rows: Dict[str, List[dict]] = {k: [] for k in variants}
    n = 0
    t0 = time.time()

    for batch in loader:
        B = batch["trajectory"].shape[0]
        cam = batch["camera_feature"]
        lid = batch["lidar_feature"]
        st0 = batch["status_feature"].numpy()
        gt_b = batch["trajectory"].numpy()

        # identity first (shared reference for mode match)
        id_modes: List[Optional[np.ndarray]] = [None] * B
        id_top1: List[int] = [-1] * B
        id_best_ade_mode: List[int] = [-1] * B

        for vname, ablate in variants.items():
            st_np = np.stack([ablate(st0[i], rng) for i in range(B)], axis=0)
            feats = {
                "camera_feature": cam.to(device, non_blocking=True),
                "lidar_feature": lid.to(device, non_blocking=True),
                "status_feature": torch.from_numpy(st_np).to(device, non_blocking=True),
            }
            pred_b = model(feats)["trajectory"].detach().cpu().numpy()

            for i in range(B):
                gt = gt_b[i]
                pred = pred_b[i]
                ade, fde = traj_metrics(pred, gt)
                row = {
                    "path": batch["path"][i],
                    "seq": batch["seq"][i],
                    "frame_idx": int(batch["frame_idx"][i]),
                    "variant": vname,
                    "ade": ade,
                    "fde": fde,
                    "min_ade20": float("nan"),
                    "min_fde20": float("nan"),
                    "best_mode_by_ade": -1,
                    "top1_mode": -1,
                    "score_entropy": float("nan"),
                    "att_mag_deg": attitude_mag_deg(st0[i]),
                    "status_roll_n": float(st_np[i, 2]),
                    "status_pitch_n": float(st_np[i, 3]),
                    "delta_ade_vs_id": float("nan"),
                    "mode_match_id": float("nan"),
                    "top1_match_id": float("nan"),
                }

                if compute_modes:
                    sample = {
                        "camera_feature": cam[i].numpy(),
                        "lidar_feature": lid[i].numpy(),
                        "status_feature": st_np[i],
                        "trajectory": gt,
                        "seq": batch["seq"][i],
                        "frame_idx": int(batch["frame_idx"][i]),
                    }
                    _, modes, scores = viz.infer_all_modes(model, sample, device)
                    min_ade, min_fde, mi = modes_metrics(modes, gt)
                    top1 = int(np.argmax(scores))
                    row["min_ade20"] = min_ade
                    row["min_fde20"] = min_fde
                    row["best_mode_by_ade"] = mi
                    row["top1_mode"] = top1
                    row["score_entropy"] = float(-(scores * np.log(scores + 1e-12)).sum())
                    if vname == "identity":
                        id_modes[i] = modes
                        id_top1[i] = top1
                        id_best_ade_mode[i] = mi

                out_rows[vname].append(row)

            if vname != "identity" and compute_modes:
                # fill match flags for this variant's last B rows
                for j in range(B):
                    r = out_rows[vname][-(B - j)]
                    r["top1_match_id"] = float(r["top1_mode"] == id_top1[j])
                    r["mode_match_id"] = float(r["best_mode_by_ade"] == id_best_ade_mode[j])

        # delta ADE vs identity (always)
        for vname in variants:
            if vname == "identity":
                for r in out_rows[vname][-B:]:
                    r["delta_ade_vs_id"] = 0.0
                    if compute_modes:
                        r["top1_match_id"] = 1.0
                        r["mode_match_id"] = 1.0
                continue
            for j in range(B):
                r = out_rows[vname][-(B - j)]
                r_id = out_rows["identity"][-(B - j)]
                r["delta_ade_vs_id"] = float(r["ade"] - r_id["ade"])

        n += B
        if n % 50 == 0 or (max_samples and n >= max_samples):
            print(f"  evaluated {n} samples  ({time.time() - t0:.1f}s)", flush=True)
        if max_samples and n >= max_samples:
            # trim extras if batch overshot
            for vname in out_rows:
                out_rows[vname] = out_rows[vname][:max_samples]
            break

    return out_rows


def write_csv(rows: List[dict], path: Path) -> None:
    keys = [
        "path",
        "seq",
        "frame_idx",
        "variant",
        "ade",
        "fde",
        "min_ade20",
        "min_fde20",
        "best_mode_by_ade",
        "top1_mode",
        "score_entropy",
        "att_mag_deg",
        "status_roll_n",
        "status_pitch_n",
        "delta_ade_vs_id",
        "mode_match_id",
        "top1_match_id",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in keys})


def plot_delta_bars(summaries: Dict[str, dict], out_path: Path, title: str) -> None:
    names = [k for k in summaries if k != "identity"]
    if not names:
        return
    id_ade = summaries["identity"]["ADE@best"]
    deltas = [summaries[k]["ADE@best"] - id_ade for k in names]
    fig, ax = plt.subplots(figsize=(8, 3.5))
    colors = ["#c44e52" if d > 0 else "#4c72b0" for d in deltas]
    ax.bar(names, deltas, color=colors, edgecolor="white")
    ax.axhline(0, color="k", lw=0.8)
    ax.set_ylabel("Δ ADE@best vs identity (m)")
    ax.set_title(title)
    ax.tick_params(axis="x", rotation=25)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--data_dir", type=Path, default=ROOT / "datasets/verti_ddv2_elevation")
    ap.add_argument("--out_dir", type=Path, required=True)
    ap.add_argument("--split", default="val", choices=["train", "val"])
    ap.add_argument("--plan_anchor_path", type=Path, default=DDV2 / "kmeans_verti_traj_20.npy")
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--num_workers", type=int, default=2)
    ap.add_argument("--max_samples", type=int, default=0, help="0 = all")
    ap.add_argument("--modes", action="store_true", help="also compute 20-mode metrics (slower)")
    ap.add_argument("--noise_deg", default="5,15", help="comma list of attitude noise σ in degrees")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument(
        "--bucket_edges",
        default="0,5,15",
        help="attitude |roll|+|pitch| deg edges for bucketed ADE",
    )
    args = ap.parse_args()

    noise_deg = [float(x) for x in args.noise_deg.split(",") if x.strip()]
    variants = make_variants(noise_deg)
    edges = [float(x) for x in args.bucket_edges.split(",") if x.strip()]

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    ds = RellisDDV2Dataset(args.data_dir, args.split)
    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate,
        pin_memory=device.type == "cuda",
    )
    print(
        f"[data] {args.data_dir} split={args.split} n={len(ds)} "
        f"variants={list(variants)} modes={args.modes}",
        flush=True,
    )

    model = load_model(args.ckpt, device, plan_anchor_path=args.plan_anchor_path)
    rows_by_v = run_ablation(
        model,
        loader,
        device,
        variants,
        seed=args.seed,
        max_samples=args.max_samples,
        compute_modes=args.modes,
    )

    summaries = {v: summarize_rows(rows) for v, rows in rows_by_v.items()}
    # identity buckets + each variant's ADE in those buckets (by identity att mag)
    id_rows = rows_by_v["identity"]
    bucketed = {"identity": bucket_summaries(id_rows, edges)}
    for v, rows in rows_by_v.items():
        if v == "identity":
            continue
        # attach identity att_mag already on each row; bucket this variant
        bucketed[v] = bucket_summaries(rows, edges)

    # paired deltas table
    delta_table = {}
    id_ade = summaries["identity"]["ADE@best"]
    for v, s in summaries.items():
        delta_table[v] = {
            "ADE@best": s["ADE@best"],
            "delta_ADE": s["ADE@best"] - id_ade,
            "FDE@best": s["FDE@best"],
            "miss@2m": s["miss_rate@2m"],
            "frac_top1_match_id": s.get("frac_top1_match_id"),
            "mean_delta_ade_vs_id": s.get("mean_delta_ade_vs_id"),
        }

    meta = {
        "ckpt": str(args.ckpt.resolve()),
        "data_dir": str(args.data_dir.resolve()),
        "split": args.split,
        "max_samples": args.max_samples,
        "modes": args.modes,
        "noise_deg": noise_deg,
        "seed": args.seed,
        "plan_anchor_path": str(args.plan_anchor_path),
        "variants": list(variants),
        "note": "camera+lidar fixed; only status_feature ablated",
    }
    report = {
        "meta": meta,
        "summaries": summaries,
        "delta_vs_identity": delta_table,
        "buckets_att_mag_deg": bucketed,
    }
    out_json = args.out_dir / "ablation_summary.json"
    out_json.write_text(json.dumps(report, indent=2))
    print(f"[wrote] {out_json}")

    all_rows = []
    for v, rows in rows_by_v.items():
        all_rows.extend(rows)
        write_csv(rows, args.out_dir / f"per_sample_{v}.csv")
    write_csv(all_rows, args.out_dir / "per_sample_all.csv")

    plot_delta_bars(
        summaries,
        args.out_dir / "delta_ade_bars.png",
        title=f"Status ablation ΔADE  |  {args.ckpt.name}",
    )

    # markdown table
    lines = [
        "# Status ablation",
        "",
        f"- ckpt: `{args.ckpt}`",
        f"- data: `{args.data_dir}` ({args.split})",
        f"- n: {summaries['identity']['num_samples']}",
        f"- modes: {args.modes}",
        "",
        "| variant | ADE@best | ΔADE | FDE@best | miss@2m | top1≡id |",
        "|---------|----------|------|----------|---------|---------|",
    ]
    for v in variants:
        s = summaries[v]
        d = delta_table[v]
        top1 = s.get("frac_top1_match_id")
        top1_s = f"{top1*100:.1f}%" if top1 == top1 else "—"
        lines.append(
            f"| {v} | {s['ADE@best']:.4f} | {d['delta_ADE']:+.4f} | "
            f"{s['FDE@best']:.4f} | {s['miss_rate@2m']*100:.1f}% | {top1_s} |"
        )
    md = args.out_dir / "ablation.md"
    md.write_text("\n".join(lines) + "\n")
    print(f"[wrote] {md}")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
