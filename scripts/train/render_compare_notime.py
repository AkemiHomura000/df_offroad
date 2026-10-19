#!/usr/bin/env python3
"""Re-render panels without the top-5 x(t) plot for several models on the trio_100
frames, then stitch them side by side: frame_XXXXXX_compare.png."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path("/home/rm/Desktop/df_offroad")
DDV2 = ROOT / "DiffusionDriveV2"
sys.path[:0] = [str(ROOT / "scripts/bridge"), str(ROOT / "scripts/rellis"), str(ROOT / "scripts/train"), str(DDV2)]

from nuplan_min_stub import install as install_nuplan_stub  # noqa: E402

install_nuplan_stub()

import viz_infer_rellis as viz  # noqa: E402
from eval_rellis_il import load_model, modes_metrics, traj_metrics  # noqa: E402

DEFAULT_MODELS = [
    "baseline=DiffusionDriveV2/ckpts/diffusiondrive_navsim_88p1_PDMS",
    "v5=log/rellis_il_finetune_v5/ckpts/best.ckpt",
    "v7_ep10=log/rellis_il_finetune_v7/ckpts/last.ckpt:DiffusionDriveV2/kmeans_rellis_traj_20.npy",
]


def resolve(p: str) -> Path:
    return Path(p) if Path(p).is_absolute() else ROOT / p


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--trio_dir", type=Path, default=ROOT / "log/rellis_il_eval_trio_100")
    ap.add_argument("--models", nargs="+", default=DEFAULT_MODELS, help="tag=ckpt[:anchor_npy]")
    ap.add_argument("--out_name", default="compare_baseline_v5_v7ep10")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sample_meta = json.loads((args.trio_dir / "sample_frames.json").read_text())
    files = [Path(s["path"]) for s in sample_meta]
    out_root = args.trio_dir / args.out_name
    panel_root = out_root / "panels"

    tags = []
    for spec in args.models:
        tag, rest = spec.split("=", 1)
        ckpt_s, _, anchor_s = rest.partition(":")
        model = load_model(resolve(ckpt_s), device, plan_anchor_path=resolve(anchor_s) if anchor_s else None)
        torch.manual_seed(args.seed)
        for i, p in enumerate(files):
            sample = viz.load_sample(p)
            fi = int(sample["frame_idx"])
            gt = sample["trajectory"].astype(np.float32)
            best, modes, scores = viz.infer_all_modes(model, sample, device)
            ade, fde = traj_metrics(best, gt)
            min_ade, _, _ = modes_metrics(modes, gt)
            title = f"{tag} | seq={sample['seq']} frame={fi} | ADE={ade:.3f} FDE={fde:.3f} minADE={min_ade:.3f}"
            viz.make_panel(
                sample, modes, scores, title,
                panel_root / tag / f"frame_{fi:06d}_{tag}.png",
                show_time_plot=False,
            )
            if (i + 1) % 25 == 0:
                print(f"  [{tag}] {i + 1}/{len(files)}")
        tags.append(tag)
        del model
        torch.cuda.empty_cache()

    n = 0
    for s in sample_meta:
        fi = int(s["frame_idx"])
        imgs = [cv2.imread(str(panel_root / t / f"frame_{fi:06d}_{t}.png")) for t in tags]
        if any(im is None for im in imgs):
            continue
        h = min(im.shape[0] for im in imgs)
        imgs = [cv2.resize(im, (int(im.shape[1] * h / im.shape[0]), h)) for im in imgs]
        gap, top = 10, 40
        width = sum(im.shape[1] for im in imgs) + gap * (len(imgs) - 1)
        canvas = np.full((h + top, width, 3), 245, np.uint8)
        x = 0
        for im in imgs:
            canvas[top:, x : x + im.shape[1]] = im
            x += im.shape[1] + gap
        label = f"frame={fi:06d}  |  " + "   ".join(f"{k + 1}: {t}" for k, t in enumerate(tags))
        cv2.putText(canvas, label, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (20, 20, 20), 2, cv2.LINE_AA)
        cv2.imwrite(str(out_root / f"frame_{fi:06d}_compare.png"), canvas)
        n += 1
    print(f"[stitch] {n} → {out_root}")


if __name__ == "__main__":
    main()
