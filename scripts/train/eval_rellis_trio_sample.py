#!/usr/bin/env python3
"""Compare baseline / v5 / v6 on val with uniform frame sampling.

- Metrics on the same N sampled frames (default 100, evenly spaced by frame_idx)
- Viz panels named by test-set frame index: frame_{frame_idx:06d}_*.png
- Triple stitch: baseline | v5 | v6 → frame_{frame_idx:06d}_compare.png
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path("/home/rm/Desktop/df_offroad")
DDV2 = ROOT / "DiffusionDriveV2"
sys.path.insert(0, str(ROOT / "scripts" / "bridge"))
sys.path.insert(0, str(ROOT / "scripts" / "rellis"))
sys.path.insert(0, str(ROOT / "scripts" / "train"))
sys.path.insert(0, str(DDV2))

from nuplan_min_stub import install as install_nuplan_stub  # noqa: E402

install_nuplan_stub()

import viz_infer_rellis as viz  # noqa: E402
from eval_rellis_il import load_model, modes_metrics, summarize, traj_metrics  # noqa: E402
from train_rellis_il import RellisDDV2Dataset  # noqa: E402


def uniform_indices(n_total: int, n_pick: int) -> np.ndarray:
    if n_pick >= n_total:
        return np.arange(n_total, dtype=int)
    return np.linspace(0, n_total - 1, num=n_pick, dtype=int)


def eval_on_files(
    model,
    files: list[Path],
    device: torch.device,
    tag: str,
    viz_dir: Path,
) -> list[dict]:
    viz_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    for i, path in enumerate(files):
        sample = viz.load_sample(path)
        frame_idx = int(sample["frame_idx"])
        seq = str(sample["seq"])
        gt = sample["trajectory"].astype(np.float32)

        feats = {
            "camera_feature": torch.from_numpy(sample["camera_feature"]).float().unsqueeze(0).to(device),
            "lidar_feature": torch.from_numpy(sample["lidar_feature"]).float().unsqueeze(0).to(device),
            "status_feature": torch.from_numpy(sample["status_feature"]).float().unsqueeze(0).to(device),
        }
        with torch.no_grad():
            pred = model(feats)["trajectory"][0].detach().cpu().numpy()
        ade, fde = traj_metrics(pred, gt)
        _, modes, scores = viz.infer_all_modes(model, sample, device)
        min_ade, min_fde, mi = modes_metrics(modes, gt)

        row = {
            "path": str(path),
            "seq": seq,
            "frame_idx": frame_idx,
            "ade": ade,
            "fde": fde,
            "min_ade20": min_ade,
            "min_fde20": min_fde,
            "best_mode_by_ade": mi,
            "score_entropy": float(-(scores * np.log(scores + 1e-12)).sum()),
        }
        rows.append(row)

        out_name = f"frame_{frame_idx:06d}_{tag}_modes.png"
        title = (
            f"{tag} | seq={seq} frame={frame_idx} | "
            f"ADE={ade:.3f} FDE={fde:.3f} minADE={min_ade:.3f}"
        )
        viz.make_panel(sample, modes, scores, title, viz_dir / out_name)

        if (i + 1) % 10 == 0 or (i + 1) == len(files):
            print(f"  [{tag}] {i+1}/{len(files)} frame={frame_idx} ADE={ade:.3f}")
    return rows


def stitch_triple(
    dirs: dict[str, Path],
    frame_idxs: list[int],
    out_dir: Path,
    order: tuple[str, ...] = ("baseline", "v5", "v6"),
) -> int:
    """dirs[tag] contains frame_{idx:06d}_{tag}_modes.png."""
    out_dir.mkdir(parents=True, exist_ok=True)
    n = 0
    for fi in frame_idxs:
        imgs = []
        ok = True
        for tag in order:
            p = dirs[tag] / f"frame_{fi:06d}_{tag}_modes.png"
            if not p.is_file():
                ok = False
                break
            im = cv2.imread(str(p))
            if im is None:
                ok = False
                break
            imgs.append(im)
        if not ok:
            continue
        h = min(im.shape[0] for im in imgs)
        w = min(im.shape[1] for im in imgs)
        imgs = [cv2.resize(im, (w, h)) for im in imgs]
        gap = 8
        canvas = np.full((h + 36, w * len(imgs) + gap * (len(imgs) - 1), 3), 245, dtype=np.uint8)
        x = 0
        for im in imgs:
            canvas[36 : 36 + h, x : x + w] = im
            x += w + gap
        label = f"frame={fi:06d}  |  LEFT={order[0]}  MID={order[1]}  RIGHT={order[2]}"
        cv2.putText(canvas, label, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (20, 20, 20), 2, cv2.LINE_AA)
        cv2.imwrite(str(out_dir / f"frame_{fi:06d}_compare.png"), canvas)
        n += 1
    print(f"[stitch] {n} triples → {out_dir}")
    return n


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data_dir", type=Path, default=ROOT / "datasets/ddv2_rellis_full")
    ap.add_argument("--split", type=str, default="val")
    ap.add_argument("--n_viz", type=int, default=100, help="uniform sample count")
    ap.add_argument("--out_dir", type=Path, default=ROOT / "log/rellis_il_eval_trio_100")
    ap.add_argument(
        "--baseline_ckpt",
        type=Path,
        default=DDV2 / "ckpts/diffusiondrive_navsim_88p1_PDMS",
    )
    ap.add_argument(
        "--v5_ckpt",
        type=Path,
        default=ROOT / "log/rellis_il_finetune_v5/ckpts/best.ckpt",
    )
    ap.add_argument(
        "--v6_ckpt",
        type=Path,
        default=ROOT / "log/rellis_il_finetune_v6/ckpts/best.ckpt",
    )
    ap.add_argument(
        "--v6_anchor",
        type=Path,
        default=DDV2 / "kmeans_rellis_traj_20.npy",
    )
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[device] {device}")

    ds = RellisDDV2Dataset(args.data_dir, args.split)
    # sort by frame_idx so linspace is temporal spacing on hold-out seq
    meta = []
    for p in ds.files:
        with np.load(p) as d:
            meta.append((int(d["frame_idx"]), str(d["seq"]), Path(p)))
    meta.sort(key=lambda x: (x[1], x[0]))
    idxs = uniform_indices(len(meta), args.n_viz)
    picked = [meta[i] for i in idxs]
    files = [p for _, _, p in picked]
    frame_idxs = [fi for fi, _, _ in picked]
    print(
        f"[sample] {args.split} total={len(meta)} pick={len(files)} "
        f"frame_idx [{frame_idxs[0]} .. {frame_idxs[-1]}] stride~{len(meta)/len(files):.1f}"
    )
    (args.out_dir / "sample_frames.json").write_text(
        json.dumps(
            [{"frame_idx": fi, "seq": seq, "path": str(p)} for fi, seq, p in picked],
            indent=2,
        )
    )

    models = [
        ("baseline", args.baseline_ckpt, None),
        ("v5", args.v5_ckpt, None),
        ("v6", args.v6_ckpt, args.v6_anchor),
    ]

    all_summaries: dict = {
        "split": args.split,
        "n_viz": len(files),
        "frame_idx_range": [frame_idxs[0], frame_idxs[-1]],
        "models": {},
    }
    viz_dirs: dict[str, Path] = {}

    for tag, ckpt, anchor in models:
        print(f"\n[eval] {tag} ckpt={ckpt}")
        t0 = time.time()
        model = load_model(ckpt, device, plan_anchor_path=anchor)
        viz_dir = args.out_dir / f"{tag}_viz"
        viz_dirs[tag] = viz_dir
        rows = eval_on_files(model, files, device, tag, viz_dir)
        summary = summarize(rows)
        summary["seconds"] = time.time() - t0
        summary["ckpt"] = str(ckpt)
        summary["plan_anchor"] = str(anchor) if anchor else "from_ckpt"
        all_summaries["models"][tag] = summary
        print(
            f"  {tag}: ADE={summary['ADE@best']:.4f} FDE={summary['FDE@best']:.4f} "
            f"minADE={summary['minADE@20']:.4f} ({summary['seconds']:.0f}s)"
        )
        del model
        torch.cuda.empty_cache()

    stitch_triple(viz_dirs, frame_idxs, args.out_dir / "compare_baseline_v5_v6")

    # compact delta table vs baseline
    base = all_summaries["models"]["baseline"]
    deltas = {}
    for tag in ("v5", "v6"):
        m = all_summaries["models"][tag]
        deltas[tag] = {
            k: float(m[k] - base[k])
            for k in ("ADE@best", "FDE@best", "minADE@20", "minFDE@20", "miss_rate@2m", "miss_rate@4m")
        }
    all_summaries["delta_vs_baseline"] = deltas

    out_json = args.out_dir / "summary.json"
    out_json.write_text(json.dumps(all_summaries, indent=2))
    print(f"\n[done] {out_json}")
    print(f"  per-model viz: {args.out_dir}/{{baseline,v5,v6}}_viz/frame_XXXXXX_*.png")
    print(f"  stitch: {args.out_dir}/compare_baseline_v5_v6/frame_XXXXXX_compare.png")


if __name__ == "__main__":
    main()
