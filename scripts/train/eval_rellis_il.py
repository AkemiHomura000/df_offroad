#!/usr/bin/env python3
"""
Evaluate DiffusionDrive IL checkpoint on RELLIS→DDV2 npz.

Metrics (ego-frame XY, meters):
  ADE@best, FDE@best, minADE@20, minFDE@20
  miss_rate@2m / @4m (FDE of best mode)
  per-sequence breakdown

Also writes histograms, summary JSON, and optional multimodal viz for
best/worst samples.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path("/home/rm/Desktop/df_offroad")
DDV2 = ROOT / "DiffusionDriveV2"
sys.path.insert(0, str(ROOT / "scripts" / "bridge"))
sys.path.insert(0, str(ROOT / "scripts" / "rellis"))
sys.path.insert(0, str(ROOT / "scripts" / "train"))
sys.path.insert(0, str(DDV2))

from nuplan_min_stub import install as install_nuplan_stub  # noqa: E402

install_nuplan_stub()

from navsim.agents.diffusiondrive.transfuser_config import TransfuserConfig  # noqa: E402
from navsim.agents.diffusiondrive.transfuser_model_v2 import V2TransfuserModel  # noqa: E402

import viz_infer_rellis as viz  # noqa: E402
from train_rellis_il import RellisDDV2Dataset, collate  # noqa: E402


def load_model(
    ckpt: Path,
    device: torch.device,
    plan_anchor_path: Path | None = None,
) -> V2TransfuserModel:
    cfg = TransfuserConfig()
    cfg.bkb_path = str(DDV2 / "ckpts/resnet34.a1_in1k/pytorch_model.bin")
    # init path is overwritten by ckpt; optional force below for RELLIS / custom banks
    cfg.plan_anchor_path = str(plan_anchor_path or (DDV2 / "kmeans_navsim_traj_20.npy"))
    model = V2TransfuserModel(cfg)
    state = torch.load(ckpt, map_location="cpu")
    if isinstance(state, dict) and "model_state_dict" in state:
        sd = state["model_state_dict"]
    elif isinstance(state, dict) and "state_dict" in state:
        sd = state["state_dict"]
    else:
        sd = state
    sd = {k.replace("agent.", "").replace("_transfuser_model.", ""): v for k, v in sd.items()}
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if plan_anchor_path is not None:
        anchors = np.load(plan_anchor_path).astype(np.float32)
        if anchors.ndim != 3 or anchors.shape[1:] != (8, 2):
            raise ValueError(f"plan anchors must be (K,8,2), got {anchors.shape}")
        with torch.no_grad():
            model._trajectory_head.plan_anchor.copy_(torch.from_numpy(anchors))
        print(f"[anchor] forced from {plan_anchor_path} shape={anchors.shape}")
    print(f"[ckpt] {ckpt}: missing={len(missing)} unexpected={len(unexpected)}")
    if isinstance(state, dict) and "ade" in state:
        print(f"  (ckpt meta ade={state['ade']}, epoch={state.get('epoch')})")
    model.to(device).eval()
    return model


def traj_metrics(pred: np.ndarray, gt: np.ndarray) -> tuple[float, float]:
    """pred/gt (T,3) → ADE, FDE on xy."""
    ade = float(np.linalg.norm(pred[:, :2] - gt[:, :2], axis=-1).mean())
    fde = float(np.linalg.norm(pred[-1, :2] - gt[-1, :2]))
    return ade, fde


def modes_metrics(modes: np.ndarray, gt: np.ndarray) -> tuple[float, float, int]:
    """modes (M,T,3) → minADE, minFDE, best_mode_idx by ADE."""
    ades = [float(np.linalg.norm(modes[i, :, :2] - gt[:, :2], axis=-1).mean()) for i in range(len(modes))]
    fdes = [float(np.linalg.norm(modes[i, -1, :2] - gt[-1, :2])) for i in range(len(modes))]
    i_ade = int(np.argmin(ades))
    i_fde = int(np.argmin(fdes))
    return ades[i_ade], fdes[i_fde], i_ade


@torch.no_grad()
def evaluate(
    model: V2TransfuserModel,
    loader: DataLoader,
    device: torch.device,
    max_samples: int = 0,
    compute_modes: bool = True,
) -> list[dict]:
    rows: list[dict] = []
    n = 0
    for batch in loader:
        B = batch["trajectory"].shape[0]
        feats = {
            "camera_feature": batch["camera_feature"].to(device, non_blocking=True),
            "lidar_feature": batch["lidar_feature"].to(device, non_blocking=True),
            "status_feature": batch["status_feature"].to(device, non_blocking=True),
        }
        out = model(feats)
        pred_b = out["trajectory"].detach().cpu().numpy()
        gt_b = batch["trajectory"].numpy()

        for i in range(B):
            gt = gt_b[i]
            pred = pred_b[i]
            ade, fde = traj_metrics(pred, gt)
            row = {
                "path": batch["path"][i],
                "seq": batch["seq"][i],
                "frame_idx": int(batch["frame_idx"][i]),
                "ade": ade,
                "fde": fde,
                "min_ade20": float("nan"),
                "min_fde20": float("nan"),
                "best_mode_by_ade": -1,
                "score_entropy": float("nan"),
            }
            if compute_modes:
                sample = {
                    "camera_feature": batch["camera_feature"][i].numpy(),
                    "lidar_feature": batch["lidar_feature"][i].numpy(),
                    "status_feature": batch["status_feature"][i].numpy(),
                    "trajectory": gt,
                    "seq": batch["seq"][i],
                    "frame_idx": int(batch["frame_idx"][i]),
                }
                _, modes, scores = viz.infer_all_modes(model, sample, device)
                min_ade, min_fde, mi = modes_metrics(modes, gt)
                row["min_ade20"] = min_ade
                row["min_fde20"] = min_fde
                row["best_mode_by_ade"] = mi
                row["score_entropy"] = float(-(scores * np.log(scores + 1e-12)).sum())
                row["_modes"] = modes
                row["_scores"] = scores
                row["_pred"] = pred
                row["_gt"] = gt
            rows.append(row)
            n += 1
            if n % 50 == 0:
                print(f"  evaluated {n} ...")
            if max_samples and n >= max_samples:
                return rows
    return rows


def summarize(rows: list[dict]) -> dict:
    def mean(key: str) -> float:
        vals = [r[key] for r in rows if np.isfinite(r[key])]
        return float(np.mean(vals)) if vals else float("nan")

    def pct(key: str, thr: float) -> float:
        vals = [r[key] for r in rows if np.isfinite(r[key])]
        if not vals:
            return float("nan")
        return float(np.mean([v > thr for v in vals]))

    summary = {
        "num_samples": len(rows),
        "ADE@best": mean("ade"),
        "FDE@best": mean("fde"),
        "minADE@20": mean("min_ade20"),
        "minFDE@20": mean("min_fde20"),
        "miss_rate@2m": pct("fde", 2.0),
        "miss_rate@4m": pct("fde", 4.0),
        "mean_entropy": mean("score_entropy"),
    }
    # per seq
    seqs = sorted({r["seq"] for r in rows})
    per_seq = {}
    for s in seqs:
        sub = [r for r in rows if r["seq"] == s]
        per_seq[s] = {
            "n": len(sub),
            "ADE@best": float(np.mean([r["ade"] for r in sub])),
            "FDE@best": float(np.mean([r["fde"] for r in sub])),
            "minADE@20": float(np.nanmean([r["min_ade20"] for r in sub])),
        }
    summary["per_seq"] = per_seq
    return summary


def plot_histograms(rows: list[dict], out_path: Path, title: str) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))
    for ax, key, name in zip(
        axes,
        ["ade", "fde", "min_ade20"],
        ["ADE@best (m)", "FDE@best (m)", "minADE@20 (m)"],
    ):
        vals = np.array([r[key] for r in rows if np.isfinite(r[key])])
        if len(vals) == 0:
            ax.set_title(f"{name} (empty)")
            continue
        ax.hist(vals, bins=40, color="steelblue", edgecolor="white")
        ax.axvline(vals.mean(), color="r", ls="--", label=f"mean={vals.mean():.3f}")
        ax.set_xlabel(name)
        ax.set_ylabel("count")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    fig.suptitle(title)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    print(f"[wrote] {out_path}")


def save_sample_viz(
    model: V2TransfuserModel,
    rows: list[dict],
    device: torch.device,
    out_dir: Path,
    k: int = 4,
    viz_all: bool = False,
) -> None:
    """Save multimodal panels. viz_all=True → every sample; else K best + K worst."""
    if (k <= 0 and not viz_all) or not rows or "_modes" not in rows[0]:
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    ranked = sorted(rows, key=lambda r: r["ade"])
    if viz_all:
        picks: list[tuple[str, list[dict]]] = [("all", ranked)]
    else:
        picks = [("best", ranked[:k]), ("worst", ranked[-k:][::-1])]
    n_done = 0
    n_total = sum(len(s) for _, s in picks)
    for tag, subset in picks:
        for rank, r in enumerate(subset):
            sample = viz.load_sample(Path(r["path"]))
            modes = r.get("_modes")
            scores = r.get("_scores")
            if modes is None:
                _, modes, scores = viz.infer_all_modes(model, sample, device)
            stem = Path(r["path"]).stem
            if tag == "all":
                # ADE-ranked so folder sorts from best→worst
                name = f"rank{rank:04d}_ade{r['ade']:.3f}_{stem}_modes.png"
                title = (
                    f"rank{rank}/{n_total-1} ADE={r['ade']:.3f} FDE={r['fde']:.3f} "
                    f"minADE={r['min_ade20']:.3f} | {stem} seq={r['seq']} frame={r['frame_idx']}"
                )
            else:
                name = f"{tag}_ade{r['ade']:.3f}_{stem}_modes.png"
                title = (
                    f"{tag} ADE={r['ade']:.3f} FDE={r['fde']:.3f} minADE={r['min_ade20']:.3f} | "
                    f"{stem} seq={r['seq']} frame={r['frame_idx']}"
                )
            viz.make_panel(sample, modes, scores, title, out_dir / name)
            n_done += 1
            if n_done % 50 == 0 or n_done == n_total:
                print(f"  [viz] {n_done}/{n_total}")
    print(f"[wrote] {n_total} panels → {out_dir}")


def write_csv(rows: list[dict], path: Path) -> None:
    keys = [
        "path",
        "seq",
        "frame_idx",
        "ade",
        "fde",
        "min_ade20",
        "min_fde20",
        "best_mode_by_ade",
        "score_entropy",
    ]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in keys})
    print(f"[wrote] {path}")


def print_summary(name: str, summary: dict) -> None:
    print(f"\n===== {name} =====")
    print(f"  n={summary['num_samples']}")
    for k in ["ADE@best", "FDE@best", "minADE@20", "minFDE@20", "miss_rate@2m", "miss_rate@4m", "mean_entropy"]:
        v = summary[k]
        if "miss" in k:
            print(f"  {k}: {v*100:.2f}%")
        else:
            print(f"  {k}: {v:.4f}")
    print("  per-seq:")
    for s, m in summary["per_seq"].items():
        print(
            f"    {s}: n={m['n']} ADE={m['ADE@best']:.4f} FDE={m['FDE@best']:.4f} minADE={m['minADE@20']:.4f}"
        )


def _sample_stem_from_viz_name(name: str) -> str | None:
    """rank0000_ade0.226_sample_001230_modes.png → sample_001230"""
    if not name.endswith("_modes.png"):
        return None
    parts = name[: -len("_modes.png")].split("_")
    # find 'sample' token
    for i, p in enumerate(parts):
        if p == "sample" and i + 1 < len(parts):
            return f"sample_{parts[i + 1]}"
    return None


def stitch_panels(baseline_dir: Path, ft_dir: Path, out_dir: Path) -> int:
    """Horizontally stitch baseline|FT panels matched by sample stem. Returns #pairs."""
    import re

    from PIL import Image, ImageDraw

    def index_dir(d: Path) -> dict[str, Path]:
        out: dict[str, Path] = {}
        if not d.is_dir():
            return out
        for p in d.glob("*_modes.png"):
            stem = _sample_stem_from_viz_name(p.name)
            if stem:
                out[stem] = p
        return out

    base_idx = index_dir(baseline_dir)
    ft_idx = index_dir(ft_dir)
    common = sorted(set(base_idx) & set(ft_idx))
    out_dir.mkdir(parents=True, exist_ok=True)
    # ADE from filenames for ranking stitch output
    ade_re = re.compile(r"_ade([0-9.]+)_")

    def ft_ade(stem: str) -> float:
        m = ade_re.search(ft_idx[stem].name)
        return float(m.group(1)) if m else 0.0

    common_sorted = sorted(common, key=ft_ade)
    for rank, stem in enumerate(common_sorted):
        left = Image.open(base_idx[stem]).convert("RGB")
        right = Image.open(ft_idx[stem]).convert("RGB")
        h = max(left.height, right.height)
        # pad to same height
        def pad(im: Image.Image, h: int) -> Image.Image:
            if im.height == h:
                return im
            canvas = Image.new("RGB", (im.width, h), (255, 255, 255))
            canvas.paste(im, (0, 0))
            return canvas

        left, right = pad(left, h), pad(right, h)
        gap = 8
        banner_h = 36
        canvas = Image.new("RGB", (left.width + gap + right.width, h + banner_h), (245, 245, 245))
        draw = ImageDraw.Draw(canvas)
        draw.rectangle([0, 0, canvas.width, banner_h], fill=(40, 40, 40))
        label = f"{stem}  |  LEFT=baseline(88p1)  RIGHT=finetune(v4)  |  rank_by_ft_ade={rank}"
        draw.text((12, 10), label, fill=(240, 240, 240))
        canvas.paste(left, (0, banner_h))
        canvas.paste(right, (left.width + gap, banner_h))
        b_ade = ade_re.search(base_idx[stem].name)
        f_ade = ade_re.search(ft_idx[stem].name)
        b_a = b_ade.group(1) if b_ade else "?"
        f_a = f_ade.group(1) if f_ade else "?"
        out_name = f"rank{rank:04d}_base{b_a}_ft{f_a}_{stem}_compare.png"
        canvas.save(out_dir / out_name, quality=92)
        if (rank + 1) % 100 == 0 or rank == 0:
            print(f"  [stitch] {rank+1}/{len(common_sorted)}")
    return len(common_sorted)


def write_compare_summary(all_summaries: dict, path: Path) -> None:
    keys = ["ADE@best", "FDE@best", "minADE@20", "minFDE@20", "miss_rate@2m", "miss_rate@4m"]
    cmp: dict = {"ckpt": all_summaries.get("ckpt"), "splits": {}}
    for split in ("val", "train"):
        ft = all_summaries.get(split)
        base = all_summaries.get(f"baseline_{split}")
        if not ft or not base:
            continue
        row = {"n": ft.get("num_samples"), "finetune": {}, "baseline": {}, "delta_ft_minus_base": {}}
        for k in keys:
            row["finetune"][k] = ft.get(k)
            row["baseline"][k] = base.get(k)
            if ft.get(k) is not None and base.get(k) is not None:
                row["delta_ft_minus_base"][k] = float(ft[k]) - float(base[k])
        cmp["splits"][split] = row
    path.write_text(json.dumps(cmp, indent=2))
    print(f"[wrote] {path}")
    for split, row in cmp["splits"].items():
        print(f"\n===== compare {split} (Δ = FT − baseline, lower ADE better) =====")
        for k in keys:
            b, f, d = row["baseline"][k], row["finetune"][k], row["delta_ft_minus_base"][k]
            if "miss" in k:
                print(f"  {k}: base={b*100:.2f}%  ft={f*100:.2f}%  Δ={d*100:+.2f}pp")
            else:
                print(f"  {k}: base={b:.4f}  ft={f:.4f}  Δ={d:+.4f}")


def _sample_stem_from_viz_name(name: str) -> str | None:
    """rank0000_ade0.123_sample_001230_modes.png → sample_001230"""
    if not name.endswith("_modes.png"):
        return None
    parts = name[: -len("_modes.png")].split("_")
    # ... sample NNNNNN
    for i, p in enumerate(parts):
        if p == "sample" and i + 1 < len(parts) and parts[i + 1].isdigit():
            return f"sample_{parts[i + 1]}"
    return None


def stitch_ft_baseline_viz(ft_dir: Path, base_dir: Path, out_dir: Path, left_label: str, right_label: str) -> int:
    """Side-by-side PNG: baseline | fine-tune, matched by sample_XXXXXX."""
    import cv2

    def index(d: Path) -> dict[str, Path]:
        out: dict[str, Path] = {}
        if not d.is_dir():
            return out
        for p in d.glob("*_modes.png"):
            stem = _sample_stem_from_viz_name(p.name)
            if stem:
                out[stem] = p
        return out

    ft_map = index(ft_dir)
    base_map = index(base_dir)
    common = sorted(set(ft_map) & set(base_map))
    out_dir.mkdir(parents=True, exist_ok=True)
    n = 0
    for stem in common:
        a = cv2.imread(str(base_map[stem]))
        b = cv2.imread(str(ft_map[stem]))
        if a is None or b is None:
            continue
        h = max(a.shape[0], b.shape[0])
        def pad(img):
            if img.shape[0] == h:
                return img
            return cv2.copyMakeBorder(img, 0, h - img.shape[0], 0, 0, cv2.BORDER_CONSTANT, value=(255, 255, 255))
        a, b = pad(a), pad(b)
        bar = np.full((h, 8, 3), 40, dtype=np.uint8)
        panel = np.concatenate([a, bar, b], axis=1)
        # labels strip
        strip = np.full((36, panel.shape[1], 3), 30, dtype=np.uint8)
        cv2.putText(strip, left_label, (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (220, 220, 220), 2, cv2.LINE_AA)
        cv2.putText(
            strip,
            right_label,
            (a.shape[1] + 20, 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (220, 220, 220),
            2,
            cv2.LINE_AA,
        )
        out = np.concatenate([strip, panel], axis=0)
        cv2.imwrite(str(out_dir / f"{stem}_baseline_vs_ft.png"), out)
        n += 1
        if n % 100 == 0 or n == len(common):
            print(f"  [stitch] {n}/{len(common)}")
    print(f"[wrote] {n} stitched panels → {out_dir}")
    return n


def main():
    ap = argparse.ArgumentParser(description="Evaluate RELLIS IL checkpoint")
    ap.add_argument(
        "--ckpt",
        type=Path,
        default=ROOT / "log/rellis_il_finetune/ckpts/best.ckpt",
    )
    ap.add_argument("--data_dir", type=Path, default=ROOT / "datasets/ddv2_rellis_full")
    ap.add_argument("--split", type=str, default="val", choices=["val", "train", "both"])
    ap.add_argument("--out_dir", type=Path, default=ROOT / "log/rellis_il_eval")
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--max_samples", type=int, default=0, help="0 = full split")
    ap.add_argument("--no_modes", action="store_true", help="skip minADE@20 (faster)")
    ap.add_argument("--viz_k", type=int, default=4, help="save K best/worst multimodal panels")
    ap.add_argument(
        "--viz_all",
        action="store_true",
        help="save multimodal panel for every sample (ADE-ranked); ignores --viz_k",
    )
    ap.add_argument(
        "--baseline_ckpt",
        type=Path,
        default=None,
        help="optional second ckpt to compare (e.g. official NAVSIM IL)",
    )
    ap.add_argument(
        "--plan_anchor_path",
        type=Path,
        default=None,
        help="force (K,8,2) anchors for --ckpt after load (e.g. RELLIS kmeans). "
        "Baseline keeps anchors from its own ckpt.",
    )
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[device] {device}")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    splits = ["val", "train"] if args.split == "both" else [args.split]
    model = load_model(args.ckpt, device, plan_anchor_path=args.plan_anchor_path)

    all_summaries = {"ckpt": str(args.ckpt), "device": str(device)}

    for split in splits:
        print(f"\n[eval] split={split} ckpt={args.ckpt.name}")
        ds = RellisDDV2Dataset(args.data_dir, split)
        loader = DataLoader(
            ds,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True,
            collate_fn=collate,
        )
        t0 = time.time()
        rows = evaluate(
            model,
            loader,
            device,
            max_samples=args.max_samples,
            compute_modes=not args.no_modes,
        )
        dt = time.time() - t0
        summary = summarize(rows)
        summary["seconds"] = dt
        print_summary(f"{args.ckpt.name} / {split}", summary)
        all_summaries[split] = summary

        tag = f"{args.ckpt.stem}_{split}"
        write_csv(rows, args.out_dir / f"{tag}_per_sample.csv")
        plot_histograms(rows, args.out_dir / f"{tag}_hist.png", f"{tag}")
        # strip heavy arrays before json
        save_sample_viz(
            model,
            rows,
            device,
            args.out_dir / f"{tag}_viz",
            k=args.viz_k,
            viz_all=args.viz_all,
        )
        for r in rows:
            r.pop("_modes", None)
            r.pop("_scores", None)
            r.pop("_pred", None)
            r.pop("_gt", None)

    ft_viz_dirs: dict[str, Path] = {}
    base_viz_dirs: dict[str, Path] = {}
    # record FT viz dirs from the loop above
    for split in splits:
        tag = f"{args.ckpt.stem}_{split}"
        ft_viz_dirs[split] = args.out_dir / f"{tag}_viz"

    if args.baseline_ckpt is not None:
        print(f"\n[eval] baseline {args.baseline_ckpt}")
        base = load_model(args.baseline_ckpt, device)
        for split in splits:
            ds = RellisDDV2Dataset(args.data_dir, split)
            loader = DataLoader(
                ds,
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=args.num_workers,
                pin_memory=True,
                collate_fn=collate,
            )
            rows = evaluate(
                base,
                loader,
                device,
                max_samples=args.max_samples,
                compute_modes=not args.no_modes,
            )
            summary = summarize(rows)
            print_summary(f"{args.baseline_ckpt.name} / {split}", summary)
            all_summaries[f"baseline_{split}"] = summary
            btag = f"{args.baseline_ckpt.stem}_{split}"
            write_csv(rows, args.out_dir / f"{btag}_per_sample.csv")
            plot_histograms(rows, args.out_dir / f"{btag}_hist.png", f"{btag}")
            save_sample_viz(
                base,
                rows,
                device,
                args.out_dir / f"{btag}_viz",
                k=args.viz_k,
                viz_all=args.viz_all,
            )
            base_viz_dirs[split] = args.out_dir / f"{btag}_viz"
            for r in rows:
                r.pop("_modes", None)
                r.pop("_scores", None)
                r.pop("_pred", None)
                r.pop("_gt", None)

        # side-by-side: baseline | finetune, keyed by sample stem
        for split in splits:
            if split not in ft_viz_dirs or split not in base_viz_dirs:
                continue
            stitch_dir = args.out_dir / f"compare_{split}_baseline_vs_ft"
            n = stitch_panels(base_viz_dirs[split], ft_viz_dirs[split], stitch_dir)
            print(f"[stitch] {n} pairs → {stitch_dir}")

        # metric comparison table
        write_compare_summary(all_summaries, args.out_dir / "compare_summary.json")

    out_json = args.out_dir / "summary.json"
    out_json.write_text(json.dumps(all_summaries, indent=2))
    print(f"\n[done] {out_json}")


if __name__ == "__main__":
    main()
