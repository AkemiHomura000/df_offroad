#!/usr/bin/env python3
"""Before/after viewpoint inference comparison panels (Scheme D showcase).

For each selected stem under --render_dir/frames/<stem>/:
  columns = as_logged | level | pitch_p15 | pitch_m15 | roll_p10
  each column: RGB (top) + ego-frame traj modes (mid) + mode score bars (bottom)

Also writes a showcase grid of high-impact stems.

  python scripts/train/viz_sensor_geometry_infer_compare.py \\
    --render_dir log/sensor_geom_ablation \\
    --ckpt log/verti_il_scratch_elevation_goal_attitude/ckpts/best.ckpt \\
    --out_dir log/sensor_geom_ablation/eval/infer_compare \\
    --stems auto --topk 8
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib import cm
from matplotlib.colors import Normalize
from PIL import Image, ImageDraw, ImageFont

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
from eval_rellis_il import load_model, traj_metrics  # noqa: E402

VARIANT_ORDER = ["as_logged", "level", "pitch_p15", "pitch_m15", "roll_p10"]
VARIANT_LABEL = {
    "as_logged": "as logged\n(before)",
    "level": "level\nr=p=0",
    "pitch_p15": "pitch +15°",
    "pitch_m15": "pitch −15°",
    "roll_p10": "roll +10°",
}


def load_variant_npz(path: Path) -> dict:
    d = np.load(path)
    out = {k: d[k] for k in d.files}
    # ensure camera_feature CHW float
    if "camera_feature" not in out and "cam_front" in out:
        cam = np.asarray(out["cam_front"], dtype=np.float32) / 255.0
        out["camera_feature"] = np.transpose(cam, (2, 0, 1))
    return out


def pick_stems(eval_dir: Path, topk: int, prefer: Optional[List[str]] = None) -> List[str]:
    rows = list(csv.DictReader(open(eval_dir / "per_sample_all.csv")))
    by = defaultdict(dict)
    for r in rows:
        by[r["stem"]][r["variant"]] = r
    scored = []
    for stem, d in by.items():
        if "as_logged" not in d:
            continue
        # prioritize mode flip + |ΔADE|
        score = 0.0
        for v in ("pitch_m15", "roll_p10", "level", "pitch_p15"):
            if v not in d:
                continue
            score = max(score, abs(float(d[v]["delta_ade_vs_logged"])))
            if float(d[v].get("top1_match_logged") or 1) < 0.5:
                score += 0.5
        scored.append((score, float(d["as_logged"].get("att_mag_deg_logged") or 0), stem))
    scored.sort(reverse=True)
    stems = [s for _, __, s in scored[:topk]]
    if prefer:
        for p in prefer:
            if p not in stems:
                stems.insert(0, p)
    # unique preserve order
    seen, out = set(), []
    for s in stems:
        if s in seen:
            continue
        seen.add(s)
        out.append(s)
    return out[: max(topk, len(prefer or []))]


@torch.no_grad()
def infer_one(model, sample: dict, device) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float, float, int]:
    pred, modes, scores = viz.infer_all_modes(model, sample, device)
    gt = np.asarray(sample["trajectory"], dtype=np.float32)
    ade, fde = traj_metrics(pred, gt)
    top1 = int(np.argmax(scores))
    return pred, modes, scores, ade, fde, top1


def draw_traj_ax(ax, gt, modes, scores, best_i):
    ax.set_aspect("equal")
    ax.set_facecolor("#f4f4f4")
    ax.plot(0, 0, marker="+", color="red", markersize=10, mew=2, zorder=10)
    if modes is not None and scores is not None:
        norm = Normalize(vmin=float(scores.min()), vmax=float(scores.max()) + 1e-8)
        cmap = cm.plasma
        order = np.argsort(scores)
        for i in order:
            if i == best_i:
                continue
            c = cmap(norm(scores[i]))
            ax.plot(modes[i, :, 1], modes[i, :, 0], "-", color=c, lw=0.9,
                    alpha=float(0.12 + 0.4 * norm(scores[i])), zorder=2)
        ax.plot(
            modes[best_i, :, 1],
            modes[best_i, :, 0],
            "--",
            color="deepskyblue",
            lw=2.4,
            marker="s",
            ms=4,
            markeredgecolor="white",
            markeredgewidth=0.6,
            zorder=6,
            label=f"top1 #{best_i}",
        )
    ax.plot(
        gt[:, 1],
        gt[:, 0],
        "-",
        color="limegreen",
        lw=2.2,
        marker="o",
        ms=4,
        markeredgecolor="black",
        markeredgewidth=0.5,
        zorder=8,
        label="GT",
    )
    # zoom
    pts = [gt[:, :2], np.zeros((1, 2))]
    if modes is not None:
        pts.append(modes[best_i, :, :2])
    pts = np.concatenate(pts, 0)
    pad = max(1.0, 0.35 * float(max(np.ptp(pts[:, 0]), np.ptp(pts[:, 1]), 0.5)))
    x0, x1 = float(pts[:, 0].min() - pad), float(pts[:, 0].max() + pad)
    y0, y1 = float(pts[:, 1].min() - pad), float(pts[:, 1].max() + pad)
    span = max(x1 - x0, y1 - y0, 2.0)
    xc, yc = 0.5 * (x0 + x1), 0.5 * (y0 + y1)
    ax.set_xlim(yc + span / 2, yc - span / 2)  # +y left
    ax.set_ylim(xc - span / 2, xc + span / 2)
    ax.grid(True, alpha=0.3)
    ax.tick_params(labelsize=7)
    ax.legend(loc="upper left", fontsize=6, framealpha=0.9)


def draw_score_ax(ax, scores: np.ndarray, best_i: int):
    """Mode softmax bars under the traj panel."""
    scores = np.asarray(scores, dtype=np.float64)
    idx = np.arange(len(scores))
    norm = Normalize(vmin=float(scores.min()), vmax=float(scores.max()) + 1e-8)
    colors = [cm.plasma(norm(s)) for s in scores]
    ax.bar(idx, scores, color=colors, edgecolor="k", linewidth=0.25, width=0.85)
    ax.bar(best_i, scores[best_i], color="deepskyblue", edgecolor="k", linewidth=0.9, width=0.85)
    ax.set_xlim(-0.5, len(scores) - 0.5)
    ax.set_ylim(0.0, max(float(scores.max()) * 1.25, 1e-3))
    ax.set_xlabel("mode id", fontsize=7)
    ax.set_ylabel("score", fontsize=7)
    ax.tick_params(labelsize=6)
    ax.set_title(f"mode scores  best=#{best_i}  p={scores[best_i]:.3f}", fontsize=7, pad=2)
    ax.grid(True, axis="y", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def render_stem_compare(
    model,
    device,
    stem: str,
    frame_dir: Path,
    out_path: Path,
    variants: List[str],
) -> dict:
    cols = [v for v in variants if (frame_dir / f"{v}.npz").is_file()]
    if "as_logged" not in cols:
        raise FileNotFoundError(f"missing as_logged in {frame_dir}")

    n = len(cols)
    # row0 RGB | row1 traj | row2 mode scores
    fig, axes = plt.subplots(
        3,
        n,
        figsize=(3.1 * n, 7.2),
        gridspec_kw={"height_ratios": [1.05, 1.15, 0.55]},
        constrained_layout=True,
    )
    if n == 1:
        axes = np.array([[axes[0]], [axes[1]], [axes[2]]])

    meta = {"stem": stem, "variants": {}}
    base_top1 = None
    base_ade = None

    for j, v in enumerate(cols):
        sample = load_variant_npz(frame_dir / f"{v}.npz")
        # status fixed (geometry-only) — already in npz
        pred, modes, scores, ade, fde, top1 = infer_one(model, sample, device)
        if v == "as_logged":
            base_top1 = top1
            base_ade = ade
        cam = np.transpose(sample["camera_feature"], (1, 2, 0))
        cam = np.clip(cam, 0, 1)
        axes[0, j].imshow(cam)
        axes[0, j].axis("off")
        roll = float(np.degrees(float(sample["roll"]))) if "roll" in sample else float("nan")
        pitch = float(np.degrees(float(sample["pitch"]))) if "pitch" in sample else float("nan")
        match = "" if base_top1 is None else ("same" if top1 == base_top1 else "CHANGED")
        dade = 0.0 if base_ade is None else ade - base_ade
        title = (
            f"{VARIANT_LABEL.get(v, v)}\n"
            f"r={roll:+.1f}° p={pitch:+.1f}°\n"
            f"ADE={ade:.2f}m (Δ{dade:+.2f})  top1=#{top1} {match}"
        )
        axes[0, j].set_title(title, fontsize=8, pad=4)
        draw_traj_ax(axes[1, j], sample["trajectory"], modes, scores, top1)
        if j == 0:
            axes[1, j].set_ylabel("x forward (m)", fontsize=8)
        axes[1, j].set_xlabel("y left (m)", fontsize=8)
        draw_score_ax(axes[2, j], scores, top1)
        meta["variants"][v] = {
            "ade": ade,
            "fde": fde,
            "top1": top1,
            "delta_ade": dade,
            "top1_match": bool(top1 == base_top1) if base_top1 is not None else True,
            "roll_deg": roll,
            "pitch_deg": pitch,
            "top1_score": float(scores[top1]),
        }

    att = float(np.load(frame_dir / "as_logged.npz")["att_mag_deg_logged"])
    fig.suptitle(
        f"{stem}  |  Chrono viewpoint ablation → model inference  |  "
        f"logged |roll|+|pitch|={att:.1f}°  |  green=GT  cyan=top1  plasma=other modes / score bars",
        fontsize=10,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    print(f"[wrote] {out_path}", flush=True)
    return meta


def make_showcase_strip(compare_pngs: List[Path], out_path: Path, max_n: int = 4):
    """Stack a few compare panels vertically for slides."""
    ims = [Image.open(p).convert("RGB") for p in compare_pngs[:max_n]]
    if not ims:
        return
    w = max(im.width for im in ims)
    resized = []
    for im in ims:
        if im.width != w:
            nh = int(im.height * w / im.width)
            im = im.resize((w, nh), Image.BILINEAR)
        resized.append(np.asarray(im))
    gap = np.full((16, w, 3), 250, dtype=np.uint8)
    parts = []
    for i, arr in enumerate(resized):
        if i:
            parts.append(gap)
        parts.append(arr)
    mosaic = np.concatenate(parts, axis=0)
    # banner
    banner_h = 42
    canvas = Image.new("RGB", (w, mosaic.shape[0] + banner_h), (32, 32, 32))
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None
    draw.text(
        (12, 12),
        "Viewpoint change → DiffusionDrive inference  |  left=as_logged (before)  |  right=re-rendered attitudes",
        fill=(240, 240, 240),
        font=font,
    )
    canvas.paste(Image.fromarray(mosaic), (0, banner_h))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path, quality=92)
    print(f"[wrote] {out_path}", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--render_dir", type=Path, default=ROOT / "log/sensor_geom_ablation")
    ap.add_argument("--ckpt", type=Path, default=ROOT / "log/verti_il_scratch_elevation_goal_attitude/ckpts/best.ckpt")
    ap.add_argument("--plan_anchor_path", type=Path, default=DDV2 / "kmeans_verti_traj_20.npy")
    ap.add_argument("--out_dir", type=Path, default=None)
    ap.add_argument("--stems", default="auto", help="comma list or 'auto'")
    ap.add_argument("--topk", type=int, default=8)
    ap.add_argument("--prefer", default="sample_000711", help="always include these stems")
    ap.add_argument("--variants", default=",".join(VARIANT_ORDER))
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    out_dir = args.out_dir or (args.render_dir / "eval" / "infer_compare")
    out_dir.mkdir(parents=True, exist_ok=True)
    variants = [v.strip() for v in args.variants.split(",") if v.strip()]
    prefer = [s.strip() for s in args.prefer.split(",") if s.strip()]

    if args.stems.strip() == "auto":
        stems = pick_stems(args.render_dir / "eval", args.topk, prefer=prefer)
    else:
        stems = [s.strip() for s in args.stems.split(",") if s.strip()]

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model = load_model(args.ckpt, device, plan_anchor_path=args.plan_anchor_path)

    metas = []
    pngs = []
    for stem in stems:
        frame_dir = args.render_dir / "frames" / stem
        if not frame_dir.is_dir():
            print(f"[skip] missing {frame_dir}", flush=True)
            continue
        out_png = out_dir / f"{stem}_before_after_infer.png"
        meta = render_stem_compare(model, device, stem, frame_dir, out_png, variants)
        metas.append(meta)
        pngs.append(out_png)

    if pngs:
        make_showcase_strip(pngs[:4], out_dir / "showcase_strip.png", max_n=4)
        # also a 2x2 contact sheet if enough
        if len(pngs) >= 4:
            ims = [Image.open(p).convert("RGB") for p in pngs[:4]]
            # normalize widths
            w = min(im.width for im in ims)
            norm = []
            for im in ims:
                nh = int(im.height * w / im.width)
                norm.append(im.resize((w, nh), Image.BILINEAR))
            h = max(im.height for im in norm)
            def pad(im):
                if im.height == h:
                    return im
                c = Image.new("RGB", (w, h), (255, 255, 255))
                c.paste(im, (0, 0))
                return c
            norm = [pad(im) for im in norm]
            sheet = Image.new("RGB", (2 * w + 8, 2 * h + 8), (245, 245, 245))
            sheet.paste(norm[0], (0, 0))
            sheet.paste(norm[1], (w + 8, 0))
            sheet.paste(norm[2], (0, h + 8))
            sheet.paste(norm[3], (w + 8, h + 8))
            sheet_path = out_dir / "showcase_2x2.png"
            sheet.save(sheet_path, quality=92)
            print(f"[wrote] {sheet_path}", flush=True)

    (out_dir / "infer_compare_meta.json").write_text(json.dumps({"stems": stems, "panels": metas}, indent=2))
    print(f"[done] {len(pngs)} panels → {out_dir}", flush=True)


if __name__ == "__main__":
    main()
