#!/usr/bin/env python3
"""Add extra checkpoints to an existing trio_100 eval dir (same sampled frames).

Writes {tag}_viz/frame_XXXXXX_{tag}_modes.png, merges metrics into summary.json,
and stitches baseline | v5 | v6 | <tags...> into compare_all/.
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
sys.path[:0] = [str(ROOT / "scripts/bridge"), str(ROOT / "scripts/rellis"), str(ROOT / "scripts/train"), str(DDV2)]

from nuplan_min_stub import install as install_nuplan_stub  # noqa: E402

install_nuplan_stub()

from eval_rellis_il import load_model, summarize  # noqa: E402
from eval_rellis_trio_sample import eval_on_files  # noqa: E402


def mode_spread(model, files: list[Path], device) -> float:
    """Mean pairwise ADE between the 20 output modes (geometric diversity)."""
    import viz_infer_rellis as viz

    vals = []
    for p in files:
        _, modes, _ = viz.infer_all_modes(model, viz.load_sample(p), device)
        xy = modes[:, :, :2]
        d = np.linalg.norm(xy[:, None] - xy[None], axis=-1).mean(-1)
        vals.append(d[np.triu_indices(len(xy), 1)].mean())
    return float(np.mean(vals))


def stitch(dirs: list[tuple[str, Path]], frame_idxs: list[int], out_dir: Path) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)
    n = 0
    for fi in frame_idxs:
        imgs = []
        for tag, d in dirs:
            im = cv2.imread(str(d / f"frame_{fi:06d}_{tag}_modes.png"))
            if im is None:
                break
            imgs.append(im)
        if len(imgs) != len(dirs):
            continue
        h = min(i.shape[0] for i in imgs)
        w = min(i.shape[1] for i in imgs)
        imgs = [cv2.resize(i, (w, h)) for i in imgs]
        # 2 columns grid keeps the panel readable
        cols = 2
        rows = (len(imgs) + cols - 1) // cols
        gap, top = 8, 36
        canvas = np.full((top + rows * h + gap * (rows - 1), cols * w + gap, 3), 245, np.uint8)
        for k, im in enumerate(imgs):
            r, c = divmod(k, cols)
            y0, x0 = top + r * (h + gap), c * (w + gap)
            canvas[y0 : y0 + h, x0 : x0 + w] = im
        label = f"frame={fi:06d} | " + "  ".join(f"[{k}]{t}" for k, (t, _) in enumerate(dirs))
        cv2.putText(canvas, label, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (20, 20, 20), 2, cv2.LINE_AA)
        cv2.imwrite(str(out_dir / f"frame_{fi:06d}_compare.png"), canvas)
        n += 1
    print(f"[stitch] {n} → {out_dir}")
    return n


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--trio_dir", type=Path, default=ROOT / "log/rellis_il_eval_trio_100")
    ap.add_argument(
        "--add",
        nargs="+",
        required=True,
        help="tag=ckpt[:anchor_npy], e.g. v7=log/.../best.ckpt:DiffusionDriveV2/kmeans_rellis_traj_20.npy",
    )
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    trio = args.trio_dir
    sample = json.loads((trio / "sample_frames.json").read_text())
    files = [Path(s["path"]) for s in sample]
    frame_idxs = [int(s["frame_idx"]) for s in sample]
    summary = json.loads((trio / "summary.json").read_text())

    new_tags = []
    for spec in args.add:
        tag, rest = spec.split("=", 1)
        ckpt_s, _, anchor_s = rest.partition(":")
        ckpt = (ROOT / ckpt_s) if not Path(ckpt_s).is_absolute() else Path(ckpt_s)
        anchor = None
        if anchor_s:
            anchor = (ROOT / anchor_s) if not Path(anchor_s).is_absolute() else Path(anchor_s)
        print(f"\n[eval] {tag} ckpt={ckpt} anchor={anchor}")
        t0 = time.time()
        model = load_model(ckpt, device, plan_anchor_path=anchor)
        rows = eval_on_files(model, files, device, tag, trio / f"{tag}_viz")
        s = summarize(rows)
        s["mode_pairwise_ade"] = mode_spread(model, files[::5], device)
        s["seconds"] = time.time() - t0
        s["ckpt"] = str(ckpt)
        s["plan_anchor"] = str(anchor) if anchor else "from_ckpt"
        ck = torch.load(ckpt, map_location="cpu")
        s["epoch"] = int(ck.get("epoch", -1)) if isinstance(ck, dict) else -1
        summary["models"][tag] = s
        base = summary["models"]["baseline"]
        summary.setdefault("delta_vs_baseline", {})[tag] = {
            k: float(s[k] - base[k])
            for k in ("ADE@best", "FDE@best", "minADE@20", "minFDE@20", "miss_rate@2m", "miss_rate@4m")
        }
        new_tags.append(tag)
        print(
            f"  {tag}: ADE={s['ADE@best']:.4f} FDE={s['FDE@best']:.4f} minADE={s['minADE@20']:.4f} "
            f"H={s['mean_entropy']:.2f} spread={s['mode_pairwise_ade']:.3f}"
        )
        del model
        torch.cuda.empty_cache()

    (trio / "summary.json").write_text(json.dumps(summary, indent=2))
    tags = ["baseline", "v5", "v6"] + new_tags
    stitch([(t, trio / f"{t}_viz") for t in tags], frame_idxs, trio / "compare_all")
    print(f"[done] {trio / 'summary.json'}")


if __name__ == "__main__":
    main()
