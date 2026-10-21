#!/usr/bin/env python3
"""Visualize RELLIS→DDV2 samples and all DiffusionDrive multimodal trajectories."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from matplotlib import cm
from matplotlib.colors import Normalize

ROOT = Path("/home/rm/Desktop/df_offroad")
DDV2 = ROOT / "DiffusionDriveV2"
sys.path.insert(0, str(ROOT / "scripts" / "bridge"))
sys.path.insert(0, str(DDV2))

from nuplan_min_stub import install as install_nuplan_stub  # noqa: E402

install_nuplan_stub()

from navsim.agents.diffusiondrive.modules.blocks import gen_sineembed_for_position  # noqa: E402
from navsim.agents.diffusiondrive.transfuser_config import TransfuserConfig  # noqa: E402
from navsim.agents.diffusiondrive.transfuser_model_v2 import V2TransfuserModel  # noqa: E402


def load_sample(path: Path) -> dict:
    d = np.load(path, allow_pickle=False)
    return {k: d[k] for k in d.files}


def _draw_trajs_on_ax(
    ax,
    gt: np.ndarray,
    modes: np.ndarray | None,
    scores: np.ndarray | None,
    *,
    with_colorbar: bool = False,
    fig=None,
) -> int | None:
    """Draw GT + multimodal trajs. Returns best mode index."""
    ax.plot(0, 0, marker="+", color="red", markersize=12, mew=2, zorder=10, label="ego")
    best_i = None
    if modes is not None and scores is not None:
        scores = np.asarray(scores, dtype=np.float64)
        modes = np.asarray(modes, dtype=np.float64)
        # NaN/Inf (diverged training) → skip mode drawing rather than crash viz.
        if not np.isfinite(scores).all() or not np.isfinite(modes).all():
            scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
            modes = np.nan_to_num(modes, nan=0.0, posinf=0.0, neginf=0.0)
            if float(scores.sum()) <= 0:
                scores = np.ones(len(scores), dtype=np.float64) / max(len(scores), 1)
        best_i = int(np.argmax(scores))
        score_floor = float(np.percentile(scores, 70))
        norm = Normalize(vmin=float(scores.min()), vmax=float(scores.max()) + 1e-8)
        cmap = cm.plasma
        for i in np.argsort(scores):
            if i == best_i:
                continue
            c = cmap(norm(scores[i]))
            alpha = float(np.clip(0.15 + 0.35 * float(norm(scores[i])), 0.0, 1.0))
            lw = 1.2 if scores[i] >= score_floor else 0.7
            ax.plot(modes[i, :, 1], modes[i, :, 0], "-", color=c, lw=lw, alpha=alpha, zorder=2)
        if with_colorbar and fig is not None:
            sm = cm.ScalarMappable(norm=norm, cmap=cmap)
            sm.set_array([])
            cbar = fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
            cbar.set_label("mode prob")
        ax.plot(
            modes[best_i, :, 1],
            modes[best_i, :, 0],
            linestyle="--",
            color="deepskyblue",
            lw=2.8,
            marker="s",
            markersize=5,
            markerfacecolor="deepskyblue",
            markeredgecolor="white",
            markeredgewidth=1.0,
            label=f"best#{best_i}",
            zorder=6,
        )
    ax.plot(
        gt[:, 1],
        gt[:, 0],
        linestyle="-",
        color="limegreen",
        lw=2.4,
        marker="o",
        markersize=6,
        markerfacecolor="limegreen",
        markeredgecolor="black",
        markeredgewidth=1.0,
        label="GT",
        zorder=8,
    )
    return best_i


def _traj_zoom_limits(gt: np.ndarray, modes: np.ndarray | None, best_i: int | None) -> tuple[float, float, float, float]:
    """Return (y0,y1,x0,x1) for traj-only panel; scale by GT + best length."""
    pts = [gt[:, :2], np.zeros((1, 2), dtype=np.float32)]
    if modes is not None and best_i is not None:
        pts.append(modes[best_i, :, :2])
    pts = np.concatenate(pts, axis=0)
    pts = pts[np.isfinite(pts).all(axis=1)]
    pad = max(1.0, 0.35 * float(max(np.ptp(pts[:, 0]), np.ptp(pts[:, 1]), 0.5)))
    x0 = float(min(0.0, pts[:, 0].min()) - pad)
    x1 = float(max(0.0, pts[:, 0].max()) + pad)
    y0 = float(min(0.0, pts[:, 1].min()) - pad)
    y1 = float(max(0.0, pts[:, 1].max()) + pad)
    span = max(x1 - x0, y1 - y0, 2.0)
    xc, yc = 0.5 * (x0 + x1), 0.5 * (y0 + y1)
    return yc - span / 2, yc + span / 2, xc - span / 2, xc + span / 2


def make_panel(
    sample: dict,
    modes: np.ndarray | None,
    scores: np.ndarray | None,
    title: str,
    out: Path,
    top_k_labels: int = 5,
    show_time_plot: bool = True,
) -> None:
    """modes: (M,8,3), scores: (M,) softmax probs.

    Layout:
      [camera]          [traj-only, auto-zoom, no map]
      [lidar BEV ±32m]
      [mode scores]     [x(t)]      (show_time_plot=False → scores span the row)
    """
    from matplotlib import gridspec

    cam = sample["camera_feature"]
    lid = sample["lidar_feature"][0]
    gt = sample["trajectory"].astype(np.float32)

    fig = plt.figure(figsize=(13, 11))
    fig.suptitle(title, fontsize=11)
    gs = gridspec.GridSpec(3, 2, figure=fig, height_ratios=[1.05, 1.05, 1.0], hspace=0.28, wspace=0.22)

    # --- left: camera (top) + lidar BEV map (bottom), stacked in old camera column ---
    ax_cam = fig.add_subplot(gs[0, 0])
    ax_cam.imshow(np.transpose(cam, (1, 2, 0)))
    ax_cam.set_title("camera_feature")
    ax_cam.axis("off")

    ax_bev = fig.add_subplot(gs[1, 0])
    # Camera-aligned BEV: +x forward (up), +y left (image-left) → invert horiz axis.
    ax_bev.set_xlim(32, -32)
    ax_bev.set_ylim(-32, 32)
    ax_bev.set_aspect("equal")
    ax_bev.set_xlabel("y (m, left+ = image left)")
    ax_bev.set_ylabel("x (m, forward+)")
    ys = np.linspace(-32, 32, lid.shape[1])
    xs = np.linspace(-32, 32, lid.shape[0])
    Y, X = np.meshgrid(ys, xs)
    ax_bev.pcolormesh(Y, X, lid, cmap="gray_r", shading="auto", zorder=0)
    best_i = _draw_trajs_on_ax(ax_bev, gt, modes, scores, with_colorbar=False)
    ax_bev.legend(loc="upper left", fontsize=7, framealpha=0.9)
    nmode = 0 if modes is None else len(modes)
    ax_bev.set_title(f"lidar BEV ±32m ({nmode} modes)")
    ax_bev.grid(True, alpha=0.2)

    # --- right top spanning 2 rows: traj-only, scaled by traj length, no map ---
    ax_traj = fig.add_subplot(gs[0:2, 1])
    ax_traj.set_aspect("equal")
    ax_traj.set_xlabel("y (m, left+ = image left)")
    ax_traj.set_ylabel("x (m, forward+)")
    ax_traj.set_facecolor("#f7f7f7")
    best_i = _draw_trajs_on_ax(ax_traj, gt, modes, scores, with_colorbar=True, fig=fig)
    y0, y1, x0, x1 = _traj_zoom_limits(gt, modes, best_i)
    # Same camera-aligned lateral axis as BEV (+y on the left).
    ax_traj.set_xlim(y1, y0)
    ax_traj.set_ylim(x0, x1)
    ax_traj.legend(loc="upper left", fontsize=8, framealpha=0.92)
    ax_traj.set_title("trajectory (no map, zoomed to GT+best)")
    ax_traj.grid(True, alpha=0.35)

    # --- mode score bars ---
    ax = fig.add_subplot(gs[2, 0] if show_time_plot else gs[2, :])
    if scores is not None:
        idx = np.arange(len(scores))
        colors = [cm.plasma(Normalize(vmin=scores.min(), vmax=scores.max() + 1e-8)(s)) for s in scores]
        ax.bar(idx, scores, color=colors, edgecolor="k", linewidth=0.3)
        if best_i is not None:
            ax.bar(best_i, scores[best_i], color="cyan", edgecolor="k", linewidth=1.0)
        ax.set_xlabel("mode id")
        ax.set_ylabel("softmax score")
        ax.set_title(f"mode scores (best={best_i}, p={scores[best_i]:.3f})")
        top = np.argsort(scores)[::-1][:top_k_labels]
        for i in top:
            ax.text(i, scores[i] + 0.01 * max(scores.max(), 1e-6), f"{i}", ha="center", fontsize=7)
    else:
        ax.set_title("mode scores (n/a)")
        ax.axis("off")
    ax.grid(True, axis="y", alpha=0.3)

    if not show_time_plot:
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=140)
        plt.close(fig)
        print(f"[wrote] {out}")
        return

    # --- top modes vs time ---
    ax = fig.add_subplot(gs[2, 1])
    t = np.arange(1, len(gt) + 1) * 0.5
    ax.plot(t, gt[:, 0], "g-o", lw=2.2, label="GT x")
    ax.plot(t, gt[:, 1], "g--", lw=1.5, label="GT y")
    if modes is not None and scores is not None:
        top = np.argsort(scores)[::-1][:5]
        for rank, i in enumerate(top):
            style = "-" if rank == 0 else ":"
            ax.plot(t, modes[i, :, 0], style, alpha=0.85 if rank == 0 else 0.55, label=f"m{i} x p={scores[i]:.2f}")
        pred = modes[best_i]
        ade = float(np.linalg.norm(pred[:, :2] - gt[:, :2], axis=-1).mean())
        ades = [float(np.linalg.norm(modes[i, :, :2] - gt[:, :2], axis=-1).mean()) for i in range(len(modes))]
        min_ade = float(np.min(ades))
        min_i = int(np.argmin(ades))
        ax.set_title(f"top-5 modes x(t)  ADE@best={ade:.2f}m  minADE@20={min_ade:.2f}m (m{min_i})")
    else:
        ax.set_title("traj vs time")
    ax.set_xlabel("t (s)")
    ax.set_ylabel("meters")
    ax.legend(fontsize=7, loc="best")
    ax.grid(True, alpha=0.3)

    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f"[wrote] {out}")


def build_model(ckpt: Path, device: torch.device) -> V2TransfuserModel:
    cfg = TransfuserConfig()
    cfg.bkb_path = str(DDV2 / "ckpts/resnet34.a1_in1k/pytorch_model.bin")
    cfg.plan_anchor_path = str(DDV2 / "kmeans_navsim_traj_20.npy")
    model = V2TransfuserModel(cfg)
    state = torch.load(ckpt, map_location="cpu")
    sd = state["state_dict"] if isinstance(state, dict) and "state_dict" in state else state
    sd = {k.replace("agent.", "").replace("_transfuser_model.", ""): v for k, v in sd.items()}
    missing, unexpected = model.load_state_dict(sd, strict=False)
    print(f"[ckpt] {ckpt.name}: loaded, missing={len(missing)} unexpected={len(unexpected)}")
    model.to(device).eval()
    return model


@torch.no_grad()
def infer_all_modes(model: V2TransfuserModel, sample: dict, device: torch.device):
    """Return best (8,3), modes (M,8,3), probs (M,)."""
    feats = {
        "camera_feature": torch.from_numpy(sample["camera_feature"]).float().unsqueeze(0).to(device),
        "lidar_feature": torch.from_numpy(sample["lidar_feature"]).float().unsqueeze(0).to(device),
        "status_feature": torch.from_numpy(sample["status_feature"]).float().unsqueeze(0).to(device),
    }

    # Backbone + decoder queries (same as V2TransfuserModel.forward)
    camera_feature = feats["camera_feature"]
    lidar_feature = feats["lidar_feature"]
    status_feature = feats["status_feature"]
    batch_size = status_feature.shape[0]

    bev_feature_upscale, bev_feature, _ = model._backbone(camera_feature, lidar_feature)
    cross_bev_feature = bev_feature_upscale
    bev_spatial_shape = bev_feature_upscale.shape[2:]
    concat_cross_bev_shape = bev_feature.shape[2:]
    bev_feature = model._bev_downscale(bev_feature).flatten(-2, -1)
    bev_feature = bev_feature.permute(0, 2, 1)
    status_encoding = model._status_encoding(status_feature)

    keyval = torch.concatenate([bev_feature, status_encoding[:, None]], dim=1)
    keyval = keyval + model._keyval_embedding.weight[None, ...]

    concat_cross_bev = (
        keyval[:, :-1].permute(0, 2, 1).contiguous().view(batch_size, -1, concat_cross_bev_shape[0], concat_cross_bev_shape[1])
    )
    concat_cross_bev = F.interpolate(concat_cross_bev, size=bev_spatial_shape, mode="bilinear", align_corners=False)
    cross_bev_feature = torch.cat([concat_cross_bev, cross_bev_feature], dim=1)
    cross_bev_feature = model.bev_proj(cross_bev_feature.flatten(-2, -1).permute(0, 2, 1))
    cross_bev_feature = (
        cross_bev_feature.permute(0, 2, 1).contiguous().view(batch_size, -1, bev_spatial_shape[0], bev_spatial_shape[1])
    )
    query = model._query_embedding.weight[None, ...].repeat(batch_size, 1, 1)
    query_out = model._tf_decoder(query, keyval)
    trajectory_query, agents_query = query_out.split(model._query_splits, dim=1)

    # TrajectoryHead.forward_test but keep all modes
    head = model._trajectory_head
    step_num = 2
    bs = trajectory_query.shape[0]
    head.diffusion_scheduler.set_timesteps(1000, device)
    step_ratio = 20 / step_num
    roll_timesteps = (np.arange(0, step_num) * step_ratio).round()[::-1].copy().astype(np.int64)
    roll_timesteps = torch.from_numpy(roll_timesteps).to(device)

    plan_anchor = head.plan_anchor.unsqueeze(0).repeat(bs, 1, 1, 1)
    img = head.norm_odo(plan_anchor)
    noise = torch.randn(img.shape, device=device)
    trunc_timesteps = torch.ones((bs,), device=device, dtype=torch.long) * 8
    img = head.diffusion_scheduler.add_noise(original_samples=img, noise=noise, timesteps=trunc_timesteps)
    ego_fut_mode = img.shape[1]
    status_enc = status_encoding[:, None]

    poses_reg = poses_cls = None
    for k in roll_timesteps:
        x_boxes = torch.clamp(img, min=-1, max=1)
        noisy_traj_points = head.denorm_odo(x_boxes)
        traj_pos_embed = gen_sineembed_for_position(noisy_traj_points, hidden_dim=64)
        traj_pos_embed = traj_pos_embed.flatten(-2)
        traj_feature = head.plan_anchor_encoder(traj_pos_embed)
        traj_feature = traj_feature.view(bs, ego_fut_mode, -1)

        timesteps = k if torch.is_tensor(k) else torch.tensor([k], dtype=torch.long, device=device)
        if timesteps.ndim == 0:
            timesteps = timesteps[None]
        timesteps = timesteps.expand(bs)
        time_embed = head.time_mlp(timesteps).view(bs, 1, -1)

        poses_reg_list, poses_cls_list = head.diff_decoder(
            traj_feature,
            noisy_traj_points,
            cross_bev_feature,
            bev_spatial_shape,
            agents_query,
            trajectory_query,
            time_embed,
            status_enc,
            None,
        )
        poses_reg = poses_reg_list[-1]
        poses_cls = poses_cls_list[-1]
        x_start = head.norm_odo(poses_reg[..., :2])
        img = head.diffusion_scheduler.step(model_output=x_start, timestep=k, sample=img).prev_sample

    # poses_reg: (B,M,T,3), poses_cls: (B,M)
    probs = torch.softmax(poses_cls[0], dim=-1)
    modes = poses_reg[0].detach().cpu().numpy()
    scores = probs.detach().cpu().numpy()
    best_i = int(scores.argmax())
    best = modes[best_i]
    return best, modes, scores


def pick_samples(data_dir: Path, n: int) -> list[Path]:
    meta = json.loads((data_dir / "meta.json").read_text())
    samples = meta["samples"]
    by_seq: dict[str, list] = {}
    for s in samples:
        by_seq.setdefault(s["seq"], []).append(s)
    picks: list[Path] = []
    seqs = sorted(by_seq)
    per_seq = max(1, (n + len(seqs) - 1) // len(seqs))
    for seq in seqs:
        lst = by_seq[seq]
        for j in range(per_seq):
            idx = int(round(j * (len(lst) - 1) / max(per_seq - 1, 1))) if per_seq > 1 else len(lst) // 2
            p = data_dir / lst[idx]["file"]
            if p.is_file() and p not in picks:
                picks.append(p)
            if len(picks) >= n:
                return picks
    return picks[:n]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=Path, default=ROOT / "datasets/ddv2_rellis_full")
    ap.add_argument("--out_dir", type=Path, default=ROOT / "log/rellis_viz")
    ap.add_argument("--ckpt", type=Path, default=DDV2 / "ckpts/diffusiondrivev2_rl.ckpt")
    ap.add_argument("--num", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--samples", type=str, default="", help="comma sample stems or paths, overrides --num pick")
    ap.add_argument("--viz_only", action="store_true")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[device] {device}")

    if args.samples:
        paths = []
        for tok in args.samples.split(","):
            tok = tok.strip()
            p = Path(tok)
            if not p.is_file():
                for split in ("train", "val"):
                    cand = args.data_dir / split / f"{tok}.npz"
                    if not cand.suffix == ".npz":
                        cand = args.data_dir / split / tok
                    if cand.is_file():
                        p = cand
                        break
                    cand = args.data_dir / split / f"{Path(tok).stem}.npz"
                    if cand.is_file():
                        p = cand
                        break
            paths.append(p)
    else:
        paths = pick_samples(args.data_dir, args.num)
    print(f"[samples] {len(paths)}")

    model = None if args.viz_only else build_model(args.ckpt, device)

    metrics = []
    for p in paths:
        sample = load_sample(p)
        seq = str(sample["seq"])
        fidx = int(sample["frame_idx"])
        modes = scores = pred = None
        if model is not None:
            pred, modes, scores = infer_all_modes(model, sample, device)
            gt = sample["trajectory"]
            ade = float(np.linalg.norm(pred[:, :2] - gt[:, :2], axis=-1).mean())
            fde = float(np.linalg.norm(pred[-1, :2] - gt[-1, :2]))
            min_ade = float(
                min(np.linalg.norm(modes[i, :, :2] - gt[:, :2], axis=-1).mean() for i in range(len(modes)))
            )
            metrics.append((p.name, seq, fidx, ade, fde, min_ade, int(scores.argmax())))
            print(
                f"  {p.name} seq={seq} frame={fidx} "
                f"ADE@best={ade:.2f}m FDE={fde:.2f}m minADE@20={min_ade:.2f}m best_mode={scores.argmax()} "
                f"entropy={float(-(scores * np.log(scores + 1e-12)).sum()):.2f}"
            )

        title = f"{p.stem}  seq={seq} frame={fidx}"
        if pred is not None:
            title += f"  | {args.ckpt.name}  | 20 modes"
        make_panel(sample, modes, scores, title, args.out_dir / f"{p.stem}_modes.png")

    if metrics:
        summary = args.out_dir / "infer_modes_summary.txt"
        lines = [f"ckpt={args.ckpt}", f"n={len(metrics)}"]
        lines.append(
            f"mean ADE@best={np.mean([m[3] for m in metrics]):.3f}m  "
            f"mean FDE={np.mean([m[4] for m in metrics]):.3f}m  "
            f"mean minADE@20={np.mean([m[5] for m in metrics]):.3f}m"
        )
        for m in metrics:
            lines.append(
                f"{m[0]}\tseq={m[1]}\tframe={m[2]}\tADE={m[3]:.3f}\tFDE={m[4]:.3f}\tminADE={m[5]:.3f}\tbest={m[6]}"
            )
        summary.write_text("\n".join(lines) + "\n")
        print(f"[summary] {summary}")
        print(lines[2])


if __name__ == "__main__":
    main()
