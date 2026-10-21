#!/usr/bin/env python3
"""Verti GRPO fine-tune on top of Verti IL ckpt, with VertiPDM replacing PDMS.

Rewards come from scripts/bridge/verti_pdm_scorer.get_verti_score_para
(same subscore keys as official PDMS so hard-failure advantage=-1 still applies).

Use --closed_loop to score Chrono-map kinematic rollouts (PDMSimulator-style)
instead of open-loop waypoints.

Does NOT import diffusiondrivev2_model_rl (needs full nuplan/PDMS). Instead:
  - backbone + TrajectoryHead from IL V2TransfuserModel
  - DDIMScheduler_with_logprob + GRPO advantage math from verti_grpo_utils
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter

ROOT = Path(__file__).resolve().parents[2]
DDV2 = ROOT / "DiffusionDriveV2"
sys.path.insert(0, str(ROOT / "scripts" / "bridge"))
sys.path.insert(0, str(ROOT / "scripts" / "train"))
sys.path.insert(0, str(ROOT / "scripts" / "rellis"))
sys.path.insert(0, str(DDV2))

from nuplan_min_stub import install as install_nuplan_stub  # noqa: E402

install_nuplan_stub()

from navsim.agents.diffusiondrive.modules.blocks import gen_sineembed_for_position  # noqa: E402
from navsim.agents.diffusiondrive.modules.scheduler import WarmupCosLR  # noqa: E402
from navsim.agents.diffusiondrive.transfuser_config import TransfuserConfig  # noqa: E402
from navsim.agents.diffusiondrive.transfuser_model_v2 import V2TransfuserModel  # noqa: E402

from verti_grpo_utils import (  # noqa: E402
    ChronoCLSimConfig,
    DDIMScheduler_with_logprob,
    VertiMetricCache,
    WorldMapBank,
    bezier_xyyaw,
    get_verti_score_para,
    grpo_advantages_torch,
)
from verti_metric_cache import load_index  # noqa: E402


class VertiGRPODataset(Dataset):
    def __init__(self, data_dir: Path, cache_dir: Path, split: str):
        self.data_dir = data_dir
        index = load_index(cache_dir / "index.jsonl")
        # token = "train/sample_000000" style without .npz → file path
        by_token = {r["token"]: r for r in index}
        meta = json.loads((data_dir / "meta.json").read_text())
        self.items = []
        for s in meta["samples"]:
            if not s["file"].startswith(f"{split}/"):
                continue
            token = s["file"].replace("/", "__").replace(".npz", "")
            # also try without double-underscore variant
            rec = by_token.get(token) or by_token.get(s["file"].replace(".npz", ""))
            if rec is None:
                # build token as in build_verti_metric_cache
                alt = s["file"].replace("/", "__").replace(".npz", "")
                rec = by_token.get(alt)
            if rec is None:
                continue
            p = data_dir / s["file"]
            if p.is_file():
                self.items.append((p, rec))
        if not self.items:
            raise FileNotFoundError(f"no {split} samples matched in {cache_dir}")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        p, rec = self.items[idx]
        d = np.load(p)
        return {
            "camera_feature": torch.from_numpy(d["camera_feature"]).float(),
            "lidar_feature": torch.from_numpy(d["lidar_feature"]).float(),
            "status_feature": torch.from_numpy(d["status_feature"]).float(),
            "trajectory": torch.from_numpy(d["trajectory"]).float(),
            "cache": rec,
            "path": str(p),
        }


def collate(batch):
    return {
        "camera_feature": torch.stack([b["camera_feature"] for b in batch], 0),
        "lidar_feature": torch.stack([b["lidar_feature"] for b in batch], 0),
        "status_feature": torch.stack([b["status_feature"] for b in batch], 0),
        "trajectory": torch.stack([b["trajectory"] for b in batch], 0),
        "cache": [b["cache"] for b in batch],
        "path": [b["path"] for b in batch],
    }


def load_plan_anchors(path: Path) -> torch.Tensor:
    anchors = np.load(path).astype(np.float32)
    if anchors.ndim != 3 or anchors.shape[1:] != (8, 2):
        raise ValueError(f"plan anchors must be (K,8,2), got {anchors.shape}")
    return torch.from_numpy(anchors)


def _plot_one(
    x,
    y,
    out_path: Path,
    *,
    title: str,
    xlabel: str,
    color: str = "C0",
    ylim=None,
    marker: Optional[str] = None,
) -> None:
    """Single metric → single PNG (one curve only)."""
    if not x or not y:
        return
    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    ax.plot(x, y, color=color, marker=marker, alpha=0.85, linewidth=1.2, markersize=5)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.grid(True, alpha=0.3)
    if ylim is not None:
        ax.set_ylim(*ylim)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def plot_curves(history: dict, out_path: Path) -> None:
    """One PNG per metric, x-axis = epoch only (no per-step curves)."""
    curves_dir = out_path.parent / "curves" if out_path.suffix == ".png" else Path(out_path)
    curves_dir.mkdir(parents=True, exist_ok=True)
    epochs = history.get("epoch") or []

    _plot_one(
        epochs, history.get("epoch_train_loss"), curves_dir / "loss.png",
        title="loss", xlabel="epoch", color="b", marker="o",
    )
    _plot_one(
        epochs, history.get("epoch_rl_loss"), curves_dir / "rl_loss.png",
        title="rl_loss", xlabel="epoch", color="c", marker="o",
    )
    _plot_one(
        epochs, history.get("epoch_il_loss"), curves_dir / "il_loss.png",
        title="il_loss", xlabel="epoch", color="m", marker="o",
    )
    _plot_one(
        epochs, history.get("epoch_reward"), curves_dir / "reward.png",
        title="reward", xlabel="epoch", color="g", marker="o", ylim=(0.0, 1.05),
    )
    _plot_one(
        epochs, history.get("epoch_sub_final"), curves_dir / "sub_final.png",
        title="sub_final", xlabel="epoch", color="olive", marker="o", ylim=(0.0, 1.05),
    )
    _plot_one(
        epochs, history.get("epoch_no_collision"), curves_dir / "no_collision.png",
        title="no_collision", xlabel="epoch", color="r", marker="o", ylim=(0.0, 1.05),
    )
    _plot_one(
        epochs, history.get("epoch_lr"), curves_dir / "lr.png",
        title="lr", xlabel="epoch", color="k", marker="o",
    )
    loss_png = curves_dir / "loss.png"
    if loss_png.is_file() and out_path.suffix == ".png":
        import shutil

        shutil.copyfile(loss_png, out_path)


def empty_history() -> dict:
    return {
        "epoch": [],
        "epoch_train_loss": [],
        "epoch_rl_loss": [],
        "epoch_il_loss": [],
        "epoch_reward": [],
        "epoch_sub_final": [],
        "epoch_no_collision": [],
        "epoch_lr": [],
    }


def build_model(ckpt: Path, device: torch.device, plan_anchor_path: Path) -> V2TransfuserModel:
    cfg = TransfuserConfig()
    cfg.bkb_path = str(DDV2 / "ckpts/resnet34.a1_in1k/pytorch_model.bin")
    cfg.plan_anchor_path = str(plan_anchor_path)
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
    anchors = load_plan_anchors(plan_anchor_path)
    with torch.no_grad():
        model._trajectory_head.plan_anchor.copy_(anchors)
    print(f"[ckpt] missing={len(missing)} unexpected={len(unexpected)} anchors={tuple(anchors.shape)}")

    # Freeze everything except trajectory head (official RL recipe).
    for name, p in model.named_parameters():
        if not name.startswith("_trajectory_head"):
            p.requires_grad = False
    for name, module in model.named_modules():
        if name and not name.startswith("_trajectory_head"):
            module.eval()
    model._trajectory_head.train()
    return model.to(device)


def encode_features(model: V2TransfuserModel, features: Dict[str, torch.Tensor]):
    """Run backbone → queries (no trajectory head)."""
    camera_feature = features["camera_feature"]
    lidar_feature = features["lidar_feature"]
    status_feature = features["status_feature"]
    batch_size = status_feature.shape[0]

    bev_feature_upscale, bev_feature, _ = model._backbone(camera_feature, lidar_feature)
    cross_bev_feature = bev_feature_upscale
    bev_spatial_shape = bev_feature_upscale.shape[2:]
    concat_cross_bev_shape = bev_feature.shape[2:]
    bev_feature = model._bev_downscale(bev_feature).flatten(-2, -1)
    bev_feature = bev_feature.permute(0, 2, 1)
    status_encoding = model._status_encoding(status_feature)

    keyval = torch.concatenate([bev_feature, status_encoding[:, None]], dim=1)
    keyval += model._keyval_embedding.weight[None, ...]

    concat_cross_bev = (
        keyval[:, :-1]
        .permute(0, 2, 1)
        .contiguous()
        .view(batch_size, -1, concat_cross_bev_shape[0], concat_cross_bev_shape[1])
    )
    concat_cross_bev = F.interpolate(
        concat_cross_bev, size=bev_spatial_shape, mode="bilinear", align_corners=False
    )
    cross_bev_feature = torch.cat([concat_cross_bev, cross_bev_feature], dim=1)
    cross_bev_feature = model.bev_proj(cross_bev_feature.flatten(-2, -1).permute(0, 2, 1))
    cross_bev_feature = (
        cross_bev_feature.permute(0, 2, 1)
        .contiguous()
        .view(batch_size, -1, bev_spatial_shape[0], bev_spatial_shape[1])
    )
    query = model._query_embedding.weight[None, ...].repeat(batch_size, 1, 1)
    query_out = model._tf_decoder(query, keyval)
    trajectory_query, agents_query = query_out.split(model._query_splits, dim=1)
    return {
        "trajectory_query": trajectory_query,
        "agents_query": agents_query,
        "cross_bev_feature": cross_bev_feature,
        "bev_spatial_shape": bev_spatial_shape,
        "status_encoding": status_encoding[:, None],
    }


@torch.no_grad()
def sample_rl_group(
    head,
    enc: dict,
    scheduler: DDIMScheduler_with_logprob,
    num_groups: int,
    step_num: int,
    eta: float,
):
    """Sample G groups × 20 modes; return denorm xy trajs + diffusion chain + log probs."""
    ego_query = enc["trajectory_query"]
    agents_query = enc["agents_query"]
    bev_feature = enc["cross_bev_feature"]
    bev_spatial_shape = enc["bev_spatial_shape"]
    status_encoding = enc["status_encoding"]
    bs = ego_query.shape[0]
    device = ego_query.device

    scheduler.set_timesteps(1000, device)
    step_ratio = 20 / step_num
    roll_timesteps = (np.arange(0, step_num) * step_ratio).round()[::-1].copy().astype(np.int64)
    roll_timesteps = torch.from_numpy(roll_timesteps).to(device)

    plan_anchor = head.plan_anchor.unsqueeze(0).unsqueeze(0).repeat(bs, num_groups, 1, 1, 1)
    plan_anchor = plan_anchor.view(bs, num_groups * head.ego_fut_mode, *plan_anchor.shape[3:])
    diffusion_output = head.norm_odo(plan_anchor)
    noise = torch.randn(diffusion_output.shape, device=device)
    trunc_timesteps = torch.ones((bs,), device=device, dtype=torch.long) * 8
    diffusion_output = scheduler.add_noise(
        original_samples=diffusion_output, noise=noise, timesteps=trunc_timesteps
    )

    all_log_probs = []
    all_diffusion_output = [diffusion_output]

    for k in roll_timesteps:
        x_boxes = torch.clamp(diffusion_output, min=-1, max=1)
        noisy_traj_points = head.denorm_odo(x_boxes)
        traj_pos_embed = gen_sineembed_for_position(noisy_traj_points, hidden_dim=64)
        traj_pos_embed = traj_pos_embed.flatten(-2)
        traj_feature = head.plan_anchor_encoder(traj_pos_embed)
        traj_feature = traj_feature.view(bs, num_groups * head.ego_fut_mode, -1)
        timesteps = k.expand(diffusion_output.shape[0])
        time_embed = head.time_mlp(timesteps).view(bs, 1, -1)
        poses_reg_list, poses_cls_list = head.diff_decoder(
            traj_feature,
            noisy_traj_points,
            bev_feature,
            bev_spatial_shape,
            agents_query,
            ego_query,
            time_embed,
            status_encoding,
            None,
        )
        poses_reg = poses_reg_list[-1]
        x_start = head.norm_odo(poses_reg[..., :2])
        prev_sample, log_prob, _ = scheduler.step(
            model_output=x_start, timestep=k, sample=diffusion_output, eta=eta
        )
        diffusion_output = prev_sample
        all_log_probs.append(log_prob)
        all_diffusion_output.append(prev_sample)

    all_log_probs = torch.stack(all_log_probs, dim=-1)
    all_diffusion_output = torch.stack(all_diffusion_output, dim=-1)
    xy = head.denorm_odo(diffusion_output)
    traj = bezier_xyyaw(xy)  # (B, G*N, 8, 3)
    return traj, all_diffusion_output, all_log_probs


def recompute_logprobs_and_loss(
    head,
    enc: dict,
    scheduler: DDIMScheduler_with_logprob,
    old_chain: torch.Tensor,
    advantages: torch.Tensor,
    targets: dict,
    num_groups: int,
    step_num: int,
    eta: float,
):
    """Second pass (trainable) for GRPO loss — mirrors get_rlloss."""
    ego_query = enc["trajectory_query"]
    agents_query = enc["agents_query"]
    bev_feature = enc["cross_bev_feature"]
    bev_spatial_shape = enc["bev_spatial_shape"]
    status_encoding = enc["status_encoding"]
    bs = ego_query.shape[0]
    device = ego_query.device

    scheduler.set_timesteps(1000, device)
    step_ratio = 20 / step_num
    roll_timesteps = (np.arange(0, step_num) * step_ratio).round()[::-1].copy().astype(np.int64)
    roll_timesteps = torch.from_numpy(roll_timesteps).to(device)

    chains = old_chain[..., :-1]
    chains_prev = old_chain[..., 1:]
    all_log_probs = []
    poses_reg_steps_list = []

    for i, k in enumerate(roll_timesteps):
        diffusion_output = chains[..., i]
        ego_fut_mode = diffusion_output.shape[1]
        x_boxes = torch.clamp(diffusion_output, min=-1, max=1)
        noisy_traj_points = head.denorm_odo(x_boxes)
        traj_pos_embed = gen_sineembed_for_position(noisy_traj_points, hidden_dim=64).flatten(-2)
        traj_feature = head.plan_anchor_encoder(traj_pos_embed).view(bs, ego_fut_mode, -1)
        timesteps = k.expand(bs)
        time_embed = head.time_mlp(timesteps).view(bs, 1, -1)
        poses_reg_list, _ = head.diff_decoder(
            traj_feature,
            noisy_traj_points,
            bev_feature,
            bev_spatial_shape,
            agents_query,
            ego_query,
            time_embed,
            status_encoding,
            None,
        )
        poses_reg_steps_list.append(poses_reg_list)
        poses_reg = poses_reg_list[-1]
        x_start = head.norm_odo(poses_reg[..., :2])
        _, log_prob, _ = scheduler.step(
            model_output=x_start,
            timestep=k,
            sample=diffusion_output,
            eta=eta,
            prev_sample=chains_prev[..., i],
        )
        all_log_probs.append(log_prob)

    all_log_probs = torch.stack(all_log_probs, dim=-1)
    per_token_logps = all_log_probs.view(bs, num_groups * head.ego_fut_mode, -1)
    per_token_loss = -torch.exp(per_token_logps - per_token_logps.detach()) * advantages

    mask_nz = per_token_loss != 0
    RL_loss_b = (per_token_loss * mask_nz).sum(dim=1) / mask_nz.sum(dim=1).clamp_min(1)
    RL_loss_b = RL_loss_b.mean(dim=-1)

    IL_loss_b = torch.zeros_like(RL_loss_b)
    target_traj = targets["trajectory"].unsqueeze(1).repeat(1, num_groups * head.ego_fut_mode, 1, 1)
    for poses_reg_list in poses_reg_steps_list:
        for poses_reg in poses_reg_list:
            traj_l1 = F.l1_loss(poses_reg[..., :2], target_traj[..., :2], reduction="none")
            IL_loss_b = IL_loss_b + traj_l1.mean()
    IL_loss_b = IL_loss_b / (len(poses_reg_steps_list) * len(poses_reg_steps_list[0]))

    has_positive = (advantages > 0).any(dim=2).any(dim=1)
    il_weight = torch.where(
        has_positive == 0,
        torch.tensor(1.0, device=device),
        torch.tensor(0.1, device=device),
    )
    loss = (RL_loss_b + il_weight * IL_loss_b).mean()
    return loss, RL_loss_b.mean().detach(), IL_loss_b.mean().detach()


def main():
    ap = argparse.ArgumentParser(description="Verti GRPO with VertiPDM rewards")
    ap.add_argument("--data_dir", type=Path, default=ROOT / "datasets/verti_ddv2_density")
    ap.add_argument("--cache_dir", type=Path, default=ROOT / "datasets/verti_metric_cache_density")
    ap.add_argument(
        "--ckpt",
        type=Path,
        default=ROOT / "log/verti_il_finetune_v2_density/ckpts/best.ckpt",
    )
    ap.add_argument(
        "--plan_anchor_path",
        type=Path,
        default=DDV2 / "kmeans_verti_traj_20.npy",
    )
    ap.add_argument("--out_dir", type=Path, default=ROOT / "log/verti_grpo_v1")
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch_size", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--num_groups", type=int, default=2)
    ap.add_argument("--step_num", type=int, default=5)
    ap.add_argument("--eta", type=float, default=1.0)
    ap.add_argument("--num_workers", type=int, default=2)
    ap.add_argument("--max_steps", type=int, default=0, help="0=full epoch")
    ap.add_argument(
        "--closed_loop",
        action="store_true",
        help="Chrono-map closed-loop kinematic rollout before VertiPDM score",
    )
    ap.add_argument("--cl_v_ref", type=float, default=4.0, help="CL sim cruise speed (m/s)")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    bank = WorldMapBank()
    cl_cfg = ChronoCLSimConfig(v_ref=float(args.cl_v_ref)) if args.closed_loop else None
    history = empty_history()
    curves_path = args.out_dir / "curves.png"
    history_path = args.out_dir / "history.json"

    train_ds = VertiGRPODataset(args.data_dir, args.cache_dir, "train")
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate,
        drop_last=True,
    )
    print(
        f"[data] train={len(train_ds)} batches/epoch≈{len(train_loader)} "
        f"closed_loop={args.closed_loop}"
    )

    model = build_model(args.ckpt, device, args.plan_anchor_path)
    head = model._trajectory_head
    scheduler = DDIMScheduler_with_logprob(
        num_train_timesteps=1000,
        steps_offset=1,
        beta_schedule="scaled_linear",
        prediction_type="sample",
    )

    opt = torch.optim.AdamW(
        [p for p in head.parameters() if p.requires_grad],
        lr=args.lr,
        weight_decay=1e-4,
    )
    warmup = 0 if args.epochs <= 1 else min(1, args.epochs - 1)
    sched = WarmupCosLR(opt, lr=args.lr, min_lr=1e-7, epochs=max(args.epochs, 1), warmup_epochs=warmup)
    writer = SummaryWriter(args.out_dir / "tb")

    (args.out_dir / "config.json").write_text(json.dumps({k: str(v) for k, v in vars(args).items()}, indent=2))

    global_step = 0
    for epoch in range(args.epochs):
        t0 = time.time()
        losses, rewards, rl_losses, il_losses, sub_finals, no_cols = [], [], [], [], [], []
        for batch in train_loader:
            feats = {
                "camera_feature": batch["camera_feature"].to(device),
                "lidar_feature": batch["lidar_feature"].to(device),
                "status_feature": batch["status_feature"].to(device),
            }
            targets = {"trajectory": batch["trajectory"].to(device)}
            caches = [VertiMetricCache.from_dict(c) for c in batch["cache"]]

            with torch.no_grad():
                enc = encode_features(model, feats)
                traj, chain, _ = sample_rl_group(
                    head, enc, scheduler, args.num_groups, args.step_num, args.eta
                )
                # append GT last (official convention) → first_is_gt=False in scorer
                gt = targets["trajectory"].unsqueeze(1)
                traj_with_gt = torch.cat([traj, gt], dim=1)
                scores, _, subs = get_verti_score_para(
                    traj_with_gt,
                    caches,
                    bank=bank,
                    first_is_gt=False,
                    closed_loop=args.closed_loop,
                    cl_cfg=cl_cfg,
                )
                # scores: (B, G*N+1) with GT last
                reward_group = scores[:, :-1]
                reward_gt = scores[:, -1:]
                batched_sub = {
                    k: torch.as_tensor(
                        np.vstack([d[k] for d in subs]),
                        device=device,
                        dtype=reward_group.dtype,
                    )
                    for k in subs[0]
                }
                advantages, reward = grpo_advantages_torch(
                    reward_group,
                    batched_sub,
                    reward_gt,
                    args.num_groups,
                    head.ego_fut_mode,
                    step_num=args.step_num,
                )

            # trainable pass (need grads through head; re-encode without no_grad on head)
            model._trajectory_head.train()
            enc = encode_features(model, feats)
            # detach encoder outputs so only head trains
            enc = {
                k: (v.detach() if torch.is_tensor(v) else v) for k, v in enc.items()
            }
            loss, rl_l, il_l = recompute_logprobs_and_loss(
                head,
                enc,
                scheduler,
                chain.detach(),
                advantages,
                targets,
                args.num_groups,
                args.step_num,
                args.eta,
            )
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
            opt.step()

            lv = float(loss.detach())
            rv = float(reward.detach()) if torch.is_tensor(reward) else float(reward)
            rl_v = float(rl_l)
            il_v = float(il_l)
            sub_final = (
                float(batched_sub["final"][:, :-1].mean())
                if "final" in batched_sub
                else float("nan")
            )
            no_col = (
                float(batched_sub["no_collision"][:, :-1].mean())
                if "no_collision" in batched_sub
                else float("nan")
            )
            losses.append(lv)
            rewards.append(rv)
            rl_losses.append(rl_v)
            il_losses.append(il_v)
            if np.isfinite(sub_final):
                sub_finals.append(sub_final)
            if np.isfinite(no_col):
                no_cols.append(no_col)
            writer.add_scalar("train/loss", lv, global_step)
            writer.add_scalar("train/reward", rv, global_step)
            writer.add_scalar("train/rl_loss", rl_v, global_step)
            writer.add_scalar("train/il_loss", il_v, global_step)
            if np.isfinite(sub_final):
                writer.add_scalar("train/sub_final", sub_final, global_step)
            if np.isfinite(no_col):
                writer.add_scalar("train/no_collision", no_col, global_step)

            global_step += 1
            if global_step % 20 == 0:
                print(
                    f"ep{epoch} step{global_step} loss={lv:.4f} "
                    f"reward={rv:.3f} rl={rl_v:.4f} il={il_v:.4f}",
                    flush=True,
                )
            if args.max_steps and global_step >= args.max_steps:
                break

        sched.step()
        ep_loss = float(np.mean(losses)) if losses else float("nan")
        ep_reward = float(np.mean(rewards)) if rewards else float("nan")
        ep_rl = float(np.mean(rl_losses)) if rl_losses else float("nan")
        ep_il = float(np.mean(il_losses)) if il_losses else float("nan")
        ep_sub = float(np.mean(sub_finals)) if sub_finals else float("nan")
        ep_ncol = float(np.mean(no_cols)) if no_cols else float("nan")
        lr = float(opt.param_groups[0]["lr"])
        history["epoch"].append(epoch)
        history["epoch_train_loss"].append(ep_loss)
        history["epoch_rl_loss"].append(ep_rl)
        history["epoch_il_loss"].append(ep_il)
        history["epoch_reward"].append(ep_reward)
        history["epoch_sub_final"].append(ep_sub)
        history["epoch_no_collision"].append(ep_ncol)
        history["epoch_lr"].append(lr)
        plot_curves(history, curves_path)
        history_path.write_text(json.dumps(history))

        ckpt_path = args.out_dir / "ckpts" / f"epoch_{epoch:03d}.ckpt"
        ckpt_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "epoch": epoch,
                "global_step": global_step,
                "history": history,
            },
            ckpt_path,
        )
        torch.save(
            {"model_state_dict": model.state_dict(), "history": history},
            args.out_dir / "ckpts" / "last.ckpt",
        )
        print(
            f"[epoch {epoch}] loss={ep_loss:.4f} reward={ep_reward:.3f} "
            f"time={time.time()-t0:.1f}s → {ckpt_path}  curves={curves_path}",
            flush=True,
        )
        if args.max_steps and global_step >= args.max_steps:
            break

    plot_curves(history, curves_path)
    history_path.write_text(json.dumps(history))
    writer.close()
    print("done", args.out_dir, "curves=", curves_path)


if __name__ == "__main__":
    main()
