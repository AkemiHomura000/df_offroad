#!/usr/bin/env python3
"""
Imitation-learning fine-tune of DiffusionDrive (V2TransfuserModel) on RELLIS.

Aligned with official DiffusionDrive / TransfuserAgent training in this repo:
  - LossComputer: focal cls + L1 reg on anchor-matched mode (per decoder layer)
  - Outer weight: config.trajectory_weight (default 12)
  - AdamW + image_encoder lr_mult=0.5
  - WarmupCosLR (official scheduler)
  - Optional AMP (official uses 16-mixed)

RELLIS adaptation (off by default unless --anchor_keep > 0):
  Non-matched modes get a weak L1 pull toward their plan_anchor xy so they
  do not collapse into zigzag garbage when GT never covers those anchors.
  Official NAVSIM IL does not need this because diverse GT hits all anchors.

NOTE: DiffusionDriveV2's documented `run_training.py agent=diffusiondrivev2_rl_agent`
is RL/GRPO on top of the IL ckpt and needs PDMS — not used here.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter

ROOT = Path("/home/rm/Desktop/df_offroad")
DDV2 = ROOT / "DiffusionDriveV2"
sys.path.insert(0, str(ROOT / "scripts" / "bridge"))
sys.path.insert(0, str(ROOT / "scripts" / "rellis"))
sys.path.insert(0, str(DDV2))

from nuplan_min_stub import install as install_nuplan_stub  # noqa: E402

install_nuplan_stub()

from navsim.agents.diffusiondrive.modules.multimodal_loss import LossComputer  # noqa: E402
from navsim.agents.diffusiondrive.modules.scheduler import WarmupCosLR  # noqa: E402
from navsim.agents.diffusiondrive.transfuser_config import TransfuserConfig  # noqa: E402
from navsim.agents.diffusiondrive.transfuser_model_v2 import V2TransfuserModel  # noqa: E402

import viz_infer_rellis as viz  # noqa: E402


class LossComputerOfficial(LossComputer):
    """Official LossComputer + optional anchor-keep for unmatched modes (RELLIS)."""

    def __init__(self, config: TransfuserConfig, anchor_keep_weight: float = 0.0):
        super().__init__(config)
        self.anchor_keep_weight = float(anchor_keep_weight)

    def forward(self, poses_reg, poses_cls, targets, plan_anchor):
        bs, num_mode, ts, d = poses_reg.shape
        target_traj = targets["trajectory"]
        dist = torch.linalg.norm(target_traj.unsqueeze(1)[..., :2] - plan_anchor, dim=-1)
        dist = dist.mean(dim=-1)
        mode_idx = torch.argmin(dist, dim=-1)  # (B,)

        # --- official matched-mode loss ---
        base = super().forward(poses_reg, poses_cls, targets, plan_anchor)

        if self.anchor_keep_weight <= 0:
            return base

        # --- RELLIS: keep unmatched modes near their anchors (xy only) ---
        # poses_reg: (B,M,T,3), plan_anchor: (B,M,T,2)
        keep = F.l1_loss(poses_reg[..., :2], plan_anchor, reduction="none").mean(dim=(-1, -2))  # (B,M)
        mask = torch.ones_like(keep)
        mask.scatter_(1, mode_idx.unsqueeze(1), 0.0)
        denom = mask.sum().clamp_min(1.0)
        keep_loss = (keep * mask).sum() / denom
        return base + self.anchor_keep_weight * keep_loss


class RellisDDV2Dataset(Dataset):
    def __init__(self, data_dir: Path, split: str):
        self.split_dir = data_dir / split
        meta = json.loads((data_dir / "meta.json").read_text())
        files = []
        for s in meta["samples"]:
            if s["file"].startswith(f"{split}/"):
                p = data_dir / s["file"]
                if p.is_file():
                    files.append(p)
        if not files:
            files = sorted(self.split_dir.glob("sample_*.npz"))
        self.files = files
        if not self.files:
            raise FileNotFoundError(f"no samples under {self.split_dir}")

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> dict:
        p = self.files[idx]
        d = np.load(p)
        return {
            "camera_feature": torch.from_numpy(d["camera_feature"]).float(),
            "lidar_feature": torch.from_numpy(d["lidar_feature"]).float(),
            "status_feature": torch.from_numpy(d["status_feature"]).float(),
            "trajectory": torch.from_numpy(d["trajectory"]).float(),
            "path": str(p),
            "seq": str(d["seq"]),
            "frame_idx": int(d["frame_idx"]),
        }


def collate(batch: list) -> dict:
    return {
        "camera_feature": torch.stack([b["camera_feature"] for b in batch], 0),
        "lidar_feature": torch.stack([b["lidar_feature"] for b in batch], 0),
        "status_feature": torch.stack([b["status_feature"] for b in batch], 0),
        "trajectory": torch.stack([b["trajectory"] for b in batch], 0),
        "path": [b["path"] for b in batch],
        "seq": [b["seq"] for b in batch],
        "frame_idx": [b["frame_idx"] for b in batch],
    }


def load_plan_anchors(path: Path) -> torch.Tensor:
    anchors = np.load(path).astype(np.float32)
    if anchors.ndim != 3 or anchors.shape[1:] != (8, 2):
        raise ValueError(f"plan anchors must be (K,8,2), got {anchors.shape} from {path}")
    return torch.from_numpy(anchors)


def build_model(
    ckpt: Path | None,
    device: torch.device,
    anchor_keep: float,
    plan_anchor_path: Path,
) -> tuple[V2TransfuserModel, TransfuserConfig]:
    cfg = TransfuserConfig()
    cfg.bkb_path = str(DDV2 / "ckpts/resnet34.a1_in1k/pytorch_model.bin")
    cfg.plan_anchor_path = str(plan_anchor_path)
    model = V2TransfuserModel(cfg)
    # replace loss computer (official + optional anchor keep)
    model._trajectory_head.loss_computer = LossComputerOfficial(cfg, anchor_keep_weight=anchor_keep)

    scratch = ckpt is None or str(ckpt) in ("", "none", "None")
    if scratch:
        # ImageNet via bkb_path for image ResNet; lidar / TF / traj head stay random.
        print("[ckpt] from-scratch (no DiffusionDrive / NAVSIM weights)")
    else:
        state = torch.load(ckpt, map_location="cpu")
        if isinstance(state, dict) and "model_state_dict" in state:
            sd = state["model_state_dict"]
        elif isinstance(state, dict) and "state_dict" in state:
            sd = state["state_dict"]
        else:
            sd = state
        sd = {k.replace("agent.", "").replace("_transfuser_model.", ""): v for k, v in sd.items()}
        missing, unexpected = model.load_state_dict(sd, strict=False)
        print(f"[ckpt] {ckpt}: missing={len(missing)} unexpected={len(unexpected)}")
    # Always force the requested plan_anchor bank (ckpt or random init both get this).
    anchors = load_plan_anchors(plan_anchor_path)
    with torch.no_grad():
        model._trajectory_head.plan_anchor.copy_(anchors)
    print(f"[anchor] forced from {plan_anchor_path} shape={tuple(anchors.shape)}")
    print(
        f"[loss] trajectory_weight={cfg.trajectory_weight} "
        f"cls_w={cfg.trajectory_cls_weight} reg_w={cfg.trajectory_reg_weight} "
        f"anchor_keep={anchor_keep}"
    )
    return model.to(device), cfg


def build_optimizer(model: V2TransfuserModel, lr: float, weight_decay: float):
    """Match TransfuserAgent.get_coslr_optimizers param groups (image_encoder lr_mult=0.5)."""
    backbone, others = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if "image_encoder" in name:
            backbone.append(p)
        else:
            others.append(p)
    opt = torch.optim.AdamW(
        [
            {"params": others, "lr": lr},
            {"params": backbone, "lr": lr * 0.5},
        ],
        lr=lr,
        weight_decay=weight_decay,
    )
    return opt


@torch.no_grad()
def eval_metrics(model: V2TransfuserModel, loader: DataLoader, device: torch.device, max_batches: int = 0) -> dict:
    model.eval()
    ades, fdes, min_ades = [], [], []
    n_batches = 0
    for batch in loader:
        feats = {
            "camera_feature": batch["camera_feature"].to(device),
            "lidar_feature": batch["lidar_feature"].to(device),
            "status_feature": batch["status_feature"].to(device),
        }
        gt = batch["trajectory"].to(device)
        pred = model(feats)["trajectory"]
        ade = torch.linalg.norm(pred[..., :2] - gt[..., :2], dim=-1).mean(dim=-1)
        fde = torch.linalg.norm(pred[:, -1, :2] - gt[:, -1, :2], dim=-1)
        ades.extend(ade.cpu().tolist())
        fdes.extend(fde.cpu().tolist())

        sample = {
            "camera_feature": batch["camera_feature"][0].cpu().numpy(),
            "lidar_feature": batch["lidar_feature"][0].cpu().numpy(),
            "status_feature": batch["status_feature"][0].cpu().numpy(),
            "trajectory": batch["trajectory"][0].cpu().numpy(),
        }
        _, modes, _ = viz.infer_all_modes(model, sample, device)
        g = sample["trajectory"]
        min_ade = min(float(np.linalg.norm(modes[i, :, :2] - g[:, :2], axis=-1).mean()) for i in range(len(modes)))
        min_ades.append(min_ade)

        n_batches += 1
        if max_batches and n_batches >= max_batches:
            break
    model.train()
    return {
        "ade": float(np.mean(ades)) if ades else float("nan"),
        "fde": float(np.mean(fdes)) if fdes else float("nan"),
        "min_ade20": float(np.mean(min_ades)) if min_ades else float("nan"),
    }


def plot_curves(history: dict, out_path: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    ax = axes[0]
    if history["train_loss"]:
        ax.plot(history["train_step"], history["train_loss"], "b-", alpha=0.3, label="step")
    if history["epoch_train_loss"]:
        ax.plot(history["epoch"], history["epoch_train_loss"], "b-o", label="epoch")
    ax.set_title("train loss (weighted)")
    ax.set_xlabel("step / epoch")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

    ax = axes[1]
    if history["epoch_val_ade"]:
        ax.plot(history["epoch"], history["epoch_val_ade"], "r-o", label="ADE@best")
        ax.plot(history["epoch"], history["epoch_val_fde"], "m--", label="FDE")
        ax.plot(history["epoch"], history["epoch_val_minade"], "g-s", label="minADE@20")
    ax.set_title("validation")
    ax.set_xlabel("epoch")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = axes[2]
    if history["epoch_lr"]:
        ax.plot(history["epoch"], history["epoch_lr"], "k-o")
    ax.set_title("lr (group0)")
    ax.set_xlabel("epoch")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def save_val_viz(model, files, device, out_dir: Path, epoch: int, tag: str) -> None:
    model.eval()
    out_dir.mkdir(parents=True, exist_ok=True)
    for p in files:
        sample = viz.load_sample(p)
        _, modes, scores = viz.infer_all_modes(model, sample, device)
        title = f"ep{epoch} {tag} | {p.stem} seq={sample['seq']} frame={int(sample['frame_idx'])}"
        viz.make_panel(sample, modes, scores, title, out_dir / f"ep{epoch:03d}_{p.stem}_modes.png")
    model.train()


def main():
    ap = argparse.ArgumentParser(description="Official-style DiffusionDrive IL on RELLIS")
    ap.add_argument("--data_dir", type=Path, default=ROOT / "datasets/ddv2_rellis_full")
    ap.add_argument(
        "--ckpt",
        type=str,
        default=str(DDV2 / "ckpts/diffusiondrive_navsim_88p1_PDMS"),
        help="IL init weights, or 'none' for from-scratch (ImageNet image backbone only)",
    )
    ap.add_argument("--out_dir", type=Path, default=ROOT / "log/rellis_il_finetune_v2")
    # official IL: lr=6e-4, epochs=100, warmup=3; finetune defaults are milder
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-4, help="official from-scratch uses 6e-4")
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--warmup_epochs", type=int, default=2)
    ap.add_argument("--min_lr", type=float, default=1e-6)
    ap.add_argument("--anchor_keep", type=float, default=1.0, help="0=pure official; >0 keeps unmatched modes near anchors")
    ap.add_argument(
        "--plan_anchor_path",
        type=Path,
        default=DDV2 / "kmeans_navsim_traj_20.npy",
        help="(K,8,2) npy; re-applied after ckpt load so old anchors are not restored",
    )
    ap.add_argument("--amp", action="store_true", help="use cuda amp (official 16-mixed)")
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--val_every", type=int, default=1)
    ap.add_argument("--viz_every", type=int, default=2)
    ap.add_argument("--val_batches", type=int, default=20)
    ap.add_argument("--max_train_steps", type=int, default=0)
    ap.add_argument("--grad_clip", type=float, default=0.0, help="official default 0; use 1.0 if from-scratch unstable")
    ap.add_argument(
        "--nan_abort_after",
        type=int,
        default=50,
        help="abort epoch/training after this many consecutive non-finite losses (0=never)",
    )
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--resume",
        type=str,
        default="",
        help="resume path, or 'last'/'best' under out_dir/ckpts. "
        "--epochs is the total target epoch (e.g. resume ep20 + --epochs 40 → train 21..40)",
    )
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "viz").mkdir(exist_ok=True)
    (args.out_dir / "ckpts").mkdir(exist_ok=True)
    with open(args.out_dir / "args.json", "w") as f:
        json.dump({k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}, f, indent=2)

    train_ds = RellisDDV2Dataset(args.data_dir, "train")
    val_ds = RellisDDV2Dataset(args.data_dir, "val")
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
        pin_memory=True, collate_fn=collate, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False, num_workers=max(1, args.num_workers // 2),
        pin_memory=True, collate_fn=collate,
    )
    print(f"[data] train={len(train_ds)} val={len(val_ds)} device={device}")

    resume_path: Path | None = None
    if args.resume:
        if args.resume in ("last", "best"):
            resume_path = args.out_dir / "ckpts" / f"{args.resume}.ckpt"
        else:
            resume_path = Path(args.resume)
        if not resume_path.is_file():
            raise SystemExit(f"--resume not found: {resume_path}")

    # Init weights from official/--ckpt unless resuming (resume loads full training state below).
    # --ckpt none → from-scratch (no NAVSIM DDV2 weights).
    if resume_path is not None:
        init_ckpt: Path | None = resume_path
    elif str(args.ckpt) in ("", "none", "None"):
        init_ckpt = None
    else:
        init_ckpt = Path(args.ckpt)
    model, cfg = build_model(init_ckpt, device, args.anchor_keep, args.plan_anchor_path)
    model.train()
    opt = build_optimizer(model, args.lr, args.weight_decay)
    sched = WarmupCosLR(
        optimizer=opt,
        lr=args.lr,
        min_lr=args.min_lr,
        epochs=args.epochs,
        warmup_epochs=args.warmup_epochs,
    )
    scaler = GradScaler(enabled=args.amp and device.type == "cuda")
    writer = SummaryWriter(log_dir=str(args.out_dir / "tb"))

    history = {
        "train_step": [], "train_loss": [], "epoch": [],
        "epoch_train_loss": [], "epoch_val_ade": [], "epoch_val_fde": [],
        "epoch_val_minade": [], "epoch_lr": [],
    }
    metrics_path = args.out_dir / "metrics.jsonl"
    global_step = 0
    best_ade = float("inf")
    start_epoch = 1

    if resume_path is not None:
        rs = torch.load(resume_path, map_location="cpu")
        start_epoch = int(rs.get("epoch", 0)) + 1
        best_ade = float(rs.get("ade", best_ade)) if rs.get("ade") is not None else best_ade
        # Prefer best.ckpt's ade if resuming from last without a better record
        best_path = args.out_dir / "ckpts" / "best.ckpt"
        if best_path.is_file():
            try:
                best_ade = min(best_ade, float(torch.load(best_path, map_location="cpu").get("ade", best_ade)))
            except Exception:
                pass
        if "optimizer_state_dict" in rs:
            opt.load_state_dict(rs["optimizer_state_dict"])
        prev_epochs = None
        if isinstance(rs.get("args"), dict):
            prev_epochs = rs["args"].get("epochs")
        # Only restore sched state if total epoch budget unchanged; else rebuild schedule.
        if (
            "scheduler_state_dict" in rs
            and prev_epochs is not None
            and int(prev_epochs) == int(args.epochs)
        ):
            sched.load_state_dict(rs["scheduler_state_dict"])
        elif start_epoch > 1:
            for _ in range(start_epoch - 1):
                sched.step()
        if rs.get("scaler_state_dict") is not None and scaler.is_enabled():
            scaler.load_state_dict(rs["scaler_state_dict"])
        global_step = int(rs.get("global_step", 0))
        if "history" in rs and isinstance(rs["history"], dict):
            for k, v in rs["history"].items():
                if k in history and isinstance(v, list):
                    history[k] = list(v)
        print(
            f"[resume] {resume_path} → start_epoch={start_epoch} "
            f"(target={args.epochs}) best_ade={best_ade:.4f} global_step={global_step}"
        )
        if start_epoch > args.epochs:
            raise SystemExit(
                f"resume start_epoch={start_epoch} > --epochs={args.epochs}; "
                f"raise --epochs (total target), e.g. --epochs {start_epoch + 19}"
            )

    viz_files = [Path(val_ds.files[i]) for i in np.linspace(0, len(val_ds) - 1, num=min(4, len(val_ds)), dtype=int)]
    if resume_path is None:
        print("[viz] epoch 0 (pretrained) ...")
        save_val_viz(model, viz_files, device, args.out_dir / "viz", 0, "pretrained")

    traj_w = float(cfg.trajectory_weight)
    nan_streak = 0

    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        t0 = time.time()
        losses = []
        nan_in_epoch = 0
        aborted_nan = False
        for bi, batch in enumerate(train_loader):
            feats = {
                "camera_feature": batch["camera_feature"].to(device, non_blocking=True),
                "lidar_feature": batch["lidar_feature"].to(device, non_blocking=True),
                "status_feature": batch["status_feature"].to(device, non_blocking=True),
            }
            targets = {"trajectory": batch["trajectory"].to(device, non_blocking=True)}

            opt.zero_grad(set_to_none=True)
            with autocast(enabled=scaler.is_enabled()):
                out = model(feats, targets=targets)
                # official transfuser_loss traj term (skip agent/BEV — no GT on RELLIS)
                loss = traj_w * out["trajectory_loss"]

            if not torch.isfinite(loss):
                nan_streak += 1
                nan_in_epoch += 1
                print(f"[warn] non-finite loss at step {global_step} (streak={nan_streak}), skip")
                if args.nan_abort_after > 0 and nan_streak >= args.nan_abort_after:
                    print(
                        f"[abort] {nan_streak} consecutive non-finite losses — "
                        f"stop training (best ADE={best_ade:.4f} kept)"
                    )
                    aborted_nan = True
                    break
                continue

            nan_streak = 0
            scaler.scale(loss).backward()
            if args.grad_clip > 0:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(opt)
            scaler.update()

            lv = float(loss.detach().cpu())
            losses.append(lv)
            history["train_step"].append(global_step)
            history["train_loss"].append(lv)
            writer.add_scalar("train/loss_step", lv, global_step)
            if "trajectory_loss_dict" in out:
                for k, v in out["trajectory_loss_dict"].items():
                    writer.add_scalar(f"train/{k}", float(v.detach().cpu()), global_step)
            global_step += 1

            if bi % 20 == 0:
                print(f"  ep{epoch} step{bi}/{len(train_loader)} loss={lv:.4f}")

            if args.max_train_steps and (bi + 1) >= args.max_train_steps:
                break

        if aborted_nan:
            break

        sched.step()
        ep_loss = float(np.mean(losses)) if losses else float("nan")
        lr = float(opt.param_groups[0]["lr"])
        history["epoch"].append(epoch)
        history["epoch_train_loss"].append(ep_loss)
        history["epoch_lr"].append(lr)
        writer.add_scalar("train/loss_epoch", ep_loss, epoch)
        writer.add_scalar("train/lr", lr, epoch)

        row = {
            "epoch": epoch,
            "train_loss": ep_loss,
            "lr": lr,
            "sec": time.time() - t0,
            "nan_skips": nan_in_epoch,
        }

        diverged = (not np.isfinite(ep_loss)) or (
            len(losses) > 0 and nan_in_epoch > max(10, len(train_loader) // 5)
        )

        if args.val_every > 0 and epoch % args.val_every == 0 and not diverged:
            print(f"[val] epoch {epoch} ...")
            m = eval_metrics(model, val_loader, device, max_batches=args.val_batches)
            history["epoch_val_ade"].append(m["ade"])
            history["epoch_val_fde"].append(m["fde"])
            history["epoch_val_minade"].append(m["min_ade20"])
            writer.add_scalar("val/ade", m["ade"], epoch)
            writer.add_scalar("val/fde", m["fde"], epoch)
            writer.add_scalar("val/min_ade20", m["min_ade20"], epoch)
            row.update(m)
            print(
                f"  train_loss={ep_loss:.4f} ADE={m['ade']:.3f} FDE={m['fde']:.3f} "
                f"minADE@20={m['min_ade20']:.3f} lr={lr:.2e} ({row['sec']:.0f}s)"
            )
            if np.isfinite(m["ade"]) and m["ade"] < best_ade:
                best_ade = m["ade"]
                sd = {f"agent._transfuser_model.{k}": v.cpu() for k, v in model.state_dict().items()}
                torch.save(
                    {
                        "epoch": epoch,
                        "state_dict": sd,
                        "model_state_dict": {k: v.cpu() for k, v in model.state_dict().items()},
                        "ade": best_ade,
                        "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                    },
                    args.out_dir / "ckpts" / "best.ckpt",
                )
                print(f"  [best] ADE={best_ade:.3f}")
            if not np.isfinite(m["ade"]):
                diverged = True
        else:
            history["epoch_val_ade"].append(float("nan"))
            history["epoch_val_fde"].append(float("nan"))
            history["epoch_val_minade"].append(float("nan"))
            if diverged:
                print(
                    f"[abort] epoch {epoch} diverged "
                    f"(train_loss={ep_loss}, nan_skips={nan_in_epoch}) — keep best ADE={best_ade:.4f}"
                )

        with open(metrics_path, "a") as f:
            f.write(json.dumps(row) + "\n")
        plot_curves(history, args.out_dir / "curves.png")
        writer.flush()
        if not diverged:
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": opt.state_dict(),
                    "scheduler_state_dict": sched.state_dict(),
                    "scaler_state_dict": scaler.state_dict() if scaler.is_enabled() else None,
                    "global_step": global_step,
                    "ade": row.get("ade", best_ade),
                    "best_ade": best_ade,
                    "history": history,
                    "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                },
                args.out_dir / "ckpts" / "last.ckpt",
            )
            if args.viz_every > 0 and epoch % args.viz_every == 0:
                print(f"[viz] epoch {epoch} ...")
                try:
                    save_val_viz(model, viz_files, device, args.out_dir / "viz", epoch, "finetune")
                except Exception as e:
                    print(f"[warn] viz failed (skipped): {e}")
        else:
            break

    writer.close()
    plot_curves(history, args.out_dir / "curves.png")
    print(f"[done] {args.out_dir}")
    print(f"  TensorBoard: tensorboard --logdir {args.out_dir / 'tb'}")
    print(f"  best: {args.out_dir / 'ckpts' / 'best.ckpt'}")


if __name__ == "__main__":
    main()
