#!/usr/bin/env python3
"""ddv2: load Chrono obs → DiffusionDrive V2TransfuserModel forward → trajectory."""

from __future__ import annotations

import os
import sys

import numpy as np
import torch

ROOT = "/home/rm/Desktop/df_offroad"
DDV2 = f"{ROOT}/DiffusionDriveV2"
sys.path.insert(0, f"{ROOT}/scripts/bridge")
sys.path.insert(0, DDV2)

from nuplan_min_stub import install as install_nuplan_stub

install_nuplan_stub()

from navsim.agents.diffusiondrive.transfuser_config import TransfuserConfig
from navsim.agents.diffusiondrive.transfuser_model_v2 import V2TransfuserModel


def main():
    obs_path = os.environ.get("BRIDGE_OBS", f"{ROOT}/log/bridge/obs.npz")
    traj_path = os.environ.get("BRIDGE_TRAJ", f"{ROOT}/log/bridge/traj.npz")
    os.makedirs(os.path.dirname(traj_path), exist_ok=True)

    data = np.load(obs_path)
    cam = torch.from_numpy(data["camera_feature"]).float().unsqueeze(0)
    lid = torch.from_numpy(data["lidar_feature"]).float().unsqueeze(0)
    status = torch.from_numpy(data["status_feature"]).float().unsqueeze(0)
    print(f"[ddv2] loaded {obs_path}")
    print(f"  camera {tuple(cam.shape)} lidar {tuple(lid.shape)} status {tuple(status.shape)}")

    cfg = TransfuserConfig()
    cfg.plan_anchor_path = f"{DDV2}/kmeans_navsim_traj_20.npy"
    # Prefer timm download; overlay unused unless needed
    cfg.bkb_path = ""

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[ddv2] building V2TransfuserModel on {device} (random init OK for bridge demo)")
    model = V2TransfuserModel(cfg).to(device)
    model.eval()

    features = {
        "camera_feature": cam.to(device),
        "lidar_feature": lid.to(device),
        "status_feature": status.to(device),
    }
    with torch.no_grad():
        out = model(features)

    traj = out["trajectory"].detach().cpu().numpy()  # (1, 8, 3)
    print(f"[ddv2] trajectory shape {traj.shape}  (B, T=8, xyθ)")
    print("  waypoints (x, y, heading):")
    for i, (x, y, h) in enumerate(traj[0]):
        print(f"    t{i}: ({x:+.3f}, {y:+.3f}, {h:+.3f})")

    np.savez_compressed(traj_path, trajectory=traj[0], obs_path=obs_path)
    print(f"[ddv2] wrote {traj_path}")
    print("[ddv2] bridge OK: Chrono obs → DiffusionDrive forward → trajectory")


if __name__ == "__main__":
    main()
