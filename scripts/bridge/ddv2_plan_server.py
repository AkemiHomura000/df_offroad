#!/usr/bin/env python3
"""DDV2 trajectory server for Verti-Bench (run in ddv2 env).

Protocol: length-prefixed npz with camera/lidar/status → trajectory (8,3).

  source ddv2_env.sh
  python scripts/bridge/ddv2_plan_server.py \\
    --ckpt log/rellis_il_finetune_v3/ckpts/best.ckpt --port 8765
"""

from __future__ import annotations

import argparse
import socket
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path("/home/rm/Desktop/df_offroad")
DDV2 = ROOT / "DiffusionDriveV2"
sys.path.insert(0, str(ROOT / "scripts" / "bridge"))
sys.path.insert(0, str(DDV2))

from ddv2_plan_protocol import recv_npz, send_npz  # noqa: E402
from nuplan_min_stub import install as install_nuplan_stub  # noqa: E402

install_nuplan_stub()

from navsim.agents.diffusiondrive.transfuser_config import TransfuserConfig  # noqa: E402
from navsim.agents.diffusiondrive.transfuser_model_v2 import V2TransfuserModel  # noqa: E402


def load_model(
    ckpt: Path,
    device: torch.device,
    plan_anchor_path: Path | None = None,
) -> V2TransfuserModel:
    cfg = TransfuserConfig()
    cfg.bkb_path = str(DDV2 / "ckpts/resnet34.a1_in1k/pytorch_model.bin")
    anchor = plan_anchor_path or (DDV2 / "kmeans_navsim_traj_20.npy")
    cfg.plan_anchor_path = str(anchor)
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
    print(f"[server] ckpt={ckpt} missing={len(missing)} unexpected={len(unexpected)}")
    if isinstance(state, dict) and "ade" in state:
        print(f"[server] meta ade={state.get('ade')} epoch={state.get('epoch')}")
    # Prefer explicit anchors when provided (Verti FT), else keep ckpt anchors.
    if plan_anchor_path is not None and plan_anchor_path.is_file():
        anchors = torch.as_tensor(np.load(plan_anchor_path), dtype=torch.float32)
        with torch.no_grad():
            model._trajectory_head.plan_anchor.copy_(anchors)
        print(f"[server] plan_anchor forced from {plan_anchor_path} shape={tuple(anchors.shape)}")
    model.to(device).eval()
    return model


@torch.no_grad()
def infer(model: V2TransfuserModel, obs: dict, device: torch.device) -> tuple[np.ndarray, float]:
    import time as _time

    feats = {
        "camera_feature": torch.from_numpy(np.asarray(obs["camera_feature"])).float().unsqueeze(0).to(device),
        "lidar_feature": torch.from_numpy(np.asarray(obs["lidar_feature"])).float().unsqueeze(0).to(device),
        "status_feature": torch.from_numpy(np.asarray(obs["status_feature"])).float().unsqueeze(0).to(device),
    }
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = _time.perf_counter()
    out = model(feats)
    if device.type == "cuda":
        torch.cuda.synchronize()
    ms = (_time.perf_counter() - t0) * 1000.0
    traj = out["trajectory"][0].detach().cpu().numpy().astype(np.float32)  # (8,3)
    return traj, ms


def handle_client(conn: socket.socket, model: V2TransfuserModel, device: torch.device) -> None:
    with conn:
        while True:
            try:
                obs = recv_npz(conn)
            except ConnectionError:
                break
            if "shutdown" in obs:
                send_npz(conn, {"ok": np.array([1], dtype=np.int32)})
                break
            traj, ms = infer(model, obs, device)
            send_npz(
                conn,
                {
                    "trajectory": traj,
                    "infer_ms": np.array([ms], dtype=np.float32),
                },
            )

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--ckpt",
        type=Path,
        default=ROOT / "log/verti_il_finetune_v3_density/ckpts/best.ckpt",
    )
    ap.add_argument(
        "--plan_anchor_path",
        type=Path,
        default=None,
        help="optional anchors npy (e.g. kmeans_verti_traj_20.npy for Verti FT)",
    )
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8765)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(args.ckpt, device, plan_anchor_path=args.plan_anchor_path)

    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind((args.host, args.port))
    srv.listen(1)
    print(f"[server] listening on {args.host}:{args.port}  device={device}")
    try:
        while True:
            conn, addr = srv.accept()
            print(f"[server] client {addr}")
            handle_client(conn, model, device)
            print("[server] client disconnected")
    except KeyboardInterrupt:
        print("\n[server] stop")
    finally:
        srv.close()


if __name__ == "__main__":
    main()
