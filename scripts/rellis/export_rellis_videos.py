#!/usr/bin/env python3
"""Export RELLIS DDV2 npz sequences to MP4 with camera + BEV GT trajectory."""

from __future__ import annotations

import argparse
import json
import subprocess
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np


CMD_LABELS = ("LEFT", "STRAIGHT", "RIGHT", "UNKNOWN")
CMD_BGR = ((80, 180, 255), (80, 220, 80), (255, 160, 80), (160, 160, 160))


def put_text(img, text, org, scale=0.55, color=(240, 240, 240), thick=1):
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thick + 2, cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)


def cmd_from_status(status: np.ndarray) -> tuple[int, str]:
    st = np.asarray(status, dtype=np.float64).reshape(-1)
    cmd = st[:4] if st.size >= 4 else np.zeros(4)
    if float(cmd.sum()) < 1e-6:
        return 3, CMD_LABELS[3]
    i = int(np.argmax(cmd))
    return i, CMD_LABELS[i]


def render_frame(
    cam: np.ndarray,
    lid: np.ndarray,
    traj: np.ndarray,
    status: np.ndarray,
    seq: str,
    frame_idx: int,
    sample_id: str,
    bev_extent: float = 32.0,
    panel_h: int = 320,
) -> np.ndarray:
    """Return BGR panel: camera | BEV+traj | ego zoom."""
    if cam.ndim == 3 and cam.shape[0] == 3:
        cam = np.transpose(cam, (1, 2, 0))
    cam = np.clip(cam, 0, 1)
    cam_u8 = (cam * 255).astype(np.uint8)
    h = panel_h
    cam_w = int(round(h * cam_u8.shape[1] / max(cam_u8.shape[0], 1)))
    cam_bgr = cv2.cvtColor(cv2.resize(cam_u8, (cam_w, h)), cv2.COLOR_RGB2BGR)

    cmd_i, cmd_name = cmd_from_status(status)
    put_text(cam_bgr, f"{seq}  frame={frame_idx}  {sample_id}", (8, 22), 0.5)
    cv2.rectangle(cam_bgr, (8, 36), (210, 78), CMD_BGR[cmd_i], -1, cv2.LINE_AA)
    put_text(cam_bgr, f"CMD: {cmd_name}", (16, 64), 0.7, (15, 15, 15), 2)

    if lid.ndim == 3:
        lid = lid[0]
    e = bev_extent
    # lid[ix,iy] with x,y ∈ [-e,e]. Camera-aligned BEV: +x up, +y left (image left).
    # row = flip x so +x at top; col = flip y so +y at left.
    bev = np.clip(lid[::-1, ::-1], 0, 1)
    bev = cv2.cvtColor((bev * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
    bev = cv2.resize(bev, (h, h), interpolation=cv2.INTER_NEAREST)

    def xy_to_px(x, y, size, span):
        # +y → image left (match forward camera); +x → image up
        u = int((span - y) / (2 * span) * (size - 1))
        v = int((size - 1) - (x + span) / (2 * span) * (size - 1))
        return u, v

    traj = np.asarray(traj, dtype=np.float64)
    pts = [xy_to_px(float(x), float(y), h, e) for x, y, *_ in traj]
    for i in range(1, len(pts)):
        cv2.line(bev, pts[i - 1], pts[i], (80, 255, 80), 2, cv2.LINE_AA)
    for p in pts:
        cv2.circle(bev, p, 3, (60, 220, 60), -1, cv2.LINE_AA)
    cv2.drawMarker(bev, xy_to_px(0, 0, h, e), (0, 0, 255), cv2.MARKER_CROSS, 16, 2)
    put_text(bev, "BEV +x up +y left", (8, 22), 0.5)

    span = max(float(np.max(np.abs(traj[:, :2]))), 1.0) * 1.25
    ego = np.full((h, h, 3), 28, dtype=np.uint8)
    epts = [xy_to_px(float(x), float(y), h, span) for x, y, *_ in traj]
    for i in range(1, len(epts)):
        cv2.line(ego, epts[i - 1], epts[i], (80, 255, 80), 2, cv2.LINE_AA)
    for p in epts:
        cv2.circle(ego, p, 4, (60, 220, 60), -1, cv2.LINE_AA)
    cv2.drawMarker(ego, xy_to_px(0, 0, h, span), (0, 0, 255), cv2.MARKER_CROSS, 18, 2)
    put_text(ego, "ego +x up +y left", (8, 22), 0.5)
    end = traj[-1]
    put_text(ego, f"end x={end[0]:.1f} y={end[1]:.1f}", (8, h - 14), 0.45, (200, 255, 200))
    return np.concatenate([cam_bgr, bev, ego], axis=1)


def collect_by_seq(data_dir: Path) -> dict[str, list[dict]]:
    meta = json.loads((data_dir / "meta.json").read_text())
    by_seq: dict[str, list[dict]] = defaultdict(list)
    for s in meta["samples"]:
        by_seq[s["seq"]].append(
            {
                "path": data_dir / s["file"],
                "frame_idx": int(s["frame_idx"]),
                "sample": Path(s["file"]).stem,
            }
        )
    for seq in by_seq:
        by_seq[seq].sort(key=lambda x: x["frame_idx"])
    return dict(by_seq)


def write_mp4_ffmpeg(frames_dir: Path, out_mp4: Path, fps: float) -> None:
    out_mp4.parent.mkdir(parents=True, exist_ok=True)
    pattern = str(frames_dir / "frame_%06d.jpg")
    cmd = [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-framerate",
        str(fps),
        "-i",
        pattern,
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-crf",
        "20",
        str(out_mp4),
    ]
    subprocess.run(cmd, check=True)


def export_seq(
    entries: list[dict],
    seq: str,
    out_dir: Path,
    fps: float,
    stride: int,
    max_frames: int,
    keep_frames: bool,
) -> Path:
    work = out_dir / f"_frames_{seq}"
    work.mkdir(parents=True, exist_ok=True)
    # clear old
    for p in work.glob("frame_*.jpg"):
        p.unlink()

    n = 0
    for e in entries[:: max(1, stride)]:
        if max_frames > 0 and n >= max_frames:
            break
        d = np.load(e["path"], allow_pickle=True)
        panel = render_frame(
            d["camera_feature"],
            d["lidar_feature"],
            d["trajectory"],
            d["status_feature"],
            seq=seq,
            frame_idx=e["frame_idx"],
            sample_id=e["sample"],
        )
        # even dims for yuv420p
        h, w = panel.shape[:2]
        if h % 2 or w % 2:
            panel = cv2.copyMakeBorder(
                panel, 0, h % 2, 0, w % 2, cv2.BORDER_CONSTANT, value=(0, 0, 0)
            )
        cv2.imwrite(str(work / f"frame_{n:06d}.jpg"), panel, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
        n += 1
        if n % 200 == 0 or n == 1:
            print(f"  [{seq}] rendered {n}/{len(entries[::max(1,stride)])}")

    out_mp4 = out_dir / f"rellis_{seq}_gt_traj.mp4"
    write_mp4_ffmpeg(work, out_mp4, fps=fps)
    print(f"[wrote] {out_mp4}  ({n} frames @ {fps} fps)")
    if not keep_frames:
        for p in work.glob("frame_*.jpg"):
            p.unlink()
        try:
            work.rmdir()
        except OSError:
            pass
    return out_mp4


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=Path, default=Path("/home/rm/Desktop/df_offroad/datasets/ddv2_rellis_full"))
    ap.add_argument("--out_dir", type=Path, default=Path("/home/rm/Desktop/df_offroad/log/rellis_videos"))
    ap.add_argument("--seqs", type=str, default="", help="comma list e.g. 00000,00001; empty=all")
    ap.add_argument("--fps", type=float, default=10.0, help="output fps (dataset lidar_hz≈10)")
    ap.add_argument("--stride", type=int, default=1, help="use every Nth sample")
    ap.add_argument("--max_frames", type=int, default=0, help="per-seq cap; 0=all")
    ap.add_argument("--keep_frames", action="store_true")
    args = ap.parse_args()

    by_seq = collect_by_seq(args.data_dir)
    want = {s.strip() for s in args.seqs.split(",") if s.strip()} if args.seqs else set(by_seq)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    written = []
    for seq in sorted(by_seq):
        if seq not in want:
            continue
        print(f"[seq {seq}] samples={len(by_seq[seq])}")
        written.append(
            export_seq(
                by_seq[seq],
                seq,
                args.out_dir,
                fps=args.fps,
                stride=args.stride,
                max_frames=args.max_frames,
                keep_frames=args.keep_frames,
            )
        )
    print(f"[done] {len(written)} videos → {args.out_dir}")


if __name__ == "__main__":
    main()
