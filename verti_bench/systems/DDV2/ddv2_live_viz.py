"""Live DDV2 panel in a *separate process* (avoids Irrlicht/Qt conflict).

Also writes live_panel.jpg next to dump dir / cwd for fallback viewing.
"""

from __future__ import annotations

import os
import time
from collections import deque
from multiprocessing import Process, Queue
from pathlib import Path
from typing import Deque, Dict, Optional, Tuple

import cv2
import numpy as np


def _to_u8_rgb(img: np.ndarray) -> np.ndarray:
    x = np.asarray(img)
    if x.dtype != np.uint8:
        if float(np.nanmax(x)) <= 1.5:
            x = (np.clip(x, 0, 1) * 255.0).astype(np.uint8)
        else:
            x = np.clip(x, 0, 255).astype(np.uint8)
    return x


def _put_text(img, text, org, scale=0.55, color=(240, 240, 240), thick=1):
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thick + 2, cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)


CMD_LABELS = ("LEFT", "STRAIGHT", "RIGHT", "UNKNOWN")
# BGR highlight colors
CMD_COLORS = (
    (80, 180, 255),   # left — orange-ish
    (80, 220, 80),    # straight — green
    (255, 160, 80),   # right — blue-ish
    (160, 160, 160),  # unknown — gray
)


def parse_driving_command(status: np.ndarray) -> Tuple[int, str, np.ndarray]:
    """status (8,) = cmd(4)+vel(2)+acc(2) → active index, label, cmd vector."""
    st = np.asarray(status, dtype=np.float64).reshape(-1)
    cmd = st[:4] if st.size >= 4 else np.zeros(4)
    if float(cmd.sum()) < 1e-6:
        idx = 3
    else:
        idx = int(np.argmax(cmd))
    return idx, CMD_LABELS[idx], cmd.astype(np.float32)


def _draw_cmd_banner(width: int, cmd_idx: int, cmd_vec: np.ndarray, vel_xy: np.ndarray, height: int = 56) -> np.ndarray:
    """Wide strip with 4 command chips; active one filled."""
    bar = np.full((height, width, 3), 22, dtype=np.uint8)
    _put_text(bar, "NAV CMD", (10, 36), 0.65, (200, 200, 200), 2)
    n = 4
    left0 = 120
    gap = 10
    chip_w = max(80, (width - left0 - 20 - gap * (n - 1)) // n)
    for i, (lab, col) in enumerate(zip(CMD_LABELS, CMD_COLORS)):
        x0 = left0 + i * (chip_w + gap)
        x1 = x0 + chip_w
        y0, y1 = 8, height - 8
        active = i == cmd_idx
        if active:
            cv2.rectangle(bar, (x0, y0), (x1, y1), col, -1, cv2.LINE_AA)
            text_col = (10, 10, 10)
        else:
            cv2.rectangle(bar, (x0, y0), (x1, y1), (70, 70, 70), 2, cv2.LINE_AA)
            text_col = (180, 180, 180)
        # soft fill by cmd weight
        w = float(cmd_vec[i]) if i < len(cmd_vec) else 0.0
        if not active and w > 0.05:
            overlay = bar.copy()
            cv2.rectangle(overlay, (x0, y0), (x1, y1), col, -1, cv2.LINE_AA)
            bar = cv2.addWeighted(overlay, 0.25 * w, bar, 1 - 0.25 * w, 0)
        tx = x0 + 12
        _put_text(bar, lab, (tx, 36), 0.55, text_col, 2 if active else 1)
    # ego speed from status
    if vel_xy is not None and len(vel_xy) >= 2:
        spd = float(np.linalg.norm(vel_xy[:2]))
        _put_text(bar, f"|v|={spd:.2f} m/s  vx={vel_xy[0]:+.2f} vy={vel_xy[1]:+.2f}", (width - 360, 36), 0.5, (200, 220, 255))
    return bar


def _compose_panel(
    obs: Dict[str, np.ndarray],
    traj_ego: np.ndarray,
    infer_ms: float,
    rtt_ms: float,
    sim_t: float,
    track_speed: float,
    infer_hist,
    rtt_hist,
    n_infer: int,
    bev_extent: float = 32.0,
    panel_h: int = 360,
) -> np.ndarray:
    status = np.asarray(obs.get("status_feature", np.zeros(8)), dtype=np.float32).reshape(-1)
    cmd_idx, cmd_name, cmd_vec = parse_driving_command(status)
    vel_xy = status[4:6] if status.size >= 6 else np.zeros(2, dtype=np.float32)

    cam = np.asarray(obs["camera_feature"])
    if cam.ndim == 3 and cam.shape[0] == 3:
        cam = np.transpose(cam, (1, 2, 0))
    cam = _to_u8_rgb(cam)
    h = panel_h
    cam_w = int(round(h * cam.shape[1] / max(cam.shape[0], 1)))
    cam_bgr = cv2.cvtColor(cv2.resize(cam, (cam_w, h)), cv2.COLOR_RGB2BGR)
    _put_text(cam_bgr, "camera", (8, 22), 0.55)
    # big command badge on camera
    badge = CMD_COLORS[cmd_idx]
    cv2.rectangle(cam_bgr, (8, 40), (8 + 200, 88), badge, -1, cv2.LINE_AA)
    _put_text(cam_bgr, f"CMD: {cmd_name}", (16, 72), 0.75, (15, 15, 15), 2)
    lid = np.asarray(obs["lidar_feature"])
    if lid.ndim == 3:
        lid = lid[0]
    e = bev_extent
    # Camera-aligned: +x up, +y left (image left) — matches forward camera L/R.
    bev = np.clip(lid[::-1, ::-1], 0, 1)
    bev = cv2.cvtColor((bev * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
    bev = cv2.resize(bev, (h, h), interpolation=cv2.INTER_NEAREST)

    def xy_to_px(x, y, size, span):
        u = int((span - y) / (2 * span) * (size - 1))  # +y → left
        v = int((size - 1) - (x + span) / (2 * span) * (size - 1))  # +x → up
        return u, v

    traj = np.asarray(traj_ego, dtype=np.float64)
    pts = [xy_to_px(float(x), float(y), h, e) for x, y, *_ in traj]
    for i in range(1, len(pts)):
        cv2.line(bev, pts[i - 1], pts[i], (255, 220, 40), 2, cv2.LINE_AA)
    for p in pts:
        cv2.circle(bev, p, 3, (255, 180, 0), -1, cv2.LINE_AA)
    cv2.drawMarker(bev, xy_to_px(0, 0, h, e), (0, 0, 255), cv2.MARKER_CROSS, 16, 2)
    _put_text(bev, "BEV +x up +y left", (8, 22), 0.55)

    span = max(float(np.max(np.abs(traj[:, :2]))), 1.0) * 1.2
    ego = np.full((h, h, 3), 25, dtype=np.uint8)
    epts = [xy_to_px(float(x), float(y), h, span) for x, y, *_ in traj]
    for i in range(1, len(epts)):
        cv2.line(ego, epts[i - 1], epts[i], (255, 220, 40), 2, cv2.LINE_AA)
    for p in epts:
        cv2.circle(ego, p, 4, (255, 180, 0), -1, cv2.LINE_AA)
    cv2.drawMarker(ego, xy_to_px(0, 0, h, span), (0, 0, 255), cv2.MARKER_CROSS, 18, 2)
    _put_text(ego, "ego +x up +y left", (8, 22), 0.55)
    _put_text(ego, f"GPU {infer_ms:.0f}ms  RTT {rtt_ms:.0f}ms", (8, h - 12), 0.5, (180, 255, 180))

    top = np.concatenate([cam_bgr, bev, ego], axis=1)
    cmd_bar = _draw_cmd_banner(top.shape[1], cmd_idx, cmd_vec, vel_xy)
    th = 140
    timing = np.full((th, top.shape[1], 3), 30, dtype=np.uint8)
    def draw_series(hist, color, label, y0):
        vals = np.array(list(hist), dtype=np.float64)
        if len(vals) < 2:
            return
        vmax = max(float(vals.max()), 1.0)
        xs = np.linspace(50, top.shape[1] - 10, len(vals)).astype(int)
        ys = (th - 24 - (vals / vmax) * (th - 55)).astype(int)
        for i in range(1, len(xs)):
            cv2.line(timing, (xs[i - 1], ys[i - 1]), (xs[i], ys[i]), color, 2, cv2.LINE_AA)
        _put_text(timing, label, (8, y0), 0.5, color)

    draw_series(infer_hist, (80, 220, 80), "GPU ms", 20)
    draw_series(rtt_hist, (255, 180, 80), "RTT ms", 42)
    mean_i = float(np.mean(infer_hist)) if infer_hist else float("nan")
    mean_r = float(np.mean(rtt_hist)) if rtt_hist else float("nan")
    _put_text(
        timing,
        f"n={n_infer}  GPU={infer_ms:.0f}(u={mean_i:.0f})  RTT={rtt_ms:.0f}(u={mean_r:.0f})  "
        f"t={sim_t:.1f}s v~{track_speed:.2f}  cmd={cmd_name}",
        (12, th - 12),
        0.5,
        (220, 220, 220),
    )
    return np.concatenate([cmd_bar, top, timing], axis=0)

def _viewer_main(queue: Queue, window_name: str = "DDV2 live") -> None:
    """Child process: only OpenCV GUI, no Chrono/Irrlicht."""
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1280, 520)
    blank = np.zeros((500, 1280, 3), dtype=np.uint8)
    _put_text(blank, "DDV2 live — waiting for first inference...", (24, 40), 0.8, (80, 220, 255))
    cv2.imshow(window_name, blank)
    cv2.waitKey(1)
    latest = blank
    while True:
        try:
            while True:
                msg = queue.get_nowait()
                if msg is None:
                    cv2.destroyAllWindows()
                    return
                latest = msg
        except Exception:
            pass
        cv2.imshow(window_name, latest)
        key = cv2.waitKey(30) & 0xFF
        if key in (27, ord("q")):
            # ignore quit from viewer; parent owns lifecycle
            pass
        time.sleep(0.01)


class DDV2LiveViz:
    def __init__(self, history: int = 60, save_path: Optional[str] = None):
        self.infer_hist: Deque[float] = deque(maxlen=history)
        self.rtt_hist: Deque[float] = deque(maxlen=history)
        self.n_infer = 0
        self.save_path = Path(save_path) if save_path else Path("/home/rm/Desktop/df_offroad/log/verti_ddv2/live_panel.jpg")
        self.save_path.parent.mkdir(parents=True, exist_ok=True)
        self._queue: Queue = Queue(maxsize=2)
        self._proc = Process(target=_viewer_main, args=(self._queue,), daemon=True)
        self._proc.start()
        print(f"[ddv2] live viz process pid={self._proc.pid}  also writing {self.save_path}")

    def update(
        self,
        obs: Dict[str, np.ndarray],
        traj_ego: np.ndarray,
        infer_ms: float,
        rtt_ms: float,
        sim_t: float,
        track_speed: float,
        local_goal: Optional[Tuple[float, float]] = None,
        vehicle_xy: Optional[Tuple[float, float]] = None,
    ) -> None:
        self.n_infer += 1
        if np.isfinite(infer_ms):
            self.infer_hist.append(float(infer_ms))
        if np.isfinite(rtt_ms):
            self.rtt_hist.append(float(rtt_ms))

        panel = _compose_panel(
            obs,
            traj_ego,
            infer_ms,
            rtt_ms,
            sim_t,
            track_speed,
            self.infer_hist,
            self.rtt_hist,
            self.n_infer,
        )
        # Always save for fallback (file viewers / IDE)
        try:
            cv2.imwrite(str(self.save_path), panel)
        except Exception:
            pass
        # Non-blocking push to viewer process
        try:
            if self._queue.full():
                try:
                    self._queue.get_nowait()
                except Exception:
                    pass
            self._queue.put_nowait(panel)
        except Exception:
            pass

    def pump(self) -> None:
        """No-op in parent; viewer process pumps itself."""
        return

    def close(self) -> None:
        try:
            self._queue.put_nowait(None)
        except Exception:
            pass
        if self._proc.is_alive():
            self._proc.join(timeout=1.0)
            if self._proc.is_alive():
                self._proc.terminate()
