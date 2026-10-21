"""
DDV2-aligned sensor suite for Verti-Bench / Chrono.

Produces observation tensors matching DiffusionDriveV2 feature sizes:
  camera_feature: (3, 256, 1024)  float32 in [0, 1]  — single front camera
  lidar_feature:  (1, 256, 256)   float32 BEV (±32 m, 4 px/m)
      default = chassis-frame max-z elevation mapped to [0, 1]
      (exp/lidar-elevation-bev; set lidar_bev_mode="density" for legacy hist)
  status_feature: (8,)            nav(4) + vel(2) + accel(2)
      default nav = ego-frame goal XY / goal_norm_m → [-1,1] in [0:2], [2:4]=0
      (scheme B; set status_nav_mode="command" for legacy L/S/R/U one-hot)

Requires working Chrono::sensor (CUDA / OptiX).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import cv2
import numpy as np
import pychrono as chrono
import pychrono.sensor as sens

from verti_bench.envs.utils.ddv2_status import (
    GOAL_NORM_M,
    encode_discrete_command,
    encode_goal_command,
    make_status_feature,
    world_to_ego_xy,
)


@dataclass
class DDV2SensorConfig:
    # Match TransfuserConfig feature sizes
    camera_out_w: int = 1024
    camera_out_h: int = 256
    # Render at target size (one front camera)
    cam_width: int = 1024
    cam_height: int = 256
    cam_hfov: float = math.radians(100.0)  # wide front FOV ≈ stitched panorama width
    cam_update_hz: float = 10.0
    cam_supersample: int = 2
    # Roof / hood mount (vehicle: X fwd, Y left, Z up)
    cam_mount: Tuple[float, float, float] = (1.2, 0.0, 1.6)
    cam_yaw: float = 0.0
    cam_pitch: float = math.radians(8.0)  # slight look-down

    lidar_update_hz: float = 10.0
    lidar_h_samples: int = 512
    lidar_v_samples: int = 32
    lidar_hfov: float = 2.0 * math.pi
    lidar_max_vert: float = math.radians(10.0)
    # Steeper look-down so ground is visible a few metres ahead of a hood mount.
    lidar_min_vert: float = math.radians(-45.0)
    lidar_max_range: float = 40.0
    # Hood / bumper: +x forward of chassis origin (was roof center 0,0,1.5).
    lidar_mount: Tuple[float, float, float] = (2.2, 0.0, 1.2)
    # Drop ego-body returns (sensor-frame range) before BEV splat.
    lidar_min_range: float = 1.5

    # Lidar BEV histogram (TransfuserConfig defaults)
    lidar_min_x: float = -32.0
    lidar_max_x: float = 32.0
    lidar_min_y: float = -32.0
    lidar_max_y: float = 32.0
    pixels_per_meter: float = 4.0
    hist_max_per_pixel: int = 5
    max_height_lidar: float = 100.0
    lidar_split_height: float = 0.2
    use_ground_plane: bool = False
    # Chrono XYZI is in sensor frame; transform to chassis before height filters.
    # If True, single-channel BEV keeps ground (z<=split); DDV2 NAVSIM default drops it.
    include_ground_in_bev: bool = True
    flip_camera_ud: bool = True  # Chrono RGBA buffer is often vertically flipped

    # What goes into lidar_feature (DDV2 backbone input).
    # "elevation": max-z BEV → [0,1] with elev_min/elev_max (matches verti_ddv2_elevation)
    # "density":   point-count histogram → [0,1] (NAVSIM / legacy)
    lidar_bev_mode: str = "elevation"
    elev_min: float = -3.0
    elev_max: float = 4.0

    # status_feature[0:4] navigation conditioning.
    # "goal":    ego-frame goal XY / goal_norm_m clipped to [-1,1] (scheme B)
    # "command": legacy one-hot [left, straight, right, unknown]
    status_nav_mode: str = "goal"
    goal_norm_m: float = GOAL_NORM_M
    command_yaw_thresh_deg: float = 20.0


class DDV2SensorSuite:
    """Attach one front camera + lidar to a Chrono chassis; export DDV2-sized features."""

    def __init__(
        self,
        system: chrono.ChSystem,
        chassis_body: chrono.ChBody,
        config: Optional[DDV2SensorConfig] = None,
    ):
        self.cfg = config or DDV2SensorConfig()
        self.system = system
        self.chassis = chassis_body
        self.manager = sens.ChSensorManager(system)
        self.manager.scene.SetAmbientLight(chrono.ChVector3f(0.4, 0.4, 0.4))
        self.manager.scene.AddPointLight(
            chrono.ChVector3f(100.0, 100.0, 100.0),
            chrono.ChColor(1.0, 1.0, 1.0),
            5000.0,
        )
        self.manager.scene.AddPointLight(
            chrono.ChVector3f(-100.0, -100.0, 80.0),
            chrono.ChColor(1.0, 1.0, 1.0),
            3000.0,
        )

        self.camera = self._make_camera()
        self.lidar = self._make_lidar()

        self._prev_vel_xy: Optional[np.ndarray] = None
        self._prev_time: Optional[float] = None
        self._accel_xy = np.zeros(2, dtype=np.float32)

    def _cam_pose(self) -> chrono.ChFramed:
        loc = chrono.ChVector3d(*self.cfg.cam_mount)
        # yaw about Z, then pitch about Y (look down)
        q_yaw = chrono.QuatFromAngleAxis(self.cfg.cam_yaw, chrono.ChVector3d(0, 0, 1))
        q_pitch = chrono.QuatFromAngleAxis(self.cfg.cam_pitch, chrono.ChVector3d(0, 1, 0))
        rot = q_yaw * q_pitch
        return chrono.ChFramed(loc, rot)

    def _offset_pose(self, xyz: Tuple[float, float, float], yaw: float) -> chrono.ChFramed:
        loc = chrono.ChVector3d(*xyz)
        rot = chrono.QuatFromAngleAxis(yaw, chrono.ChVector3d(0, 0, 1))
        return chrono.ChFramed(loc, rot)

    def _make_camera(self) -> sens.ChCameraSensor:
        cam = sens.ChCameraSensor(
            self.chassis,
            self.cfg.cam_update_hz,
            self._cam_pose(),
            self.cfg.cam_width,
            self.cfg.cam_height,
            self.cfg.cam_hfov,
            self.cfg.cam_supersample,
        )
        cam.SetName("cam_front")
        cam.SetLag(0)
        cam.SetCollectionWindow(0)
        cam.PushFilter(sens.ChFilterRGBA8Access())
        self.manager.AddSensor(cam)
        return cam

    def _make_lidar(self) -> sens.ChLidarSensor:
        lidar = sens.ChLidarSensor(
            self.chassis,
            self.cfg.lidar_update_hz,
            self._offset_pose(self.cfg.lidar_mount, 0.0),
            self.cfg.lidar_h_samples,
            self.cfg.lidar_v_samples,
            self.cfg.lidar_hfov,
            self.cfg.lidar_max_vert,
            self.cfg.lidar_min_vert,
            self.cfg.lidar_max_range,
        )
        lidar.SetName("lidar_ddv2")
        lidar.SetLag(0)
        lidar.SetCollectionWindow(1.0 / self.cfg.lidar_update_hz)
        lidar.PushFilter(sens.ChFilterPCfromDepth())
        lidar.PushFilter(sens.ChFilterXYZIAccess())
        self.manager.AddSensor(lidar)
        return lidar

    def update(self) -> None:
        self.manager.Update()

    def _read_rgb(self, cam: sens.ChCameraSensor) -> Optional[np.ndarray]:
        buf = cam.GetMostRecentRGBA8Buffer()
        if not buf.HasData():
            return None
        rgba = np.array(buf.GetRGBA8Data(), dtype=np.uint8)
        rgba = rgba.reshape((buf.Height, buf.Width, 4))
        rgb = rgba[:, :, :3].copy()
        if self.cfg.flip_camera_ud:
            rgb = np.flipud(rgb)
        return np.ascontiguousarray(rgb)

    def _to_camera_feature(self, rgb: np.ndarray) -> np.ndarray:
        """Ensure (3, 256, 1024) float32 in [0, 1]."""
        h, w = self.cfg.camera_out_h, self.cfg.camera_out_w
        if rgb.shape[0] != h or rgb.shape[1] != w:
            rgb = cv2.resize(rgb, (w, h))
        return np.transpose(rgb.astype(np.float32) / 255.0, (2, 0, 1))

    def _read_lidar_xyz(self) -> Optional[np.ndarray]:
        """XYZI in sensor frame → chassis frame (add lidar mount)."""
        buf = self.lidar.GetMostRecentXYZIBuffer()
        if not buf.HasData():
            return None
        xyzi = np.array(buf.GetXYZIData(), dtype=np.float32).reshape(-1, 4)
        rng = np.linalg.norm(xyzi[:, :3], axis=1)
        valid = np.isfinite(xyzi[:, 0]) & (rng > self.cfg.lidar_min_range)
        pts = xyzi[valid, :3]
        # Identity rotation mount: p_chassis = p_sensor + t_mount
        t = np.array(self.cfg.lidar_mount, dtype=np.float32)
        return pts + t

    def _bev_bins(self) -> Tuple[np.ndarray, np.ndarray]:
        cfg = self.cfg
        xbins = np.linspace(
            cfg.lidar_min_x,
            cfg.lidar_max_x,
            int((cfg.lidar_max_x - cfg.lidar_min_x) * cfg.pixels_per_meter) + 1,
        )
        ybins = np.linspace(
            cfg.lidar_min_y,
            cfg.lidar_max_y,
            int((cfg.lidar_max_y - cfg.lidar_min_y) * cfg.pixels_per_meter) + 1,
        )
        return xbins, ybins

    def _splat_lidar_density(self, points_xyz: np.ndarray) -> np.ndarray:
        """BEV point-count histogram in chassis frame → (1|2, H, W) float32 in [0, 1]."""
        cfg = self.cfg
        pts = points_xyz[points_xyz[:, 2] < cfg.max_height_lidar]
        above = pts[pts[:, 2] > cfg.lidar_split_height]
        below = pts[pts[:, 2] <= cfg.lidar_split_height]
        xbins, ybins = self._bev_bins()

        def splat(pc: np.ndarray) -> np.ndarray:
            if pc.size == 0:
                return np.zeros((len(xbins) - 1, len(ybins) - 1), dtype=np.float32)
            hist = np.histogramdd(pc[:, :2], bins=(xbins, ybins))[0]
            hist[hist > cfg.hist_max_per_pixel] = cfg.hist_max_per_pixel
            return (hist / cfg.hist_max_per_pixel).astype(np.float32)

        if cfg.use_ground_plane:
            return np.stack([splat(below), splat(above)], axis=0)
        if cfg.include_ground_in_bev:
            return splat(pts)[None, ...]
        return splat(above)[None, ...]

    def _splat_lidar_elevation(self, points_xyz: np.ndarray) -> np.ndarray:
        """BEV max-z (chassis frame, z-up) per cell → (1, H, W). Empty cells = 0.

        Density==0 marks invalid cells. z is relative to the chassis origin, so
        vehicle roll/pitch tilts the map with the body (sensor-centric).
        """
        cfg = self.cfg
        pts = points_xyz[np.isfinite(points_xyz).all(axis=1)]
        pts = pts[pts[:, 2] < cfg.max_height_lidar]
        xbins, ybins = self._bev_bins()
        h, w = len(xbins) - 1, len(ybins) - 1
        elev = np.zeros((h, w), dtype=np.float32)
        if pts.size == 0:
            return elev[None, ...]
        ix = np.clip(np.digitize(pts[:, 0], xbins) - 1, 0, h - 1)
        iy = np.clip(np.digitize(pts[:, 1], ybins) - 1, 0, w - 1)
        # Max reduce into grid (lexsort by z ascending, then write so last=max)
        order = np.argsort(pts[:, 2])
        elev[ix[order], iy[order]] = pts[order, 2]
        return elev[None, ...]

    def _normalize_elevation(
        self, elev: np.ndarray, dens: np.ndarray
    ) -> np.ndarray:
        """Raw max-z BEV → [0,1]; empty (dens≈0) cells stay 0. Matches dataset build."""
        cfg = self.cfg
        out = np.zeros_like(elev, dtype=np.float32)
        mask = dens > 1e-6
        span = max(float(cfg.elev_max - cfg.elev_min), 1e-3)
        out[mask] = np.clip(
            (elev[mask] - cfg.elev_min) / span, 0.0, 1.0
        ).astype(np.float32)
        return out

    def _splat_lidar(self, points_xyz: np.ndarray) -> np.ndarray:
        """Alias for current lidar_feature BEV (elevation by default)."""
        dens = self._splat_lidar_density(points_xyz)
        if self.cfg.lidar_bev_mode == "density":
            return dens
        elev = self._splat_lidar_elevation(points_xyz)
        return self._normalize_elevation(elev, dens)

    def _update_kinematics(self) -> Tuple[np.ndarray, np.ndarray]:
        """Ego-frame vx/vy (+x forward, +y left) — matches DDV2/NAVSIM status."""
        vel = self.chassis.GetPosDt()
        rot = self.chassis.GetRot()
        x_axis = rot.GetAxisX()
        heading = math.atan2(x_axis.y, x_axis.x)
        c, s = math.cos(heading), math.sin(heading)
        # world → ego: R^T
        vel_xy = np.array(
            [c * vel.x + s * vel.y, -s * vel.x + c * vel.y],
            dtype=np.float32,
        )
        t = self.system.GetChTime()
        if self._prev_vel_xy is not None and self._prev_time is not None:
            dt = max(t - self._prev_time, 1e-3)
            self._accel_xy = (vel_xy - self._prev_vel_xy) / dt
        self._prev_vel_xy = vel_xy.copy()
        self._prev_time = t
        return vel_xy, self._accel_xy.astype(np.float32)

    def _chassis_pose_xyh(self) -> Tuple[float, float, float]:
        pos = self.chassis.GetPos()
        x_axis = self.chassis.GetRot().GetAxisX()
        heading = math.atan2(x_axis.y, x_axis.x)
        return float(pos.x), float(pos.y), heading

    def _nav_command(self, goal: Optional[chrono.ChVector3d]) -> np.ndarray:
        """Fill status[0:4] per ``status_nav_mode`` (goal XY or legacy one-hot)."""
        cfg = self.cfg
        if goal is None:
            if cfg.status_nav_mode == "command":
                return encode_discrete_command(None)
            return encode_goal_command(None, goal_norm_m=cfg.goal_norm_m)

        pose = self._chassis_pose_xyh()
        goal_xy = (float(goal.x), float(goal.y))
        if cfg.status_nav_mode == "command":
            dx, dy = goal_xy[0] - pose[0], goal_xy[1] - pose[1]
            desired = math.atan2(dy, dx)
            err = (desired - pose[2] + math.pi) % (2 * math.pi) - math.pi
            return encode_discrete_command(
                err, thresh_rad=math.radians(cfg.command_yaw_thresh_deg)
            )

        ego = world_to_ego_xy(pose, goal_xy)
        return encode_goal_command(ego, goal_norm_m=cfg.goal_norm_m)

    # Back-compat alias used by older call sites / comments.
    def _driving_command(
        self, chassis: chrono.ChBody, goal: Optional[chrono.ChVector3d]
    ) -> np.ndarray:
        del chassis  # use self.chassis
        return self._nav_command(goal)

    def get_observation(
        self,
        goal: Optional[chrono.ChVector3d] = None,
    ) -> Optional[Dict[str, np.ndarray]]:
        """Returns dict with DDV2 feature keys, or None if buffers not ready yet."""
        rgb = self._read_rgb(self.camera)
        if rgb is None:
            return None
        pts = self._read_lidar_xyz()
        if pts is None:
            return None

        camera_feature = self._to_camera_feature(rgb)
        lidar_density = self._splat_lidar_density(pts)
        lidar_elevation = self._splat_lidar_elevation(pts)  # raw max-z (m)
        if self.cfg.lidar_bev_mode == "density":
            lidar_feature = lidar_density
        else:
            lidar_feature = self._normalize_elevation(lidar_elevation, lidar_density)
        vel_xy, accel_xy = self._update_kinematics()
        nav4 = self._nav_command(goal)
        status_feature = make_status_feature(nav4, vel_xy, accel_xy)
        goal_xy_ego = None
        if goal is not None and self.cfg.status_nav_mode == "goal":
            goal_xy_ego = world_to_ego_xy(self._chassis_pose_xyh(), (goal.x, goal.y))

        out = {
            "camera_feature": camera_feature,
            # lidar_feature = elevation [0,1] by default (exp/lidar-elevation-bev)
            "lidar_feature": lidar_feature,
            "lidar_density": lidar_density,
            "lidar_elevation": lidar_elevation,
            "status_feature": status_feature,
            "cam_front": rgb,
            "lidar_xyz": pts,
        }
        if goal_xy_ego is not None:
            out["goal_xy_ego"] = goal_xy_ego
        return out
