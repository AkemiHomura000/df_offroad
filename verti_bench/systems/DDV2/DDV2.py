"""Local planner: DDV2 ego trajectory → world waypoints → PID track."""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import numpy as np

from verti_bench.systems.PID.PID import PIDPlanner


def ego_traj_to_world(
    traj: np.ndarray,
    vehicle_pos: Tuple[float, float],
    heading: float,
) -> List[Tuple[float, float]]:
    """traj (T,3) ego +x fwd +y left → Chrono world XY polyline."""
    c, s = math.cos(heading), math.sin(heading)
    path: List[Tuple[float, float]] = []
    for x, y, _yaw in np.asarray(traj, dtype=np.float64):
        wx = vehicle_pos[0] + c * x - s * y
        wy = vehicle_pos[1] + s * x + c * y
        path.append((float(wx), float(wy)))
    return path


class DDV2Planner(PIDPlanner):
    """Replace A* with DiffusionDrive local traj; keep PID lateral/longitudinal."""

    def __init__(self, terrain_manager, plan_client, look_ahead_m: Optional[float] = None):
        super().__init__(terrain_manager)
        self.client = plan_client
        # Local horizon is ~4 s / ~few meters; shorter look-ahead than A*
        self.look_ahead_distance = (
            look_ahead_m
            if look_ahead_m is not None
            else max(2.0, 4.0 * self.terrain_manager.scale_factor)
        )
        self.last_traj: Optional[np.ndarray] = None
        self.last_path: Optional[List[Tuple[float, float]]] = None
        self.last_infer_ms: float = float("nan")
        self.last_rtt_ms: float = float("nan")

    def plan_local(
        self,
        obs: Dict[str, np.ndarray],
        vehicle_pos: Tuple[float, float],
        heading: float,
    ) -> List[Tuple[float, float]]:
        out = self.client.infer(obs)
        if isinstance(out, tuple):
            traj, meta = out
            self.last_infer_ms = float(meta.get("infer_ms", float("nan")))
            self.last_rtt_ms = float(meta.get("rtt_ms", float("nan")))
        else:
            traj = out
        self.last_traj = traj
        path = ego_traj_to_world(traj, vehicle_pos, heading)
        self.last_path = path
        return path

    def suggested_speed(self, default_speed: float, horizon_s: float = 4.0) -> float:
        """Scale target speed from predicted progress; fall back to config speed."""
        if self.last_traj is None:
            return default_speed
        dist = float(np.linalg.norm(self.last_traj[-1, :2]))
        v = dist / max(horizon_s, 1e-3)
        # Soft clip around configured cruise speed
        return float(np.clip(v, 0.5 * default_speed, 1.25 * default_speed))
