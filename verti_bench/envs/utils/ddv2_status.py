"""DDV2 status_feature helpers (no Chrono dependency).

Layout (8,):
  [0:4]  navigation conditioning
         - mode "goal" (default, scheme B): ego-frame goal
           [0]=clip(gx/norm,-1,1), [1]=clip(gy/norm,-1,1),
           [2]=clip(||g||/norm,0,1), [3]=0
         - mode "command": legacy one-hot [left, straight, right, unknown]
  [4:6]  ego-frame velocity  vx, vy  (+x forward, +y left)
  [6:8]  ego-frame acceleration ax, ay

goal_norm_m defaults to 32 m to match LiDAR BEV extent.
"""

from __future__ import annotations

from typing import Optional, Sequence, Tuple, Union

import numpy as np

GOAL_NORM_M = 32.0
ArrayLike = Union[Sequence[float], np.ndarray]


def world_to_ego_xy(
    pose_xyh: ArrayLike,
    goal_xy: ArrayLike,
) -> np.ndarray:
    """World XY goal → ego frame (+x forward, +y left).

    pose_xyh: (x, y, heading) or (x, y, z, heading) — heading = yaw of +X.
    """
    p = np.asarray(pose_xyh, dtype=np.float64).reshape(-1)
    g = np.asarray(goal_xy, dtype=np.float64).reshape(-1)
    if p.size < 3 or g.size < 2:
        return np.zeros(2, dtype=np.float32)
    x0, y0 = float(p[0]), float(p[1])
    h0 = float(p[3] if p.size >= 4 else p[2])
    dx, dy = float(g[0]) - x0, float(g[1]) - y0
    c, s = np.cos(h0), np.sin(h0)
    return np.array([c * dx + s * dy, -s * dx + c * dy], dtype=np.float32)


def encode_goal_command(
    goal_xy_ego: Optional[ArrayLike],
    goal_norm_m: float = GOAL_NORM_M,
) -> np.ndarray:
    """Ego goal XY → 4-dim slot: [gx_n, gy_n, range_n, 0].

    gx,gy clipped to [-1,1]; range_n = clip(||g||/norm, 0, 1). Slot [2] distinguishes
    this from a pure one-hot (which never carries a continuous range).
    None / non-finite → zeros (no goal).
    """
    out = np.zeros(4, dtype=np.float32)
    if goal_xy_ego is None:
        return out
    g = np.asarray(goal_xy_ego, dtype=np.float64).reshape(-1)
    if g.size < 2 or not np.isfinite(g[:2]).all():
        return out
    span = max(float(goal_norm_m), 1e-3)
    gx, gy = float(g[0]), float(g[1])
    out[0] = float(np.clip(gx / span, -1.0, 1.0))
    out[1] = float(np.clip(gy / span, -1.0, 1.0))
    out[2] = float(np.clip(np.hypot(gx, gy) / span, 0.0, 1.0))
    out[3] = 0.0
    return out


def encode_discrete_command(
    yaw_err_rad: Optional[float],
    thresh_rad: float = np.radians(20.0),
) -> np.ndarray:
    """Legacy one-hot [left, straight, right, unknown] from heading error to goal.

    yaw_err_rad=None → unknown. Positive err (CCW) → left.
    """
    cmd = np.zeros(4, dtype=np.float32)
    if yaw_err_rad is None or not np.isfinite(yaw_err_rad):
        cmd[3] = 1.0
        return cmd
    if yaw_err_rad > thresh_rad:
        cmd[0] = 1.0
    elif yaw_err_rad < -thresh_rad:
        cmd[2] = 1.0
    else:
        cmd[1] = 1.0
    return cmd


def make_status_feature(
    nav4: ArrayLike,
    vel_xy: ArrayLike,
    accel_xy: ArrayLike,
) -> np.ndarray:
    """Concatenate nav(4)+vel(2)+acc(2) → (8,) float32."""
    return np.concatenate(
        [
            np.asarray(nav4, dtype=np.float32).reshape(4),
            np.asarray(vel_xy, dtype=np.float32).reshape(2),
            np.asarray(accel_xy, dtype=np.float32).reshape(2),
        ]
    ).astype(np.float32)


def rewrite_status_nav_goal(
    status: ArrayLike,
    pose_xyh: ArrayLike,
    goal_xy_world: ArrayLike,
    goal_norm_m: float = GOAL_NORM_M,
) -> np.ndarray:
    """Keep vel/acc from ``status``; replace [0:4] with normalized ego goal."""
    st = np.asarray(status, dtype=np.float32).reshape(-1).copy()
    if st.size < 8:
        st = np.pad(st, (0, 8 - st.size)).astype(np.float32)
    ego = world_to_ego_xy(pose_xyh, goal_xy_world)
    st[:4] = encode_goal_command(ego, goal_norm_m=goal_norm_m)
    return st[:8]


def decode_goal_from_status(
    status: ArrayLike,
    goal_norm_m: float = GOAL_NORM_M,
) -> Tuple[np.ndarray, float]:
    """Inverse of encode (ignores clip saturation): returns (ego_xy_m, range_m)."""
    st = np.asarray(status, dtype=np.float64).reshape(-1)
    if st.size < 2:
        return np.zeros(2, dtype=np.float32), 0.0
    ego = (st[:2] * float(goal_norm_m)).astype(np.float32)
    return ego, float(np.hypot(ego[0], ego[1]))
