"""DDV2 status_feature helpers (no Chrono dependency).

Layout (8,):
  [0:4]  navigation / attitude conditioning
         - mode "goal" (default, scheme B):
           [0]=clip(gx/norm,-1,1), [1]=clip(gy/norm,-1,1),
           [2]=clip(roll/π,-1,1),  [3]=clip(pitch/π,-1,1)
           (roll/pitch = Chrono Cardan XYZ, radians)
         - mode "command": legacy one-hot [left, straight, right, unknown]
  [4:6]  ego-frame velocity  vx, vy  (+x forward, +y left)
  [6:8]  ego-frame acceleration ax, ay

goal_norm_m defaults to 32 m to match LiDAR BEV extent.
"""

from __future__ import annotations

from typing import Optional, Sequence, Tuple, Union

import numpy as np

GOAL_NORM_M = 32.0
STATUS_DIM = 8
ATTITUDE_NORM_RAD = float(np.pi)  # roll/pitch → [-1,1] via /π
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


def encode_attitude(
    roll_rad: float = 0.0,
    pitch_rad: float = 0.0,
) -> np.ndarray:
    """Cardan roll/pitch (rad) → 2-dim in [-1, 1] via /π."""
    span = ATTITUDE_NORM_RAD
    return np.array(
        [
            float(np.clip(float(roll_rad) / span, -1.0, 1.0)),
            float(np.clip(float(pitch_rad) / span, -1.0, 1.0)),
        ],
        dtype=np.float32,
    )


def encode_goal_command(
    goal_xy_ego: Optional[ArrayLike],
    goal_norm_m: float = GOAL_NORM_M,
    roll_rad: float = 0.0,
    pitch_rad: float = 0.0,
) -> np.ndarray:
    """Ego goal XY + chassis attitude → 4-dim: [gx_n, gy_n, roll_n, pitch_n].

    gx,gy clipped to [-1,1]; roll/pitch = Cardan XYZ / π → [-1,1].
    None / non-finite goal → gx,gy=0 (attitude still written).
    """
    out = np.zeros(4, dtype=np.float32)
    out[2:4] = encode_attitude(roll_rad, pitch_rad)
    if goal_xy_ego is None:
        return out
    g = np.asarray(goal_xy_ego, dtype=np.float64).reshape(-1)
    if g.size < 2 or not np.isfinite(g[:2]).all():
        return out
    span = max(float(goal_norm_m), 1e-3)
    gx, gy = float(g[0]), float(g[1])
    out[0] = float(np.clip(gx / span, -1.0, 1.0))
    out[1] = float(np.clip(gy / span, -1.0, 1.0))
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
    roll_rad: Optional[float] = None,
    pitch_rad: Optional[float] = None,
) -> np.ndarray:
    """Replace goal XY in [0:2]; set/keep attitude in [2:4]; keep vel/acc.

    - roll_rad / pitch_rad given → write those (rad).
    - both omitted → keep existing status[2:4] (sensor-filled attitude, or
      legacy range_n/0 until data is recollected).
    """
    st = np.asarray(status, dtype=np.float32).reshape(-1).copy()
    if st.size < STATUS_DIM:
        st = np.pad(st, (0, STATUS_DIM - st.size)).astype(np.float32)
    ego = world_to_ego_xy(pose_xyh, goal_xy_world)
    if roll_rad is None and pitch_rad is None:
        att = st[2:4].copy()
    else:
        att = encode_attitude(
            0.0 if roll_rad is None else float(roll_rad),
            0.0 if pitch_rad is None else float(pitch_rad),
        )
    st[:4] = encode_goal_command(
        ego,
        goal_norm_m=goal_norm_m,
        roll_rad=float(att[0]) * ATTITUDE_NORM_RAD,
        pitch_rad=float(att[1]) * ATTITUDE_NORM_RAD,
    )
    return st[:STATUS_DIM]


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


def decode_attitude_from_status(status: ArrayLike) -> Tuple[float, float]:
    """status[2:4] → (roll_rad, pitch_rad)."""
    st = np.asarray(status, dtype=np.float64).reshape(-1)
    if st.size < 4:
        return 0.0, 0.0
    return float(st[2]) * ATTITUDE_NORM_RAD, float(st[3]) * ATTITUDE_NORM_RAD
