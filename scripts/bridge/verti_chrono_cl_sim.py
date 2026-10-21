"""Chrono-map closed-loop proposal simulator (PDM-style, no PyChrono).

Rolls out a batch of ego-frame waypoint proposals with a kinematic bicycle +
pure-pursuit tracker on Verti/Chrono world coordinates. Used as the closed-loop
front-end for Verti GRPO rewards (same role as NAVSIM PDMSimulator).

Full ProjectChrono physics stays in chrono9 eval; this runs inside the lean ddv2
training env against the same height/obstacle rasters.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from verti_metric_cache import VertiMetricCache, WorldMaps, ego_to_world, sample_obstacle


@dataclass
class ChronoCLSimConfig:
    dt: float = 0.1
    horizon_s: float = 4.0  # matches DDV2 8×0.5s
    v_ref: float = 4.0
    wheelbase: float = 2.8
    max_steer: float = 0.6  # rad
    lookahead_s: float = 1.0
    track_end_eps: float = 1.5  # m — stop / hold when near last wp


def _estimate_yaw_ego(xy: np.ndarray) -> np.ndarray:
    """xy (G,T,2) → yaw (G,T)."""
    d = np.diff(xy, axis=1)
    yaw_mid = np.arctan2(d[..., 1], d[..., 0])
    yaw0 = yaw_mid[..., :1]
    return np.concatenate([yaw0, yaw_mid], axis=-1)


def _wrap(a: np.ndarray) -> np.ndarray:
    return np.arctan2(np.sin(a), np.cos(a))


def proposals_to_world(
    trajs_ego: np.ndarray,
    pose: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """(G,8,2|3) ego → world xy (G,8,2) and world yaw (G,8)."""
    trajs = np.asarray(trajs_ego, dtype=np.float64)
    xy_ego = trajs[..., :2]
    if trajs.shape[-1] >= 3:
        yaw_ego = trajs[..., 2]
    else:
        yaw_ego = _estimate_yaw_ego(xy_ego)
    h0 = float(pose[3])
    xy_w = ego_to_world(xy_ego, pose)
    yaw_w = h0 + yaw_ego
    return xy_w, yaw_w


def world_to_ego(xy_w: np.ndarray, pose: np.ndarray) -> np.ndarray:
    """(G,T,2) world → ego xy."""
    x0, y0, h0 = float(pose[0]), float(pose[1]), float(pose[3])
    c, s = np.cos(h0), np.sin(h0)
    dx = xy_w[..., 0] - x0
    dy = xy_w[..., 1] - y0
    ex = c * dx + s * dy
    ey = -s * dx + c * dy
    return np.stack([ex, ey], axis=-1)


def simulate_proposals(
    trajs_ego: np.ndarray,
    sample: VertiMetricCache,
    world: Optional[WorldMaps] = None,
    cfg: Optional[ChronoCLSimConfig] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Closed-loop rollout of G proposals.

    Args:
      trajs_ego: (G, 8, 2|3) planned waypoints in ego frame
      sample: pose / goal context
      world: optional maps — if given, collision flags use obstacle raster
      cfg: sim parameters

    Returns:
      sim_ego: (G, 8, 3) rolled-out ego-frame states at 0.5 s
      crashed: (G,) bool — hit obstacle or left map during rollout
    """
    cfg = cfg or ChronoCLSimConfig()
    trajs = np.asarray(trajs_ego, dtype=np.float64)
    if trajs.ndim != 3 or trajs.shape[1] != 8:
        raise ValueError(f"expected (G,8,C), got {trajs.shape}")
    G = trajs.shape[0]
    pose = np.asarray(sample.pose, dtype=np.float64)
    ref_xy, _ = proposals_to_world(trajs, pose)  # (G,8,2)

    n_steps = int(round(cfg.horizon_s / cfg.dt))
    sample_every = max(1, int(round(0.5 / cfg.dt)))  # 0.5 s → waypoint index
    n_out = 8

    # state: x,y,heading,v
    x = np.full(G, pose[0], dtype=np.float64)
    y = np.full(G, pose[1], dtype=np.float64)
    h = np.full(G, pose[3], dtype=np.float64)
    v = np.full(G, cfg.v_ref, dtype=np.float64)
    crashed = np.zeros(G, dtype=bool)

    # store dense then subsample
    hist_xy = np.zeros((G, n_steps + 1, 2), dtype=np.float64)
    hist_h = np.zeros((G, n_steps + 1), dtype=np.float64)
    hist_xy[:, 0, 0] = x
    hist_xy[:, 0, 1] = y
    hist_h[:, 0] = h

    # cumulative arc length of reference for lookahead
    dseg = np.linalg.norm(np.diff(ref_xy, axis=1), axis=-1)  # (G,7)
    cum = np.concatenate([np.zeros((G, 1)), np.cumsum(dseg, axis=1)], axis=1)  # (G,8)

    L = max(cfg.v_ref * cfg.lookahead_s, 2.0)

    for t in range(1, n_steps + 1):
        # nearest ref index by distance
        d2 = (ref_xy[..., 0] - x[:, None]) ** 2 + (ref_xy[..., 1] - y[:, None]) ** 2
        nearest = np.argmin(d2, axis=1)  # (G,)
        s_near = cum[np.arange(G), nearest]
        s_tgt = s_near + L
        # interpolate lookahead point along polyline
        tgt = ref_xy[np.arange(G), -1].copy()
        for g in range(G):
            sg = s_tgt[g]
            if sg >= cum[g, -1]:
                tgt[g] = ref_xy[g, -1]
                continue
            # find segment
            j = int(np.searchsorted(cum[g], sg, side="right") - 1)
            j = int(np.clip(j, 0, 6))
            span = cum[g, j + 1] - cum[g, j]
            alpha = 0.0 if span < 1e-6 else (sg - cum[g, j]) / span
            tgt[g] = (1 - alpha) * ref_xy[g, j] + alpha * ref_xy[g, j + 1]

        # pure pursuit curvature
        dx = tgt[:, 0] - x
        dy = tgt[:, 1] - y
        local_x = np.cos(h) * dx + np.sin(h) * dy
        local_y = -np.sin(h) * dx + np.cos(h) * dy
        ld2 = np.maximum(local_x**2 + local_y**2, 1e-3)
        curvature = 2.0 * local_y / ld2
        steer = np.arctan(cfg.wheelbase * curvature)
        steer = np.clip(steer, -cfg.max_steer, cfg.max_steer)

        # near end → slow down
        dist_end = np.linalg.norm(ref_xy[:, -1, :] - np.stack([x, y], axis=-1), axis=-1)
        v_cmd = np.where(dist_end < cfg.track_end_eps, 0.5, cfg.v_ref)
        v = 0.8 * v + 0.2 * v_cmd

        # bicycle integrate
        x = x + v * np.cos(h) * cfg.dt
        y = y + v * np.sin(h) * cfg.dt
        h = _wrap(h + v * np.tan(steer) / cfg.wheelbase * cfg.dt)

        hist_xy[:, t, 0] = x
        hist_xy[:, t, 1] = y
        hist_h[:, t] = h

        if world is not None:
            xy = np.stack([x, y], axis=-1)
            hit = sample_obstacle(world, xy)
            crashed |= hit

    # subsample to 8 poses at 0.5 s (skip t=0; take t=sample_every ... )
    idxs = [min(n_steps, (i + 1) * sample_every) for i in range(n_out)]
    xy_w = hist_xy[:, idxs, :]
    yaw_w = hist_h[:, idxs]
    xy_ego = world_to_ego(xy_w, pose)
    yaw_ego = _wrap(yaw_w - float(pose[3]))
    sim_ego = np.concatenate([xy_ego, yaw_ego[..., None]], axis=-1).astype(np.float32)
    return sim_ego, crashed
