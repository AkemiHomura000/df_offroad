"""VertiPDM: open-loop off-road trajectory scorer (PDMS surrogate).

Interface mirrors DiffusionDriveV2 PDMS subscores so GRPO hard-penalty logic
(no_collision / drivable_area → advantage -1) works unchanged:

  final = no_collision * drivable_area * weighted(progress, ttc, comfort, dir_weighted)

Aliases:
  no_collision  ← no_crash (obstacle / OOB)
  drivable_area ← traversable (max slope)
  ttc           ← tip_margin
  dir_weighted  ← goal_align
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from verti_metric_cache import (
    VertiMetricCache,
    WorldMapBank,
    WorldMaps,
    ego_to_world,
    sample_height,
    sample_obstacle,
)

H_DT = 0.5  # seconds between waypoints (matches DDV2 horizon)


@dataclass
class VertiPDMConfig:
    slope_max_deg: float = 30.0
    tip_max_deg: float = 30.0
    footprint_length: float = 4.5
    footprint_width: float = 2.2
    progress_distance_threshold: float = 2.0
    w_progress: float = 5.0
    w_tip: float = 3.0
    w_comfort: float = 2.0
    w_align: float = 2.0
    comfort_accel_ref: float = 3.0  # m/s^2
    comfort_curv_ref: float = 0.25  # 1/m


def _footprint_corners(xy: np.ndarray, yaw: np.ndarray, length: float, width: float) -> np.ndarray:
    """xy (...,2), yaw (...) → corners (...,4,2)."""
    hl, hw = 0.5 * length, 0.5 * width
    local = np.array(
        [[hl, hw], [hl, -hw], [-hl, -hw], [-hl, hw]],
        dtype=np.float64,
    )  # (4,2)
    c = np.cos(yaw)[..., None]
    s = np.sin(yaw)[..., None]
    # rotate local → world offset
    rx = c * local[..., 0] - s * local[..., 1]
    ry = s * local[..., 0] + c * local[..., 1]
    return np.stack([xy[..., 0:1] + rx, xy[..., 1:2] + ry], axis=-1)


def _estimate_yaw(xy: np.ndarray) -> np.ndarray:
    """xy (G,T,2) → yaw (G,T) from finite differences; prepend first delta."""
    d = np.diff(xy, axis=1)
    yaw_mid = np.arctan2(d[..., 1], d[..., 0])  # (G,T-1)
    # pad first with first delta yaw
    yaw0 = yaw_mid[..., :1]
    return np.concatenate([yaw0, yaw_mid], axis=-1)


def _pairwise_progress(
    raw_prog: np.ndarray,
    multi_prod: np.ndarray,
    thresh: float,
) -> np.ndarray:
    """PDMS-style pairwise progress vs index-0 GT. raw_prog/multi_prod shape (G,)."""
    raw = raw_prog * multi_prod
    raw_gt = raw[0]
    cand = raw[1:]
    prod = multi_prod[1:]
    max_pair = np.maximum(raw_gt, cand)
    return np.where(
        max_pair > thresh,
        cand / (max_pair + 1e-6),
        np.where(prod == 0.0, 0.0, 1.0),
    ).astype(np.float64)


def score_batch(
    trajs_ego: np.ndarray,
    sample: VertiMetricCache,
    world: WorldMaps,
    cfg: Optional[VertiPDMConfig] = None,
    *,
    first_is_gt: bool = True,
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """Score G ego-frame trajectories.

    Args:
      trajs_ego: (G, 8, 2|3) — if first_is_gt, index 0 is GT (for progress pairwise).
      sample: pose / goal context
      world: height + obstacle maps
      first_is_gt: PDMS convention (GT at index 0); returned scores are for [1:] if True,
                   else for all G.

    Returns:
      final_scores (G' ,) float32
      subscores dict with PDMS key names, each (G',)
    """
    cfg = cfg or VertiPDMConfig()
    trajs = np.asarray(trajs_ego, dtype=np.float64)
    if trajs.ndim != 3 or trajs.shape[1] != 8:
        raise ValueError(f"expected (G,8,C), got {trajs.shape}")
    G = trajs.shape[0]
    xy_ego = trajs[..., :2]
    if trajs.shape[-1] >= 3:
        yaw_ego = trajs[..., 2]
    else:
        yaw_ego = _estimate_yaw(xy_ego)

    pose = np.asarray(sample.pose, dtype=np.float64)
    h0 = float(pose[3])
    # ego yaw → world yaw
    yaw_w = h0 + yaw_ego
    xy_w = ego_to_world(xy_ego, pose)  # (G,8,2)

    # --- footprint obstacle / OOB ---
    corners = _footprint_corners(
        xy_w, yaw_w, cfg.footprint_length, cfg.footprint_width
    )  # (G,8,4,2)
    flat = corners.reshape(G, -1, 2)
    hit = sample_obstacle(world, flat).reshape(G, -1)
    # also check waypoint centers
    hit_c = sample_obstacle(world, xy_w.reshape(-1, 2)).reshape(G, 8)
    no_crash = (~hit.any(axis=1) & ~hit_c.any(axis=1)).astype(np.float64)

    # --- slope / tip from heights ---
    z = sample_height(world, xy_w.reshape(-1, 2)).reshape(G, 8)
    # nan ⇒ OOB / missing → not traversable
    nan_mask = np.isnan(z)
    z_fill = np.where(nan_mask, 0.0, z)
    ds = np.linalg.norm(np.diff(xy_w, axis=1), axis=-1).clip(min=1e-3)  # (G,7)
    dz = np.diff(z_fill, axis=1)
    slope = np.abs(np.arctan2(np.abs(dz), ds))  # radians
    slope_max = np.deg2rad(cfg.slope_max_deg)
    tip_max = np.deg2rad(cfg.tip_max_deg)
    # local tip estimate: also use lateral height at footprint width
    # simpler: max |dz|/ds along path
    traversable = (
        (~nan_mask.any(axis=1))
        & (slope.max(axis=1) <= slope_max)
        & (no_crash > 0.5)  # crash already counted separately; keep independent
    )
    # make traversable independent of crash: only slope/height validity
    traversable = ((~nan_mask.any(axis=1)) & (slope.max(axis=1) <= slope_max)).astype(np.float64)

    tip_ratio = np.clip(slope.max(axis=1) / tip_max, 0.0, 1.0)
    tip_margin = 1.0 - tip_ratio

    # --- progress toward goal ---
    goal = np.asarray(sample.goal_xy, dtype=np.float64)
    start_xy = pose[:2]
    dist0 = np.linalg.norm(start_xy - goal)
    end_xy = xy_w[:, -1, :]
    dist_end = np.linalg.norm(end_xy - goal[None, :], axis=-1)
    raw_prog = np.maximum(dist0 - dist_end, 0.0)  # meters closed toward goal

    # --- comfort: accel + curvature proxies ---
    # speed from ds / dt
    v = ds / H_DT  # (G,7)
    # pad to 8 for accel
    v_pad = np.concatenate([v[:, :1], v], axis=1)  # (G,8)
    accel = np.abs(np.diff(v_pad, axis=1)) / H_DT  # (G,7)
    # curvature κ ≈ |Δheading| / ds
    dyaw = np.abs(np.diff(np.unwrap(yaw_w, axis=1), axis=1))
    curv = dyaw / ds
    comfort = np.exp(
        -accel.mean(axis=1) / max(cfg.comfort_accel_ref, 1e-3)
        - curv.mean(axis=1) / max(cfg.comfort_curv_ref, 1e-3)
    )
    comfort = np.clip(comfort, 0.0, 1.0)

    # --- goal align: mean cos of heading vs goal bearing ---
    to_goal = goal[None, None, :] - xy_w  # (G,8,2)
    bearing = np.arctan2(to_goal[..., 1], to_goal[..., 0])
    ang = np.arctan2(np.sin(yaw_w - bearing), np.cos(yaw_w - bearing))
    goal_align = np.clip(np.cos(ang).mean(axis=1), 0.0, 1.0)

    multi = no_crash * traversable  # (G,)

    wsum = cfg.w_progress + cfg.w_tip + cfg.w_comfort + cfg.w_align
    if first_is_gt and G >= 2:
        progress = np.ones(G, dtype=np.float64)
        progress[1:] = _pairwise_progress(raw_prog, multi, cfg.progress_distance_threshold)
        # GT itself gets 1.0 for logging; scoring uses [1:]
    else:
        # normalize by max in group (no GT)
        denom = max(float(raw_prog.max()), cfg.progress_distance_threshold)
        progress = np.clip(raw_prog / denom, 0.0, 1.0)

    weighted = (
        cfg.w_progress * progress
        + cfg.w_tip * tip_margin
        + cfg.w_comfort * comfort
        + cfg.w_align * goal_align
    ) / wsum

    final = multi * weighted

    if first_is_gt and G >= 2:
        sl = slice(1, None)
    else:
        sl = slice(None)

    subscores = {
        "no_collision": no_crash[sl].astype(np.float32),
        "drivable_area": traversable[sl].astype(np.float32),
        "progress": progress[sl].astype(np.float32),
        "ttc": tip_margin[sl].astype(np.float32),
        "comfort": comfort[sl].astype(np.float32),
        "dir_weighted": goal_align[sl].astype(np.float32),
        "final": final[sl].astype(np.float32),
    }
    return final[sl].astype(np.float32), subscores


def get_verti_score_para(
    trajectory: Union[np.ndarray, "torch.Tensor"],
    caches: Sequence[Union[VertiMetricCache, dict]],
    bank: Optional[WorldMapBank] = None,
    cfg: Optional[VertiPDMConfig] = None,
    *,
    first_is_gt: bool = True,
):
    """Drop-in replacement for TrajectoryHead.get_pdm_score_para.

    Args:
      trajectory: (B, G, 8, 3) — last of G may be GT if first_is_gt=False and GT appended,
                  OR if first_is_gt=True, index 0 should be GT (caller cats GT at dim1=0).
                  Official DDV2 cats GT at the *end*; we accept that via first_is_gt=False
                  and reorder internally... Actually official does:
                    diffusion_output_with_gt = cat(diffusion, gt)  # GT last
                    then pairwise uses index 0 as GT in scorer.
                  Official pdm_score puts GT as proposal 0. Our train path will cat GT first.
      caches: length B VertiMetricCache (or dicts)
      first_is_gt: True if traj[:,0] is GT.

    Returns:
      scores (B, G_out) torch/numpy matching trajectory device if torch
      caches_out: list of caches
      sub_scores: list[dict] length B
    """
    try:
        import torch
        is_torch = isinstance(trajectory, torch.Tensor)
    except ImportError:
        torch = None
        is_torch = False

    if is_torch:
        device = trajectory.device
        traj_np = trajectory.detach().cpu().numpy()
    else:
        device = None
        traj_np = np.asarray(trajectory)

    bank = bank or WorldMapBank()
    cfg = cfg or VertiPDMConfig()
    B, G = traj_np.shape[:2]
    scores = []
    subs = []
    caches_out = []
    for b in range(B):
        c = caches[b]
        if isinstance(c, dict):
            c = VertiMetricCache.from_dict(c)
        caches_out.append(c)
        world = bank.get_for_sample(c)
        tb = traj_np[b]
        if not first_is_gt:
            # official appends GT last → move to front for pairwise
            tb = np.concatenate([tb[-1:], tb[:-1]], axis=0)
            sc, sub = score_batch(tb, c, world, cfg, first_is_gt=True)
            # sc/sub are for original candidates in order [0..G-2] matching diffusion then we need
            # to append GT score: recompute full including GT at end for reward_gt
            # After reorder tb=[GT, cand...], score_batch returns cand scores only.
            # Rebuild full (cand..., GT) for drop-in with official slicing reward[:,:-1]
            # Official: reward_group includes GT as last column.
            # Our score_batch with first_is_gt returns only candidates (G-1,).
            # Score GT alone for last column:
            sc_gt, sub_gt = score_batch(tb[:1], c, world, cfg, first_is_gt=False)
            sc_full = np.concatenate([sc, sc_gt], axis=0)
            sub_full = {k: np.concatenate([sub[k], sub_gt[k]], axis=0) for k in sub}
            scores.append(sc_full)
            subs.append(sub_full)
        else:
            sc, sub = score_batch(tb, c, world, cfg, first_is_gt=True)
            # return with GT score prepended for logging consistency → (G,) with GT at 0
            sc_gt, sub_gt = score_batch(tb[:1], c, world, cfg, first_is_gt=False)
            sc_full = np.concatenate([sc_gt, sc], axis=0)
            sub_full = {k: np.concatenate([sub_gt[k], sub[k]], axis=0) for k in sub}
            scores.append(sc_full)
            subs.append(sub_full)

    scores_np = np.stack(scores, axis=0).astype(np.float32)
    if is_torch:
        return torch.from_numpy(scores_np).to(device), caches_out, subs
    return scores_np, caches_out, subs


def compute_grpo_advantages(
    reward_group: np.ndarray,
    subscores: Dict[str, np.ndarray],
    reward_gt: np.ndarray,
    num_groups: int,
    ego_fut_mode: int,
) -> np.ndarray:
    """Mirror diffusiondrivev2_model_rl Inter-Anchor Truncated GRPO advantages.

    Args:
      reward_group: (B, G*N) candidate finals (no GT)
      subscores: each (B, G*N)
      reward_gt: (B, 1)
      returns advantages (B, G*N) with negatives truncated and hard -1 on crash
    """
    B = reward_group.shape[0]
    rg = reward_group.reshape(B, num_groups, ego_fut_mode)
    if num_groups >= 2:
        mean = rg.mean(axis=1, keepdims=True)
        std = np.nan_to_num(rg.std(axis=1, keepdims=True), nan=0.0)
        adv = (rg - mean) / (std + 1e-4)
    else:
        mean = rg.mean(axis=2, keepdims=True)
        std = np.nan_to_num(rg.std(axis=2, keepdims=True), nan=0.0)
        adv = (rg - mean) / (std + 1e-4)
    mask_pos = rg > (reward_gt.reshape(B, 1, 1) - 1e-6)
    adv = np.clip(adv, 0.0, None) * mask_pos.astype(np.float32)
    adv = np.nan_to_num(adv, nan=0.0)
    for k in ("no_collision", "drivable_area"):
        if k not in subscores:
            continue
        v = subscores[k].reshape(B, num_groups, ego_fut_mode)
        zero = v != 1
        adv = np.where(zero, -1.0, adv)
    return adv.reshape(B, num_groups * ego_fut_mode).astype(np.float32)
