"""VertiMetricCache: light-weight PDMS MetricCache replacement for off-road GRPO.

Stores per-world height/obstacle rasters and per-sample ego pose / goal / GT traj.
Maps use the same coordinate convention as verti_bench TerrainManager.
"""
from __future__ import annotations

import glob
import json
import os
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import yaml
from PIL import Image

MAPS_DIR = Path(
    "/home/rm/Desktop/df_offroad/verti_bench/envs/data/BenchMaps/sampled_maps/Configs/Final"
)


def _load_config(world_id: int) -> dict:
    matches = sorted(glob.glob(str(MAPS_DIR / f"config{world_id}_*.yaml")))
    if not matches:
        raise FileNotFoundError(f"no config for world {world_id} under {MAPS_DIR}")
    with open(matches[0]) as f:
        return yaml.safe_load(f)


def _load_height(world_id: int) -> np.ndarray:
    matches = sorted(glob.glob(str(MAPS_DIR / f"height{world_id}_*.npy")))
    if not matches:
        raise FileNotFoundError(f"no height map for world {world_id}")
    hd = np.load(matches[0]).astype(np.float32)
    # Match TerrainManager._load_high_res_data orientation.
    hd = np.flip(hd, axis=1)
    hd = np.rot90(hd, k=1, axes=(1, 0))
    hd = np.rot90(hd, k=1, axes=(1, 0))
    return np.ascontiguousarray(hd)


def _load_obstacle(world_id: int, difficulty: str) -> np.ndarray:
    path = MAPS_DIR / f"obs{world_id}_{difficulty}.bmp"
    if not path.is_file():
        raise FileNotFoundError(path)
    obs = np.array(Image.open(path).convert("L"))
    # 255 = obstacle (same as PID A*).
    return (obs == 255).astype(np.uint8)


@dataclass
class WorldMaps:
    world_id: int
    difficulty: str
    terrain_length: float
    terrain_width: float
    min_height: float
    max_height: float
    scale_factor: float
    height: np.ndarray  # (H,W) meters before scale
    obstacle: np.ndarray  # (h,w) uint8 {0,1}

    @property
    def height_m(self) -> np.ndarray:
        return self.height * self.scale_factor


def load_world_maps(world_id: int, scale_factor: float = 1.0, difficulty: Optional[str] = None) -> WorldMaps:
    cfg = _load_config(world_id)
    terr = cfg["terrain"]
    diff = difficulty or terr["difficulty"]
    return WorldMaps(
        world_id=int(world_id),
        difficulty=str(diff),
        terrain_length=float(terr["length"]) * scale_factor,
        terrain_width=float(terr["width"]) * scale_factor,
        min_height=float(terr["min_height"]) * scale_factor,
        max_height=float(terr["max_height"]) * scale_factor,
        scale_factor=float(scale_factor),
        height=_load_height(world_id),
        obstacle=_load_obstacle(world_id, diff),
    )


def chrono_to_grid(
    xy: np.ndarray,
    grid_h: int,
    grid_w: int,
    terrain_length: float,
    terrain_width: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """World (x,y) → floating grid (col,row) for a raster of shape (grid_h, grid_w)."""
    xy = np.asarray(xy, dtype=np.float64)
    s_norm_x = grid_w / (2.0 * terrain_length)
    s_norm_y = grid_h / (2.0 * terrain_width)
    vehicle_x = xy[..., 0]
    vehicle_y = -xy[..., 1]
    col = (vehicle_x + terrain_length) * s_norm_x
    row = (vehicle_y + terrain_width) * s_norm_y
    return col, row


def sample_height(world: WorldMaps, xy: np.ndarray) -> np.ndarray:
    """Bilinear sample height (m) at world xy; out-of-bounds → nan."""
    hmap = world.height_m
    H, W = hmap.shape
    col, row = chrono_to_grid(xy, H, W, world.terrain_length, world.terrain_width)
    out = np.full(xy.shape[:-1], np.nan, dtype=np.float64)
    finite = np.isfinite(col) & np.isfinite(row)
    if not np.any(finite):
        return out
    c0 = np.floor(np.where(finite, col, 0.0)).astype(np.int64)
    r0 = np.floor(np.where(finite, row, 0.0)).astype(np.int64)
    c1 = c0 + 1
    r1 = r0 + 1
    valid = finite & (c0 >= 0) & (r0 >= 0) & (c1 < W) & (r1 < H)
    if not np.any(valid):
        return out
    c0v, r0v, c1v, r1v = c0[valid], r0[valid], c1[valid], r1[valid]
    dc = col[valid] - c0v
    dr = row[valid] - r0v
    z00 = hmap[r0v, c0v]
    z10 = hmap[r0v, c1v]
    z01 = hmap[r1v, c0v]
    z11 = hmap[r1v, c1v]
    out[valid] = (
        z00 * (1 - dc) * (1 - dr)
        + z10 * dc * (1 - dr)
        + z01 * (1 - dc) * dr
        + z11 * dc * dr
    )
    return out


def sample_obstacle(world: WorldMaps, xy: np.ndarray) -> np.ndarray:
    """Nearest obstacle occupancy at world xy; OOB → True (treat as crash)."""
    obs = world.obstacle
    H, W = obs.shape
    col, row = chrono_to_grid(xy, H, W, world.terrain_length, world.terrain_width)
    finite = np.isfinite(col) & np.isfinite(row)
    hit = np.ones(xy.shape[:-1], dtype=bool)
    if not np.any(finite):
        return hit
    ci = np.where(finite, np.rint(col), 0).astype(np.int64)
    ri = np.where(finite, np.rint(row), 0).astype(np.int64)
    valid = finite & (ci >= 0) & (ri >= 0) & (ci < W) & (ri < H)
    hit[valid] = obs[ri[valid], ci[valid]].astype(bool)
    return hit


@dataclass
class VertiMetricCache:
    """Per-sample scoring context (maps referenced by world_id)."""

    token: str
    world_id: int
    difficulty: str
    scale_factor: float
    pose: np.ndarray  # (4,) x,y,z,heading
    goal_xy: np.ndarray  # (2,)
    gt_traj: np.ndarray  # (8,3) ego-frame
    seq: str = ""
    frame_idx: int = -1
    ep_status: str = ""

    def to_dict(self) -> dict:
        d = asdict(self)
        for k in ("pose", "goal_xy", "gt_traj"):
            d[k] = np.asarray(d[k]).tolist()
        return d

    @staticmethod
    def from_dict(d: dict) -> "VertiMetricCache":
        return VertiMetricCache(
            token=d["token"],
            world_id=int(d["world_id"]),
            difficulty=str(d["difficulty"]),
            scale_factor=float(d.get("scale_factor", 1.0)),
            pose=np.asarray(d["pose"], dtype=np.float64),
            goal_xy=np.asarray(d["goal_xy"], dtype=np.float64),
            gt_traj=np.asarray(d["gt_traj"], dtype=np.float32),
            seq=d.get("seq", ""),
            frame_idx=int(d.get("frame_idx", -1)),
            ep_status=d.get("ep_status", ""),
        )


class WorldMapBank:
    """Lazy cache of WorldMaps keyed by (world_id, difficulty, scale)."""

    def __init__(self):
        self._bank: Dict[Tuple[int, str, float], WorldMaps] = {}

    def get(self, world_id: int, difficulty: str, scale_factor: float = 1.0) -> WorldMaps:
        key = (int(world_id), str(difficulty), float(scale_factor))
        if key not in self._bank:
            self._bank[key] = load_world_maps(world_id, scale_factor, difficulty)
        return self._bank[key]

    def get_for_sample(self, sample: VertiMetricCache) -> WorldMaps:
        return self.get(sample.world_id, sample.difficulty, sample.scale_factor)


def ego_to_world(traj_ego: np.ndarray, pose: np.ndarray) -> np.ndarray:
    """(G,T,2|3) ego → world xy. pose = (x,y,z,heading)."""
    traj = np.asarray(traj_ego, dtype=np.float64)
    xy = traj[..., :2]
    x0, y0, h0 = float(pose[0]), float(pose[1]), float(pose[3])
    c, s = np.cos(h0), np.sin(h0)
    # ego: x forward, y left → world
    wx = x0 + c * xy[..., 0] - s * xy[..., 1]
    wy = y0 + s * xy[..., 0] + c * xy[..., 1]
    return np.stack([wx, wy], axis=-1)


def save_index(path: Path, records: list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


def load_index(path: Path) -> list:
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows
