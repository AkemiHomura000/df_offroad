"""Collect expert trajectories in Verti-Bench with its built-in planners.

Rows are appended + flushed to traj.csv every sample, so Ctrl+C (or SIGTERM)
at any moment loses at most the current row; the episode meta.json is then
marked "interrupted".

Layout:
  OUT/index.jsonl                     one line per finished/interrupted episode
  OUT/ep_{N:05d}_w{world}_{system}/
      meta.json                       config, start/goal, status, result
      traj.csv                        t,x,y,z,roll,pitch,yaw,heading,speed,vx,vy,vz,wz,steer,throttle,brake,goal_x,goal_y,dist_goal
      astar_path.npy                  initial global path (N,2), chrono coords
      obs/obs_{k:05d}.npz             (--sensors) at --obs_hz:
                                      cam_front uint8 (256,1024,3),
                                      lidar_density (1,256,256) count hist in [0,1],
                                      lidar_elevation (1,256,256) max-z chassis frame (empty=0),
                                      lidar_feature (=density, DDV2 compat),
                                      status_feature (8,), pose [x,y,z,heading], t, speed
      snapshot.png                    mid-episode cam | density | elevation collage

Turn episodes into DDV2 training samples with build_verti_ddv2_dataset.py.

Example:
  python collect_expert_verti.py --systems mppi6,pid --worlds low --goal_mode random --episodes 0
"""
import argparse
import csv
import glob
import json
import os
import random
import re
import signal
import subprocess
import sys
import time as wall

import numpy as np
from PIL import Image
from scipy.ndimage import binary_dilation

from verti_bench.envs.utils.utils import SetChronoDataDirectories

ASTAR_SYSTEMS = ['pid', 'eh', 'mppi', 'mppi6', 'tal', 'tnt', 'wmvct']
CFG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       '../../verti_bench/envs/data/BenchMaps/sampled_maps/Configs/Final')
FIELDS = ['t', 'x', 'y', 'z', 'roll', 'pitch', 'yaw', 'heading', 'speed', 'vx', 'vy', 'vz', 'wz',
          'steer', 'throttle', 'brake', 'goal_x', 'goal_y', 'dist_goal']


def make_sim(system, config):
    if system == 'pid':
        from verti_bench.systems.PID.PID_sim import PIDSim as S
    elif system == 'eh':
        from verti_bench.systems.EH.EH_sim import EHSim as S
    elif system == 'mppi':
        from verti_bench.systems.MPPI.MPPI_sim import MPPISim as S
    elif system == 'mppi6':
        from verti_bench.systems.MPPI6.MPPI6_sim import MPPI6Sim as S
    elif system == 'tal':
        from verti_bench.systems.TAL.TAL_sim import TALSim as S
    elif system == 'tnt':
        from verti_bench.systems.TNT.TNT_sim import TNTSim as S
    elif system == 'wmvct':
        from verti_bench.systems.WMVCT.WMVCT_sim import WMVCTSim as S
    else:
        raise ValueError(f'unsupported system {system}; choose from {ASTAR_SYSTEMS}')
    return S(config)


def parse_worlds(spec):
    """'1,5,9' | '3-12' | 'low' | 'mid' | 'high' | 'all' (comma-combinable)."""
    worlds = []
    for tok in spec.split(','):
        tok = tok.strip()
        if tok in ('low', 'mid', 'high', 'all'):
            pat = '*' if tok == 'all' else tok
            for f in glob.glob(os.path.join(CFG_DIR, f'config*_{pat}.yaml')):
                worlds.append(int(re.search(r'config(\d+)_', os.path.basename(f)).group(1)))
        elif '-' in tok:
            a, b = map(int, tok.split('-'))
            worlds += list(range(a, b + 1))
        elif tok:
            worlds.append(int(tok))
    return sorted(set(worlds))


def parse_xy(s):
    return [float(v) for v in s.split(',')] if s else None


def sample_random_task(tm, rng, min_dist, max_dist, margin, inflate):
    """Sample start/goal (scaled chrono coords) on obstacle-free cells."""
    obs = np.array(Image.open(tm.obs_path).convert('L'))
    occ = binary_dilation(obs == 255, structure=np.ones((2 * inflate + 1,) * 2))
    h, w = occ.shape
    m = int(np.ceil(margin / (2 * tm.terrain_length) * w))
    free = np.argwhere(~occ[m:h - m, m:w - m]) + m  # (row, col)
    for _ in range(500):
        a, b = free[rng.integers(len(free))], free[rng.integers(len(free))]
        (sx, sy), (gx, gy) = tm.transform_to_chrono([(a[1], a[0]), (b[1], b[0])])
        d = np.hypot(gx - sx, gy - sy)
        if min_dist <= d <= max_dist:
            return [sx, sy], [gx, gy]
    raise RuntimeError('could not sample a start/goal pair; relax --min_dist/--max_dist')


def choose_task(args, sim, rng, ep):
    tm, s = sim.terrain_manager, sim.scale_factor
    z = 1.0 * s
    if args.goal_mode == 'fixed':
        st, gl = parse_xy(args.start), parse_xy(args.goal)
        return st[:2] + [z], gl[:2] + [z], 'fixed'
    if args.goal_mode == 'yaml':
        pairs = tm.positions
        i = ep % len(pairs) if args.yaml_cycle else int(rng.integers(len(pairs)))
        return ([v * s for v in pairs[i]['start']], [v * s for v in pairs[i]['goal']], f'yaml#{i}')
    for _ in range(20):
        st, gl = sample_random_task(tm, rng, args.min_dist * s, args.max_dist * s,
                                    args.margin * s, args.inflate)
        st, gl = st + [z], gl + [z]
        if not args.check_path or sim.planner.astar_path(tm.obs_path, st, gl) is not None:
            return st, gl, 'random'
    raise RuntimeError('no A*-feasible random task after 20 tries')


def chassis_heading(body):
    ax = body.GetRot().GetAxisX()
    return float(np.arctan2(ax.y, ax.x))


BEV_RANGE, BEV_PPM = 32.0, 4.0


def _bev_to_rgb(arr: np.ndarray, cmap: str = 'gray', vmin=None, vmax=None) -> np.ndarray:
    """(H,W) float → (H,W,3) uint8 for snapshot. Forward is up in the image."""
    import matplotlib
    a = np.asarray(arr, dtype=np.float32)
    if a.ndim == 3:
        a = a[0]
    a = a[::-1, ::-1]  # fwd-up, left-left
    lo = float(np.nanmin(a)) if vmin is None else vmin
    hi = float(np.nanmax(a)) if vmax is None else vmax
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo, hi = 0.0, 1.0
    norm = np.clip((a - lo) / (hi - lo), 0, 1)
    rgb = (matplotlib.colormaps[cmap](norm)[..., :3] * 255).astype(np.uint8)
    return rgb


def _load_traj_csv(path):
    with open(path) as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return None
    return {k: np.array([float(r[k]) for r in rows]) for k in rows[0]}


def _world_to_ego(xy, pose):
    """World XY (N,2) → ego frame at pose [x,y,z,heading]."""
    x0, y0, h0 = float(pose[0]), float(pose[1]), float(pose[3])
    c, s = np.cos(h0), np.sin(h0)
    dx, dy = xy[:, 0] - x0, xy[:, 1] - y0
    return np.stack([c * dx + s * dy, -s * dx + c * dy], 1)


def _ego_to_bev_px(ego_xy, size):
    """Ego (x fwd, y left) → pixel (col, row) on a size×size fwd-up image."""
    scale = size / (2 * BEV_RANGE)
    col = (BEV_RANGE - ego_xy[:, 1]) * scale
    row = (BEV_RANGE - ego_xy[:, 0]) * scale
    return np.stack([col, row], 1)


def _draw_polyline(draw, pts, color, width=2):
    if pts is None or len(pts) < 2:
        return
    xy = [(float(p[0]), float(p[1])) for p in pts]
    draw.line(xy, fill=color, width=width)


def _overlay_ego_traj(rgb, pose, tr, t0, goal_xy=None):
    """Draw past (cyan) + future (lime) ego-frame traj and goal on a BEV RGB image."""
    from PIL import Image, ImageDraw
    img = Image.fromarray(rgb.copy())
    draw = ImageDraw.Draw(img)
    size = img.size[0]
    if tr is None:
        return np.array(img)
    xy = np.stack([tr['x'], tr['y']], 1)
    k = int(np.searchsorted(tr['t'], t0, side='right'))
    past = _ego_to_bev_px(_world_to_ego(xy[: max(k, 1)], pose), size)
    fut = _ego_to_bev_px(_world_to_ego(xy[k:], pose), size) if k < len(xy) else None
    _draw_polyline(draw, past, (0, 220, 255), 2)
    _draw_polyline(draw, fut, (50, 255, 50), 2)
    # ego marker
    c0, r0 = _ego_to_bev_px(np.array([[0.0, 0.0]]), size)[0]
    r = 4
    draw.ellipse([c0 - r, r0 - r, c0 + r, r0 + r], fill=(255, 40, 40))
    if goal_xy is not None:
        ge = _ego_to_bev_px(_world_to_ego(np.asarray(goal_xy, dtype=np.float64).reshape(1, 2), pose), size)[0]
        if 0 <= ge[0] < size and 0 <= ge[1] < size:
            draw.regular_polygon((ge[0], ge[1], 6), 3, fill=(255, 215, 0))
    return np.array(img)


def _world_traj_panel(size, tr, pose, start, goal, astar=None):
    """Top-down world map with A*, driven path, start/goal, current pose."""
    from PIL import Image, ImageDraw
    img = Image.new('RGB', (size, size), (30, 30, 30))
    draw = ImageDraw.Draw(img)
    pts = []
    if tr is not None:
        pts.append(np.stack([tr['x'], tr['y']], 1))
    if start is not None:
        pts.append(np.asarray(start[:2], dtype=np.float64).reshape(1, 2))
    if goal is not None:
        pts.append(np.asarray(goal[:2], dtype=np.float64).reshape(1, 2))
    if astar is not None and len(astar):
        pts.append(np.asarray(astar)[:, :2])
    if not pts:
        return np.array(img)
    all_xy = np.concatenate(pts, 0)
    pad = 8.0
    xmin, ymin = all_xy.min(0) - pad
    xmax, ymax = all_xy.max(0) + pad
    span = max(xmax - xmin, ymax - ymin, 1.0)
    cx, cy = 0.5 * (xmin + xmax), 0.5 * (ymin + ymax)

    def to_px(xy):
        col = (xy[:, 0] - (cx - 0.5 * span)) / span * (size - 1)
        row = ((cy + 0.5 * span) - xy[:, 1]) / span * (size - 1)  # y-up
        return np.stack([col, row], 1)

    if astar is not None and len(astar):
        _draw_polyline(draw, to_px(np.asarray(astar)[:, :2]), (180, 180, 180), 1)
    if tr is not None:
        xy = np.stack([tr['x'], tr['y']], 1)
        _draw_polyline(draw, to_px(xy), (80, 80, 120), 1)
        d2 = (xy[:, 0] - pose[0]) ** 2 + (xy[:, 1] - pose[1]) ** 2
        k = int(np.argmin(d2)) + 1
        _draw_polyline(draw, to_px(xy[:k]), (40, 140, 255), 3)
    if start is not None:
        p = to_px(np.asarray(start[:2], dtype=np.float64).reshape(1, 2))[0]
        draw.ellipse([p[0] - 5, p[1] - 5, p[0] + 5, p[1] + 5], fill=(0, 220, 80))
    if goal is not None:
        p = to_px(np.asarray(goal[:2], dtype=np.float64).reshape(1, 2))[0]
        draw.regular_polygon((p[0], p[1], 7), 3, fill=(255, 60, 60))
    # current pose arrow
    p = to_px(np.asarray(pose[:2], dtype=np.float64).reshape(1, 2))[0]
    hdg = float(pose[3])
    L = 14
    tip = (p[0] + L * np.cos(hdg), p[1] - L * np.sin(hdg))
    draw.line([(p[0], p[1]), tip], fill=(255, 220, 0), width=3)
    draw.ellipse([p[0] - 3, p[1] - 3, p[0] + 3, p[1] + 3], fill=(255, 220, 0))
    return np.array(img)


def write_episode_snapshot(ep_dir, obs_idx=None):
    """Collage cam | dens+traj | elev+traj | world map → ep_dir/snapshot.png."""
    from PIL import Image, ImageDraw, ImageFont
    obs_files = sorted(glob.glob(os.path.join(ep_dir, 'obs', 'obs_*.npz')))
    if not obs_files:
        return None
    if obs_idx is None:
        obs_idx = len(obs_files) // 2
    obs_idx = int(np.clip(obs_idx, 0, len(obs_files) - 1))
    d = np.load(obs_files[obs_idx])
    cam = d['cam_front']
    pose = d['pose']
    t0 = float(d['t'])
    dens = d['lidar_density'] if 'lidar_density' in d.files else d['lidar_feature']
    elev = d['lidar_elevation'] if 'lidar_elevation' in d.files else np.zeros_like(dens)
    elev_vis = elev[0].copy()
    mask = dens[0] > 1e-6
    if mask.any():
        vals = elev_vis[mask]
        vmin, vmax = float(vals.min()), float(vals.max())
    else:
        vmin, vmax = -2.0, 2.0
    elev_vis = np.where(mask, elev_vis, vmin)

    tr = _load_traj_csv(os.path.join(ep_dir, 'traj.csv'))
    meta = {}
    mp = os.path.join(ep_dir, 'meta.json')
    if os.path.isfile(mp):
        meta = json.load(open(mp))
    start, goal = meta.get('start'), meta.get('goal')
    astar = None
    ap = os.path.join(ep_dir, 'astar_path.npy')
    if os.path.isfile(ap):
        astar = np.load(ap)

    h = cam.shape[0]
    dens_rgb = np.array(Image.fromarray(_bev_to_rgb(dens, 'gray', 0, 1)).resize((h, h), Image.BILINEAR))
    elev_rgb = np.array(Image.fromarray(_bev_to_rgb(elev_vis, 'terrain', vmin, vmax)).resize((h, h), Image.BILINEAR))
    goal_xy = goal[:2] if goal is not None else None
    dens_rgb = _overlay_ego_traj(dens_rgb, pose, tr, t0, goal_xy)
    elev_rgb = _overlay_ego_traj(elev_rgb, pose, tr, t0, goal_xy)
    world = _world_traj_panel(h, tr, pose, start, goal, astar)

    gap = np.full((h, 8, 3), 32, dtype=np.uint8)
    collage = np.concatenate([cam, gap, dens_rgb, gap, elev_rgb, gap, world], axis=1)
    img = Image.fromarray(collage)
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None
    x0 = 0
    panels = [
        (cam.shape[1], 'camera'),
        (h, 'density + traj'),
        (h, f'elev [{vmin:.1f},{vmax:.1f}]m + traj'),
        (h, 'world: A*(gray) driven(blue) start(g) goal(r)'),
    ]
    x = 8
    for w, text in panels:
        draw.text((x, 4), text, fill=(255, 255, 0), font=font)
        x += w + 8
    out = os.path.join(ep_dir, 'snapshot.png')
    img.save(out)
    return out


class SensorRecorder:
    """Camera + lidar density/elevation; one atomic npz per obs under ep_dir/obs/."""

    def __init__(self, sim, ep_dir, hz):
        from verti_bench.envs.utils.ddv2_sensors import DDV2SensorSuite
        self.sim, self.dt, self.last_t = sim, 1.0 / hz, -1e9
        self.ep_dir = ep_dir
        self.dir = os.path.join(ep_dir, 'obs')
        os.makedirs(self.dir, exist_ok=True)
        self.suite = DDV2SensorSuite(sim.system, sim.vehicle_manager.chassis_body)
        self.n = 0

    def step(self, t):
        self.suite.update()
        if t - self.last_t < self.dt - 1e-9:
            return
        vm = self.sim.vehicle_manager
        obs = self.suite.get_observation(goal=vm.goal)
        if obs is None:  # sensor buffers not filled yet; retry next step
            return
        self.last_t = t
        body = vm.chassis_body
        p = body.GetPos()
        dens = obs['lidar_density'].astype(np.float32)
        elev = obs['lidar_elevation'].astype(np.float32)
        path = os.path.join(self.dir, f'obs_{self.n:05d}.npz')
        with open(path + '.tmp', 'wb') as f:
            np.savez_compressed(
                f,
                cam_front=obs['cam_front'].astype(np.uint8),
                lidar_density=dens,
                lidar_elevation=elev,
                lidar_feature=dens,  # DDV2 / legacy alias
                status_feature=obs['status_feature'].astype(np.float32),
                pose=np.array([p.x, p.y, p.z, chassis_heading(body)], dtype=np.float64),
                t=np.float64(t),
                speed=np.float32(vm.get_speed()),
            )
        os.replace(path + '.tmp', path)
        self.n += 1

    def write_snapshot(self):
        return write_episode_snapshot(self.ep_dir)


class Recorder:
    def __init__(self, sim, ep_dir, hz, sensors=None):
        self.sim, self.dt, self.last_t = sim, 1.0 / hz, -1e9
        self.sensors = sensors
        self.f = open(os.path.join(ep_dir, 'traj.csv'), 'w', newline='')
        self.w = csv.writer(self.f)
        self.w.writerow(FIELDS)
        self.f.flush()
        self.n = 0

    def hook(self, orig_advance):
        def advance(step):
            orig_advance(step)
            t = self.sim.system.GetChTime()
            if self.sensors is not None:
                self.sensors.step(t)
            if t - self.last_t >= self.dt - 1e-9:
                self.last_t = t
                self.sample(t)
        return advance

    def sample(self, t):
        vm = self.sim.vehicle_manager
        p = vm.get_position()
        e = vm.get_rotation()
        body = vm.chassis_body
        v = body.GetPosDt()
        wz = body.GetAngVelParent().z
        di = self.sim.driver_inputs
        g = vm.goal
        self.w.writerow([f'{t:.3f}', f'{p.x:.4f}', f'{p.y:.4f}', f'{p.z:.4f}',
                         f'{e.x:.5f}', f'{e.y:.5f}', f'{e.z:.5f}', f'{chassis_heading(body):.5f}',
                         f'{vm.get_speed():.4f}',
                         f'{v.x:.4f}', f'{v.y:.4f}', f'{v.z:.4f}', f'{wz:.5f}',
                         f'{di.m_steering:.4f}', f'{di.m_throttle:.4f}', f'{di.m_braking:.4f}',
                         f'{g.x:.3f}', f'{g.y:.3f}', f'{(g - p).Length():.3f}'])
        self.f.flush()
        self.n += 1

    def close(self):
        if not self.f.closed:
            self.f.flush()
            os.fsync(self.f.fileno())
            self.f.close()


def write_json(path, obj):
    tmp = path + '.tmp'
    with open(tmp, 'w') as f:
        json.dump(obj, f, indent=2)
    os.replace(tmp, path)


def next_episode_id(out):
    ids = [int(m.group(1)) for d in os.listdir(out) if (m := re.match(r'ep_(\d+)_', d))]
    return max(ids) + 1 if ids else 0


def run_episode(args, ep, world, system, rng):
    config = dict(vehicle=args.vehicle, speed=args.speed, system=system, world_id=world,
                  max_time=args.max_time, scale_factor=args.scale_factor, render=args.render,
                  use_gui=False)
    sim = make_sim(system, config)
    start, goal, src = choose_task(args, sim, rng, ep)

    ep_dir = os.path.join(args.out, f'ep_{ep:05d}_w{world}_{system}')
    os.makedirs(ep_dir, exist_ok=True)
    meta = dict(episode=ep, world=world, difficulty=sim.terrain_manager.difficulty,
                system=system, vehicle=args.vehicle, speed=args.speed,
                scale_factor=args.scale_factor, max_time=args.max_time,
                task_source=src, start=start, goal=goal, sample_hz=args.hz,
                status='running', wall_start=wall.time())
    write_json(os.path.join(ep_dir, 'meta.json'), meta)
    print(f'[ep {ep}] world={world} ({meta["difficulty"]}) system={system} '
          f'start={np.round(start[:2], 1)} goal={np.round(goal[:2], 1)} [{src}]', flush=True)

    rec = srec = None
    try:
        sim.initialize(start_pos=start, goal_pos=goal)
        if getattr(sim, 'chrono_path', None) is None:
            meta.update(status='no_path')
            return meta
        np.save(os.path.join(ep_dir, 'astar_path.npy'), np.asarray(sim.chrono_path, dtype=np.float32))
        if args.sensors:
            srec = SensorRecorder(sim, ep_dir, args.obs_hz)
        rec = Recorder(sim, ep_dir, args.hz, srec)
        sim.vehicle_manager.advance = rec.hook(sim.vehicle_manager.advance)
        ttg, success, roll, pitch = sim.run()
        meta.update(status='success' if success else 'fail', time=ttg,
                    avg_roll_deg=float(roll), avg_pitch_deg=float(pitch))
        return meta
    except KeyboardInterrupt:
        meta.update(status='interrupted')
        raise
    except Exception as ex:  # keep collecting on per-episode crashes
        meta.update(status='error', error=repr(ex))
        print(f'[ep {ep}] error: {ex!r}', flush=True)
        return meta
    finally:
        if rec is not None:
            rec.close()
            meta['n_samples'] = rec.n
        if srec is not None:
            meta['n_obs'] = srec.n
            try:
                snap = srec.write_snapshot()
                if snap:
                    meta['snapshot'] = os.path.basename(snap)
            except Exception as ex:
                print(f'[ep {ep}] snapshot failed: {ex!r}', flush=True)
        meta['wall_time'] = wall.time() - meta['wall_start']
        write_json(os.path.join(ep_dir, 'meta.json'), meta)
        with open(os.path.join(args.out, 'index.jsonl'), 'a') as f:
            f.write(json.dumps({k: meta.get(k) for k in
                                ('episode', 'world', 'system', 'status', 'time', 'n_samples', 'n_obs',
                                 'start', 'goal', 'task_source')}) + '\n')
        if args.render and getattr(sim, 'vis', None) is not None:
            try:
                sim.vis.Quit()
            except Exception:
                pass
        print(f'[ep {ep}] -> {meta["status"]} samples={meta.get("n_samples", 0)} '
              f'obs={meta.get("n_obs", 0)} dir={ep_dir}',
              flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default='/home/rm/Desktop/df_offroad/log/verti_expert')
    ap.add_argument('--systems', default='mppi6', help=f'comma list from {ASTAR_SYSTEMS}')
    ap.add_argument('--worlds', default='low', help="e.g. '5,9,11' | '1-20' | low|mid|high|all")
    ap.add_argument('--episodes', type=int, default=0, help='total episodes; 0 = until Ctrl+C')
    ap.add_argument('--goal_mode', choices=['random', 'yaml', 'fixed'], default='random')
    ap.add_argument('--yaml_cycle', action='store_true', help='yaml mode: cycle pairs instead of random')
    ap.add_argument('--start', default='', help='fixed mode: "x,y" (unscaled chrono m)')
    ap.add_argument('--goal', default='', help='fixed mode: "x,y"')
    ap.add_argument('--min_dist', type=float, default=40.0, help='random mode start-goal min dist (m)')
    ap.add_argument('--max_dist', type=float, default=100.0)
    ap.add_argument('--margin', type=float, default=8.0, help='keep away from map border (m)')
    ap.add_argument('--inflate', type=int, default=2, help='obstacle inflation (px) when sampling')
    ap.add_argument('--check_path', type=lambda x: x.lower() == 'true', default=True,
                    help='random mode: reject tasks without an A* path')
    ap.add_argument('--vehicle', default='hmmwv')
    ap.add_argument('--speed', type=float, default=4.0)
    ap.add_argument('--max_time', type=float, default=60.0)
    ap.add_argument('--scale_factor', type=float, default=1.0)
    ap.add_argument('--render', type=lambda x: x.lower() == 'true', default=False)
    ap.add_argument('--hz', type=float, default=10.0, help='traj.csv recording rate')
    ap.add_argument('--sensors', type=lambda x: x.lower() == 'true', default=True,
                    help='record DDV2 camera/lidar/status observations (needs Chrono::Sensor GPU)')
    ap.add_argument('--obs_hz', type=float, default=2.0,
                    help='observation rate; 2 Hz matches the 0.5 s waypoint spacing')
    ap.add_argument('--seed', type=int, default=None)
    ap.add_argument('--hang_timeout', type=float, default=180.0,
                    help='kill an episode whose files stop updating for this many seconds')
    ap.add_argument('--_worker', default='', help=argparse.SUPPRESS)
    args = ap.parse_args()

    if args.goal_mode == 'fixed' and not (args.start and args.goal):
        ap.error('--goal_mode fixed needs --start and --goal')
    if args.goal_mode == 'fixed':
        args.start = ','.join(str(v * args.scale_factor) for v in parse_xy(args.start))
        args.goal = ','.join(str(v * args.scale_factor) for v in parse_xy(args.goal))
    systems = [s.strip().lower() for s in args.systems.split(',') if s.strip()]
    for s in systems:
        if s not in ASTAR_SYSTEMS:
            ap.error(f'unsupported system {s}; choose from {ASTAR_SYSTEMS}')
    worlds = parse_worlds(args.worlds)
    if not worlds:
        ap.error(f'no worlds matched {args.worlds}')

    def on_term(*_):
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, on_term)
    signal.signal(signal.SIGINT, on_term)

    if args._worker:
        run_worker(args)
        return

    os.makedirs(args.out, exist_ok=True)
    seed = args.seed if args.seed is not None else int(wall.time())
    rng = np.random.default_rng(seed)

    print(f'out={args.out} systems={systems} worlds={worlds} goal_mode={args.goal_mode} seed={seed}')
    ep, done = next_episode_id(args.out), 0
    stats = {}
    try:
        while args.episodes <= 0 or done < args.episodes:
            world = worlds[int(rng.integers(len(worlds)))]
            system = systems[done % len(systems)]
            status = launch_episode(args, ep, world, system, int(rng.integers(2**31)))
            stats[status] = stats.get(status, 0) + 1
            ep += 1
            done += 1
            print(f'[progress] {done} episodes  {stats}', flush=True)
    except KeyboardInterrupt:
        print(f'\n[stopped] interrupted after {done} complete episodes  {stats}', flush=True)
        sys.exit(0)


def run_worker(args):
    """One episode per process: Chrono::Sensor/OptiX state is not reclaimed between
    episodes in a single process and eventually deadlocks sensor creation."""
    ep, world, system, seed = args._worker.split(',')
    SetChronoDataDirectories()
    random.seed(int(seed))
    try:
        run_episode(args, int(ep), int(world), system, np.random.default_rng(int(seed)))
    except KeyboardInterrupt:
        pass
    sys.stdout.flush()
    os._exit(0)  # skip Chrono/OptiX destructors, which can hang at teardown


def latest_mtime(ep_dir):
    paths = [os.path.join(ep_dir, n) for n in ('meta.json', 'traj.csv', 'obs')]
    return max((os.path.getmtime(p) for p in paths if os.path.exists(p)), default=0.0)


def mark_episode(args, ep_dir, status):
    mp = os.path.join(ep_dir, 'meta.json')
    try:
        meta = json.load(open(mp))
    except (OSError, ValueError):
        return
    if meta.get('status') != 'running':
        return
    meta['status'] = status
    write_json(mp, meta)
    with open(os.path.join(args.out, 'index.jsonl'), 'a') as f:
        f.write(json.dumps({k: meta.get(k) for k in
                            ('episode', 'world', 'system', 'status', 'start', 'goal',
                             'task_source')}) + '\n')


def launch_episode(args, ep, world, system, seed):
    ep_dir = os.path.join(args.out, f'ep_{ep:05d}_w{world}_{system}')
    argv = [a for a in sys.argv[1:]]
    cmd = [sys.executable, os.path.abspath(__file__), *argv, '--_worker', f'{ep},{world},{system},{seed}']
    proc = subprocess.Popen(cmd, start_new_session=True)
    t_start = wall.time()
    try:
        while proc.poll() is None:
            wall.sleep(2.0)
            last = max(latest_mtime(ep_dir), t_start)
            if wall.time() - last > args.hang_timeout:
                print(f'[ep {ep}] no progress for {args.hang_timeout:.0f}s -> kill (hang)', flush=True)
                proc.kill()
                proc.wait()
                mark_episode(args, ep_dir, 'hang')
                return 'hang'
    except KeyboardInterrupt:
        proc.send_signal(signal.SIGINT)
        try:
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
        mark_episode(args, ep_dir, 'interrupted')
        raise
    mark_episode(args, ep_dir, 'crash')
    try:
        return json.load(open(os.path.join(ep_dir, 'meta.json'))).get('status', 'crash')
    except (OSError, ValueError):
        return 'crash'


if __name__ == '__main__':
    main()
