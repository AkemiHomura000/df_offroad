"""Convert collect_expert_verti.py episodes into DDV2 training samples.

Can emit one or both lidar variants (same frames / same train–val split):
  --lidar_mode elevation   → lidar_feature = chassis-frame max-z BEV, mapped to [0,1] (default)
  --lidar_mode density     → lidar_feature = point-count BEV in [0,1]
  --lidar_mode both        → write two out dirs in one pass

Output layout (matches RELLIS / train_rellis_il.py):
  OUT/{train,val}/sample_XXXXXX.npz
      camera_feature (3,256,1024) f32
      lidar_feature  (1,256,256) f32
      status_feature (8,) f32   # goal XY [0:2] + roll/pitch/π [2:4] + vel/acc
      trajectory     (8,3) f32  ego-frame x,y,yaw at 0.5..4.0 s
  OUT/meta.json

By default ``--status_nav goal`` rewrites status[0:2] from episode goal
into ego-frame XY / --goal_norm_m; [2:4] = roll/pitch from traj.csv (or obs pose).
Use ``--status_nav command`` to keep the one-hot stored in expert obs.

Example:
  python build_verti_ddv2_dataset.py --src log/verti_expert --lidar_mode both \\
    --out_density datasets/verti_ddv2_density \\
    --out_elevation datasets/verti_ddv2_elevation
"""
import argparse
import csv
import glob
import json
import os
import shutil
import sys

from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from verti_bench.envs.utils.ddv2_status import (  # noqa: E402
    GOAL_NORM_M,
    rewrite_status_nav_goal,
)

H_STEPS, H_DT = 8, 0.5


def episode_goal_xy(meta, traj_path):
    """World-frame goal XY from meta.json or first traj.csv row."""
    g = meta.get("goal")
    if isinstance(g, (list, tuple)) and len(g) >= 2:
        return float(g[0]), float(g[1])
    with open(traj_path) as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise KeyError("no goal in meta and empty traj.csv")
    return float(rows[0]["goal_x"]), float(rows[0]["goal_y"])


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def load_traj(path):
    """Return t, xy, heading, roll, pitch (roll/pitch unwrapped lightly via interp)."""
    with open(path) as f:
        rows = list(csv.DictReader(f))
    t = np.array([float(r['t']) for r in rows])
    xy = np.array([[float(r['x']), float(r['y'])] for r in rows])
    hd = np.unwrap(np.array([float(r['heading']) for r in rows]))
    roll = np.array([float(r['roll']) for r in rows], dtype=np.float64)
    pitch = np.array([float(r['pitch']) for r in rows], dtype=np.float64)
    return t, xy, hd, roll, pitch


def attitude_at(t, roll, pitch, t0):
    """Interpolate chassis roll/pitch (rad) at observation time t0."""
    if len(t) == 0:
        return 0.0, 0.0
    r = float(np.interp(t0, t, roll))
    p = float(np.interp(t0, t, pitch))
    return r, p


def pose_attitude(pose):
    """Extract (roll, pitch) from pose if present: [x,y,z,heading,roll,pitch]."""
    p = np.asarray(pose, dtype=np.float64).reshape(-1)
    if p.size >= 6:
        return float(p[4]), float(p[5])
    return None, None


def future_labels(t, xy, hd, t0, pose0, clip=False):
    """clip=True keeps the available prefix instead of rejecting a short future."""
    ts = t0 + H_DT * np.arange(1, H_STEPS + 1)
    if ts[-1] > t[-1]:
        if not clip:
            return None
        ts = ts[ts <= t[-1]]
        if len(ts) == 0:
            return None
    fx, fy = np.interp(ts, t, xy[:, 0]), np.interp(ts, t, xy[:, 1])
    fh = np.interp(ts, t, hd)
    x0, y0, h0 = pose0[0], pose0[1], pose0[3]
    c, s = np.cos(h0), np.sin(h0)
    dx, dy = fx - x0, fy - y0
    return np.stack([c * dx + s * dy, -s * dx + c * dy, wrap(fh - h0)], 1).astype(np.float32)


def get_density(d):
    if 'lidar_density' in d.files:
        return np.asarray(d['lidar_density'], dtype=np.float32)
    return np.asarray(d['lidar_feature'], dtype=np.float32)


def get_elevation_norm(d, elev_min, elev_max, dens=None):
    """Chassis max-z BEV → (1,H,W) in [0,1]; empty cells stay 0."""
    if 'lidar_elevation' not in d.files:
        raise KeyError('lidar_elevation missing (re-collect with hood lidar pipeline)')
    elev = np.asarray(d['lidar_elevation'], dtype=np.float32)
    if dens is None:
        dens = get_density(d)
    out = np.zeros_like(elev, dtype=np.float32)
    mask = dens > 1e-6
    span = max(elev_max - elev_min, 1e-3)
    out[mask] = np.clip((elev[mask] - elev_min) / span, 0.0, 1.0)
    return out


def prepare_out(path):
    if os.path.isdir(path):
        # wipe previous samples so counts stay consistent
        for sp in ('train', 'val'):
            sp_dir = os.path.join(path, sp)
            if os.path.isdir(sp_dir):
                shutil.rmtree(sp_dir)
        meta = os.path.join(path, 'meta.json')
        if os.path.isfile(meta):
            os.remove(meta)
    for sp in ('train', 'val'):
        os.makedirs(os.path.join(path, sp), exist_ok=True)


def save_sample(out_root, split, sid, cam, lidar, status, traj, name, fidx, meta):
    fn = f'sample_{sid:06d}.npz'
    np.savez_compressed(
        os.path.join(out_root, split, fn),
        camera_feature=cam,
        lidar_feature=lidar,
        status_feature=status,
        trajectory=traj,
        seq=name,
        frame_idx=fidx,
        world=meta['world'],
        system=meta['system'],
    )
    return dict(file=f'{split}/{fn}', seq=name, frame_idx=fidx,
                world=meta['world'], system=meta['system'])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--src', default=str(ROOT / "log/verti_expert"))
    ap.add_argument('--lidar_mode', choices=['density', 'elevation', 'both'], default='elevation')
    ap.add_argument('--out', default='', help='single-mode output dir')
    ap.add_argument('--out_density', default=str(ROOT / "datasets/verti_ddv2_density"))
    ap.add_argument('--out_elevation', default=str(ROOT / "datasets/verti_ddv2_elevation"))
    ap.add_argument('--status', default='success')
    ap.add_argument('--min_t', type=float, default=1.0)
    ap.add_argument('--min_progress', type=float, default=0.5)
    ap.add_argument('--val_every', type=int, default=10)
    ap.add_argument('--elev_min', type=float, default=-3.0, help='chassis-z low for [0,1] map')
    ap.add_argument('--elev_max', type=float, default=4.0, help='chassis-z high for [0,1] map')
    ap.add_argument(
        '--status_nav',
        choices=['goal', 'command'],
        default='goal',
        help='goal: rewrite status[0:2] as ego XY/norm (+ attitude [2:4]); command: keep one-hot',
    )
    ap.add_argument(
        '--goal_norm_m',
        type=float,
        default=GOAL_NORM_M,
        help='ego goal XY ÷ this, then clip to [-1,1] (default=BEV 32 m)',
    )
    args = ap.parse_args()

    keep = set(s.strip() for s in args.status.split(',') if s.strip())
    modes = []
    if args.lidar_mode in ('density', 'both'):
        dens_out = args.out if (args.lidar_mode == 'density' and args.out) else args.out_density
        modes.append(('density', dens_out))
    if args.lidar_mode in ('elevation', 'both'):
        elev_out = args.out if (args.lidar_mode == 'elevation' and args.out) else args.out_elevation
        modes.append(('elevation', elev_out))

    for _, path in modes:
        prepare_out(path)

    samples = {m: [] for m, _ in modes}
    sid = 0
    stats = dict(episodes=0, skipped_eps=0, obs=0, no_future=0, stuck=0, no_elev=0, no_goal=0)
    # Split by kept-episode order so train/val membership is identical for both outs.
    kept_ei = 0
    ep_dirs = sorted(glob.glob(os.path.join(args.src, 'ep_*')))
    for ed in ep_dirs:
        name = os.path.basename(ed)
        try:
            meta = json.load(open(os.path.join(ed, 'meta.json')))
        except (OSError, ValueError):
            stats['skipped_eps'] += 1
            continue
        obs_files = sorted(glob.glob(os.path.join(ed, 'obs', 'obs_*.npz')))
        traj_path = os.path.join(ed, 'traj.csv')
        if meta.get('status') not in keep or not obs_files or not os.path.isfile(traj_path):
            stats['skipped_eps'] += 1
            continue
        t, xy, hd, roll_s, pitch_s = load_traj(traj_path)
        if len(t) < 2:
            stats['skipped_eps'] += 1
            continue
        goal_xy = None
        if args.status_nav == 'goal':
            try:
                goal_xy = episode_goal_xy(meta, traj_path)
            except (KeyError, ValueError, OSError):
                stats['no_goal'] += 1
                stats['skipped_eps'] += 1
                continue

        split = 'val' if kept_ei % args.val_every == args.val_every - 1 else 'train'
        kept_ei += 1
        stats['episodes'] += 1
        n_before = sid

        for of in obs_files:
            stats['obs'] += 1
            d = np.load(of)
            t0 = float(d['t'])
            if t0 < args.min_t:
                continue
            traj = future_labels(t, xy, hd, t0, d['pose'])
            if traj is None:
                stats['no_future'] += 1
                continue
            if np.hypot(*traj[-1, :2]) < args.min_progress:
                stats['stuck'] += 1
                continue

            dens = get_density(d)
            try:
                lids = {}
                for mode, _ in modes:
                    if mode == 'density':
                        lids[mode] = dens
                    else:
                        lids[mode] = get_elevation_norm(d, args.elev_min, args.elev_max, dens)
            except KeyError:
                stats['no_elev'] += 1
                continue

            cam = np.transpose(d['cam_front'].astype(np.float32) / 255.0, (2, 0, 1))
            status = np.asarray(d['status_feature'], dtype=np.float32)
            # Prefer pose[4:6] (new collect); else interpolate traj.csv roll/pitch
            roll_rad, pitch_rad = pose_attitude(d['pose'])
            if roll_rad is None:
                roll_rad, pitch_rad = attitude_at(t, roll_s, pitch_s, t0)
            if args.status_nav == 'goal':
                status = rewrite_status_nav_goal(
                    status,
                    d['pose'],
                    goal_xy,
                    goal_norm_m=args.goal_norm_m,
                    roll_rad=roll_rad,
                    pitch_rad=pitch_rad,
                )
            fidx = int(os.path.basename(of)[4:9])
            for mode, out_root in modes:
                samples[mode].append(
                    save_sample(out_root, split, sid, cam, lids[mode], status, traj, name, fidx, meta)
                )
            sid += 1

        print(f'[{name}] {meta.get("status")} -> {split}  +{sid - n_before}  total={sid}', flush=True)

    for mode, out_root in modes:
        out_meta = dict(
            source=os.path.abspath(args.src),
            lidar_mode=mode,
            status_nav=args.status_nav,
            status_layout='goal_xy[0:2]+roll_pitch_over_pi[2:4]+vel[4:6]+acc[6:8]',
            goal_norm_m=args.goal_norm_m if args.status_nav == 'goal' else None,
            elev_min=args.elev_min if mode == 'elevation' else None,
            elev_max=args.elev_max if mode == 'elevation' else None,
            horizon_s=H_STEPS * H_DT,
            interval_s=H_DT,
            num_train=sum(s['file'].startswith('train/') for s in samples[mode]),
            num_val=sum(s['file'].startswith('val/') for s in samples[mode]),
            stats=stats,
            samples=samples[mode],
        )
        with open(os.path.join(out_root, 'meta.json'), 'w') as f:
            json.dump(out_meta, f, indent=1)
        print(f'done [{mode}]: train={out_meta["num_train"]} val={out_meta["num_val"]} -> {out_root}')
    print(f'stats {stats}')


if __name__ == '__main__':
    main()
