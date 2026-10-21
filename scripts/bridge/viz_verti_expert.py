"""Render Verti-Bench expert episodes (collect_expert_verti.py output) to mp4.

Each video frame = one recorded observation:
  top:     front camera
  bottom:  lidar BEV + future 4 s GT trajectory | world map (height, obstacles, A* path,
           driven path, pose, start/goal) | speed / steer / throttle vs time

Examples:
  python viz_verti_expert.py --ep 21                       # one episode by id
  python viz_verti_expert.py --ep 3,7,10-15                # several
  python viz_verti_expert.py --ep log/verti_expert/ep_00003_w5_pid
  python viz_verti_expert.py --all --status success        # every episode in --src
"""
from pathlib import Path
import argparse
import csv
import glob
import json
import os
import subprocess
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import yaml
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_verti_ddv2_dataset import future_labels  # noqa: E402

ROOT = str(Path(__file__).resolve().parents[2])
MAPS = os.path.join(ROOT, 'verti_bench/envs/data/BenchMaps/sampled_maps')
BEV_RANGE, BEV_PPM = 32.0, 4.0


def parse_ep_spec(spec, src):
    dirs = []
    for tok in spec.split(','):
        tok = tok.strip()
        if not tok:
            continue
        if os.path.isdir(tok):
            dirs.append(os.path.abspath(tok))
            continue
        a, b = (tok.split('-') + [None])[:2] if '-' in tok else (tok, tok)
        for i in range(int(a), int(b) + 1):
            dirs += glob.glob(os.path.join(src, f'ep_{i:05d}_*'))
    return sorted(set(dirs))


def load_csv(path):
    with open(path) as f:
        rows = list(csv.DictReader(f))
    return {k: np.array([float(r[k]) for r in rows]) for k in rows[0]} if rows else None


def load_world(world, scale):
    cfg_path = glob.glob(os.path.join(MAPS, 'Configs/Final', f'config{world}_*.yaml'))[0]
    cfg = yaml.safe_load(open(cfg_path))
    L = cfg['terrain']['length'] * scale
    hmin, hmax = cfg['terrain']['min_height'], cfg['terrain']['max_height']
    height = np.array(Image.open(os.path.join(MAPS, 'Worlds', f'{world}.bmp')).convert('L'), np.float32)
    height = hmin + height / 255.0 * (hmax - hmin)
    obs = np.array(Image.open(os.path.join(MAPS, 'Configs/Final',
                                           f'obs{world}_{cfg["terrain"]["difficulty"]}.bmp')).convert('L'))
    return L, height, obs == 255


def bev_px(x, y):
    """Ego (x fwd, y left) -> BEV image (col, row) with forward up, left on the left."""
    return (BEV_RANGE - y) * BEV_PPM, (BEV_RANGE - x) * BEV_PPM


def render_episode(ep_dir, out_dir, fps, dpi, max_frames):
    meta = json.load(open(os.path.join(ep_dir, 'meta.json')))
    obs_files = sorted(glob.glob(os.path.join(ep_dir, 'obs', 'obs_*.npz')))
    tr_path = os.path.join(ep_dir, 'traj.csv')
    tr = load_csv(tr_path) if os.path.isfile(tr_path) else None
    if not obs_files or tr is None or len(tr['t']) < 2:
        print(f'[skip] {os.path.basename(ep_dir)}: no obs/traj ({meta.get("status")})')
        return None
    if max_frames and len(obs_files) > max_frames:
        obs_files = [obs_files[i] for i in np.linspace(0, len(obs_files) - 1, max_frames).astype(int)]

    name = os.path.basename(ep_dir)
    L, height, occ = load_world(meta['world'], meta.get('scale_factor', 1.0))
    ap_path = os.path.join(ep_dir, 'astar_path.npy')
    astar = np.load(ap_path) if os.path.isfile(ap_path) else None
    xy = np.stack([tr['x'], tr['y']], 1)
    hd = np.unwrap(tr['heading'])
    start, goal = meta['start'], meta['goal']

    fig = plt.figure(figsize=(16, 9), dpi=dpi)
    gs = fig.add_gridspec(2, 3, height_ratios=[1, 1.9], width_ratios=[1, 1.15, 1.1],
                          hspace=0.18, wspace=0.18, left=0.04, right=0.98, top=0.93, bottom=0.06)
    ax_cam = fig.add_subplot(gs[0, :])
    ax_bev = fig.add_subplot(gs[1, 0])
    ax_map = fig.add_subplot(gs[1, 1])
    sub = gs[1, 2].subgridspec(2, 1, hspace=0.3)
    ax_v, ax_c = fig.add_subplot(sub[0]), fig.add_subplot(sub[1])

    d0 = np.load(obs_files[0])
    im_cam = ax_cam.imshow(d0['cam_front'])
    ax_cam.set_axis_off()

    n_bev = int(2 * BEV_RANGE * BEV_PPM)
    im_bev = ax_bev.imshow(np.zeros((n_bev, n_bev)), cmap='gray', vmin=0, vmax=1)
    bev_traj, = ax_bev.plot([], [], 'o-', color='lime', ms=4, lw=2, label='GT future 4 s')
    c0, r0 = bev_px(0, 0)
    ax_bev.plot(c0, r0, marker='^', color='red', ms=10)
    ax_bev.set_xlim(0, n_bev)
    ax_bev.set_ylim(n_bev, 0)
    ax_bev.set_title('LiDAR BEV (±32 m, fwd up)')
    ax_bev.set_xticks([])
    ax_bev.set_yticks([])
    ax_bev.legend(loc='lower right', fontsize=8)

    ext = [-L, L, -L, L]
    ax_map.imshow(height, cmap='terrain', extent=ext, origin='upper')
    ax_map.imshow(np.ma.masked_where(~occ, occ), cmap='autumn', extent=ext, origin='upper', alpha=0.9)
    if astar is not None and len(astar):
        ax_map.plot(astar[:, 0], astar[:, 1], '--', color='white', lw=1.2, label='A* path')
    ax_map.plot(xy[:, 0], xy[:, 1], color='black', lw=0.8, alpha=0.35)
    driven, = ax_map.plot([], [], color='blue', lw=2, label='driven')
    ax_map.plot(*start[:2], 'go', ms=9, label='start')
    ax_map.plot(*goal[:2], 'r*', ms=14, label='goal')
    ego = ax_map.quiver([start[0]], [start[1]], [1], [0], color='red', scale=18, width=0.012)
    ax_map.set_xlim(-L, L)
    ax_map.set_ylim(-L, L)
    ax_map.set_aspect('equal')
    ax_map.set_title(f'world {meta["world"]} ({meta.get("difficulty", "?")})')
    ax_map.legend(loc='lower left', fontsize=7)

    t = tr['t']
    ax_v.plot(t, tr['speed'], color='tab:blue')
    ax_v.set_ylabel('speed (m/s)')
    ax_v.grid(alpha=0.3)
    ax_c.plot(t, tr['steer'], label='steer')
    ax_c.plot(t, tr['throttle'], label='throttle')
    ax_c.plot(t, tr['brake'], label='brake')
    ax_c.set_ylim(-1.05, 1.05)
    ax_c.set_xlabel('t (s)')
    ax_c.legend(fontsize=7, loc='lower left')
    ax_c.grid(alpha=0.3)
    cur_v = ax_v.axvline(0, color='red')
    cur_c = ax_c.axvline(0, color='red')
    title = fig.suptitle('')

    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, f'{name}.mp4')
    w, h = fig.canvas.get_width_height()
    ff = subprocess.Popen(['ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo', '-pix_fmt', 'rgba',
                           '-s', f'{w}x{h}', '-r', str(fps), '-i', '-', '-pix_fmt', 'yuv420p',
                           '-vcodec', 'libx264', '-crf', '23', out], stdin=subprocess.PIPE)
    try:
        from verti_bench.systems.DDV2.ddv2_live_viz import parse_nav_status
    except ImportError:
        parse_nav_status = None
    cmd_names = ['LEFT', 'STRAIGHT', 'RIGHT', 'UNKNOWN']
    for of in obs_files:
        d = np.load(of)
        t0 = float(d['t'])
        pose = d['pose']
        im_cam.set_data(d['cam_front'])
        im_bev.set_data(d['lidar_feature'][0][::-1, ::-1])
        fut = future_labels(t, xy, hd, t0, pose, clip=True)
        if fut is not None and len(fut):
            cs, rs = bev_px(np.r_[0, fut[:, 0]], np.r_[0, fut[:, 1]])
            bev_traj.set_data(cs, rs)
        else:
            bev_traj.set_data([], [])
        k = int(np.searchsorted(t, t0, side='right'))
        driven.set_data(xy[:k, 0], xy[:k, 1])
        ego.set_offsets([[pose[0], pose[1]]])
        ego.set_UVC([np.cos(pose[3])], [np.sin(pose[3])])
        cur_v.set_xdata([t0, t0])
        cur_c.set_xdata([t0, t0])
        st = d['status_feature']
        dist = np.hypot(goal[0] - pose[0], goal[1] - pose[1])
        if parse_nav_status is not None:
            _mode, nav_lab, _ = parse_nav_status(st)
            nav_txt = nav_lab
        else:
            nav_txt = f'cmd={cmd_names[int(np.argmax(st[:4]))]}'
        title.set_text(f'{name}   system={meta["system"]}   status={meta.get("status")}   '
                       f't={t0:5.1f}s   v={float(d["speed"]):4.2f} m/s   '
                       f'{nav_txt}   dist_goal={dist:5.1f} m')
        fig.canvas.draw()
        ff.stdin.write(fig.canvas.buffer_rgba().tobytes())
    ff.stdin.close()
    ff.wait()
    plt.close(fig)
    print(f'[ok] {out}  frames={len(obs_files)}', flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--src', default=os.path.join(ROOT, 'log/verti_expert'))
    ap.add_argument('--out', default=None, help='default: SRC/videos')
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument('--ep', help="episode ids '21' | '3,7,10-15' | episode dir path(s)")
    g.add_argument('--all', action='store_true', help='all episodes under --src')
    ap.add_argument('--status', default='', help='--all filter, e.g. success,fail')
    ap.add_argument('--fps', type=float, default=4.0, help='obs are 2 Hz by default; 4 fps = 2x speed')
    ap.add_argument('--dpi', type=int, default=80)
    ap.add_argument('--max_frames', type=int, default=0, help='subsample long episodes (0 = all)')
    ap.add_argument('--skip_existing', action='store_true')
    args = ap.parse_args()

    dirs = sorted(glob.glob(os.path.join(args.src, 'ep_*'))) if args.all else parse_ep_spec(args.ep, args.src)
    if not dirs:
        sys.exit(f'no episodes matched under {args.src}')
    keep = {s.strip() for s in args.status.split(',') if s.strip()}
    out_dir = args.out or os.path.join(args.src, 'videos')
    for d in dirs:
        try:
            meta = json.load(open(os.path.join(d, 'meta.json')))
        except (OSError, ValueError):
            continue
        if keep and meta.get('status') not in keep:
            continue
        if args.skip_existing and os.path.isfile(os.path.join(out_dir, os.path.basename(d) + '.mp4')):
            continue
        try:
            render_episode(d, out_dir, args.fps, args.dpi, args.max_frames)
        except Exception as ex:
            print(f'[error] {os.path.basename(d)}: {ex!r}', flush=True)


if __name__ == '__main__':
    main()
