#!/usr/bin/env python3
"""Smoke-test Chrono-map closed-loop scoring + one GRPO train step.

1) Score a few metric-cache samples open-loop vs closed-loop (sanity).
2) Run train_verti_grpo for 1 step with --closed_loop (optional --skip_train).

Usage (ddv2 env):
  python scripts/train/smoke_verti_grpo_chrono.py
  python scripts/train/smoke_verti_grpo_chrono.py --skip_train
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "bridge"))

from verti_chrono_cl_sim import ChronoCLSimConfig, simulate_proposals  # noqa: E402
from verti_metric_cache import VertiMetricCache, WorldMapBank, load_index  # noqa: E402
from verti_pdm_scorer import VertiPDMConfig, get_verti_score_para, score_batch  # noqa: E402


def make_crash_traj(gt: np.ndarray, lateral_m: float = 8.0) -> np.ndarray:
    out = gt.copy()
    out[:, 1] += lateral_m
    return out


def score_smoke(cache_dir: Path, n: int, seed: int) -> dict:
    rows = load_index(cache_dir / "index.jsonl")
    if not rows:
        raise FileNotFoundError(f"empty cache: {cache_dir}")
    rng = np.random.default_rng(seed)
    pick = rows if len(rows) <= n else list(rng.choice(rows, size=n, replace=False))
    bank = WorldMapBank()
    cfg = VertiPDMConfig()
    cl_cfg = ChronoCLSimConfig(v_ref=4.0)

    ol_final, cl_final = [], []
    ol_crash, cl_crash = [], []
    bad_ol, bad_cl = [], []
    t_ol = t_cl = 0.0

    for r in pick:
        c = VertiMetricCache.from_dict(r)
        world = bank.get_for_sample(c)
        gt = c.gt_traj.astype(np.float32)
        bad = make_crash_traj(gt)

        t0 = time.time()
        sc_ol, sub_ol = score_batch(gt[None], c, world, cfg, first_is_gt=False, closed_loop=False)
        t_ol += time.time() - t0
        t0 = time.time()
        sc_cl, sub_cl = score_batch(
            gt[None], c, world, cfg, first_is_gt=False, closed_loop=True, cl_cfg=cl_cfg
        )
        t_cl += time.time() - t0

        ol_final.append(float(sc_ol[0]))
        cl_final.append(float(sc_cl[0]))
        ol_crash.append(1.0 - float(sub_ol["no_collision"][0]))
        cl_crash.append(1.0 - float(sub_cl["no_collision"][0]))

        # batch API with GT + bad (pairwise)
        tb = np.stack([gt, bad], 0)
        sc_b_ol, _ = score_batch(tb, c, world, cfg, first_is_gt=True, closed_loop=False)
        sc_b_cl, _ = score_batch(
            tb, c, world, cfg, first_is_gt=True, closed_loop=True, cl_cfg=cl_cfg
        )
        bad_ol.append(float(sc_b_ol[0]))
        bad_cl.append(float(sc_b_cl[0]))

        # ensure sim returns finite
        sim, crashed = simulate_proposals(gt[None], c, world=world, cfg=cl_cfg)
        assert sim.shape == (1, 8, 3), sim.shape
        assert np.isfinite(sim).all()

    # get_verti_score_para torch path
    import torch

    batch = pick[: min(2, len(pick))]
    caches = [VertiMetricCache.from_dict(r) for r in batch]
    traj = torch.stack(
        [torch.from_numpy(c.gt_traj.astype(np.float32)) for c in caches], 0
    ).unsqueeze(1)  # (B,1,8,3)
    # append a noisy second mode
    noise = traj + torch.randn_like(traj) * 0.5
    traj2 = torch.cat([noise, traj], dim=1)  # GT last
    scores, _, subs = get_verti_score_para(
        traj2, caches, bank=bank, first_is_gt=False, closed_loop=True, cl_cfg=cl_cfg
    )
    assert scores.shape == (len(batch), 2), scores.shape

    out = {
        "n": len(pick),
        "open_loop": {
            "final_mean": float(np.mean(ol_final)),
            "crash_frac": float(np.mean(ol_crash)),
            "bad_final_mean": float(np.mean(bad_ol)),
            "sec": t_ol,
        },
        "closed_loop": {
            "final_mean": float(np.mean(cl_final)),
            "crash_frac": float(np.mean(cl_crash)),
            "bad_final_mean": float(np.mean(bad_cl)),
            "sec": t_cl,
        },
        "batch_score_shape": list(scores.shape),
        "batch_score_mean": float(scores.mean()),
    }
    # basic discriminability: lateral crash should score worse than GT on avg
    ok_disc = out["closed_loop"]["bad_final_mean"] < out["closed_loop"]["final_mean"] + 0.05
    out["ok_discriminate"] = bool(ok_disc)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--cache_dir",
        type=Path,
        default=ROOT / "datasets/verti_metric_cache_density",
    )
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--skip_train", action="store_true")
    ap.add_argument(
        "--out_dir",
        type=Path,
        default=ROOT / "log/verti_grpo_chrono_smoke",
    )
    ap.add_argument(
        "--ckpt",
        type=Path,
        default=ROOT / "log/verti_il_finetune_v2_density/ckpts/best.ckpt",
    )
    ap.add_argument(
        "--data_dir",
        type=Path,
        default=ROOT / "datasets/verti_ddv2_density",
    )
    args = ap.parse_args()

    print("========== Chrono-CL score smoke ==========")
    summary = score_smoke(args.cache_dir, args.n, args.seed)
    print(json.dumps(summary, indent=2))
    if not summary["ok_discriminate"]:
        print(
            "[warn] closed-loop bad traj not clearly worse than GT "
            f"(bad={summary['closed_loop']['bad_final_mean']:.3f} "
            f"gt={summary['closed_loop']['final_mean']:.3f})"
        )
    else:
        print("[ok] closed-loop discriminates lateral crash vs GT")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "score_smoke.json").write_text(json.dumps(summary, indent=2))

    if args.skip_train:
        print("skip train step")
        return

    if not args.ckpt.is_file():
        print(f"[warn] missing ckpt {args.ckpt}; skip train smoke")
        return

    print("========== GRPO 1-step closed_loop smoke ==========")
    cmd = [
        sys.executable,
        str(ROOT / "scripts/train/train_verti_grpo.py"),
        "--data_dir",
        str(args.data_dir),
        "--cache_dir",
        str(args.cache_dir),
        "--ckpt",
        str(args.ckpt),
        "--out_dir",
        str(args.out_dir),
        "--epochs",
        "1",
        "--batch_size",
        "1",
        "--num_groups",
        "1",
        "--step_num",
        "2",
        "--max_steps",
        "1",
        "--num_workers",
        "0",
        "--closed_loop",
        "--cl_v_ref",
        "4.0",
    ]
    print(" ".join(cmd))
    rc = subprocess.call(cmd)
    if rc != 0:
        raise SystemExit(rc)
    print("done →", args.out_dir)


if __name__ == "__main__":
    main()
