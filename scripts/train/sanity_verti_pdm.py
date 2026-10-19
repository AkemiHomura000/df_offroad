#!/usr/bin/env python3
"""Sanity-check VertiPDM: success vs fail expert GT, plus noisy crash modes.

Expects datasets/verti_metric_cache_* /index.jsonl (from build_verti_metric_cache.py).
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path("/home/rm/Desktop/df_offroad")
sys.path.insert(0, str(ROOT / "scripts" / "bridge"))

from verti_metric_cache import VertiMetricCache, WorldMapBank, load_index  # noqa: E402
from verti_pdm_scorer import VertiPDMConfig, score_batch  # noqa: E402


def make_crash_traj(gt: np.ndarray, lateral_m: float = 8.0) -> np.ndarray:
    """Shift GT sideways so it should hit obstacles / leave corridor."""
    out = gt.copy()
    out[:, 1] += lateral_m
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache_dir", type=Path, default=ROOT / "datasets/verti_metric_cache_density")
    ap.add_argument("--max_per_status", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rows = load_index(args.cache_dir / "index.jsonl")
    by_status = defaultdict(list)
    for r in rows:
        by_status[r.get("ep_status", "")].append(r)

    rng = np.random.default_rng(args.seed)
    bank = WorldMapBank()
    cfg = VertiPDMConfig()

    summary = {}
    for status in ("success", "fail"):
        pool = by_status.get(status, [])
        if not pool:
            print(f"[warn] no samples with ep_status={status}")
            continue
        pick = pool if len(pool) <= args.max_per_status else list(
            rng.choice(pool, size=args.max_per_status, replace=False)
        )
        finals, crash_rates, trav_rates = [], [], []
        crash_mode_finals = []
        for r in pick:
            c = VertiMetricCache.from_dict(r)
            world = bank.get_for_sample(c)
            gt = c.gt_traj
            # score GT alone (no pairwise)
            sc, sub = score_batch(gt[None], c, world, cfg, first_is_gt=False)
            finals.append(float(sc[0]))
            crash_rates.append(1.0 - float(sub["no_collision"][0]))
            trav_rates.append(float(sub["drivable_area"][0]))
            # crash mode
            bad = make_crash_traj(gt)
            sc_b, sub_b = score_batch(
                np.stack([gt, bad], 0), c, world, cfg, first_is_gt=True
            )
            crash_mode_finals.append(float(sc_b[0]))  # score of bad vs GT

        summary[status] = {
            "n": len(pick),
            "final_mean": float(np.mean(finals)),
            "final_std": float(np.std(finals)),
            "crash_frac": float(np.mean(crash_rates)),
            "traversable_frac": float(np.mean(trav_rates)),
            "lateral_shift_final_mean": float(np.mean(crash_mode_finals)),
        }
        print(
            f"[{status}] n={len(pick)} final={summary[status]['final_mean']:.3f}±"
            f"{summary[status]['final_std']:.3f} crash_frac={summary[status]['crash_frac']:.3f} "
            f"trav={summary[status]['traversable_frac']:.3f} "
            f"shift8m_final={summary[status]['lateral_shift_final_mean']:.3f}",
            flush=True,
        )

    if "success" in summary and "fail" in summary:
        delta = summary["success"]["final_mean"] - summary["fail"]["final_mean"]
        print(f"Δ(success-fail) final_mean = {delta:.3f}")
        ok = delta > 0.02 or summary["success"]["final_mean"] > summary["fail"]["final_mean"]
        # crash modes should score worse than typical success GT
        ok = ok and (
            summary["success"]["lateral_shift_final_mean"]
            < summary["success"]["final_mean"] - 0.05
        )
        print("SANITY", "PASS" if ok else "WEAK/FAIL")
    out = args.cache_dir / "sanity_verti_pdm.json"
    out.write_text(json.dumps(summary, indent=2))
    print("wrote", out)


if __name__ == "__main__":
    main()
