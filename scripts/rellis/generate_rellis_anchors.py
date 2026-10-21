#!/usr/bin/env python3
"""K-Means plan anchors from RELLIS DDV2 train trajectories.

Matches DiffusionDrive / DDV2 format: save (K, 8, 2) float32 npy of ego-frame
future XY waypoints (horizon 4s @ 0.5s). Uses train split only so val (00004)
does not leak into the prior.

Example:
  python scripts/rellis/generate_rellis_anchors.py \\
      --data_dir datasets/ddv2_rellis_full --k 8 --min_sep 1.0
  # -> DiffusionDriveV2/kmeans_rellis_traj_8.npy
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans

ROOT = Path(__file__).resolve().parents[2]
CMD_NAMES = ("L", "S", "R")


def load_train_xy(data_dir: Path) -> np.ndarray:
    """Load all train trajectories as (N, 8, 2) float32."""
    train_dir = data_dir / "train"
    paths = sorted(train_dir.glob("sample_*.npz"))
    if not paths:
        raise FileNotFoundError(f"no samples under {train_dir}")

    trajs = []
    for p in paths:
        with np.load(p) as d:
            t = np.asarray(d["trajectory"], dtype=np.float32)
        if t.shape != (8, 3):
            raise ValueError(f"{p.name}: expected trajectory (8,3), got {t.shape}")
        trajs.append(t[:, :2])
    return np.stack(trajs, axis=0)


def cmd_bucket(end_xy: np.ndarray, straight_thresh: float = 0.5) -> np.ndarray:
    """Coarse L/S/R from end lateral offset (+y = left in NAVSIM)."""
    y = end_xy[:, 1]
    out = np.ones(len(y), dtype=np.int64)  # S
    out[y > straight_thresh] = 0  # L
    out[y < -straight_thresh] = 2  # R
    return out


def within_cluster_ade(xy: np.ndarray, centers: np.ndarray, labels: np.ndarray) -> float:
    """Mean ADE (m) of each sample to its assigned center."""
    pred = centers[labels]  # (N, 8, 2)
    return float(np.linalg.norm(xy - pred, axis=-1).mean())


def pairwise_ade(centers: np.ndarray) -> np.ndarray:
    """(K,8,2) → (K,K) mean waypoint distance, diagonal = inf."""
    d = np.linalg.norm(centers[:, None] - centers[None], axis=-1).mean(axis=-1)
    np.fill_diagonal(d, np.inf)
    return d


def lateral_balance_weights(xy: np.ndarray, n_bins: int, power: float) -> np.ndarray:
    """Up-weight rare lateral offsets so turns are not swallowed by straight clusters."""
    end_y = xy[:, -1, 1]
    edges = np.quantile(end_y, [0.0, 1.0])
    bins = np.linspace(edges[0], edges[1] + 1e-6, n_bins + 1)
    idx = np.clip(np.digitize(end_y, bins) - 1, 0, n_bins - 1)
    counts = np.bincount(idx, minlength=n_bins).astype(np.float64)
    w = (1.0 / np.maximum(counts[idx], 1.0)) ** power
    return (w / w.mean()).astype(np.float64)


def assign(xy: np.ndarray, centers: np.ndarray) -> np.ndarray:
    return np.linalg.norm(xy[:, None] - centers[None], axis=-1).mean(-1).argmin(1)


def select_separated_centers(
    xy: np.ndarray,
    weights: np.ndarray,
    k: int,
    min_sep: float,
    over_k: int,
    seed: int,
    lloyd_iters: int = 20,
) -> np.ndarray:
    """Over-cluster to over_k, greedily keep the heaviest candidates that are >= min_sep
    from every kept one, then refine with weighted Lloyd steps that preserve min_sep."""
    km = KMeans(n_clusters=over_k, n_init=10, random_state=seed, max_iter=500)
    km.fit(xy.reshape(len(xy), -1), sample_weight=weights)
    cand = km.cluster_centers_.reshape(over_k, 8, 2)
    mass = np.bincount(assign(xy, cand), weights=weights, minlength=over_k)

    sep = min_sep
    while True:
        chosen: list[int] = []
        for c in np.argsort(-mass):
            if all(np.linalg.norm(cand[c] - cand[s], axis=-1).mean() >= sep for s in chosen):
                chosen.append(int(c))
            if len(chosen) == k:
                break
        if len(chosen) == k:
            break
        sep *= 0.9
        print(f"[warn] only {len(chosen)} candidates at sep; relaxing to {sep:.3f} m")
    centers = cand[chosen].copy()

    for _ in range(lloyd_iters):
        labels = assign(xy, centers)
        updated = centers.copy()
        for j in range(k):
            m = labels == j
            if m.any():
                updated[j] = np.average(xy[m], axis=0, weights=weights[m])
        if pairwise_ade(updated).min() < sep:
            break
        centers = updated
    return centers


def nn_ade_to_bank(xy: np.ndarray, bank: np.ndarray) -> tuple[float, np.ndarray]:
    """Nearest-neighbor ADE of each traj to a fixed anchor bank (K,8,2)."""
    # dist: (N, K)
    d = np.linalg.norm(xy[:, None] - bank[None], axis=-1).mean(axis=-1)
    nn = d.argmin(axis=1)
    return float(d.min(axis=1).mean()), nn


def infer_dataset_name(data_dir: Path) -> str:
    n = data_dir.name.lower()
    if "verti" in n:
        return "Verti-Bench"
    if "rellis" in n:
        return "RELLIS"
    return data_dir.name


def plot_anchors(
    centers: np.ndarray,
    xy: np.ndarray,
    labels: np.ndarray,
    navsim: np.ndarray | None,
    out_png: Path,
    dataset_name: str = "RELLIS",
) -> None:
    fig, axes = plt.subplots(1, 2 if navsim is not None else 1, figsize=(12, 5))
    if navsim is None:
        axes = [axes]

    ax = axes[0]
    # light cloud of GT ends
    ends = xy[:, -1]
    ax.scatter(ends[:, 0], ends[:, 1], s=2, c="0.75", alpha=0.25, label="GT end")
    for i, c in enumerate(centers):
        c0 = np.vstack([np.zeros((1, 2)), c])  # anchors start at t=0.5s; prepend ego origin for display
        ax.plot(c0[:, 0], c0[:, 1], "-", lw=1.5, alpha=0.9)
        ax.scatter(c[-1, 0], c[-1, 1], s=28, zorder=3)
        ax.text(c[-1, 0], c[-1, 1], str(i), fontsize=7, ha="left", va="bottom")
    ax.set_aspect("equal", adjustable="datalim")
    ax.set_xlabel("x forward (m)")
    ax.set_ylabel("y left (m)")
    ax.set_title(f"{dataset_name} K-Means anchors (K={len(centers)})")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right", fontsize=8)

    if navsim is not None:
        ax = axes[1]
        for c in navsim:
            ax.plot(c[:, 0], c[:, 1], "-", color="C0", lw=1.0, alpha=0.7)
        for c in centers:
            ax.plot(c[:, 0], c[:, 1], "-", color="C3", lw=1.5, alpha=0.9)
        ax.set_aspect("equal", adjustable="datalim")
        ax.set_xlabel("x forward (m)")
        ax.set_ylabel("y left (m)")
        ax.set_title(f"NAVSIM (blue) vs {dataset_name} (red)")
        ax.grid(True, alpha=0.3)

    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=140)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data_dir", type=Path, default=ROOT / "datasets" / "ddv2_rellis_full")
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument(
        "--min_sep",
        type=float,
        default=1.0,
        help="Minimum pairwise ADE (m) between anchors; closer pairs get re-seeded.",
    )
    ap.add_argument("--over_k", type=int, default=40, help="candidate clusters before separation pick")
    ap.add_argument("--balance_bins", type=int, default=12, help="end_y bins for turn up-weighting")
    ap.add_argument(
        "--balance_power",
        type=float,
        default=0.7,
        help="0=plain KMeans, 1=equal weight per end_y bin",
    )
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--min_end_x",
        type=float,
        default=0.5,
        help="Drop near-stationary samples with end_x < this (meters). 0=keep all.",
    )
    ap.add_argument(
        "--max_end_x",
        type=float,
        default=0.0,
        help="Drop samples with end_x > this (m). 0=disabled.",
    )
    ap.add_argument(
        "--max_end_abs_y",
        type=float,
        default=0.0,
        help="Drop samples with |end_y| > this (m). 0=disabled.",
    )
    ap.add_argument(
        "--max_end_r",
        type=float,
        default=0.0,
        help="Drop samples with hypot(end_x,end_y) > this (m). 0=disabled.",
    )
    ap.add_argument(
        "--end_r_percentile",
        type=float,
        default=0.0,
        help="If >0, also drop ends above this percentile of radius (after other filters). e.g. 99.",
    )
    ap.add_argument(
        "--min_cluster_size",
        type=int,
        default=0,
        help="After clustering, warn if any center has fewer than this many assigned train samples.",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=None,
        help="default: DiffusionDriveV2/kmeans_rellis_traj_{k}.npy",
    )
    ap.add_argument(
        "--navsim_anchors",
        type=Path,
        default=ROOT / "DiffusionDriveV2" / "kmeans_navsim_traj_20.npy",
    )
    ap.add_argument(
        "--report_dir",
        type=Path,
        default=ROOT / "log" / "rellis_anchors",
    )
    ap.add_argument(
        "--dataset_name",
        type=str,
        default="",
        help="Plot title tag (default: infer from --data_dir, e.g. Verti-Bench / RELLIS)",
    )
    args = ap.parse_args()
    if not args.dataset_name:
        args.dataset_name = infer_dataset_name(args.data_dir)

    xy_all = load_train_xy(args.data_dir)
    end = xy_all[:, -1]
    print(f"loaded train trajs: N={len(xy_all)} shape={xy_all.shape}")
    print(
        f"  end_x: mean={end[:, 0].mean():.2f} max={end[:, 0].max():.2f}  "
        f"end_|y|: mean={np.abs(end[:, 1]).mean():.2f} max={np.abs(end[:, 1]).max():.2f}"
    )

    keep = end[:, 0] >= args.min_end_x
    if args.max_end_x > 0:
        keep &= end[:, 0] <= args.max_end_x
    if args.max_end_abs_y > 0:
        keep &= np.abs(end[:, 1]) <= args.max_end_abs_y
    if args.max_end_r > 0:
        keep &= np.hypot(end[:, 0], end[:, 1]) <= args.max_end_r
    xy = xy_all[keep]
    print(
        f"after end filters "
        f"(min_x>={args.min_end_x}, max_x={args.max_end_x or 'off'}, "
        f"max_|y|={args.max_end_abs_y or 'off'}, max_r={args.max_end_r or 'off'}): "
        f"N={len(xy)} (dropped {(~keep).sum()})"
    )
    if args.end_r_percentile and args.end_r_percentile > 0:
        r = np.hypot(xy[:, -1, 0], xy[:, -1, 1])
        thr = float(np.percentile(r, args.end_r_percentile))
        m = r <= thr
        print(f"after end_r_percentile<={args.end_r_percentile} (thr={thr:.2f} m): "
              f"N={m.sum()} (dropped {(~m).sum()})")
        xy = xy[m]

    if len(xy) < args.k * 5:
        raise SystemExit(f"too few trajectories after filtering: N={len(xy)}")

    if args.out is None:
        args.out = ROOT / "DiffusionDriveV2" / f"kmeans_rellis_traj_{args.k}.npy"

    weights = lateral_balance_weights(xy, args.balance_bins, args.balance_power)
    centers = select_separated_centers(
        xy, weights, args.k, args.min_sep, args.over_k, args.seed
    ).astype(np.float32)
    min_pair = float(pairwise_ade(centers).min())
    print(f"min pairwise ADE={min_pair:.3f} m (target {args.min_sep})")
    labels = assign(xy, centers)

    # sort anchors by end_x then end_y for stable indexing / viz
    order = np.lexsort((centers[:, -1, 1], centers[:, -1, 0]))
    centers = centers[order]
    # remap labels
    inv = np.empty_like(order)
    inv[order] = np.arange(len(order))
    labels = inv[labels]

    ade = within_cluster_ade(xy, centers, labels)
    buckets = cmd_bucket(centers[:, -1])
    counts = np.bincount(labels, minlength=args.k)
    cmd_counts = np.bincount(buckets, minlength=3)

    print(f"\nKMeans k={args.k}  within-cluster ADE={ade:.4f} m")
    print(f"  center L/S/R counts: {dict(zip(CMD_NAMES, cmd_counts.tolist()))}")
    print(f"  center end_x range: [{centers[:, -1, 0].min():.2f}, {centers[:, -1, 0].max():.2f}]")
    print(f"  center end_y range: [{centers[:, -1, 1].min():.2f}, {centers[:, -1, 1].max():.2f}]")
    print(f"  samples/cluster: min={counts.min()} max={counts.max()} mean={counts.mean():.1f}")
    if args.min_cluster_size > 0 and counts.min() < args.min_cluster_size:
        print(f"[warn] {int((counts < args.min_cluster_size).sum())} clusters have "
              f"< {args.min_cluster_size} samples (possible residual outliers)")

    end_r = np.hypot(centers[:, -1, 0], centers[:, -1, 1])
    print(f"  center end_r: min={end_r.min():.2f} max={end_r.max():.2f} median={np.median(end_r):.2f}")

    navsim = None
    if args.navsim_anchors.is_file():
        navsim = np.load(args.navsim_anchors).astype(np.float32)
        assert navsim.ndim == 3 and navsim.shape[1:] == (8, 2), navsim.shape
        nn_ade, nn_idx = nn_ade_to_bank(xy, navsim)
        used = np.unique(nn_idx)
        print(f"\nNAVSIM bank NN-ADE on same train: {nn_ade:.4f} m  unique anchors used={len(used)}/{len(navsim)}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.out, centers)
    print(f"\nsaved anchors {centers.shape} -> {args.out}")

    report_dir = args.report_dir
    report_dir.mkdir(parents=True, exist_ok=True)
    plot_anchors(
        centers, xy, labels, navsim, report_dir / f"anchors_k{args.k}.png",
        dataset_name=args.dataset_name,
    )

    report = {
        "data_dir": str(args.data_dir),
        "dataset_name": args.dataset_name,
        "split": "train",
        "k": args.k,
        "seed": args.seed,
        "min_end_x": args.min_end_x,
        "max_end_x": args.max_end_x,
        "max_end_abs_y": args.max_end_abs_y,
        "max_end_r": args.max_end_r,
        "end_r_percentile": args.end_r_percentile,
        "n_all": int(len(xy_all)),
        "n_used": int(len(xy)),
        "within_cluster_ade_m": ade,
        "min_sep_target_m": args.min_sep,
        "min_pairwise_ade_m": float(pairwise_ade(centers).min()),
        "mean_pairwise_ade_m": float(pairwise_ade(centers)[np.isfinite(pairwise_ade(centers))].mean()),
        "balance_power": args.balance_power,
        "center_cmd_LSR": {n: int(c) for n, c in zip(CMD_NAMES, cmd_counts)},
        "samples_per_cluster": counts.tolist(),
        "centers_end_xy": centers[:, -1].tolist(),
        "centers_end_r": end_r.tolist(),
        "out": str(args.out),
    }
    if navsim is not None:
        report["navsim_nn_ade_m"] = float(nn_ade)
        report["navsim_unique_used"] = int(len(used))
    with open(report_dir / f"report_k{args.k}.json", "w") as f:
        json.dump(report, f, indent=2)
    print(f"report + plot -> {report_dir}")


if __name__ == "__main__":
    main()
