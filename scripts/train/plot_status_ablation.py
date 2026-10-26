#!/usr/bin/env python3
"""Plot status-ablation results (no bar charts).

Reads OUT/ablation_summary.json + per_sample_all.csv and writes OUT/figures/*.png

  python scripts/train/plot_status_ablation.py \\
    --out_dir log/status_ablation_attitude_modes
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

ORDER = [
    "identity",
    "noise_att_5deg",
    "zero_att",
    "flip_att",
    "noise_att_15deg",
    "zero_goal",
]
LABELS = {
    "identity": "identity",
    "noise_att_5deg": "noise 5°",
    "zero_att": "zero att",
    "flip_att": "flip att",
    "noise_att_15deg": "noise 15°",
    "zero_goal": "zero goal",
}
COLORS = {
    "identity": "#2f2f2f",
    "noise_att_5deg": "#4c78a8",
    "zero_att": "#f58518",
    "flip_att": "#e45756",
    "noise_att_15deg": "#72b7b2",
    "zero_goal": "#54a24b",
}


def plot_all(out_dir: Path) -> list[Path]:
    summary = json.loads((out_dir / "ablation_summary.json").read_text())
    rows = list(csv.DictReader(open(out_dir / "per_sample_all.csv")))
    by_v: dict[str, list] = defaultdict(list)
    for r in rows:
        by_v[r["variant"]].append(r)
    order = [v for v in ORDER if v in by_v]
    plot_vs = [v for v in order if v != "identity"]
    id_by_path = {r["path"]: r for r in by_v["identity"]}
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    def farr(key: str, variant: str) -> np.ndarray:
        return np.array([float(r[key]) for r in by_v[variant]], dtype=np.float64)

    # 1) profiles
    fig, axes = plt.subplots(1, 3, figsize=(11.2, 3.4), sharex=True)
    xs = np.arange(len(order))
    specs = [
        ("ADE@best", "ADE@best (m)", False),
        ("minADE@20", "minADE@20 (m)", False),
        ("frac_top1_match_id", "top-1 ≡ identity", True),
    ]
    for ax, (key, ylab, is_frac) in zip(axes, specs):
        ys = []
        for v in order:
            val = summary["summaries"][v].get(key, float("nan"))
            ys.append(val * 100.0 if is_frac and val == val else val)
        ys = np.asarray(ys, dtype=np.float64)
        ax.plot(xs, ys, "-o", color="#333333", lw=1.6, ms=7, zorder=2)
        for i, v in enumerate(order):
            ax.plot(xs[i], ys[i], "o", color=COLORS[v], ms=9, zorder=3)
            if np.isfinite(ys[i]):
                txt = f"{ys[i]:.1f}%" if is_frac else f"{ys[i]:.3f}"
                ax.annotate(txt, (xs[i], ys[i]), textcoords="offset points",
                            xytext=(0, 8), ha="center", fontsize=7)
        ax.set_xticks(xs)
        ax.set_xticklabels([LABELS[v] for v in order], rotation=28, ha="right", fontsize=8)
        ax.set_ylabel(ylab)
        ax.grid(True, axis="y", alpha=0.35)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    fig.suptitle("Status ablation profiles (camera + lidar fixed)", fontsize=11, y=1.02)
    fig.tight_layout()
    p = fig_dir / "01_metric_profiles.png"
    fig.savefig(p, dpi=160, bbox_inches="tight")
    plt.close(fig)
    written.append(p)

    # 2) ECDF
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    for v in plot_vs:
        dade = farr("delta_ade_vs_id", v)
        dade = dade[np.isfinite(dade)]
        xs_s = np.sort(dade)
        ys = np.arange(1, len(xs_s) + 1) / len(xs_s)
        ax.plot(xs_s, ys, color=COLORS[v], lw=1.8, label=LABELS[v])
    ax.axvline(0, color="#666", ls="--", lw=1)
    ax.set_xlabel("ΔADE vs identity (m)")
    ax.set_ylabel("ECDF")
    ax.set_title("Per-sample ADE change under status ablation")
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    ax.grid(True, alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    p = fig_dir / "02_delta_ade_ecdf.png"
    fig.savefig(p, dpi=160, bbox_inches="tight")
    plt.close(fig)
    written.append(p)

    # 3) paired scatter
    fig, axes = plt.subplots(2, 3, figsize=(10.5, 6.8))
    axes = axes.ravel()
    lim = 6.0
    for ax, v in zip(axes, plot_vs):
        ade_id, ade_v = [], []
        for r in by_v[v]:
            rid = id_by_path.get(r["path"])
            if rid is None:
                continue
            ade_id.append(float(rid["ade"]))
            ade_v.append(float(r["ade"]))
        ade_id = np.asarray(ade_id)
        ade_v = np.asarray(ade_v)
        ax.scatter(ade_id, ade_v, s=8, alpha=0.35, c=COLORS[v], edgecolors="none")
        ax.plot([0, lim], [0, lim], "--", color="#888", lw=1)
        ax.set_xlim(0, lim)
        ax.set_ylim(0, lim)
        ax.set_aspect("equal")
        ax.set_xlabel("identity ADE (m)")
        ax.set_ylabel(f"{LABELS[v]} ADE (m)")
        ax.set_title(LABELS[v], fontsize=10)
        ax.text(0.04, 0.96, f"mean Δ={float(np.mean(ade_v - ade_id)):+.3f} m",
                transform=ax.transAxes, va="top", fontsize=8)
        ax.grid(True, alpha=0.25)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    for ax in axes[len(plot_vs) :]:
        ax.axis("off")
    fig.suptitle("Paired ADE: identity vs ablated status", fontsize=11)
    fig.tight_layout()
    p = fig_dir / "03_paired_ade_scatter.png"
    fig.savefig(p, dpi=160, bbox_inches="tight")
    plt.close(fig)
    written.append(p)

    # 4) slope
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.0))
    id_ade = summary["summaries"]["identity"]["ADE@best"]
    ax = axes[0]
    for i, v in enumerate(plot_vs):
        y = len(plot_vs) - 1 - i
        ade = summary["summaries"][v]["ADE@best"]
        ax.plot([id_ade, ade], [y, y], color="#bbbbbb", lw=1.5, zorder=1)
        ax.scatter([id_ade], [y], s=36, c=COLORS["identity"], zorder=2)
        ax.scatter([ade], [y], s=48, c=COLORS[v], zorder=3)
        ax.text(max(id_ade, ade) + 0.01, y, f"{ade:.3f} ({ade - id_ade:+.3f})",
                va="center", fontsize=8)
    ax.set_yticks(range(len(plot_vs)))
    ax.set_yticklabels([LABELS[v] for v in plot_vs][::-1])
    ax.set_xlabel("ADE@best (m)")
    ax.set_title("ADE slope (identity → variant)")
    ax.axvline(id_ade, color=COLORS["identity"], ls=":", lw=1, alpha=0.7)
    ax.grid(True, axis="x", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax = axes[1]
    for i, v in enumerate(plot_vs):
        y = len(plot_vs) - 1 - i
        m = summary["summaries"][v].get("frac_top1_match_id", float("nan"))
        if m != m:
            continue
        m *= 100
        ax.plot([100, m], [y, y], color="#bbbbbb", lw=1.5, zorder=1)
        ax.scatter([100], [y], s=36, c=COLORS["identity"], zorder=2)
        ax.scatter([m], [y], s=48, c=COLORS[v], zorder=3)
        ax.text(min(m, 100) - 1, y, f"{m:.1f}%", va="center", ha="right", fontsize=8)
    ax.set_yticks(range(len(plot_vs)))
    ax.set_yticklabels([LABELS[v] for v in plot_vs][::-1])
    ax.set_xlabel("top-1 mode ≡ identity (%)")
    ax.set_title("Mode agreement slope")
    ax.set_xlim(55, 105)
    ax.grid(True, axis="x", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.legend(
        handles=[
            Line2D([0], [0], marker="o", color="w", markerfacecolor=COLORS["identity"],
                   markersize=8, label="identity"),
            Line2D([0], [0], marker="o", color="w", markerfacecolor="#888888",
                   markersize=8, label="ablated"),
        ],
        loc="upper center",
        ncol=2,
        frameon=False,
        bbox_to_anchor=(0.5, 1.02),
    )
    fig.tight_layout()
    p = fig_dir / "04_slope_ade_mode.png"
    fig.savefig(p, dpi=160, bbox_inches="tight")
    plt.close(fig)
    written.append(p)

    # 5) attitude buckets
    buckets = summary.get("buckets_att_mag_deg", {})
    bnames = list(buckets.get("identity", {}).keys())
    if bnames:
        fig, ax = plt.subplots(figsize=(7.5, 4.0))
        for v in order:
            if v not in buckets:
                continue
            ys = []
            for bn in bnames:
                s = buckets[v].get(bn, {})
                ys.append(s.get("ADE@best", float("nan")) if s.get("num_samples", 0) else float("nan"))
            ax.plot(range(len(bnames)), ys, "-o", color=COLORS[v], lw=1.6, ms=6, label=LABELS[v])
        ax.set_xticks(range(len(bnames)))
        ax.set_xticklabels(bnames)
        ax.set_xlabel("|roll|+|pitch| bucket (deg)")
        ax.set_ylabel("ADE@best (m)")
        ax.set_title("ADE by chassis attitude magnitude")
        ax.legend(frameon=False, fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        fig.tight_layout()
        p = fig_dir / "05_ade_by_attitude_bucket.png"
        fig.savefig(p, dpi=160, bbox_inches="tight")
        plt.close(fig)
        written.append(p)

    return written


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", type=Path, required=True)
    args = ap.parse_args()
    paths = plot_all(args.out_dir)
    for p in paths:
        print(f"[wrote] {p}")


if __name__ == "__main__":
    main()
