#!/usr/bin/env python3
"""Non-bar charts for Scheme D sensor-geometry ablation."""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from PIL import Image

ORDER = ["as_logged", "level", "pitch_p15", "pitch_m15", "roll_p10"]
LABELS = {
    "as_logged": "as logged",
    "level": "level (r=p=0)",
    "pitch_p15": "pitch +15°",
    "pitch_m15": "pitch −15°",
    "roll_p10": "roll +10°",
}
COLORS = {
    "as_logged": "#2f2f2f",
    "level": "#4c78a8",
    "pitch_p15": "#f58518",
    "pitch_m15": "#e45756",
    "roll_p10": "#54a24b",
}


def plot_all(render_dir: Path, eval_dir: Path) -> list[Path]:
    summary = json.loads((eval_dir / "geometry_ablation_summary.json").read_text())
    rows = list(csv.DictReader(open(eval_dir / "per_sample_all.csv")))
    by_v = defaultdict(list)
    for r in rows:
        by_v[r["variant"]].append(r)
    order = [v for v in ORDER if v in by_v] or sorted(by_v.keys())
    plot_vs = [v for v in order if v != "as_logged"]
    fig_dir = eval_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    written = []

    # 1) metric profiles
    fig, axes = plt.subplots(1, 2 if "frac_top1_match_logged" in summary["summaries"][order[0]] else 1,
                             figsize=(9.5, 3.4))
    if not isinstance(axes, np.ndarray):
        axes = np.array([axes])
    xs = np.arange(len(order))
    ades = [summary["summaries"][v]["ADE@best"] for v in order]
    axes[0].plot(xs, ades, "-o", color="#333", lw=1.6, ms=7)
    for i, v in enumerate(order):
        axes[0].plot(xs[i], ades[i], "o", color=COLORS.get(v, "#333"), ms=9)
        axes[0].annotate(f"{ades[i]:.3f}", (xs[i], ades[i]), textcoords="offset points",
                         xytext=(0, 8), ha="center", fontsize=7)
    axes[0].set_xticks(xs)
    axes[0].set_xticklabels([LABELS.get(v, v) for v in order], rotation=25, ha="right", fontsize=8)
    axes[0].set_ylabel("ADE@best (m)")
    axes[0].set_title("Open-loop ADE vs attitude re-render")
    axes[0].grid(True, axis="y", alpha=0.35)
    axes[0].spines["top"].set_visible(False)
    axes[0].spines["right"].set_visible(False)

    if len(axes) > 1:
        top1 = []
        for v in order:
            t = summary["summaries"][v].get("frac_top1_match_logged", float("nan"))
            top1.append(t * 100 if t == t else float("nan"))
        axes[1].plot(xs, top1, "-o", color="#333", lw=1.6, ms=7)
        for i, v in enumerate(order):
            if np.isfinite(top1[i]):
                axes[1].plot(xs[i], top1[i], "o", color=COLORS.get(v, "#333"), ms=9)
                axes[1].annotate(f"{top1[i]:.1f}%", (xs[i], top1[i]), textcoords="offset points",
                                 xytext=(0, 8), ha="center", fontsize=7)
        axes[1].set_xticks(xs)
        axes[1].set_xticklabels([LABELS.get(v, v) for v in order], rotation=25, ha="right", fontsize=8)
        axes[1].set_ylabel("top-1 ≡ as_logged (%)")
        axes[1].set_title("Mode stability under viewpoint change")
        axes[1].grid(True, axis="y", alpha=0.35)
        axes[1].spines["top"].set_visible(False)
        axes[1].spines["right"].set_visible(False)
    fig.suptitle("Chrono sensor-geometry ablation (status fixed)", fontsize=11, y=1.02)
    fig.tight_layout()
    p = fig_dir / "01_metric_profiles.png"
    fig.savefig(p, dpi=160, bbox_inches="tight")
    plt.close(fig)
    written.append(p)

    # 2) ECDF ΔADE
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    for v in plot_vs:
        dade = np.array([float(r["delta_ade_vs_logged"]) for r in by_v[v]], dtype=np.float64)
        dade = dade[np.isfinite(dade)]
        if len(dade) == 0:
            continue
        xs_s = np.sort(dade)
        ys = np.arange(1, len(xs_s) + 1) / len(xs_s)
        ax.plot(xs_s, ys, color=COLORS.get(v, "#333"), lw=1.8, label=LABELS.get(v, v))
    ax.axvline(0, color="#666", ls="--", lw=1)
    ax.set_xlabel("ΔADE vs as_logged (m)")
    ax.set_ylabel("ECDF")
    ax.set_title("Per-sample ADE change from attitude re-render")
    ax.legend(frameon=False, fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    p = fig_dir / "02_delta_ade_ecdf.png"
    fig.savefig(p, dpi=160, bbox_inches="tight")
    plt.close(fig)
    written.append(p)

    # 3) slope chart
    fig, ax = plt.subplots(figsize=(7.0, 3.6))
    base = summary["summaries"]["as_logged"]["ADE@best"]
    for i, v in enumerate(plot_vs):
        y = len(plot_vs) - 1 - i
        ade = summary["summaries"][v]["ADE@best"]
        ax.plot([base, ade], [y, y], color="#bbb", lw=1.5)
        ax.scatter([base], [y], s=36, c=COLORS["as_logged"], zorder=2)
        ax.scatter([ade], [y], s=48, c=COLORS.get(v, "#333"), zorder=3)
        ax.text(max(base, ade) + 0.01, y, f"{ade:.3f} ({ade-base:+.3f})", va="center", fontsize=8)
    ax.set_yticks(range(len(plot_vs)))
    ax.set_yticklabels([LABELS.get(v, v) for v in plot_vs][::-1])
    ax.set_xlabel("ADE@best (m)")
    ax.set_title("ADE slope: as_logged → re-rendered attitude")
    ax.axvline(base, color=COLORS["as_logged"], ls=":", lw=1, alpha=0.7)
    ax.grid(True, axis="x", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.legend(
        handles=[
            Line2D([0], [0], marker="o", color="w", markerfacecolor=COLORS["as_logged"], ms=8, label="as_logged"),
            Line2D([0], [0], marker="o", color="w", markerfacecolor="#888", ms=8, label="re-render"),
        ],
        loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.05),
    )
    fig.tight_layout()
    p = fig_dir / "03_slope_ade.png"
    fig.savefig(p, dpi=160, bbox_inches="tight")
    plt.close(fig)
    written.append(p)

    # 4) mosaic of RGB panels (up to 6)
    panels = sorted((render_dir / "frames").glob("*/panel_rgb.png"))[:6]
    if panels:
        ims = [Image.open(p).convert("RGB") for p in panels]
        w = max(im.width for im in ims)
        # resize to common width
        resized = []
        for im in ims:
            if im.width != w:
                nh = int(im.height * w / im.width)
                im = im.resize((w, nh), Image.BILINEAR)
            resized.append(np.asarray(im))
        gap = np.full((12, w, 3), 245, dtype=np.uint8)
        stack = []
        for i, arr in enumerate(resized):
            if i:
                stack.append(gap)
            stack.append(arr)
        mosaic = np.concatenate(stack, axis=0)
        p = fig_dir / "04_rgb_panels_mosaic.png"
        Image.fromarray(mosaic).save(p)
        written.append(p)

    # 5) ADE vs logged attitude magnitude (as_logged vs level)
    if "as_logged" in by_v and "level" in by_v:
        fig, ax = plt.subplots(figsize=(6.5, 4.0))
        mags, dade = [], []
        id_map = {r["stem"]: r for r in by_v["as_logged"]}
        for r in by_v["level"]:
            b = id_map.get(r["stem"])
            if not b:
                continue
            mags.append(float(b["att_mag_deg_logged"]))
            dade.append(float(r["delta_ade_vs_logged"]))
        ax.scatter(mags, dade, s=28, c=COLORS["level"], alpha=0.75, edgecolors="none")
        ax.axhline(0, color="#666", ls="--", lw=1)
        ax.set_xlabel("logged |roll|+|pitch| (deg)")
        ax.set_ylabel("ΔADE: level − as_logged (m)")
        ax.set_title("Does leveling hurt more on steep frames?")
        ax.grid(True, alpha=0.3)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        fig.tight_layout()
        p = fig_dir / "05_delta_vs_logged_attitude.png"
        fig.savefig(p, dpi=160, bbox_inches="tight")
        plt.close(fig)
        written.append(p)

    return written


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--render_dir", type=Path, required=True)
    ap.add_argument("--eval_dir", type=Path, default=None)
    args = ap.parse_args()
    eval_dir = args.eval_dir or (args.render_dir / "eval")
    for p in plot_all(args.render_dir, eval_dir):
        print(f"[wrote] {p}")


if __name__ == "__main__":
    main()
