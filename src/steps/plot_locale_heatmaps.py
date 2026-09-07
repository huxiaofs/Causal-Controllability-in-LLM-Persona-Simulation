"""
Generate Model × Region heatmaps from persona_locale_map.json.

Produces 3 figures:
  fig_locale_q_heatmap.png  — local quality  q = v - l  (most important)
  fig_locale_l_heatmap.png  — local mismatch (twist)
  fig_locale_v_heatmap.png  — local diversity (collapse)

Usage:
    .venv/bin/python src/steps/plot_locale_heatmaps.py
    .venv/bin/python src/steps/plot_locale_heatmaps.py --input output/persona_locale_map.json
"""
import argparse
import json
import pathlib

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

ROOT = pathlib.Path(__file__).resolve().parent.parent.parent

MODEL_SHORT = {
    "qwen2.5-7b": "Qwen-2.5-7B",
    "deepseek-r1-distill-qwen-7b": "DeepSeek-R1-7B",
    "mistral-7b-instruct-v0.3": "Mistral-7B-v0.3",
    "llama3.1-8b": "Llama-3.1-8B",
}

REGION_ORDER_GROUPS = {
    "L1: Extremes": [
        "High-H", "Low-H", "High-E", "Low-E", "High-X", "Low-X",
    ],
    "L2: Combinations": [
        "High-H+A", "High-X,Low-H", "High-C,Low-A", "High-E,Low-C",
    ],
    "L3: Near-boundary": [
        "Near-H", "Near-E", "Near-X", "Near-A", "Near-C", "Near-O",
    ],
}


def plot_heatmap(
    values: np.ndarray,
    model_names: list,
    region_names: list,
    metric: str,
    out_path: pathlib.Path,
):
    titles = {
        "q": "Local Quality  $q = v - l$  (higher = better)",
        "l": "Local Mismatch  $l$  (lower = better)",
        "v": "Local Diversity  $v = \\frac{1}{2}\\log\\det(h)$  (higher = better)",
    }
    cmaps = {"q": "RdYlGn", "l": "RdYlGn_r", "v": "YlGnBu"}

    # reorder regions by group
    ordered_regions = []
    group_boundaries = []
    for gname, members in REGION_ORDER_GROUPS.items():
        start = len(ordered_regions)
        for r in members:
            if r in region_names:
                ordered_regions.append(r)
        if len(ordered_regions) > start:
            group_boundaries.append((start, len(ordered_regions), gname))

    col_idx = [region_names.index(r) for r in ordered_regions]
    mat = values[:, col_idx]
    display_names = [MODEL_SHORT.get(m, m) for m in model_names]

    fig, ax = plt.subplots(figsize=(max(10, len(ordered_regions) * 0.75), len(model_names) * 0.9 + 2.5))

    vmin, vmax = np.nanmin(mat), np.nanmax(mat)
    if metric == "q":
        vcenter = 0.0 if vmin < 0 < vmax else (vmin + vmax) / 2
        norm = TwoSlopeNorm(vmin=vmin, vcenter=vcenter, vmax=vmax)
    else:
        norm = None

    im = ax.imshow(mat, aspect="auto", cmap=cmaps[metric], norm=norm,
                   vmin=None if norm else vmin, vmax=None if norm else vmax)

    ax.set_xticks(range(len(ordered_regions)))
    ax.set_xticklabels(ordered_regions, rotation=55, ha="right", fontsize=8)
    ax.set_yticks(range(len(display_names)))
    ax.set_yticklabels(display_names, fontsize=10)

    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            val = mat[i, j]
            if np.isnan(val):
                continue
            text_color = "white" if abs(val - np.nanmean(mat)) > np.nanstd(mat) else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                    fontsize=7, color=text_color)

    # group separators
    for start, end, gname in group_boundaries:
        if start > 0:
            ax.axvline(start - 0.5, color="gray", linewidth=1.5, linestyle="--")

    cbar = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
    cbar.set_label(metric, fontsize=10)

    ax.set_title(titles.get(metric, metric), fontsize=12, pad=12)
    fig.tight_layout()
    fig.savefig(str(out_path), dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="output/persona_locale_map.json")
    parser.add_argument("--outdir", default="output/figures")
    args = parser.parse_args()

    data = json.loads((ROOT / args.input).read_text(encoding="utf-8"))
    out_dir = ROOT / args.outdir
    out_dir.mkdir(parents=True, exist_ok=True)

    for metric in ["q", "l", "v"]:
        hm = data["heatmaps"][metric]
        values = np.array(hm["values"], dtype=float)
        plot_heatmap(
            values,
            hm["model_names"],
            hm["region_names"],
            metric,
            out_dir / f"fig_locale_{metric}_heatmap.png",
        )


if __name__ == "__main__":
    main()
