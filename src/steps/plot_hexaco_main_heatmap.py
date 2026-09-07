import argparse
import csv
import json
import pathlib
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np


DIM_NAMES = [
    "Honesty-Humility",
    "Emotionality",
    "Extraversion",
    "Agreeableness",
    "Conscientiousness",
    "Openness",
]
DIM_SHORT = ["H", "E", "X", "A", "C", "O"]


def load_model_arrays(output_dir: pathlib.Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    pred_path = output_dir / "persona_hexaco_pred.jsonl"
    score_path = output_dir / "questionnaire_scores.jsonl"

    if not pred_path.exists():
        raise FileNotFoundError(f"missing file: {pred_path}")
    if not score_path.exists():
        raise FileNotFoundError(f"missing file: {score_path}")

    labels: List[List[float]] = []
    templates: List[int] = []
    for i, line in enumerate(pred_path.read_text(encoding="utf-8").splitlines()):
        if not line.strip():
            continue
        row = json.loads(line)
        label = row.get("hexaco_label")
        if not isinstance(label, list) or len(label) != 6:
            raise ValueError(f"invalid hexaco_label at {pred_path} line {i + 1}")
        labels.append([float(v) for v in label])
        templates.append(i % 3)

    score_map: Dict[Tuple[int, int], List[float]] = {}
    for i, line in enumerate(score_path.read_text(encoding="utf-8").splitlines()):
        if not line.strip():
            continue
        row = json.loads(line)
        pid = int(row["persona_id"])
        pidx = int(row["prompt_idx"])
        dims = row["scores"]["hexaco"]["dimensions"]
        vec = [float(dims[k]) for k in DIM_NAMES]
        score_map[(pid, pidx)] = vec

    scores: List[List[float]] = []
    for i in range(len(labels)):
        persona_id = i // 3
        prompt_idx = i % 3
        if (persona_id, prompt_idx) not in score_map:
            raise ValueError(
                f"missing score for persona_id={persona_id}, prompt_idx={prompt_idx} in {score_path}"
            )
        scores.append(score_map[(persona_id, prompt_idx)])

    return np.array(labels, dtype=float), np.array(scores, dtype=float), np.array(templates, dtype=int)


def compute_dimension_metrics(
    labels: np.ndarray,
    scores: np.ndarray,
    templates: np.ndarray,
    quantile: float,
    center: float,
    neutral_band: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str], List[Tuple[int, str]]]:
    q_low = np.quantile(labels, quantile, axis=0)
    q_high = np.quantile(labels, 1.0 - quantile, axis=0)

    mismatch_rows = []
    direction_rows = []
    count_rows = []
    row_labels = []
    row_meta: List[Tuple[int, str]] = []

    for template_idx in (0, 1, 2):
        template_mask = templates == template_idx
        for direction in ("High", "Low"):
            mismatch_vals = []
            direction_vals = []
            count_vals = []

            for d in range(6):
                region_mask = labels[:, d] >= q_high[d] if direction == "High" else labels[:, d] <= q_low[d]
                mask = template_mask & region_mask
                n = int(mask.sum())
                count_vals.append(n)

                if n == 0:
                    mismatch_vals.append(np.nan)
                    direction_vals.append(np.nan)
                    continue

                mismatch = float(np.mean(np.abs(scores[mask, d] - labels[mask, d])))

                label_side = labels[mask, d] - center
                score_side = scores[mask, d] - center
                valid = np.abs(label_side) > neutral_band
                if int(valid.sum()) == 0:
                    direction_consistency = np.nan
                else:
                    direction_consistency = float(
                        np.mean(np.sign(label_side[valid]) == np.sign(score_side[valid]))
                    )

                mismatch_vals.append(mismatch)
                direction_vals.append(direction_consistency)

            mismatch_rows.append(mismatch_vals)
            direction_rows.append(direction_vals)
            count_rows.append(count_vals)
            row_labels.append(f"T{template_idx + 1} | {direction}")
            row_meta.append((template_idx, direction))

    return (
        np.array(mismatch_rows, dtype=float),
        np.array(direction_rows, dtype=float),
        np.array(count_rows, dtype=float),
        row_labels,
        row_meta,
    )


def draw_heatmap(
    ax,
    data: np.ndarray,
    counts: np.ndarray,
    row_labels: List[str],
    col_labels: List[str],
    title: str,
    cmap: str,
    vmin: float,
    vmax: float,
    fmt: str,
) -> None:
    cm = plt.get_cmap(cmap).copy()
    cm.set_bad("#E6E6E6")

    im = ax.imshow(data, aspect="auto", cmap=cm, vmin=vmin, vmax=vmax)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_xticks(np.arange(len(col_labels)))
    ax.set_xticklabels(col_labels, fontsize=10)
    ax.set_yticks(np.arange(len(row_labels)))
    ax.set_yticklabels(row_labels, fontsize=8)

    ax.set_xticks(np.arange(-0.5, len(col_labels), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(row_labels), 1), minor=True)
    ax.grid(which="minor", color="white", linestyle="-", linewidth=0.8)
    ax.tick_params(which="minor", bottom=False, left=False)

    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            val = data[i, j]
            if np.isnan(val):
                txt = "-"
                color = "black"
            else:
                txt = format(val, fmt)
                color = "white" if val > (vmin + vmax) / 2 else "black"
            ax.text(j, i, txt, ha="center", va="center", fontsize=8, color=color)

    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cbar.ax.tick_params(labelsize=9)


def export_long_csv(
    out_csv: pathlib.Path,
    model_names: List[str],
    row_metas: List[List[Tuple[int, str]]],
    mismatch: np.ndarray,
    direction: np.ndarray,
    counts: np.ndarray,
) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "model",
                "template_idx",
                "direction_group",
                "dimension",
                "mismatch_abs",
                "direction_consistency",
                "sample_count",
            ]
        )

        row_base = 0
        for model_idx, model_name in enumerate(model_names):
            metas = row_metas[model_idx]
            for local_row, (template_idx, direction_group) in enumerate(metas):
                global_row = row_base + local_row
                for d, dim in enumerate(DIM_SHORT):
                    writer.writerow(
                        [
                            model_name,
                            template_idx,
                            direction_group,
                            dim,
                            float(mismatch[global_row, d]) if not np.isnan(mismatch[global_row, d]) else "",
                            float(direction[global_row, d]) if not np.isnan(direction[global_row, d]) else "",
                            int(counts[global_row, d]) if not np.isnan(counts[global_row, d]) else 0,
                        ]
                    )
            row_base += len(metas)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot HEXACO main-result dual heatmaps across three model outputs")
    parser.add_argument("--llama-dir", type=str, default="output")
    parser.add_argument("--qwen-dir", type=str, default="output_qwen2.5-7b")
    parser.add_argument("--deepseek-dir", type=str, default="output_deepseek-r1-distill-qwen-7b")
    parser.add_argument("--quantile", type=float, default=0.35, help="high/low personality region split quantile")
    parser.add_argument("--center", type=float, default=3.0, help="neutral center on 1-5 HEXACO scale")
    parser.add_argument("--neutral-band", type=float, default=0.1, help="ignore near-center labels for direction consistency")
    parser.add_argument("--dpi", type=int, default=240)
    parser.add_argument("--fig-path", type=str, default="output/figures/hexaco_main_dual_heatmap.png")
    parser.add_argument("--csv-path", type=str, default="output/figures/hexaco_main_dual_heatmap_metrics.csv")
    args = parser.parse_args()

    model_inputs = [
        ("Meta-Llama-3-8B-Instruct", pathlib.Path(args.llama_dir)),
        ("Qwen2.5-7B-Instruct", pathlib.Path(args.qwen_dir)),
        ("DeepSeek-R1-Distill-Qwen-7B", pathlib.Path(args.deepseek_dir)),
    ]

    all_mismatch = []
    all_direction = []
    all_counts = []
    all_row_labels = []
    all_row_meta: List[List[Tuple[int, str]]] = []

    for model_name, out_dir in model_inputs:
        labels, scores, templates = load_model_arrays(out_dir)
        mismatch, direction, counts, row_labels, row_meta = compute_dimension_metrics(
            labels=labels,
            scores=scores,
            templates=templates,
            quantile=float(args.quantile),
            center=float(args.center),
            neutral_band=float(args.neutral_band),
        )

        all_mismatch.append(mismatch)
        all_direction.append(direction)
        all_counts.append(counts)
        all_row_labels.extend([f"{model_name} | {row}" for row in row_labels])
        all_row_meta.append(row_meta)

    mismatch_mat = np.vstack(all_mismatch)
    direction_mat = np.vstack(all_direction)
    counts_mat = np.vstack(all_counts)

    finite_mismatch = mismatch_mat[np.isfinite(mismatch_mat)]
    mismatch_vmax = float(np.percentile(finite_mismatch, 95)) if finite_mismatch.size else 1.0
    mismatch_vmax = max(mismatch_vmax, 0.5)

    fig, axes = plt.subplots(1, 2, figsize=(24, 12), constrained_layout=True)

    draw_heatmap(
        ax=axes[0],
        data=mismatch_mat,
        counts=counts_mat,
        row_labels=all_row_labels,
        col_labels=DIM_SHORT,
        title="(a) Geometric Mismatch (|Questionnaire - Target|)",
        cmap="YlOrRd_r",
        vmin=0.0,
        vmax=mismatch_vmax,
        fmt=".2f",
    )

    draw_heatmap(
        ax=axes[1],
        data=direction_mat,
        counts=counts_mat,
        row_labels=all_row_labels,
        col_labels=DIM_SHORT,
        title="(b) Direction Consistency (higher is better)",
        cmap="YlGnBu",
        vmin=0.0,
        vmax=1.0,
        fmt=".2f",
    )

    rows_per_model = 6
    for ax in axes:
        for k in range(1, len(model_inputs)):
            ax.axhline(k * rows_per_model - 0.5, color="black", linewidth=1.4)

    fig.suptitle(
        "HEXACO Main Result Heatmaps Across Models / Prompt Templates / Personality Regions",
        fontsize=15,
        fontweight="bold",
    )

    out_fig = pathlib.Path(args.fig_path)
    out_fig.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_fig, dpi=int(args.dpi), bbox_inches="tight")

    export_long_csv(
        out_csv=pathlib.Path(args.csv_path),
        model_names=[m[0] for m in model_inputs],
        row_metas=all_row_meta,
        mismatch=mismatch_mat,
        direction=direction_mat,
        counts=counts_mat,
    )

    print(f"saved figure to {out_fig}")
    print(f"saved metric table to {pathlib.Path(args.csv_path)}")


if __name__ == "__main__":
    main()
