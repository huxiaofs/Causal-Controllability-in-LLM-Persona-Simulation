import argparse
import csv
import json
import math
import pathlib
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

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


@dataclass
class ModelInput:
    name: str
    out_dir: pathlib.Path


@dataclass
class SampleRow:
    model: str
    persona_id: int
    template_idx: int
    dimension: str
    target: float
    observed: float
    mismatch_abs: float
    direction_consistency: Optional[float]
    intervention_strength: float


def _safe_float(x: Optional[str]) -> Optional[float]:
    if x is None:
        return None
    x = str(x).strip()
    if not x:
        return None
    return float(x)


def _mean(values: Iterable[float]) -> Optional[float]:
    vals = [float(v) for v in values if v is not None and not math.isnan(float(v))]
    if not vals:
        return None
    return float(np.mean(vals))


def _std(values: Iterable[float], ddof: int = 1) -> Optional[float]:
    vals = [float(v) for v in values if v is not None and not math.isnan(float(v))]
    if not vals:
        return None
    if len(vals) <= ddof:
        return 0.0
    return float(np.std(vals, ddof=ddof))


def _sem(values: Iterable[float]) -> Optional[float]:
    vals = [float(v) for v in values if v is not None and not math.isnan(float(v))]
    n = len(vals)
    if n == 0:
        return None
    if n == 1:
        return 0.0
    return float(np.std(vals, ddof=1) / math.sqrt(n))


def _ci95(values: Iterable[float]) -> Optional[float]:
    sem = _sem(values)
    if sem is None:
        return None
    return float(1.96 * sem)


def load_heatmap_long_csv(path: pathlib.Path) -> List[Dict]:
    if not path.exists():
        raise FileNotFoundError(f"missing file: {path}")
    rows: List[Dict] = []
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def build_template_stability_tables(
    heatmap_rows: List[Dict],
    out_summary_csv: pathlib.Path,
    out_per_template_csv: pathlib.Path,
) -> None:
    out_summary_csv.parent.mkdir(parents=True, exist_ok=True)

    per_key = defaultdict(list)
    per_template_rows: List[Dict] = []

    for row in heatmap_rows:
        model = row["model"]
        template_idx = int(row["template_idx"])
        direction_group = row["direction_group"]
        dim = row["dimension"]

        mismatch = _safe_float(row.get("mismatch_abs"))
        direction = _safe_float(row.get("direction_consistency"))
        n = int(row.get("sample_count", 0) or 0)

        per_template_rows.append(
            {
                "model": model,
                "template_idx": template_idx,
                "direction_group": direction_group,
                "dimension": dim,
                "mismatch_abs": mismatch,
                "direction_consistency": direction,
                "sample_count": n,
            }
        )

        per_key[(model, direction_group, dim)].append(
            {
                "template_idx": template_idx,
                "mismatch_abs": mismatch,
                "direction_consistency": direction,
                "sample_count": n,
            }
        )

    with out_per_template_csv.open("w", encoding="utf-8", newline="") as f:
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
        for row in sorted(
            per_template_rows,
            key=lambda x: (x["model"], x["direction_group"], x["dimension"], x["template_idx"]),
        ):
            writer.writerow(
                [
                    row["model"],
                    row["template_idx"],
                    row["direction_group"],
                    row["dimension"],
                    "" if row["mismatch_abs"] is None else row["mismatch_abs"],
                    "" if row["direction_consistency"] is None else row["direction_consistency"],
                    row["sample_count"],
                ]
            )

    with out_summary_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "model",
                "direction_group",
                "dimension",
                "template_count",
                "sample_count_mean",
                "mismatch_mean",
                "mismatch_std",
                "mismatch_ci95",
                "direction_consistency_mean",
                "direction_consistency_std",
                "direction_consistency_ci95",
            ]
        )

        for key in sorted(per_key.keys()):
            model, direction_group, dim = key
            vals = per_key[key]
            mismatch_vals = [v["mismatch_abs"] for v in vals if v["mismatch_abs"] is not None]
            direction_vals = [v["direction_consistency"] for v in vals if v["direction_consistency"] is not None]
            count_vals = [v["sample_count"] for v in vals]

            writer.writerow(
                [
                    model,
                    direction_group,
                    dim,
                    len(vals),
                    _mean(count_vals),
                    _mean(mismatch_vals),
                    _std(mismatch_vals),
                    _ci95(mismatch_vals),
                    _mean(direction_vals),
                    _std(direction_vals),
                    _ci95(direction_vals),
                ]
            )


def load_label_score_rows(model: ModelInput, center: float, neutral_band: float) -> List[SampleRow]:
    pred_path = model.out_dir / "persona_hexaco_pred.jsonl"
    score_path = model.out_dir / "questionnaire_scores.jsonl"

    if not pred_path.exists():
        raise FileNotFoundError(f"missing file: {pred_path}")
    if not score_path.exists():
        raise FileNotFoundError(f"missing file: {score_path}")

    labels: List[List[float]] = []
    for i, line in enumerate(pred_path.read_text(encoding="utf-8").splitlines()):
        if not line.strip():
            continue
        row = json.loads(line)
        label = row.get("hexaco_label")
        if not isinstance(label, list) or len(label) != 6:
            raise ValueError(f"invalid hexaco_label at {pred_path} line {i + 1}")
        labels.append([float(v) for v in label])

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

    rows: List[SampleRow] = []
    for i, label_vec in enumerate(labels):
        persona_id = i // 3
        template_idx = i % 3

        score_vec = score_map.get((persona_id, template_idx))
        if score_vec is None:
            raise ValueError(
                f"missing score for model={model.name}, persona_id={persona_id}, prompt_idx={template_idx}"
            )

        for d, dim in enumerate(DIM_SHORT):
            target = float(label_vec[d])
            observed = float(score_vec[d])
            strength = abs(target - center)

            if abs(target - center) <= neutral_band:
                dc = None
            else:
                dc = float(np.sign(target - center) == np.sign(observed - center))

            rows.append(
                SampleRow(
                    model=model.name,
                    persona_id=persona_id,
                    template_idx=template_idx,
                    dimension=dim,
                    target=target,
                    observed=observed,
                    mismatch_abs=abs(observed - target),
                    direction_consistency=dc,
                    intervention_strength=strength,
                )
            )

    return rows


def build_strength_tables(
    sample_rows: List[SampleRow],
    out_raw_csv: pathlib.Path,
    out_by_model_csv: pathlib.Path,
    out_by_dimension_csv: pathlib.Path,
    bins: int,
    max_strength: float,
) -> None:
    out_raw_csv.parent.mkdir(parents=True, exist_ok=True)

    edges = np.linspace(0.0, max_strength, bins + 1)

    def assign_bin(x: float) -> Tuple[int, float, float, float]:
        idx = int(np.digitize([x], edges, right=False)[0]) - 1
        if idx < 0:
            idx = 0
        if idx >= bins:
            idx = bins - 1
        left = float(edges[idx])
        right = float(edges[idx + 1])
        center = float((left + right) / 2.0)
        return idx, left, right, center

    raw_rows = []
    for r in sample_rows:
        bin_idx, left, right, bin_center = assign_bin(r.intervention_strength)
        raw_rows.append(
            {
                "model": r.model,
                "persona_id": r.persona_id,
                "template_idx": r.template_idx,
                "dimension": r.dimension,
                "target": r.target,
                "observed": r.observed,
                "mismatch_abs": r.mismatch_abs,
                "direction_consistency": r.direction_consistency,
                "intervention_strength": r.intervention_strength,
                "strength_bin_idx": bin_idx,
                "strength_bin_left": left,
                "strength_bin_right": right,
                "strength_bin_center": bin_center,
            }
        )

    with out_raw_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "model",
                "persona_id",
                "template_idx",
                "dimension",
                "target",
                "observed",
                "mismatch_abs",
                "direction_consistency",
                "intervention_strength",
                "strength_bin_idx",
                "strength_bin_left",
                "strength_bin_right",
                "strength_bin_center",
            ]
        )
        for row in raw_rows:
            writer.writerow(
                [
                    row["model"],
                    row["persona_id"],
                    row["template_idx"],
                    row["dimension"],
                    row["target"],
                    row["observed"],
                    row["mismatch_abs"],
                    "" if row["direction_consistency"] is None else row["direction_consistency"],
                    row["intervention_strength"],
                    row["strength_bin_idx"],
                    row["strength_bin_left"],
                    row["strength_bin_right"],
                    row["strength_bin_center"],
                ]
            )

    def aggregate(rows: List[Dict], keys: Tuple[str, ...], out_csv: pathlib.Path) -> None:
        grouped = defaultdict(list)
        for row in rows:
            grouped[tuple(row[k] for k in keys)].append(row)

        with out_csv.open("w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                list(keys)
                + [
                    "strength_bin_left",
                    "strength_bin_right",
                    "strength_bin_center",
                    "n_total",
                    "n_direction_valid",
                    "mismatch_mean",
                    "mismatch_std",
                    "mismatch_sem",
                    "mismatch_ci95",
                    "direction_consistency_mean",
                    "direction_consistency_std",
                    "direction_consistency_sem",
                    "direction_consistency_ci95",
                ]
            )

            for gkey in sorted(grouped.keys()):
                vals = grouped[gkey]
                mismatch_vals = [v["mismatch_abs"] for v in vals]
                direction_vals = [v["direction_consistency"] for v in vals if v["direction_consistency"] is not None]

                first = vals[0]
                writer.writerow(
                    list(gkey)
                    + [
                        first["strength_bin_left"],
                        first["strength_bin_right"],
                        first["strength_bin_center"],
                        len(vals),
                        len(direction_vals),
                        _ci95(mismatch_vals)
                    ]
                )

    aggregate(raw_rows, keys=("model", "strength_bin_idx"), out_csv=out_by_model_csv)
    aggregate(raw_rows, keys=("model", "dimension", "strength_bin_idx"), out_csv=out_by_dimension_csv)


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare plot-support tables for template stability and intervention strength trends")
    parser.add_argument("--llama-dir", type=str, default="output")
    parser.add_argument("--qwen-dir", type=str, default="output_qwen2.5-7b")
    parser.add_argument("--deepseek-dir", type=str, default="output_deepseek-r1-distill-qwen-7b")
    parser.add_argument("--heatmap-csv", type=str, default="output/figures/hexaco_main_dual_heatmap_metrics.csv")
    parser.add_argument("--out-dir", type=str, default="output/figures")
    parser.add_argument("--center", type=float, default=3.0)
    parser.add_argument("--neutral-band", type=float, default=0.1)
    parser.add_argument("--strength-bins", type=int, default=8)
    parser.add_argument("--max-strength", type=float, default=2.0)
    args = parser.parse_args()

    model_inputs = [
        ModelInput("Meta-Llama-3-8B-Instruct", pathlib.Path(args.llama_dir)),
        ModelInput("Qwen2.5-7B-Instruct", pathlib.Path(args.qwen_dir)),
        ModelInput("DeepSeek-R1-Distill-Qwen-7B", pathlib.Path(args.deepseek_dir)),
    ]

    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    heatmap_rows = load_heatmap_long_csv(pathlib.Path(args.heatmap_csv))
    build_template_stability_tables(
        heatmap_rows=heatmap_rows,
        out_summary_csv=out_dir / "template_stability_errorbar_summary.csv",
        out_per_template_csv=out_dir / "template_stability_per_template.csv",
    )

    all_sample_rows: List[SampleRow] = []
    for model in model_inputs:
        all_sample_rows.extend(
            load_label_score_rows(
                model=model,
                center=float(args.center),
                neutral_band=float(args.neutral_band),
            )
        )

    build_strength_tables(
        sample_rows=all_sample_rows,
        out_raw_csv=out_dir / "intervention_strength_raw_samples.csv",
        out_by_model_csv=out_dir / "intervention_strength_trend_by_model.csv",
        out_by_dimension_csv=out_dir / "intervention_strength_trend_by_model_dimension.csv",
        bins=int(args.strength_bins),
        max_strength=float(args.max_strength),
    )

    print(f"saved template stability table: {out_dir / 'template_stability_errorbar_summary.csv'}")
    print(f"saved per-template table: {out_dir / 'template_stability_per_template.csv'}")
    print(f"saved strength raw samples: {out_dir / 'intervention_strength_raw_samples.csv'}")
    print(f"saved strength trend by model: {out_dir / 'intervention_strength_trend_by_model.csv'}")
    print(f"saved strength trend by model+dimension: {out_dir / 'intervention_strength_trend_by_model_dimension.csv'}")


if __name__ == "__main__":
    main()
