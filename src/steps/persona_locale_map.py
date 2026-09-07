"""
Persona Locale Map — local controllability landscape over the personality space.

From per-point (g, h) Fisher metrics, construct:
  1. Local indicators  l_i (mismatch), v_i (diversity), q_i (quality)
  2. Structured region partition (3 levels, 10-15 regions)
  3. Model × Region aggregates for heatmap visualization
  4. Candidate point selection for downstream open-ended tasks

All heatmaps, region aggregates, and point selection use uncalibrated local
indicators (q, l, v) — pure measures of local controllability without the
global volume term (vol_theta) that EI_g includes.  EI_g provides the global
ranking; the locale map complements it with local detail.

Usage:
    .venv/bin/python src/steps/persona_locale_map.py
    .venv/bin/python src/steps/persona_locale_map.py --models qwen2.5-7b llama3.1-8b
"""
import argparse
import json
import pathlib
from typing import Dict, List, Tuple

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent.parent

HEXACO_DIMS = ["H", "E", "X", "A", "C", "O"]
HEXACO_FULL = [
    "Honesty-Humility", "Emotionality", "Extraversion",
    "Agreeableness", "Conscientiousness", "Openness",
]

MODEL_OUTPUT_DIRS = {
    "qwen2.5-7b": ROOT / "output_qwen2.5-7b",
    "deepseek-r1-distill-qwen-7b": ROOT / "output_deepseek-r1-distill-qwen-7b",
    "mistral-7b-instruct-v0.3": ROOT / "output_mistral-7b-instruct-v0.3",
    "llama3.1-8b": ROOT / "output_llama3.1-8b",
}

# ── coordinate range: HEXACO labels are on [1, 5] scale ──
HIGH_THRESH = 3.8   # top ~30% of [1, 5]
LOW_THRESH  = 2.2   # bottom ~30% of [1, 5]
NEAR_LO = 2.7       # near-boundary band
NEAR_HI = 3.3


# ═══════════════════════════════════════════════════════════
#  Step 1 — Local indicators
# ═══════════════════════════════════════════════════════════

def compute_local_indicators(
    data: Dict[str, np.ndarray],
) -> Dict[str, np.ndarray]:
    """
    Compute per-point indicators using the same weighting as EI_g (strategy B).

    w_i = sqrt(max(det(h_i), 0))  — same as EI_g
    v_i = log(w_i) = 0.5 * log(max(det(h_i), floor))
    l_i = mismatch (from data['l'])
    q_i = v_i - l_i  (uncalibrated local quality)
    healthy_i = True if det(h_i) > 0 and h_i is PSD
    """
    l = data["l"]                # [N,]
    h = data["h"]                # [N, d, d]
    N, d, _ = h.shape

    det_h = np.empty(N)
    min_eigval = np.empty(N)
    for i in range(N):
        det_h[i] = np.linalg.det(h[i])
        min_eigval[i] = np.linalg.eigvalsh(h[i])[0]

    healthy = (det_h > 0) & (min_eigval > -1e-6)
    det_floor = 1e-30
    v = 0.5 * np.log(np.maximum(det_h, det_floor))
    v[~healthy] = np.nan

    w = np.sqrt(np.maximum(det_h, 0.0))
    w[~healthy] = 0.0

    q = v - l
    return {"l": l, "v": v, "q": q, "w": w, "healthy": healthy}


# ═══════════════════════════════════════════════════════════
#  Step 2 — Structured region partition
# ═══════════════════════════════════════════════════════════

def assign_regions(x: np.ndarray) -> Dict[str, np.ndarray]:
    """
    Assign each point to named regions.
    Returns {region_name: boolean mask [N,]}.
    A point can belong to multiple regions.

    Level 1 — single-dim extreme (6 regions)
    Level 2 — pairwise combination (4-6 regions)
    Level 3 — near-boundary bands (6 regions)
    """
    N = x.shape[0]
    H, E, X, A, C, O = [x[:, i] for i in range(6)]
    regions = {}

    # ── Level 1: single-dim extremes ──
    regions["High-H"] = H > HIGH_THRESH
    regions["Low-H"]  = H < LOW_THRESH
    regions["High-E"] = E > HIGH_THRESH
    regions["Low-E"]  = E < LOW_THRESH
    regions["High-X"] = X > HIGH_THRESH
    regions["Low-X"]  = X < LOW_THRESH

    # ── Level 2: pairwise combinations ──
    regions["High-H+A"] = (H > HIGH_THRESH) & (A > HIGH_THRESH)
    regions["High-X,Low-H"] = (X > HIGH_THRESH) & (H < LOW_THRESH)
    regions["High-C,Low-A"] = (C > HIGH_THRESH) & (A < LOW_THRESH)
    regions["High-E,Low-C"] = (E > HIGH_THRESH) & (C < LOW_THRESH)

    # ── Level 3: near-boundary (hardest for fine-grained control) ──
    for i, dim in enumerate(HEXACO_DIMS):
        regions[f"Near-{dim}"] = (x[:, i] >= NEAR_LO) & (x[:, i] <= NEAR_HI)

    return regions


# ═══════════════════════════════════════════════════════════
#  Step 3 — Model × Region aggregates
# ═══════════════════════════════════════════════════════════

def aggregate_regions(
    indicators: Dict[str, np.ndarray],
    regions: Dict[str, np.ndarray],
) -> Dict[str, Dict[str, float]]:
    """
    For each region, compute w-weighted mean l, v, q (consistent with EI_g strategy B).
    Falls back to unweighted mean for l when all weights are zero.
    """
    w = indicators["w"]
    healthy = indicators["healthy"]
    out = {}
    for rname, mask in regions.items():
        n_total = int(mask.sum())
        if n_total == 0:
            out[rname] = {"count": 0, "n_healthy": 0,
                          "l": None, "v": None, "q": None}
            continue
        hmask = mask & healthy
        n_h = int(hmask.sum())
        w_r = w[hmask]
        w_sum = w_r.sum()
        if n_h == 0 or w_sum < 1e-30:
            out[rname] = {
                "count": n_total, "n_healthy": 0,
                "l": float(indicators["l"][mask].mean()),
                "v": None, "q": None,
            }
            continue
        wn = w_r / w_sum
        out[rname] = {
            "count": n_total,
            "n_healthy": n_h,
            "l": float(np.sum(wn * indicators["l"][hmask])),
            "v": float(np.sum(wn * indicators["v"][hmask])),
            "q": float(np.sum(wn * indicators["q"][hmask])),
            "l_std": float(np.sqrt(np.sum(wn * (indicators["l"][hmask] - np.sum(wn * indicators["l"][hmask]))**2))),
            "v_std": float(np.sqrt(np.sum(wn * (indicators["v"][hmask] - np.sum(wn * indicators["v"][hmask]))**2))),
            "q_std": float(np.sqrt(np.sum(wn * (indicators["q"][hmask] - np.sum(wn * indicators["q"][hmask]))**2))),
        }
    return out


# ═══════════════════════════════════════════════════════════
#  Step 4 — Point selection for downstream tasks
# ═══════════════════════════════════════════════════════════

def select_downstream_points(
    all_model_regions: Dict[str, Dict[str, Dict[str, float]]],
    all_model_indicators: Dict[str, Dict[str, np.ndarray]],
    x_raw: np.ndarray,
    top_k: int = 6,
) -> Dict[str, List]:
    """
    Select 3 categories of points:
      A: max inter-model variance in q (regions where models disagree most)
      B: near-boundary regions (fine-grained discrimination)
      C: universally hard (all models low q)
    """
    model_names = list(all_model_regions.keys())
    region_names = list(next(iter(all_model_regions.values())).keys())

    # build q matrix [models × regions] — uncalibrated local quality
    q_matrix = {}
    for rname in region_names:
        qs = []
        for mname in model_names:
            val = all_model_regions[mname][rname].get("q")
            if val is not None:
                qs.append(val)
        if len(qs) == len(model_names):
            q_matrix[rname] = qs

    # Category A: max inter-model variance in local q
    var_scores = {r: float(np.var(qs)) for r, qs in q_matrix.items()}
    cat_a = sorted(var_scores.items(), key=lambda x: -x[1])[:2]

    # Category B: near-boundary regions (hardest for fine-grained control)
    near_regions = [r for r in region_names if r.startswith("Near-")]
    cat_b_scores = {r: float(np.mean(q_matrix[r])) for r in near_regions if r in q_matrix}
    cat_b = sorted(cat_b_scores.items(), key=lambda x: x[1])[:3]

    # Category C: universally hard (lowest mean q across all models)
    mean_q = {r: float(np.mean(qs)) for r, qs in q_matrix.items()}
    cat_c = sorted(mean_q.items(), key=lambda x: x[1])[:1]

    def pick_representative(region_name, model_name=None):
        mname = model_name or model_names[0]
        indicators = all_model_indicators[mname]
        regions = assign_regions(x_raw)
        mask = regions.get(region_name)
        if mask is None or mask.sum() == 0:
            return None
        hmask = mask & indicators["healthy"]
        use_mask = hmask if hmask.sum() > 0 else mask
        pts = x_raw[use_mask]
        centroid = pts.mean(axis=0)
        dists = np.linalg.norm(pts - centroid, axis=1)
        local_idx = np.argmin(dists)
        global_idx = np.where(use_mask)[0][local_idx]
        vi = indicators["v"][global_idx]
        qi = indicators["q"][global_idx]
        return {
            "global_idx": int(global_idx),
            "x_raw": x_raw[global_idx].tolist(),
            "l": float(indicators["l"][global_idx]),
            "v": float(vi) if not np.isnan(vi) else None,
            "q": float(qi) if not np.isnan(qi) else None,
        }

    selections = {
        "cat_A_divergent": [
            {"region": r, "var": v, "point": pick_representative(r)}
            for r, v in cat_a
        ],
        "cat_B_near_boundary": [
            {"region": r, "mean_q": mq, "point": pick_representative(r)}
            for r, mq in cat_b
        ],
        "cat_C_hard": [
            {"region": r, "mean_q": mq, "point": pick_representative(r)}
            for r, mq in cat_c
        ],
    }
    return selections


# ═══════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════

def _load_eig_result(out_dir: pathlib.Path) -> float:
    """Load EI_g from the model's eig_metric_result.json."""
    p = out_dir / "eig_metric_result.json"
    if not p.exists():
        return float("nan")
    return json.loads(p.read_text(encoding="utf-8"))["EI_g"]


def _weighted_global_stats(ind: Dict[str, np.ndarray]) -> Dict[str, float]:
    """w-weighted global stats, matching EI_g strategy B."""
    w = ind["w"]
    healthy = ind["healthy"]
    N = len(w)
    n_h = int(healthy.sum())
    wh = w[healthy]
    ws = wh.sum()
    if ws < 1e-30:
        return {"l_mean": float(ind["l"].mean()), "v_mean": None, "q_mean": None,
                "n_total": N, "n_healthy": n_h}
    wn = wh / ws
    l_w = float(np.sum(wn * ind["l"][healthy]))
    v_w = float(np.sum(wn * ind["v"][healthy]))
    q_w = float(v_w - l_w)
    return {"l_mean": l_w, "v_mean": v_w, "q_mean": q_w,
            "l_std": float(np.sqrt(np.sum(wn * (ind["l"][healthy] - l_w)**2))),
            "v_std": float(np.sqrt(np.sum(wn * (ind["v"][healthy] - v_w)**2))),
            "q_std": float(np.sqrt(np.sum(wn * (ind["q"][healthy] - q_w)**2))),
            "n_total": N, "n_healthy": n_h}


def run_locale_map(model_names: List[str]) -> Dict:
    all_indicators = {}
    all_regions_agg = {}
    x_raw_ref = None
    eig_values = {}

    for mname in model_names:
        out_dir = MODEL_OUTPUT_DIRS.get(mname)
        if out_dir is None:
            print(f"[skip] unknown model: {mname}")
            continue
        npz_path = out_dir / "eig_per_point.npz"
        if not npz_path.exists():
            print(f"[skip] {mname}: no eig_per_point.npz")
            continue

        data = dict(np.load(npz_path))
        indicators = compute_local_indicators(data)
        all_indicators[mname] = indicators
        eig_values[mname] = _load_eig_result(out_dir)

        x_raw = data["x_raw"]
        if x_raw_ref is None:
            x_raw_ref = x_raw

        regions = assign_regions(x_raw)
        agg = aggregate_regions(indicators, regions)
        all_regions_agg[mname] = agg

    if not all_indicators:
        raise RuntimeError("No valid model data found")

    region_names = list(next(iter(all_regions_agg.values())).keys())

    # build heatmap matrices [models × regions]
    heatmaps = {}
    for metric in ["q", "l", "v"]:
        matrix = []
        for mname in all_indicators:
            row = []
            for rname in region_names:
                val = all_regions_agg[mname][rname].get(metric)
                row.append(val if val is not None else float("nan"))
            matrix.append(row)
        heatmaps[metric] = {
            "model_names": list(all_indicators.keys()),
            "region_names": region_names,
            "values": matrix,
        }

    # point selection (using uncalibrated q)
    selections = select_downstream_points(
        all_regions_agg, all_indicators, x_raw_ref
    )

    # per-model global stats
    global_stats = {}
    for mname, ind in all_indicators.items():
        gs = _weighted_global_stats(ind)
        gs["EI_g"] = eig_values.get(mname, None)
        global_stats[mname] = gs

    return {
        "global_stats": global_stats,
        "region_aggregates": {
            mname: all_regions_agg[mname] for mname in all_indicators
        },
        "heatmaps": heatmaps,
        "selections": selections,
    }


def main():
    parser = argparse.ArgumentParser(description="Persona locale map analysis")
    parser.add_argument("--models", nargs="*", default=None)
    parser.add_argument("--output", type=str, default="output/persona_locale_map.json")
    args = parser.parse_args()

    model_names = args.models or list(MODEL_OUTPUT_DIRS.keys())
    result = run_locale_map(model_names)

    out_path = ROOT / args.output
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"Saved locale map to {out_path}")

    # ── summary print ──
    print("\n" + "=" * 70)
    print("  GLOBAL STATS  (w-weighted, uncalibrated local indicators)")
    print("=" * 70)
    for mname, st in result["global_stats"].items():
        eig = st.get("EI_g")
        eig_str = f"{eig:.3f}" if eig is not None else "N/A"
        q_str = f"{st['q_mean']:+.3f}" if st["q_mean"] is not None else "N/A"
        v_str = f"{st['v_mean']:.3f}" if st["v_mean"] is not None else "N/A"
        print(f"  {mname:40s}  EI_g={eig_str:>8}  q={q_str:>8}  l={st['l_mean']:.3f}"
              f"  v={v_str:>8}  healthy={st['n_healthy']}/{st['n_total']}")

    print("\n" + "=" * 70)
    print("  SELECTED POINTS FOR DOWNSTREAM TASKS")
    print("=" * 70)
    for cat, items in result["selections"].items():
        print(f"\n  {cat}:")
        for item in items:
            pt = item.get("point") or {}
            coords = pt.get("x_raw", [])
            coord_str = " ".join(f"{HEXACO_DIMS[i]}={c:.2f}" for i, c in enumerate(coords)) if coords else "N/A"
            q_val = pt.get("q")
            q_str = f"{q_val:>8.3f}" if q_val is not None else "     N/A"
            print(f"    {item['region']:25s}  q={q_str}  [{coord_str}]")


if __name__ == "__main__":
    main()
