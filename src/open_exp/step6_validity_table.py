"""
Step 6 — Build the external-validity report linking EI_g to open-task
behavior under the framing of *separability and stability* rather than raw
score prediction.

This step does not call any LLM.  It integrates:

    Geometric metrics (per model)
        EI_g, log_VI, l_mean        from output_<model>/eig_metric_result.json

    Classical baselines (per model)
        MI(x,y), R^2(x->y), R^2(theta->y), CCA1, CCA_mean, RV
                                    from output/baselines_all_models.json

    Behavioral indicators (per model and per (model, task) when applicable)
        Alignment_RAS               GPT-4o single-pass judge mean (1-5)
        Score_full                  GPT-4o (RAS+BSS+RCS)/3 mean
        Stability                   1 - normalised RAS std across seeds
        Separability_pair           pairwise judge accuracy (step5)
        Separability_top1           6-AFC identification top-1 accuracy (step5b)
        Separability_p              6-AFC mean prob assigned to true persona
        Separability_logp           6-AFC mean log prob (continuous, no ceiling)
        Separability_margin         score margin (true vs best other)
        Separability_mrr            mean reciprocal rank of true persona

For correlations we report **both Pearson r and Spearman rho** at two
granularities:

    Coarse  (n=4, model-level):   per-model means
    Fine    (n=36, model-task):   per-(model, task) cells; here baselines
                                  and EI_g are repeated 9 times per model
                                  but Separability is genuinely task-level,
                                  so the correlation is over real variation.

We also bootstrap (per-judgement resampling) the per-model 6-AFC numbers and
report 95% CIs.

Outputs:
    output/open_exp/validity_table.csv
    output/open_exp/validity_table.json
    output/open_exp/figures/fig5_external_validity.png
    output/open_exp/figures/fig6_validity_pearson_bootstrap.png
"""
from __future__ import annotations

import json
import os
import pathlib
import sys
from collections import defaultdict

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

import matplotlib.font_manager as fm
_CJK_FONT = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
if os.path.exists(_CJK_FONT):
    fm.fontManager.addfont(_CJK_FONT)
    mpl.rcParams["font.sans-serif"] = [fm.FontProperties(fname=_CJK_FONT).get_name(),
                                        "DejaVu Sans"]
else:
    mpl.rcParams["font.sans-serif"] = ["DejaVu Sans"]
mpl.rcParams["axes.unicode_minus"] = False

MODEL_ORDER = [
    "qwen2.5-7b",
    "deepseek-r1-distill-qwen-7b",
    "mistral-7b-instruct-v0.3",
    "llama3.1-8b",
    "mixtral-8x7b",
    "llama3.1-70b",
    "qwen3-30b-a3b",
]
MODEL_LABELS = {
    "qwen2.5-7b": "Qwen2.5-7B",
    "deepseek-r1-distill-qwen-7b": "DeepSeek-R1-Distill",
    "mistral-7b-instruct-v0.3": "Mistral-v0.3",
    "llama3.1-8b": "Llama3.1-8B",
    "mixtral-8x7b": "Mixtral-8x7B",
    "llama3.1-70b": "Llama3.1-70B",
    "qwen3-30b-a3b": "Qwen3-30B-A3B",
}


# ──────────────────────────────────────────────────────────────────
#  Loaders
# ──────────────────────────────────────────────────────────────────
def load_global_metrics() -> dict:
    """Per-model geometric + baseline metrics."""
    out = {}
    for m in MODEL_ORDER:
        p = ROOT / f"output_{m}/eig_metric_result.json"
        if not p.exists():
            print(f"[warn] missing {p}")
            out[m] = {"EI_g": float("nan"), "log_VI": float("nan"),
                      "l_mean": float("nan")}
            continue
        d = json.loads(p.read_text(encoding="utf-8"))
        VI = d.get("VI", 0.0)
        out[m] = {
            "EI_g": float(d.get("EI_g", float("nan"))),
            "log_VI": float(np.log(VI)) if VI > 0 else float("nan"),
            "l_mean": float(d.get("l_mean", float("nan"))),
        }
    bp = ROOT / "output/baselines_all_models.json"
    if bp.exists():
        bd = json.loads(bp.read_text(encoding="utf-8"))
        for m in MODEL_ORDER:
            if m in bd:
                out[m].update({
                    "MI": float(bd[m].get("MI_x_y", float("nan"))),
                    "R2_xy": float(bd[m].get("R2_x_to_y", float("nan"))),
                    "R2_theta_y": float(bd[m].get("R2_theta_to_y", float("nan"))),
                    "CCA_first": float(bd[m].get("CCA_first", float("nan"))),
                    "CCA_mean": float(bd[m].get("CCA_mean", float("nan"))),
                    "RV": float(bd[m].get("RV", float("nan"))),
                })
    else:
        print(f"[warn] baselines file missing; run compute_baselines_all_models.py")
        for m in MODEL_ORDER:
            for k in ["MI", "R2_xy", "R2_theta_y", "CCA_first", "CCA_mean", "RV"]:
                out[m][k] = float("nan")
    return out


def load_alignment_and_stability():
    judg_path = ROOT / "output/open_exp/judgments.jsonl"
    rows = [json.loads(l) for l in judg_path.open("r", encoding="utf-8")]
    align_bucket = defaultdict(list)
    seed_bucket = defaultdict(list)
    full_score_bucket = defaultdict(list)
    align_per_mt = defaultdict(list)        # (model, task) -> RAS list
    score_per_mt = defaultdict(list)
    for r in rows:
        m = r["model"]
        t = r["task_id"]
        align_bucket[m].append(r["RAS"])
        full_score_bucket[m].append(r["score"])
        align_per_mt[(m, t)].append(r["RAS"])
        score_per_mt[(m, t)].append(r["score"])
        seed_bucket[(m, r["persona"], t, r["prompt_type"])].append(r["RAS"])

    align = {m: float(np.mean(v)) for m, v in align_bucket.items()}
    full = {m: float(np.mean(v)) for m, v in full_score_bucket.items()}
    align_mt = {k: float(np.mean(v)) for k, v in align_per_mt.items()}
    score_mt = {k: float(np.mean(v)) for k, v in score_per_mt.items()}

    seed_std = defaultdict(list)
    for (m, p, t, pt), vs in seed_bucket.items():
        if len(vs) >= 2:
            seed_std[m].append(float(np.std(vs)))
    stability_raw = {m: float(np.mean(v)) for m, v in seed_std.items()}
    stability = {m: float(1.0 - min(stability_raw[m] / 1.5, 1.0)) for m in stability_raw}
    return ({"alignment_RAS": align, "score_full": full,
             "alignment_RAS_mt": align_mt, "score_full_mt": score_mt},
            {"stability_norm": stability, "stability_raw_std": stability_raw})


def load_separability_pair() -> dict:
    """Pairwise (2-AFC) results from step5."""
    p = ROOT / "output/open_exp/separability.json"
    if not p.exists():
        return {}
    d = json.loads(p.read_text(encoding="utf-8"))
    out_per_model = {m: float(v["separability"]) for m, v in d["per_model"].items()}
    out_per_mt = {}
    for k, v in d.get("per_model_task", {}).items():
        m, t = k.split("|", 1)
        out_per_mt[(m, t)] = float(v["separability"])
    return {"per_model": out_per_model, "per_model_task": out_per_mt}


def load_separability_nway() -> dict:
    """6-AFC identification results from step5b."""
    p = ROOT / "output/open_exp/identification.json"
    if not p.exists():
        return {}
    d = json.loads(p.read_text(encoding="utf-8"))
    pm = d["per_model"]
    out_per_model = {
        m: {
            "top1": float(v["top1_acc"]),
            "p_correct": float(v["p_correct_mean"]),
            "log_p": float(v["log_p_correct_mean"]),
            "margin": float(v["margin_mean"]),
            "mrr": float(v["mrr"]),
        }
        for m, v in pm.items()
    }
    out_per_mt = {}
    for k, v in d.get("per_model_task", {}).items():
        m, t = k.split("|", 1)
        out_per_mt[(m, t)] = {
            "top1": float(v["top1_acc"]),
            "p_correct": float(v["p_correct_mean"]),
            "log_p": float(v["log_p_correct_mean"]),
            "margin": float(v["margin_mean"]),
            "mrr": float(v["mrr"]),
        }
    return {"per_model": out_per_model, "per_model_task": out_per_mt}


def load_id_rows() -> list[dict]:
    """Raw 6-AFC rows for bootstrapping."""
    p = ROOT / "output/open_exp/identification.jsonl"
    if not p.exists():
        return []
    return [json.loads(l) for l in p.open("r", encoding="utf-8")]


# ──────────────────────────────────────────────────────────────────
#  Build table + correlations (Pearson and Spearman, two N's)
# ──────────────────────────────────────────────────────────────────
def build_table(global_metrics, align_dict, stab_dict,
                sep_pair_dict, sep_nway_dict) -> list[dict]:
    align = align_dict["alignment_RAS"]
    score_full = align_dict["score_full"]
    stab = stab_dict["stability_norm"]
    stab_raw = stab_dict["stability_raw_std"]
    sep_p_pm = sep_pair_dict.get("per_model", {})
    sep_n_pm = sep_nway_dict.get("per_model", {})
    rows = []
    for m in MODEL_ORDER:
        g = global_metrics.get(m, {})
        sep_p = sep_p_pm.get(m, float("nan"))
        sn = sep_n_pm.get(m, {})
        rows.append({
            "model": MODEL_LABELS[m],
            "alias": m,
            "EI_g": g.get("EI_g", float("nan")),
            "log_VI": g.get("log_VI", float("nan")),
            "l_mean": g.get("l_mean", float("nan")),
            "MI_xy": g.get("MI", float("nan")),
            "R2_xy": g.get("R2_xy", float("nan")),
            "R2_theta_y": g.get("R2_theta_y", float("nan")),
            "CCA_first": g.get("CCA_first", float("nan")),
            "CCA_mean": g.get("CCA_mean", float("nan")),
            "RV": g.get("RV", float("nan")),
            "Alignment_RAS": align.get(m, float("nan")),
            "Score_full": score_full.get(m, float("nan")),
            "Stability": stab.get(m, float("nan")),
            "Stability_raw_std": stab_raw.get(m, float("nan")),
            "Sep_pair": float(sep_p) if sep_p == sep_p else float("nan"),
            "Sep_top1": float(sn.get("top1", float("nan"))),
            "Sep_p": float(sn.get("p_correct", float("nan"))),
            "Sep_logp": float(sn.get("log_p", float("nan"))),
            "Sep_margin": float(sn.get("margin", float("nan"))),
            "Sep_mrr": float(sn.get("mrr", float("nan"))),
        })
    return rows


def pearson(xs, ys):
    xs = np.asarray(xs, float); ys = np.asarray(ys, float)
    mask = np.isfinite(xs) & np.isfinite(ys)
    if mask.sum() < 2:
        return float("nan"), int(mask.sum())
    from scipy.stats import pearsonr
    r, _ = pearsonr(xs[mask], ys[mask])
    return float(r), int(mask.sum())


def spearman(xs, ys):
    xs = np.asarray(xs, float); ys = np.asarray(ys, float)
    mask = np.isfinite(xs) & np.isfinite(ys)
    if mask.sum() < 2:
        return float("nan"), int(mask.sum())
    from scipy.stats import spearmanr
    rho, _ = spearmanr(xs[mask], ys[mask])
    return float(rho), int(mask.sum())


CORR_X_COLS = ["EI_g", "log_VI", "l_mean", "MI_xy",
               "R2_xy", "R2_theta_y", "CCA_first", "CCA_mean", "RV"]
CORR_Y_COLS = ["Sep_logp", "Sep_p", "Sep_top1", "Sep_mrr",
               "Sep_pair", "Alignment_RAS", "Stability", "Score_full"]


def compute_correlations_coarse(rows) -> dict:
    """Per-model correlations.  n=4."""
    out = {}
    for cx in CORR_X_COLS:
        out[cx] = {}
        for cy in CORR_Y_COLS:
            xs = [r[cx] for r in rows]
            ys = [r[cy] for r in rows]
            r, n = pearson(xs, ys)
            rho, _ = spearman(xs, ys)
            out[cx][cy] = {"pearson_r": r, "spearman_rho": rho, "n": n}
    return out


def compute_correlations_fine(global_metrics, align_dict,
                              sep_pair_dict, sep_nway_dict) -> dict:
    """Per-(model, task) correlations.  n up to 36 (4 models * 9 tasks).

    Geometric / baseline metrics are repeated 9 times per model (since they
    are model-level scalars).  This is a mixed-effects design; the resulting
    correlation is over the joint cell variation.  We report Pearson r and
    Spearman rho.
    """
    align_mt = align_dict["alignment_RAS_mt"]
    score_mt = align_dict["score_full_mt"]
    sep_pair_mt = sep_pair_dict.get("per_model_task", {})
    sep_nway_mt = sep_nway_dict.get("per_model_task", {})
    cells = []
    for m in MODEL_ORDER:
        g = global_metrics.get(m, {})
        for t in sorted({tt for (mm, tt) in align_mt.keys() if mm == m}):
            sn = sep_nway_mt.get((m, t), {})
            cells.append({
                "model": m, "task": t,
                "EI_g": g.get("EI_g", float("nan")),
                "log_VI": g.get("log_VI", float("nan")),
                "l_mean": g.get("l_mean", float("nan")),
                "MI_xy": g.get("MI", float("nan")),
                "R2_xy": g.get("R2_xy", float("nan")),
                "R2_theta_y": g.get("R2_theta_y", float("nan")),
                "CCA_first": g.get("CCA_first", float("nan")),
                "CCA_mean": g.get("CCA_mean", float("nan")),
                "RV": g.get("RV", float("nan")),
                "Sep_pair": float(sep_pair_mt.get((m, t), float("nan"))),
                "Sep_top1": float(sn.get("top1", float("nan"))),
                "Sep_p": float(sn.get("p_correct", float("nan"))),
                "Sep_logp": float(sn.get("log_p", float("nan"))),
                "Sep_mrr": float(sn.get("mrr", float("nan"))),
                "Alignment_RAS": float(align_mt.get((m, t), float("nan"))),
                "Score_full": float(score_mt.get((m, t), float("nan"))),
                "Stability": float("nan"),  # not defined per task here
            })
    out = {}
    for cx in CORR_X_COLS:
        out[cx] = {}
        for cy in CORR_Y_COLS:
            xs = [c[cx] for c in cells]
            ys = [c[cy] for c in cells]
            r, n = pearson(xs, ys)
            rho, _ = spearman(xs, ys)
            out[cx][cy] = {"pearson_r": r, "spearman_rho": rho, "n": n}
    return out, cells


# ──────────────────────────────────────────────────────────────────
#  Bootstrap separability per model
# ──────────────────────────────────────────────────────────────────
def bootstrap_nway(id_rows, B: int = 1000, seed: int = 42) -> dict:
    """Bootstrap per-model nway means (top1, p_correct, log_p, mrr) and
    return (point, lo, hi) for each.  Uses per-judgement resampling.
    """
    rng = np.random.default_rng(seed)
    by_model = defaultdict(list)
    for r in id_rows:
        by_model[r["model"]].append(r)
    results = {}
    for m, rows in by_model.items():
        n = len(rows)
        if n == 0:
            continue
        top1 = np.array([r["top1_correct"] for r in rows], dtype=float)
        pcor = np.array([r["p_correct"] for r in rows], dtype=float)
        logp = np.array([r["log_p_correct"] for r in rows], dtype=float)
        rrk = np.array([1.0 / r["true_rank"] for r in rows], dtype=float)
        boot = {"top1": [], "p": [], "logp": [], "mrr": []}
        for _ in range(B):
            idx = rng.integers(0, n, size=n)
            boot["top1"].append(top1[idx].mean())
            boot["p"].append(pcor[idx].mean())
            boot["logp"].append(logp[idx].mean())
            boot["mrr"].append(rrk[idx].mean())
        results[m] = {
            "top1": (float(top1.mean()),
                     float(np.percentile(boot["top1"], 2.5)),
                     float(np.percentile(boot["top1"], 97.5))),
            "p": (float(pcor.mean()),
                  float(np.percentile(boot["p"], 2.5)),
                  float(np.percentile(boot["p"], 97.5))),
            "logp": (float(logp.mean()),
                     float(np.percentile(boot["logp"], 2.5)),
                     float(np.percentile(boot["logp"], 97.5))),
            "mrr": (float(rrk.mean()),
                    float(np.percentile(boot["mrr"], 2.5)),
                    float(np.percentile(boot["mrr"], 97.5))),
        }
    return results


def bootstrap_correlations(rows, id_rows, B: int = 1000, seed: int = 42) -> dict:
    """Bootstrap Pearson/Spearman of EI_g vs Sep_logp/Sep_top1 by resampling
    the per-judgement nway data per model.  Geometric metrics stay fixed.
    """
    rng = np.random.default_rng(seed)
    by_model = defaultdict(list)
    for r in id_rows:
        by_model[r["model"]].append(r)
    out = {}
    for cy_field, key in [("log_p_correct", "logp"), ("top1_correct", "top1"),
                           ("p_correct", "p")]:
        for cx in ["EI_g", "log_VI", "MI_xy", "R2_xy", "CCA_mean", "RV"]:
            r_boot, rho_boot = [], []
            for _ in range(B):
                xs, ys = [], []
                for row in rows:
                    m = row["alias"]
                    samples = by_model.get(m, [])
                    if not samples:
                        continue
                    idx = rng.integers(0, len(samples), size=len(samples))
                    arr = np.array([samples[i][cy_field] for i in idx], dtype=float)
                    ys.append(float(arr.mean()))
                    xs.append(float(row[cx]))
                xs = np.asarray(xs); ys = np.asarray(ys)
                mask = np.isfinite(xs) & np.isfinite(ys)
                if mask.sum() < 3:
                    continue
                from scipy.stats import pearsonr, spearmanr
                r, _ = pearsonr(xs[mask], ys[mask])
                rho, _ = spearmanr(xs[mask], ys[mask])
                r_boot.append(r); rho_boot.append(rho)
            if r_boot:
                out[f"{cx}_vs_Sep_{key}"] = {
                    "pearson_mean": float(np.mean(r_boot)),
                    "pearson_lo": float(np.percentile(r_boot, 2.5)),
                    "pearson_hi": float(np.percentile(r_boot, 97.5)),
                    "spearman_mean": float(np.mean(rho_boot)),
                    "spearman_lo": float(np.percentile(rho_boot, 2.5)),
                    "spearman_hi": float(np.percentile(rho_boot, 97.5)),
                    "n_boot": len(r_boot),
                }
    return out


# ──────────────────────────────────────────────────────────────────
#  Writers
# ──────────────────────────────────────────────────────────────────
COL_ORDER = [
    "model", "EI_g", "log_VI", "l_mean",
    "MI_xy", "R2_xy", "R2_theta_y", "CCA_first", "CCA_mean", "RV",
    "Alignment_RAS", "Sep_pair", "Sep_top1", "Sep_p", "Sep_logp",
    "Sep_margin", "Sep_mrr", "Stability", "Score_full",
]


def write_csv(rows, path: pathlib.Path) -> None:
    import csv
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COL_ORDER, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{v:.4f}" if isinstance(v, float) else v)
                        for k, v in r.items() if k in COL_ORDER})


def print_table(rows) -> None:
    print("\n" + "=" * 165)
    print(f"{'Model':22s} {'EI_g':>7s} {'logVI':>7s} {'lmean':>7s} "
          f"{'MI':>6s} {'R2xy':>6s} {'CCA1':>6s} {'CCAm':>6s} {'RV':>6s}  "
          f"{'Pair':>6s} {'Top1':>6s} {'p_cor':>6s} {'logp':>7s} {'mrr':>6s}  "
          f"{'Align':>6s} {'Score':>6s} {'Stab':>6s}")
    print("-" * 165)
    for r in rows:
        print(
            f"{r['model']:22s} "
            f"{r['EI_g']:7.3f} {r['log_VI']:7.3f} {r['l_mean']:7.3f} "
            f"{r['MI_xy']:6.3f} {r['R2_xy']:6.3f} {r['CCA_first']:6.3f} {r['CCA_mean']:6.3f} {r['RV']:6.3f}  "
            f"{r['Sep_pair']:6.3f} {r['Sep_top1']:6.3f} {r['Sep_p']:6.3f} {r['Sep_logp']:+7.3f} {r['Sep_mrr']:6.3f}  "
            f"{r['Alignment_RAS']:6.3f} {r['Score_full']:6.3f} {r['Stability']:6.3f}"
        )
    print("=" * 165)


def print_correlations(corr, label: str) -> None:
    print(f"\n=== {label} ===")
    print(f"{'metric':12s}" + "".join(f"{c:>20s}" for c in CORR_Y_COLS))
    for cx in CORR_X_COLS:
        cells = []
        for cy in CORR_Y_COLS:
            r = corr[cx][cy]["pearson_r"]
            rho = corr[cx][cy]["spearman_rho"]
            if np.isfinite(r):
                cells.append(f"r={r:+.3f}/ρ={rho:+.3f}".rjust(20))
            else:
                cells.append("NaN".rjust(20))
        print(f"{cx:12s}" + "".join(cells))


def print_bootstrap_corrs(corrs_boot: dict) -> None:
    print("\n=== Bootstrap CIs of EI_g/baselines vs Sep_logp ===")
    print(f"{'pair':28s}  {'pearson r (95% CI)':>26s}  {'spearman ρ (95% CI)':>26s}")
    for k, v in corrs_boot.items():
        if "Sep_logp" not in k: continue
        pe = f"{v['pearson_mean']:+.3f} [{v['pearson_lo']:+.3f}, {v['pearson_hi']:+.3f}]"
        sp = f"{v['spearman_mean']:+.3f} [{v['spearman_lo']:+.3f}, {v['spearman_hi']:+.3f}]"
        print(f"{k:28s}  {pe:>26s}  {sp:>26s}")


# ──────────────────────────────────────────────────────────────────
#  Figures
# ──────────────────────────────────────────────────────────────────
def fig_main(rows, corr_coarse, corr_fine, fine_cells, out_path: pathlib.Path) -> None:
    fig = plt.figure(figsize=(15, 4.6))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.1, 1.1, 1.6])

    eig = np.array([r["EI_g"] for r in rows])
    sep_logp = np.array([r["Sep_logp"] for r in rows])
    sep_top1 = np.array([r["Sep_top1"] for r in rows])
    align = np.array([r["Alignment_RAS"] for r in rows])
    labels = [r["model"] for r in rows]
    colors = ["#d62728", "#1f77b4", "#2ca02c", "#9467bd",
              "#ff7f0e", "#8c564b", "#e377c2"]

    # (a) EI_g vs Sep_logp (continuous, the headline metric)
    ax = fig.add_subplot(gs[0, 0])
    for x, y, c, lbl in zip(eig, sep_logp, colors, labels):
        if np.isfinite(x) and np.isfinite(y):
            ax.scatter(x, y, s=160, color=c, edgecolor="k", linewidth=0.8)
            ax.annotate(lbl, (x, y), xytext=(6, 6), textcoords="offset points",
                        fontsize=8)
    r_p = corr_coarse["EI_g"]["Sep_logp"]["pearson_r"]
    rho = corr_coarse["EI_g"]["Sep_logp"]["spearman_rho"]
    ax.set_xlabel(r"$EI_g$")
    ax.set_ylabel("Sep_logp  (mean log P[true persona])")
    ax.set_title(f"(a) $EI_g$ vs Sep_logp  (n=4)\nPearson r={r_p:+.3f}, "
                 f"Spearman ρ={rho:+.3f}")
    ax.axhline(np.log(1.0/6), color="gray", linestyle="--", linewidth=0.8,
               alpha=0.7)
    ax.text(0.02, 0.04, "chance", color="gray", transform=ax.transAxes,
            fontsize=8)
    ax.grid(alpha=0.3)

    # (b) per-(model,task) cells: EI_g vs Sep_logp, n=36, Pearson is now continuous
    ax = fig.add_subplot(gs[0, 1])
    xs = np.array([c["EI_g"] for c in fine_cells])
    ys = np.array([c["Sep_logp"] for c in fine_cells])
    cs = np.array([colors[MODEL_ORDER.index(c["model"])] if c["model"] in MODEL_ORDER else "#999999"
                   for c in fine_cells])
    ax.scatter(xs, ys, c=cs, s=40, edgecolor="k", linewidth=0.4, alpha=0.85)
    r_fine = corr_fine["EI_g"]["Sep_logp"]["pearson_r"]
    rho_fine = corr_fine["EI_g"]["Sep_logp"]["spearman_rho"]
    n_fine = corr_fine["EI_g"]["Sep_logp"]["n"]
    if np.isfinite(r_fine):
        z = np.polyfit(xs[np.isfinite(xs) & np.isfinite(ys)],
                       ys[np.isfinite(xs) & np.isfinite(ys)], 1)
        x_grid = np.linspace(np.nanmin(xs), np.nanmax(xs), 50)
        ax.plot(x_grid, np.poly1d(z)(x_grid), "k--", lw=1, alpha=0.6)
    ax.set_xlabel(r"$EI_g$  (model-level, repeated per task)")
    ax.set_ylabel("Sep_logp  (per (model, task) cell)")
    ax.set_title(f"(b) Cell-level $EI_g$ vs Sep_logp  (n={n_fine})\n"
                 f"Pearson r={r_fine:+.3f}, Spearman ρ={rho_fine:+.3f}")
    ax.axhline(np.log(1.0/6), color="gray", linestyle="--", linewidth=0.8,
               alpha=0.7)
    handles = [plt.Line2D([0], [0], marker="o", color="w",
                          markerfacecolor=colors[i], markeredgecolor="k",
                          markersize=8, label=MODEL_LABELS[m])
               for i, m in enumerate(MODEL_ORDER)]
    ax.legend(handles=handles, fontsize=7, loc="lower right")
    ax.grid(alpha=0.3)

    # (c) Pearson r heatmap: x metrics vs y metrics
    ax = fig.add_subplot(gs[0, 2])
    M = np.array([[corr_coarse[cx][cy]["pearson_r"] for cy in CORR_Y_COLS]
                  for cx in CORR_X_COLS])
    im = ax.imshow(M, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    for i in range(len(CORR_X_COLS)):
        for j in range(len(CORR_Y_COLS)):
            v = M[i, j]
            if np.isfinite(v):
                ax.text(j, i, f"{v:+.2f}", ha="center", va="center",
                        color="white" if abs(v) > 0.55 else "black", fontsize=7.5)
    ax.set_xticks(range(len(CORR_Y_COLS)))
    ax.set_xticklabels(CORR_Y_COLS, rotation=25, ha="right", fontsize=7.5)
    ax.set_yticks(range(len(CORR_X_COLS)))
    ax.set_yticklabels(CORR_X_COLS, fontsize=8)
    ax.set_title("(c) Pearson r matrix  (n=4 model-level)")
    plt.colorbar(im, ax=ax, fraction=0.04, pad=0.02, label="Pearson r")

    plt.suptitle("External validity: EI$_g$ and open-task separability / alignment  (6-AFC + pairwise)",
                 y=1.03, fontsize=12)
    plt.tight_layout()
    plt.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close()
    print(f"Saved figure -> {out_path}")


def fig_bootstrap(boot_per_model, boot_corrs, out_path: pathlib.Path) -> None:
    """Two-panel: (a) per-model Sep_logp with bootstrap CIs;
    (b) bootstrap distributions of EI_g vs Sep_logp Pearson r vs baselines.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    # (a)
    ax = axes[0]
    models_sorted = sorted(boot_per_model.keys(),
                           key=lambda m: -boot_per_model[m]["logp"][0])
    xs = np.arange(len(models_sorted))
    means = [boot_per_model[m]["logp"][0] for m in models_sorted]
    los = [boot_per_model[m]["logp"][1] for m in models_sorted]
    his = [boot_per_model[m]["logp"][2] for m in models_sorted]
    yerr = np.array([[m - l for m, l in zip(means, los)],
                     [h - m for m, h in zip(means, his)]])
    colors = ["#d62728", "#1f77b4", "#2ca02c", "#9467bd",
              "#ff7f0e", "#8c564b", "#e377c2"]
    bar_colors = [colors[MODEL_ORDER.index(m)] if m in MODEL_ORDER else "#999999"
                  for m in models_sorted]
    ax.bar(xs, means, yerr=yerr, capsize=5, color=bar_colors, edgecolor="k",
           linewidth=0.6, alpha=0.85)
    ax.set_xticks(xs)
    ax.set_xticklabels([MODEL_LABELS[m] for m in models_sorted],
                       rotation=15, ha="right")
    ax.set_ylabel("Sep_logp  (mean log P[true persona])")
    ax.axhline(np.log(1.0/6), color="gray", linestyle="--", linewidth=0.8)
    ax.text(0.02, 0.95, "chance log(1/6)", color="gray",
            transform=ax.transAxes, va="top", fontsize=8)
    ax.set_title("(a) 6-AFC Separability per model with 95% bootstrap CI")
    ax.grid(alpha=0.3, axis="y")

    # (b) bootstrap of Pearson r: EI_g vs Sep_logp + 5 baselines
    ax = axes[1]
    keys = ["EI_g_vs_Sep_logp", "log_VI_vs_Sep_logp",
            "MI_xy_vs_Sep_logp", "R2_xy_vs_Sep_logp",
            "CCA_mean_vs_Sep_logp", "RV_vs_Sep_logp"]
    keys = [k for k in keys if k in boot_corrs]
    labels = [k.replace("_vs_Sep_logp", "") for k in keys]
    means = [boot_corrs[k]["pearson_mean"] for k in keys]
    los = [boot_corrs[k]["pearson_lo"] for k in keys]
    his = [boot_corrs[k]["pearson_hi"] for k in keys]
    ys = np.arange(len(keys))
    xerr = np.array([[m - l for m, l in zip(means, los)],
                     [h - m for m, h in zip(means, his)]])
    cmap = ["#d62728" if "EI_g" in k or "log_VI" in k else "#888888" for k in keys]
    ax.errorbar(means, ys, xerr=xerr, fmt="o", color="k", ecolor="gray",
                capsize=4, markersize=8, markerfacecolor="white", linewidth=1.5)
    for y, m_, c in zip(ys, means, cmap):
        ax.scatter([m_], [y], s=120, color=c, edgecolor="k", zorder=5)
    ax.axvline(0, color="gray", linewidth=0.6)
    ax.set_yticks(ys)
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("Pearson r vs Sep_logp  (bootstrap mean ± 95% CI)")
    ax.set_title("(b) EI$_g$ versus baselines for predicting Sep_logp")
    ax.grid(alpha=0.3, axis="x")
    ax.set_xlim(-1.1, 1.1)

    plt.tight_layout()
    plt.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close()
    print(f"Saved figure -> {out_path}")


# ──────────────────────────────────────────────────────────────────
#  Main
# ──────────────────────────────────────────────────────────────────
def main():
    global_metrics = load_global_metrics()
    align_dict, stab_dict = load_alignment_and_stability()
    sep_pair_dict = load_separability_pair()
    sep_nway_dict = load_separability_nway()
    id_rows = load_id_rows()

    rows = build_table(global_metrics, align_dict, stab_dict,
                        sep_pair_dict, sep_nway_dict)
    corr_coarse = compute_correlations_coarse(rows)
    corr_fine, fine_cells = compute_correlations_fine(
        global_metrics, align_dict, sep_pair_dict, sep_nway_dict)

    boot_pm = bootstrap_nway(id_rows, B=1000)
    boot_corrs = bootstrap_correlations(rows, id_rows, B=1000)

    out_dir = ROOT / "output/open_exp"
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    write_csv(rows, out_dir / "validity_table.csv")
    (out_dir / "validity_table.json").write_text(
        json.dumps({"rows": rows,
                    "correlations_coarse": corr_coarse,
                    "correlations_fine": corr_fine,
                    "fine_cells": fine_cells,
                    "bootstrap_per_model": boot_pm,
                    "bootstrap_correlations": boot_corrs},
                   ensure_ascii=False, indent=2),
        encoding="utf-8")

    print_table(rows)
    print_correlations(corr_coarse, "Per-model correlations  (n=4, Pearson r / Spearman ρ)")
    if fine_cells:
        print_correlations(corr_fine,
                            f"Per-(model, task) correlations  (n={corr_fine['EI_g']['Sep_logp']['n']}, Pearson r / Spearman ρ)")
    if boot_corrs:
        print_bootstrap_corrs(boot_corrs)

    fig_main(rows, corr_coarse, corr_fine, fine_cells,
             fig_dir / "fig5_external_validity.png")
    if boot_pm and boot_corrs:
        fig_bootstrap(boot_pm, boot_corrs,
                      fig_dir / "fig6_validity_pearson_bootstrap.png")
    print(f"\nCSV  -> {out_dir / 'validity_table.csv'}")
    print(f"JSON -> {out_dir / 'validity_table.json'}")


if __name__ == "__main__":
    main()
