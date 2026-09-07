"""
Step 4 — Aggregate scores, link to q / EI_g / MI, produce 4 figures.

Inputs:
    output/open_exp/judgments.jsonl
    output/persona_locale_map.json     (provides q per (model, persona))
    output_<model>/eig_per_point.npz   (provides per-point l, h, q)
    output/eig_results.json (or per-model)  (global EI_g, MI, l_mean)

Outputs:
    output/open_exp/scores.json
    output/open_exp/figures/fig1_behavior_heatmap.png
    output/open_exp/figures/fig2_q_vs_score.png
    output/open_exp/figures/fig3_model_level.png
    output/open_exp/figures/fig4_neighbor_discrimination.png
"""
from __future__ import annotations

import json
import os
import pathlib
import sys
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

# Register CJK font explicitly via file path
import matplotlib.font_manager as fm
_CJK_FONT = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
if os.path.exists(_CJK_FONT):
    fm.fontManager.addfont(_CJK_FONT)
    _cjk_name = fm.FontProperties(fname=_CJK_FONT).get_name()
    mpl.rcParams["font.sans-serif"] = [_cjk_name, "DejaVu Sans"]
else:
    mpl.rcParams["font.sans-serif"] = ["DejaVu Sans"]
mpl.rcParams["axes.unicode_minus"] = False

MODEL_ORDER = ["qwen2.5-7b", "deepseek-r1-distill-qwen-7b",
               "mistral-7b-instruct-v0.3", "llama3.1-8b"]
MODEL_LABELS = {
    "qwen2.5-7b": "Qwen2.5",
    "deepseek-r1-distill-qwen-7b": "DeepSeek-R1-Qwen",
    "mistral-7b-instruct-v0.3": "Mistral-v0.3",
    "llama3.1-8b": "Llama3.1",
}


def load_data():
    judgments = [json.loads(l) for l in (ROOT / "output/open_exp/judgments.jsonl").open("r", encoding="utf-8")]
    personas = json.loads((ROOT / "data/open_exp/personas.json").read_text(encoding="utf-8"))["personas"]
    persona_meta = {p["id"]: p for p in personas}
    return judgments, persona_meta, personas


def aggregate_scores(judgments):
    """Return dict[(model, persona)] -> {RAS, BSS, RCS, score, n, std}."""
    bucket = defaultdict(list)
    for j in judgments:
        bucket[(j["model"], j["persona"])].append(j["score"])
    out = {}
    for k, vals in bucket.items():
        arr = np.array(vals)
        out[k] = {"score_mean": float(arr.mean()),
                  "score_std": float(arr.std()),
                  "n": len(arr)}
    return out


def aggregate_by_prompt_seed(judgments):
    """Return dict[(model, persona, prompt_type)] -> seed-mean & dict[(m,p)] -> seed_var."""
    by_pt = defaultdict(list)
    for j in judgments:
        by_pt[(j["model"], j["persona"], j["prompt_type"])].append(j["score"])
    pt_mean = {k: float(np.mean(v)) for k, v in by_pt.items()}
    # seed var = within-(m,p,pt) std averaged across pts; prompt-stab = |A - B|
    seed_var = defaultdict(list)
    by_seed = defaultdict(list)
    for j in judgments:
        by_seed[(j["model"], j["persona"], j["prompt_type"], j["task_id"])].append(j["score"])
    for k, v in by_seed.items():
        if len(v) >= 2:
            seed_var[(k[0], k[1])].append(float(np.std(v)))
    seed_var_mean = {k: float(np.mean(v)) for k, v in seed_var.items()}
    return pt_mean, seed_var_mean


def load_q_per_persona(personas):
    """Return dict[(model, persona)] -> q,l,v,EI_g,MI."""
    out = {}
    for m in MODEL_ORDER:
        npz_path = ROOT / f"output_{m}/eig_per_point.npz"
        if not npz_path.exists():
            print(f"[warn] missing {npz_path}")
            continue
        d = dict(np.load(npz_path))
        h = d["h"]; l = d["l"]
        det_h = np.array([np.linalg.det(h[i]) for i in range(h.shape[0])])
        v_arr = 0.5 * np.log(np.maximum(det_h, 1e-30))
        v_arr[det_h <= 0] = np.nan
        for p in personas:
            gid = p["global_idx"]
            out[(m, p["id"])] = {
                "q": float(v_arr[gid] - l[gid]),
                "l": float(l[gid]),
                "v": float(v_arr[gid]),
            }
    return out


def load_global_metrics():
    """Return dict[model] -> {EI_g, MI, l_mean, log_VI}.

    EI_g, l_mean, VI come from `output_<model>/eig_metric_result.json`.
    MI is read from `output/exp3_discriminant_validity.json` when available
    (keyed by display name); falls back to NaN otherwise.
    """
    out = {}
    for m in MODEL_ORDER:
        cand = ROOT / f"output_{m}/eig_metric_result.json"
        if cand.exists():
            d = json.loads(cand.read_text(encoding="utf-8"))
            out[m] = {
                "EI_g": float(d.get("EI_g", float("nan"))),
                "MI": float("nan"),
                "l_mean": float(d.get("l_mean", float("nan"))),
                "log_VI": float(np.log(d["VI"])) if d.get("VI", 0) > 0 else float("nan"),
            }
    # patch in MI from exp3 when available
    exp3 = ROOT / "output/exp3_discriminant_validity.json"
    if exp3.exists():
        try:
            d3 = json.loads(exp3.read_text(encoding="utf-8"))
            name_map = {"Qwen2.5-7B": "qwen2.5-7b", "DeepSeek-R1-Distill": "deepseek-r1-distill-qwen-7b"}
            for disp, alias in name_map.items():
                if disp in d3 and alias in out:
                    out[alias]["MI"] = float(d3[disp]["original"].get("MI_x_y", float("nan")))
        except Exception:
            pass
    return out


# ════════════════ Figures ════════════════

def fig1_behavior_heatmap(scores, personas, out):
    pids = [p["id"] for p in personas]
    plabels = [p["region"] for p in personas]
    M = np.full((len(MODEL_ORDER), len(pids)), np.nan)
    for i, m in enumerate(MODEL_ORDER):
        for j, pid in enumerate(pids):
            if (m, pid) in scores:
                M[i, j] = scores[(m, pid)]["score_mean"]
    fig, ax = plt.subplots(figsize=(11, 4.5))
    im = ax.imshow(M, aspect="auto", cmap="YlGnBu", vmin=1, vmax=5)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            v = M[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                        color="white" if v > 3.0 else "black", fontsize=9)
    ax.set_xticks(range(len(pids)))
    ax.set_xticklabels(plabels, rotation=20, ha="right", fontsize=8)
    ax.set_yticks(range(len(MODEL_ORDER)))
    ax.set_yticklabels([MODEL_LABELS[m] for m in MODEL_ORDER])
    ax.set_title("Fig. 1: Behavioral score heatmap  Model × Persona")
    plt.colorbar(im, ax=ax, label="Mean score (1–5)")
    plt.tight_layout()
    plt.savefig(out, dpi=140, bbox_inches="tight")
    plt.close()


def fig2_q_vs_score(scores, q_data, personas, out):
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = {m: c for m, c in zip(MODEL_ORDER, ["#d62728", "#1f77b4", "#2ca02c", "#9467bd"])}
    markers = {p["id"]: m for p, m in zip(personas, ["o", "s", "^", "D", "v", "P"])}
    xs, ys = [], []
    for m in MODEL_ORDER:
        for p in personas:
            k = (m, p["id"])
            if k in scores and k in q_data:
                x = q_data[k]["q"]
                y = scores[k]["score_mean"]
                if np.isnan(x) or np.isnan(y): continue
                ax.scatter(x, y, s=80, c=colors[m], marker=markers[p["id"]],
                           edgecolor="k", linewidth=0.5, alpha=0.85)
                xs.append(x); ys.append(y)
    if xs:
        from scipy.stats import spearmanr, pearsonr
        rho, prho = spearmanr(xs, ys)
        r, pr = pearsonr(xs, ys)
        ax.set_title(f"Fig. 2: q vs Score  (Spearman rho={rho:.3f}, p={prho:.3g} | Pearson r={r:.3f})")
        z = np.polyfit(xs, ys, 1)
        xx = np.linspace(min(xs), max(xs), 50)
        ax.plot(xx, np.poly1d(z)(xx), "k--", alpha=0.4, lw=1)
    ax.set_xlabel("Local q  (= v - l)")
    ax.set_ylabel("Behavior Score (1-5, mean)")
    h_models = [plt.Line2D([0],[0], marker="o", color="w", markerfacecolor=colors[m],
                            markeredgecolor="k", markersize=10, label=MODEL_LABELS[m])
                for m in MODEL_ORDER]
    h_personas = [plt.Line2D([0],[0], marker=markers[p["id"]], color="w", markerfacecolor="gray",
                              markeredgecolor="k", markersize=10, label=p["region"])
                  for p in personas]
    leg1 = ax.legend(handles=h_models, loc="upper left", fontsize=8, title="Model")
    ax.add_artist(leg1)
    ax.legend(handles=h_personas, loc="lower right", fontsize=8, title="Persona")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out, dpi=140, bbox_inches="tight")
    plt.close()


def fig3_model_level(scores, q_data, global_metrics, out):
    score_mean = {m: np.mean([scores[(m,pid)]["score_mean"]
                              for (mm,pid) in scores if mm == m]) for m in MODEL_ORDER}
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for ax, key, label in zip(axes, ["EI_g", "MI"], ["EI_g (geometric)", "MI"]):
        xs = [global_metrics.get(m,{}).get(key, np.nan) for m in MODEL_ORDER]
        ys = [score_mean[m] for m in MODEL_ORDER]
        for m, x, y in zip(MODEL_ORDER, xs, ys):
            if np.isnan(x) or np.isnan(y): continue
            ax.scatter(x, y, s=180, label=MODEL_LABELS[m])
            ax.annotate(MODEL_LABELS[m], (x, y), xytext=(5, 5),
                        textcoords="offset points", fontsize=9)
        ax.set_xlabel(label)
        ax.set_ylabel("Mean Score (1-5)")
        if all(not np.isnan(x) for x in xs):
            from scipy.stats import spearmanr
            rho, _ = spearmanr(xs, ys)
            ax.set_title(f"{label} vs Score  (ρ={rho:.3f})")
        else:
            ax.set_title(f"{label} vs Score")
        ax.grid(alpha=0.3)
    plt.suptitle("Fig. 3: Model-level consistency  EI_g vs MI vs behavior")
    plt.tight_layout()
    plt.savefig(out, dpi=140, bbox_inches="tight")
    plt.close()


def fig4_neighbor(scores, personas, out):
    near_ids = [p["id"] for p in personas if p["category"] == "B_boundary"]
    if not near_ids:
        return
    M = np.full((len(MODEL_ORDER), len(near_ids)), np.nan)
    for i, m in enumerate(MODEL_ORDER):
        for j, pid in enumerate(near_ids):
            if (m, pid) in scores:
                M[i, j] = scores[(m, pid)]["score_mean"]
    spread = np.nanmax(M, axis=1) - np.nanmin(M, axis=1)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    ax = axes[0]
    x = np.arange(len(near_ids))
    width = 0.18
    for i, m in enumerate(MODEL_ORDER):
        ax.bar(x + (i - 1.5)*width, M[i], width, label=MODEL_LABELS[m])
    ax.set_xticks(x)
    ax.set_xticklabels([next(p["region"] for p in personas if p["id"]==pid) for pid in near_ids])
    ax.set_ylabel("Score")
    ax.set_title("Persona alignment on neighboring points")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, axis="y")

    ax = axes[1]
    ax.bar([MODEL_LABELS[m] for m in MODEL_ORDER], spread,
           color=["#d62728","#1f77b4","#2ca02c","#9467bd"])
    ax.set_ylabel("Score spread (max − min) on B-boundary")
    ax.set_title("Neighbor discrimination")
    for i, v in enumerate(spread):
        if not np.isnan(v):
            ax.text(i, v + 0.02, f"{v:.2f}", ha="center", fontsize=10)
    plt.suptitle("Fig. 4: Fine-grained persona boundaries - neighbor discrimination")
    plt.tight_layout()
    plt.savefig(out, dpi=140, bbox_inches="tight")
    plt.close()


def main():
    judgments, persona_meta, personas = load_data()
    print(f"Loaded {len(judgments)} judgments")

    scores = aggregate_scores(judgments)
    pt_mean, seed_var = aggregate_by_prompt_seed(judgments)
    q_data = load_q_per_persona(personas)
    global_metrics = load_global_metrics()

    out_dir = ROOT / "output/open_exp"
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    summary = {"per_model_persona": {f"{m}|{pid}": v for (m,pid), v in scores.items()},
               "per_model_persona_q": {f"{m}|{pid}": v for (m,pid), v in q_data.items()},
               "global_metrics": global_metrics,
               "prompt_type_mean": {f"{m}|{pid}|{pt}": v for (m,pid,pt), v in pt_mean.items()},
               "seed_var_mean": {f"{m}|{pid}": v for (m,pid), v in seed_var.items()},
               "n_judgments": len(judgments)}
    (out_dir / "scores.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved → {out_dir/'scores.json'}")

    fig1_behavior_heatmap(scores, personas, fig_dir / "fig1_behavior_heatmap.png")
    fig2_q_vs_score(scores, q_data, personas, fig_dir / "fig2_q_vs_score.png")
    fig3_model_level(scores, q_data, global_metrics, fig_dir / "fig3_model_level.png")
    fig4_neighbor(scores, personas, fig_dir / "fig4_neighbor_discrimination.png")
    print(f"Saved 4 figures → {fig_dir}")


if __name__ == "__main__":
    main()
