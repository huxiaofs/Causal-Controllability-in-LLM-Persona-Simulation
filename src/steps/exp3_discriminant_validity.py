"""Experiment 3 — Discriminant Validity: EI_g vs MI(x,y)

Analyses whether EI_g provides independent information beyond MI(x,y).
Uses E2 permutation results (200 × 3 types × 2 models) plus original values.

Outputs:
  1. Correlation table  (Spearman ρ between EI_g and MI across permutations)
  2. Drop-magnitude comparison  (Δ EI_g vs Δ MI for each perm type)
  3. Divergence analysis  (cases where EI_g rank ≠ MI rank)
  4. Scatter plot  (EI_g vs MI, coloured by perm type)
"""
import json
import pathlib
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr

CODE = pathlib.Path(__file__).resolve().parent.parent.parent

PERM_LABELS = {
    "perm_x": "Perm-x",
    "perm_y": "Perm-y",
    "perm_both": "Perm-both",
}
PERM_KEYS = list(PERM_LABELS.keys())


def load_e2_results() -> dict:
    p = CODE / "output" / "exp2_permutation_test.json"
    with p.open() as f:
        return json.load(f)


def extract_vectors(model_data: dict):
    """Return arrays (eig, mi, log_vi, l_mean, labels) pooling all perm types."""
    eig, mi, lv, lm, labels = [], [], [], [], []
    for pk in PERM_KEYS:
        recs = model_data[pk]["records"]
        for r in recs:
            eig.append(r["EI_g"])
            mi.append(r["MI_x_y"])
            lv.append(r["log_VI"])
            lm.append(r["l_mean"])
            labels.append(pk)
    return np.array(eig), np.array(mi), np.array(lv), np.array(lm), labels


def main():
    data = load_e2_results()
    models = [k for k in data if k != "metadata"]

    all_analysis = {}
    fig, axes = plt.subplots(1, len(models), figsize=(6 * len(models), 5),
                             squeeze=False)

    for col, model_name in enumerate(models):
        md = data[model_name]
        orig = md["original"]
        var_y_x = md.get("var_y_x", float("nan"))

        print(f"\n{'=' * 70}")
        print(f"  Model: {model_name}")
        print(f"{'=' * 70}")
        print(f"  Original  EI_g={orig['EI_g']:.4f}  MI(x,y)={orig['MI_x_y']:.4f}"
              f"  Var(y|x)={var_y_x:.4f}")

        eig_all, mi_all, lv_all, lm_all, labels = extract_vectors(md)

        # ── 1. overall Spearman across all 600 permutations ──
        rho_all, p_all = spearmanr(eig_all, mi_all)
        print(f"\n  [1] Overall Spearman (600 perm instances):")
        print(f"      ρ = {rho_all:.4f}   p = {p_all:.2e}")

        # ── 2. drop-magnitude comparison per perm type ──
        print(f"\n  [2] Drop magnitudes (original − perm mean):")
        print(f"      {'Type':<12} {'Δ EI_g':>10} {'Δ MI':>10}"
              f" {'Δ log(VI)':>10} {'Δ l_mean':>10}")
        print(f"      {'-'*52}")
        type_stats = {}
        for pk in PERM_KEYS:
            recs = md[pk]["records"]
            e_arr = np.array([r["EI_g"] for r in recs])
            m_arr = np.array([r["MI_x_y"] for r in recs])
            lv_arr = np.array([r["log_VI"] for r in recs])
            lm_arr = np.array([r["l_mean"] for r in recs])
            d_eig = orig["EI_g"] - e_arr.mean()
            d_mi = orig["MI_x_y"] - m_arr.mean()
            d_lv = orig["log_VI"] - lv_arr.mean()
            d_lm = orig["l_mean"] - lm_arr.mean()
            rho_t, p_t = spearmanr(e_arr, m_arr)
            type_stats[pk] = {
                "d_eig": d_eig, "d_mi": d_mi,
                "d_log_vi": d_lv, "d_l_mean": d_lm,
                "eig_mean": float(e_arr.mean()), "eig_std": float(e_arr.std()),
                "mi_mean": float(m_arr.mean()), "mi_std": float(m_arr.std()),
                "spearman_rho": rho_t, "spearman_p": p_t,
            }
            print(f"      {PERM_LABELS[pk]:<12} {d_eig:>+10.4f} {d_mi:>+10.4f}"
                  f" {d_lv:>+10.4f} {d_lm:>+10.4f}")

        # ── 3. differential sensitivity: Perm-x vs Perm-y ──
        print(f"\n  [3] Differential sensitivity (Perm-x vs Perm-y):")
        dx = type_stats["perm_x"]
        dy = type_stats["perm_y"]
        print(f"      Perm-x EI_g drop: {dx['d_eig']:+.4f}   MI drop: {dx['d_mi']:+.4f}")
        print(f"      Perm-y EI_g drop: {dy['d_eig']:+.4f}   MI drop: {dy['d_mi']:+.4f}")
        ratio_eig = dx["d_eig"] / max(dy["d_eig"], 1e-8)
        ratio_mi = dx["d_mi"] / max(dy["d_mi"], 1e-8)
        print(f"      Ratio (x/y): EI_g {ratio_eig:.2f}x   MI {ratio_mi:.2f}x")
        if abs(ratio_eig - ratio_mi) > 0.1:
            print(f"      → EI_g and MI respond DIFFERENTLY to x- vs y-disruption")
        else:
            print(f"      → EI_g and MI respond similarly to x- vs y-disruption")

        # ── 4. divergence analysis: where EI_g rank ≠ MI rank ──
        print(f"\n  [4] Divergence analysis (perm-type ranking):")
        eig_order = sorted(PERM_KEYS, key=lambda k: type_stats[k]["eig_mean"], reverse=True)
        mi_order = sorted(PERM_KEYS, key=lambda k: type_stats[k]["mi_mean"], reverse=True)
        print(f"      EI_g ranking: {' > '.join(PERM_LABELS[k] for k in eig_order)}")
        print(f"      MI   ranking: {' > '.join(PERM_LABELS[k] for k in mi_order)}")
        if eig_order != mi_order:
            print(f"      → Rankings DIFFER — EI_g captures structure MI misses")
        else:
            print(f"      → Rankings agree (check magnitudes for nuance)")

        # ── 5. per-type Spearman ──
        print(f"\n  [5] Within-type Spearman ρ(EI_g, MI):")
        for pk in PERM_KEYS:
            ts = type_stats[pk]
            print(f"      {PERM_LABELS[pk]:<12}  ρ = {ts['spearman_rho']:.4f}"
                  f"  (p = {ts['spearman_p']:.2e})")

        # ── scatter plot ──
        ax = axes[0, col]
        colors = {"perm_x": "#e74c3c", "perm_y": "#3498db", "perm_both": "#2ecc71"}
        for pk in PERM_KEYS:
            recs = md[pk]["records"]
            ex = [r["EI_g"] for r in recs]
            mx = [r["MI_x_y"] for r in recs]
            ax.scatter(mx, ex, s=12, alpha=0.4, c=colors[pk],
                       label=PERM_LABELS[pk], edgecolors="none")
        ax.scatter(orig["MI_x_y"], orig["EI_g"], s=120, c="gold",
                   edgecolors="k", linewidths=1.2, zorder=5, marker="*",
                   label="Original")
        ax.set_xlabel("MI(x, y)", fontsize=11)
        ax.set_ylabel("EI_g", fontsize=11)
        ax.set_title(model_name, fontsize=12, fontweight="bold")
        ax.legend(fontsize=8, loc="upper left")
        ax.grid(True, alpha=0.3)

        all_analysis[model_name] = {
            "original": orig,
            "var_y_x": var_y_x,
            "overall_spearman": {"rho": rho_all, "p": p_all},
            "per_type": type_stats,
        }

    fig.tight_layout()
    fig_path = CODE / "output" / "figures" / "exp3_eig_vs_mi_scatter.png"
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    print(f"\n  Scatter plot saved to {fig_path}")

    # ── cross-model summary table ──
    print(f"\n\n{'=' * 90}")
    print("  E3 — Discriminant Validity Summary")
    print(f"{'=' * 90}")
    hdr = (f"{'Model':<22} {'Type':<12} {'EI_g(mean)':>10} {'MI(mean)':>10}"
           f" {'Δ EI_g':>10} {'Δ MI':>10} {'ρ(EI_g,MI)':>11}")
    print(hdr)
    print("-" * 90)
    for mn in models:
        a = all_analysis[mn]
        for pk in PERM_KEYS:
            ts = a["per_type"][pk]
            print(f"{mn:<22} {PERM_LABELS[pk]:<12}"
                  f" {ts['eig_mean']:>10.4f} {ts['mi_mean']:>10.4f}"
                  f" {ts['d_eig']:>+10.4f} {ts['d_mi']:>+10.4f}"
                  f" {ts['spearman_rho']:>11.4f}")
    print("-" * 90)

    print("\nKey findings:")
    for mn in models:
        a = all_analysis[mn]
        rho = a["overall_spearman"]["rho"]
        dx_e = a["per_type"]["perm_x"]["d_eig"]
        dy_e = a["per_type"]["perm_y"]["d_eig"]
        dx_m = a["per_type"]["perm_x"]["d_mi"]
        dy_m = a["per_type"]["perm_y"]["d_mi"]
        print(f"  [{mn}]")
        print(f"    Overall ρ(EI_g, MI) = {rho:.3f}"
              f"  → {'correlated but not equivalent' if 0.3 < abs(rho) < 0.9 else 'high overlap' if abs(rho) >= 0.9 else 'largely independent'}")
        print(f"    Perm-x: Δ EI_g={dx_e:+.2f}, Δ MI={dx_m:+.4f}")
        print(f"    Perm-y: Δ EI_g={dy_e:+.2f}, Δ MI={dy_m:+.4f}")

    # ── save ──
    save_path = CODE / "output" / "exp3_discriminant_validity.json"
    with save_path.open("w") as f:
        json.dump(all_analysis, f, ensure_ascii=False, indent=2,
                  default=lambda o: float(o) if isinstance(o, (np.floating, np.integer)) else o)
    print(f"\nResults saved to {save_path}")


if __name__ == "__main__":
    main()
