"""
Compute classical input-output coupling baselines for all 4 models so they
can be compared head-to-head against EI_g on the open-task external-validity
panel.

Baselines:

    MI_x_y          KSG mutual information between target persona x and the
                    questionnaire-derived observation y.  Reuses the KSG
                    estimator already validated in exp2_permutation_test.py.

    R2_xy           Multi-output linear R^2 of x -> y, fitted on the same
                    aligned (x, y) pairs.  Reported as the unweighted mean
                    of dimension-wise R^2.

    R2_theta_y      R^2 of theta -> y on the same aligned pairs.  Tells us
                    how much variance in y is recoverable from the bottleneck
                    representation.

    CCA1            First canonical correlation between x and y (i.e. the
                    largest singular value of the cross-correlation
                    structure).

    CCA_mean        Mean of all six canonical correlations between x and y.

    RV              Escoufier's RV coefficient between x and y, a
                    multivariate generalisation of squared correlation that
                    is invariant to orthogonal rotations of either side.

All baselines are computed on the *standardised* x and y (z-score per
dimension) to match the EI_g preprocessing in eig_metric.py.

Outputs (idempotent, overwrites):
    output/baselines_all_models.json
"""
from __future__ import annotations

import json
import pathlib
import sys

import numpy as np
from scipy.special import digamma
from sklearn.cross_decomposition import CCA
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import BallTree, NearestNeighbors

ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

HEXACO_DIMS = [
    "Honesty-Humility", "Emotionality", "Extraversion",
    "Agreeableness", "Conscientiousness", "Openness",
]

MODELS = {
    "qwen2.5-7b": ROOT / "output_qwen2.5-7b",
    "deepseek-r1-distill-qwen-7b": ROOT / "output_deepseek-r1-distill-qwen-7b",
    "mistral-7b-instruct-v0.3": ROOT / "output_mistral-7b-instruct-v0.3",
    "llama3.1-8b": ROOT / "output_llama3.1-8b",
    "mixtral-8x7b": ROOT / "output_mixtral-8x7b",
    "llama3.1-70b": ROOT / "output_llama3.1-70b",
    "qwen3-30b-a3b": ROOT / "output_qwen3-30b-a3b",
}


def load_xy_theta(out_dir: pathlib.Path):
    x_list, theta_list = [], []
    with (out_dir / "persona_hexaco_pred.jsonl").open("r") as f:
        for line in f:
            row = json.loads(line)
            x_list.append(row["hexaco_label"])
            theta_list.append(row["hexaco_pred"])
    x_raw = np.asarray(x_list, dtype=float)
    theta_raw = np.asarray(theta_list, dtype=float)

    y_dict = {}
    with (out_dir / "questionnaire_scores.jsonl").open("r") as f:
        for line in f:
            row = json.loads(line)
            try:
                yv = [row["scores"]["hexaco"]["dimensions"][k] for k in HEXACO_DIMS]
            except Exception:
                continue
            y_dict[(row["persona_id"], row["prompt_idx"])] = yv

    n = len(x_list)
    keys = [(i // 3, i % 3) for i in range(n)]
    aligned = [k in y_dict for k in keys]
    keep = np.array(aligned, dtype=bool)
    x_raw = x_raw[keep]
    theta_raw = theta_raw[keep]
    y_raw = np.array([y_dict[(k[0], k[1])] for k in keys if k in y_dict],
                     dtype=float)
    assert y_raw.shape[0] == x_raw.shape[0]
    return x_raw, theta_raw, y_raw


def zscore(a: np.ndarray) -> np.ndarray:
    m = a.mean(0, keepdims=True)
    s = a.std(0, keepdims=True)
    s = np.where(s < 1e-6, 1.0, s)
    return (a - m) / s


def mi_knn_ksg(X: np.ndarray, Y: np.ndarray, k: int = 7) -> float:
    """KSG (1) MI estimator with Chebyshev metric.  Mirrors exp2."""
    n = X.shape[0]
    XY = np.hstack([X, Y])
    nn = NearestNeighbors(n_neighbors=k + 1, metric="chebyshev").fit(XY)
    dists, _ = nn.kneighbors(XY)
    eps = dists[:, k]
    tx = BallTree(X, metric="chebyshev")
    ty = BallTree(Y, metric="chebyshev")
    nx = np.maximum(np.array(tx.query_radius(X, r=eps * (1 - 1e-10),
                                              count_only=True)) - 1, 1).astype(float)
    ny = np.maximum(np.array(ty.query_radius(Y, r=eps * (1 - 1e-10),
                                              count_only=True)) - 1, 1).astype(float)
    return float(max(digamma(k) - np.mean(digamma(nx) + digamma(ny))
                     + digamma(n), 0.0))


def linreg_r2(X: np.ndarray, Y: np.ndarray) -> float:
    """Multi-output linear regression R^2 (mean over output dims)."""
    reg = LinearRegression().fit(X, Y)
    pred = reg.predict(X)
    return float(r2_score(Y, pred, multioutput="uniform_average"))


def cca_corrs(X: np.ndarray, Y: np.ndarray, n_components: int = 6) -> np.ndarray:
    """Return the canonical correlations between X and Y."""
    n_components = min(n_components, X.shape[1], Y.shape[1])
    cca = CCA(n_components=n_components, max_iter=2000)
    cca.fit(X, Y)
    Xc, Yc = cca.transform(X, Y)
    Xc = Xc - Xc.mean(0); Yc = Yc - Yc.mean(0)
    Xc = Xc / np.maximum(Xc.std(0), 1e-12)
    Yc = Yc / np.maximum(Yc.std(0), 1e-12)
    n = X.shape[0]
    corrs = (Xc * Yc).sum(0) / (n - 1)
    return np.clip(corrs, -1.0, 1.0)


def rv_coefficient(X: np.ndarray, Y: np.ndarray) -> float:
    """Escoufier's RV coefficient.

    RV(X, Y) = trace(SXY @ SYX) / sqrt(trace(SXX^2) * trace(SYY^2))

    where SXY = X.T @ Y / (n-1).  Bounded in [0, 1] for centred inputs.
    """
    Xc = X - X.mean(0)
    Yc = Y - Y.mean(0)
    SXX = Xc.T @ Xc
    SYY = Yc.T @ Yc
    SXY = Xc.T @ Yc
    num = float(np.trace(SXY @ SXY.T))
    den = float(np.sqrt(np.trace(SXX @ SXX) * np.trace(SYY @ SYY)))
    return num / den if den > 1e-12 else float("nan")


def compute_for_model(out_dir: pathlib.Path) -> dict:
    x_raw, theta_raw, y_raw = load_xy_theta(out_dir)
    x = zscore(x_raw)
    y = zscore(y_raw)
    th = zscore(theta_raw)
    mi = mi_knn_ksg(x, y, k=7)
    r2_xy = linreg_r2(x, y)
    r2_ty = linreg_r2(th, y)
    r2_tx = linreg_r2(th, x)
    cca = cca_corrs(x, y, n_components=6)
    rv = rv_coefficient(x, y)
    out = {
        "n": int(x.shape[0]),
        "MI_x_y": float(mi),
        "R2_x_to_y": float(r2_xy),
        "R2_theta_to_y": float(r2_ty),
        "R2_theta_to_x": float(r2_tx),
        "CCA_first": float(cca[0]) if len(cca) else float("nan"),
        "CCA_mean": float(cca.mean()) if len(cca) else float("nan"),
        "CCA_all": [float(c) for c in cca],
        "RV": float(rv),
    }
    return out


def main():
    results = {}
    for alias, out_dir in MODELS.items():
        if not (out_dir / "persona_hexaco_pred.jsonl").exists():
            print(f"[skip] {alias}: missing persona_hexaco_pred.jsonl")
            continue
        if not (out_dir / "questionnaire_scores.jsonl").exists():
            print(f"[skip] {alias}: missing questionnaire_scores.jsonl")
            continue
        try:
            r = compute_for_model(out_dir)
        except Exception as e:
            print(f"[err]  {alias}: {e}")
            continue
        print(f"{alias:35s}  n={r['n']:>4d}  "
              f"MI={r['MI_x_y']:.3f}  R2(x->y)={r['R2_x_to_y']:+.3f}  "
              f"R2(theta->y)={r['R2_theta_to_y']:+.3f}  "
              f"CCA1={r['CCA_first']:.3f}  CCA_mean={r['CCA_mean']:.3f}  "
              f"RV={r['RV']:.3f}")
        results[alias] = r

    out_path = ROOT / "output" / "baselines_all_models.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, ensure_ascii=False, indent=2),
                        encoding="utf-8")
    print(f"\nSaved -> {out_path}")


if __name__ == "__main__":
    main()
