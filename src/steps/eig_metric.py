"""
EI_g (Geometric Effective Information) computation module.

Implements the local-geometric EI_g pipeline used by the experiment scripts.
"""
import json
import pathlib
from typing import Dict, Literal, Optional, Tuple

import numpy as np
from scipy.linalg import LinAlgError, cholesky, svd
from sklearn.neighbors import NearestNeighbors


def compute_local_jacobian_regression(
    theta: np.ndarray, y: np.ndarray, k: int, weights: Optional[np.ndarray] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """Fit a local linear map theta -> y and return coef/intercept."""
    from sklearn.linear_model import LinearRegression

    reg = LinearRegression()
    if weights is not None:
        reg.fit(theta, y, sample_weight=weights)
    else:
        reg.fit(theta, y)
    return reg.coef_, reg.intercept_


def estimate_covariance(
    residuals: np.ndarray,
    cov_type: Literal["full", "diag", "shrinkage"] = "full",
    alpha: float = 0.05,
    cov_reg: float = 0.0,
) -> np.ndarray:
    """Estimate residual covariance."""
    if cov_type == "full":
        cov = np.cov(residuals, rowvar=False)
    elif cov_type == "diag":
        v = np.var(residuals, axis=0)
        cov = np.diag(v)
    elif cov_type == "shrinkage":
        v = np.var(residuals, axis=0)
        full = np.cov(residuals, rowvar=False)
        cov = alpha * np.diag(v) + (1 - alpha) * full
    else:
        raise ValueError(f"Unknown cov_type: {cov_type}")
    if cov_reg and cov_reg > 0:
        cov = cov + cov_reg * np.eye(cov.shape[0])
    return cov


def stable_inverse(mat: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Numerically stable inverse."""
    return np.linalg.inv(mat + eps * np.eye(mat.shape[0]))


def logdet_stable(mat: np.ndarray, eta: float = 1e-8) -> float:
    """Stable log determinant, Cholesky first and SVD fallback."""
    try:
        L = cholesky(mat + eta * np.eye(mat.shape[0]), lower=True)
        return 2 * np.sum(np.log(np.diag(L)))
    except LinAlgError:
        _, s, _ = svd(mat + eta * np.eye(mat.shape[0]))
        return np.sum(np.log(s))


def compute_eig(
    x: np.ndarray,
    theta: np.ndarray,
    y: np.ndarray,
    k: int = 30,
    cov_type: Literal["full", "diag", "shrinkage"] = "full",
    alpha: float = 0.05,
    cov_reg: float = 0.0,
    eps: float = 1e-6,
    eta: float = 1e-8,
    volume_strategy: Literal["A", "B"] = "A",
    vol_x: float = 1.0,
    vol_theta: float = 1.0,
    delta_const: Optional[np.ndarray] = None,
    weight_type: Literal["none", "uniform", "gaussian"] = "none",
    weight_sigma: Optional[float] = None,
    winsorize_pct: float = 0.0,
    weight_sqrt: Optional[bool] = None,
) -> Dict:
    """Compute EI_g and diagnostic local matrices."""
    n_samples, d_theta = theta.shape
    if n_samples <= 1:
        raise ValueError("Need at least 2 samples to compute EI_g")

    k_eff = max(2, min(int(k), n_samples))
    nbrs = NearestNeighbors(n_neighbors=k_eff, algorithm="auto").fit(theta)
    _, knn_idx = nbrs.kneighbors(theta)

    a_list, l_list, g_list, h_list, eigvals_list = [], [], [], [], []
    delta_list, w_list = [], []
    if weight_sqrt is None:
        weight_sqrt = volume_strategy == "B"

    for i in range(n_samples):
        idx = knn_idx[i]
        theta_k = theta[idx]
        y_k = y[idx]
        x_k = x[idx]

        weights = None
        if weight_type != "none":
            d = np.linalg.norm(theta_k - theta[i], axis=1)
            if weight_type == "uniform":
                weights = np.ones_like(d)
            elif weight_type == "gaussian":
                if weight_sigma is None or weight_sigma <= 0:
                    nonzero = d[d > 0]
                    sigma = float(np.median(nonzero)) if nonzero.size > 0 else 1.0
                else:
                    sigma = float(weight_sigma)
                weights = np.exp(-(d ** 2) / (2 * sigma ** 2))
            else:
                raise ValueError(f"Unknown weight_type: {weight_type}")

        ay, by = compute_local_jacobian_regression(theta_k, y_k, k_eff, weights=weights)
        ax, bx = compute_local_jacobian_regression(theta_k, x_k, k_eff, weights=weights)

        ry = y_k - (theta_k @ ay.T + by)
        rx = x_k - (theta_k @ ax.T + bx)

        sy = estimate_covariance(ry, cov_type, alpha, cov_reg)
        sx = estimate_covariance(rx, cov_type, alpha, cov_reg)
        if cov_reg is None or cov_reg <= 0:
            tr_y = np.trace(sy)
            tr_x = np.trace(sx)
            sy = sy + (1e-6 * tr_y / max(1, sy.shape[0])) * np.eye(sy.shape[0])
            sx = sx + (1e-6 * tr_x / max(1, sx.shape[0])) * np.eye(sx.shape[0])

        ey = stable_inverse(sy, eps)
        dx = stable_inverse(sx, eps)
        g = ay.T @ ey @ ay
        h = ax.T @ dx @ ax
        g_list.append(g)
        h_list.append(h)
        delta_list.append(dx)

        # Eq. (16): use the symmetric congruence transform
        # (g + eps I)^(-1/2) h (g + eps I)^(-1/2).  Clipping the eigenvalues
        # of the regularized g matrix keeps the inverse square root stable.
        g_reg = 0.5 * (g + g.T) + eps * np.eye(g.shape[0])
        g_vals, g_vecs = np.linalg.eigh(g_reg)
        g_vals = np.maximum(g_vals, eps)
        g_inv_sqrt = (g_vecs * (1.0 / np.sqrt(g_vals))) @ g_vecs.T
        a_sym = g_inv_sqrt @ (0.5 * (h + h.T)) @ g_inv_sqrt
        a_list.append(a_sym)

        eigvals = np.linalg.eigvalsh(a_sym)
        eigvals = np.clip(eigvals, -0.99, None)
        l_val = 0.5 * np.sum(np.log1p(eigvals))
        l_list.append(l_val)
        eigvals_list.append(eigvals)

        w = np.linalg.det(h)
        if weight_sqrt:
            w = float(np.sqrt(max(w, 0.0)))
        w_list.append(w)

    l_arr = np.array(l_list)
    if winsorize_pct and winsorize_pct > 0:
        lo = np.percentile(l_arr, winsorize_pct * 100)
        hi = np.percentile(l_arr, (1 - winsorize_pct) * 100)
        l_arr = np.clip(l_arr, lo, hi)

    w_arr = np.array(w_list)
    if volume_strategy == "A":
        l_mean = l_arr.mean()
        if delta_const is None and len(delta_list) > 0:
            delta_const = np.mean(np.stack(delta_list, axis=0), axis=0)
        if delta_const is not None:
            delta_const = 0.5 * (delta_const + delta_const.T)
            vi = vol_x * np.linalg.det(delta_const + eta * np.eye(delta_const.shape[0]))
        else:
            vi = vol_x
    else:
        l_mean = np.sum(w_arr * l_arr) / np.sum(w_arr)
        vi = vol_theta / n_samples * np.sum(w_arr)

    ei_g = np.log(vi / ((2 * np.pi * np.e) ** (d_theta / 2))) - l_mean
    return {
        "A_list": a_list,
        "l_list": l_list,
        "g_list": g_list,
        "h_list": h_list,
        "eigvals_list": eigvals_list,
        "EI_g": ei_g,
        "l_mean": l_mean,
        "VI": vi,
    }


def apply_output_suffix(cfg: Dict) -> Dict:
    suffix = str(cfg.get("output_suffix", "")).strip()
    if not suffix:
        return cfg
    base = str(cfg.get("output_dir", "output"))
    if not base.endswith(f"_{suffix}") and not base.endswith(f"/{suffix}"):
        base = f"{base}_{suffix}"
    cfg["output_dir"] = base
    return cfg


def main(cfg_path: str | None = None):
    import argparse

    parser = argparse.ArgumentParser(description="Compute EI_g metric")
    parser.add_argument("--config", type=str, default=cfg_path, help="path to model config json")
    parser.add_argument("--output-suffix", type=str, default=None, help="override output_suffix")
    args, _ = parser.parse_known_args()

    params = dict(
        k=20,
        cov_type="shrinkage",
        alpha=0.1,
        cov_reg=1e-5,
        eps=1e-5,
        eta=1e-8,
        volume_strategy="B",
        weight_type="gaussian",
        weight_sigma=None,
        winsorize_pct=0.01,
    )

    output_dir = "output"
    if args.config:
        cfg_path = pathlib.Path(args.config)
    else:
        cfg_path = pathlib.Path("config/ae_deepseek-r1-distill-qwen-7b.json")
        if not cfg_path.exists():
            cfg_path = pathlib.Path("config/ae.json")
    if cfg_path.exists():
        cfg_raw = json.loads(cfg_path.read_text(encoding="utf-8"))
        if args.output_suffix:
            cfg_raw["output_suffix"] = args.output_suffix
        cfg = apply_output_suffix(cfg_raw)
        output_dir = cfg.get("output_dir", output_dir)

    output_path = pathlib.Path(output_dir)
    pred_path = output_path / "persona_hexaco_pred.jsonl"
    score_path = output_path / "questionnaire_scores.jsonl"

    x_list, theta_list = [], []
    with pred_path.open("r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            x_list.append(row["hexaco_label"])
            theta_list.append(row["hexaco_pred"])

    x = np.array(x_list)
    theta = np.array(theta_list)
    x_raw = x.copy()
    theta_raw = theta.copy()

    x_mean = x.mean(axis=0, keepdims=True)
    x_std = x.std(axis=0, keepdims=True)
    x_std = np.where(x_std < 1e-6, 1.0, x_std)
    x = (x - x_mean) / x_std

    theta_mean = theta.mean(axis=0, keepdims=True)
    theta_std = theta.std(axis=0, keepdims=True)
    theta_std = np.where(theta_std < 1e-6, 1.0, theta_std)
    theta_z = (theta - theta_mean) / theta_std
    cov_theta = np.cov(theta_z, rowvar=False)
    evals, evecs = np.linalg.eigh(cov_theta)
    evals = np.clip(evals, 1e-6, None)
    w_mat = evecs @ np.diag(1.0 / np.sqrt(evals)) @ evecs.T
    theta = theta_z @ w_mat

    x_range = np.maximum(x_raw.max(axis=0) - x_raw.min(axis=0), 1e-12)
    theta_range = np.maximum(theta_raw.max(axis=0) - theta_raw.min(axis=0), 1e-12)
    vol_x_raw = float(np.prod(x_range))
    vol_theta_raw = float(np.prod(theta_range))

    vol_x = vol_x_raw / float(np.prod(x_std.squeeze()))
    scale_mat = np.diag(1.0 / theta_std.squeeze())
    _, logdet_scale = np.linalg.slogdet(scale_mat)
    _, logdet_whiten = np.linalg.slogdet(w_mat)
    vol_theta = vol_theta_raw * float(np.exp(logdet_scale + logdet_whiten))

    cov_theta_after = np.cov(theta, rowvar=False)
    eig_after = np.linalg.eigvalsh(cov_theta_after)
    cond_after = eig_after.max() / max(eig_after.min(), 1e-12)
    print(f"theta cov cond after whitening: {cond_after:.4f}")

    y_dict = {}
    with score_path.open("r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            pid = row["persona_id"]
            pidx = row["prompt_idx"]
            try:
                y_vec = [
                    row["scores"]["hexaco"]["dimensions"][k]
                    for k in [
                        "Honesty-Humility",
                        "Emotionality",
                        "Extraversion",
                        "Agreeableness",
                        "Conscientiousness",
                        "Openness",
                    ]
                ]
            except Exception:
                continue
            y_dict[(pid, pidx)] = y_vec

    y_list = []
    for i in range(len(x_list)):
        persona_id = i // 3
        prompt_idx = i % 3
        y_vec = y_dict.get((persona_id, prompt_idx))
        if y_vec is None:
            raise ValueError(f"Missing y for persona_id={persona_id}, prompt_idx={prompt_idx}")
        y_list.append(y_vec)
    y = np.array(y_list)

    y_mean = y.mean(axis=0, keepdims=True)
    y_std = y.std(axis=0, keepdims=True)
    y_std = np.where(y_std < 1e-6, 1.0, y_std)
    y = (y - y_mean) / y_std

    result = compute_eig(x=x, theta=theta, y=y, vol_x=vol_x, vol_theta=vol_theta, **params)
    print(f"EI_g: {result['EI_g']:.6f}")
    print(f"l_mean: {result['l_mean']:.6f}")
    print(f"VI: {result['VI']:.6f}")
    print(f"A_list shape: {np.array(result['A_list']).shape}")

    save_path = output_path / "eig_metric_result.json"
    out_json = {
        "EI_g": float(result["EI_g"]),
        "l_mean": float(result["l_mean"]),
        "VI": float(result["VI"]),
        "A_list": np.array(result["A_list"]).round(4).tolist(),
    }
    with save_path.open("w", encoding="utf-8") as fout:
        json.dump(out_json, fout, ensure_ascii=False, indent=2)
    print(f"Saved EI_g and A_list to {save_path}")


if __name__ == "__main__":
    main()
