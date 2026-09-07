import json
import pathlib
import sys
from typing import Dict, List

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent
PARENT = ROOT.parent
for p in (ROOT, PARENT):
    if str(p) not in sys.path:
        sys.path.append(str(p))

from steps.eig_metric import compute_eig


def load_data(output_dir: pathlib.Path):
    pred_path = output_dir / "persona_hexaco_pred.jsonl"
    x_list, theta_list = [], []
    with pred_path.open("r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            x_list.append(row["hexaco_label"])
            theta_list.append(row["hexaco_pred"])

    x = np.array(x_list)
    theta = np.array(theta_list)

    score_path = output_dir / "questionnaire_scores.jsonl"
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

    return x, theta, y


def whiten_theta(theta: np.ndarray) -> np.ndarray:
    mean = theta.mean(axis=0, keepdims=True)
    std = theta.std(axis=0, keepdims=True)
    std = np.where(std < 1e-6, 1.0, std)
    theta_z = (theta - mean) / std
    cov_theta = np.cov(theta_z, rowvar=False)
    evals, evecs = np.linalg.eigh(cov_theta)
    evals = np.clip(evals, 1e-6, None)
    W = evecs @ np.diag(1.0 / np.sqrt(evals)) @ evecs.T
    return theta_z @ W


def summary_stats(arr: np.ndarray) -> Dict:
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "p5": float(np.percentile(arr, 5)),
        "p50": float(np.percentile(arr, 50)),
        "p95": float(np.percentile(arr, 95)),
        "p99": float(np.percentile(arr, 99)),
        "max": float(np.max(arr)),
    }


def main():
    output_dir = pathlib.Path("output")
    x, theta, y = load_data(output_dir)
    theta = whiten_theta(theta)

    base_params = dict(
        k=20,
        cov_type="shrinkage",
        alpha=0.1,
        cov_reg=1e-5,
        eps=1e-5,
        eta=1e-8,
        volume_strategy="A",
        weight_type="gaussian",
        weight_sigma=None,
        winsorize_pct=0.01,
    )

    res = compute_eig(x=x, theta=theta, y=y, **base_params)
    l_arr = np.array(res["l_list"])
    eigvals_arr = np.array(res["eigvals_list"])
    max_eigs = eigvals_arr.max(axis=1)
    traceA = np.array([np.trace(a) for a in res["A_list"]])
    logdet_IpA = 2 * l_arr

    diagnostics = {
        "base": {
            "EI_g": float(res["EI_g"]),
            "l_mean": float(res["l_mean"]),
            "VI": float(res["VI"]),
        },
        "distributions": {
            "lambda_max": summary_stats(max_eigs),
            "trace_A": summary_stats(traceA),
            "logdet_I_plus_A": summary_stats(logdet_IpA),
        },
    }

    # sensitivity curves
    ks = [20, 30, 40, 60]
    eps_list = [1e-6, 1e-5, 1e-4]
    cov_types = ["diag", "shrinkage"]
    sensitivity: List[Dict] = []
    for k in ks:
        for eps in eps_list:
            for cov_type in cov_types:
                res_s = compute_eig(
                    x=x,
                    theta=theta,
                    y=y,
                    k=k,
                    cov_type=cov_type,
                    alpha=0.1,
                    cov_reg=1e-5,
                    eps=eps,
                    eta=1e-8,
                    volume_strategy="A",
                    weight_type="gaussian",
                    weight_sigma=None,
                    winsorize_pct=0.01,
                )
                sensitivity.append(
                    {
                        "k": k,
                        "eps": eps,
                        "cov_type": cov_type,
                        "EI_g": float(res_s["EI_g"]),
                        "l_mean": float(res_s["l_mean"]),
                        "VI": float(res_s["VI"]),
                    }
                )
    diagnostics["sensitivity"] = sensitivity

    # bootstrap CI
    B = 50
    rng = np.random.default_rng(42)
    boot = []
    N = x.shape[0]
    for _ in range(B):
        idx = rng.integers(0, N, size=N)
        res_b = compute_eig(x=x[idx], theta=theta[idx], y=y[idx], **base_params)
        boot.append(res_b["EI_g"])
    boot = np.array(boot)
    diagnostics["bootstrap"] = {
        "B": B,
        "mean": float(np.mean(boot)),
        "std": float(np.std(boot)),
        "p2_5": float(np.percentile(boot, 2.5)),
        "p97_5": float(np.percentile(boot, 97.5)),
    }

    out_path = output_dir / "eig_metric_diagnostics.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(diagnostics, f, ensure_ascii=False, indent=2)

    print(f"saved diagnostics to {out_path}")


if __name__ == "__main__":
    main()
