import json
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent
PARENT = ROOT.parent
for p in (ROOT, PARENT):
    if str(p) not in sys.path:
        sys.path.append(str(p))

from steps.eig_metric import compute_eig


def main():
    # Keep arguments aligned with eig_metric.py::main().
    k = 20
    cov_type = "shrinkage"
    alpha = 0.1
    cov_reg = 1e-5
    eps = 1e-5
    eta = 1e-8
    volume_strategy = "B"
    weight_type = "gaussian"
    weight_sigma = None
    winsorize_pct = 0.01
    output_dir = "output"

    # Load persona_hexaco_pred.jsonl.
    pred_path = pathlib.Path(output_dir) / "persona_hexaco_pred.jsonl"
    x_list, theta_list = [], []
    with pred_path.open("r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            x_list.append(row["hexaco_label"])
            theta_list.append(row["hexaco_pred"])

    x = np.array(x_list)
    theta = np.array(theta_list)

    # Put x on a common scale by standardizing it.
    x_mean = x.mean(axis=0, keepdims=True)
    x_std = x.std(axis=0, keepdims=True)
    x_std = np.where(x_std < 1e-6, 1.0, x_std)
    x = (x - x_mean) / x_std

    # Standardize/whiten theta to reduce anisotropy.
    theta_mean = theta.mean(axis=0, keepdims=True)
    theta_std = theta.std(axis=0, keepdims=True)
    theta_std = np.where(theta_std < 1e-6, 1.0, theta_std)
    theta_z = (theta - theta_mean) / theta_std
    cov_theta = np.cov(theta_z, rowvar=False)
    evals, evecs = np.linalg.eigh(cov_theta)
    evals = np.clip(evals, 1e-6, None)
    W = evecs @ np.diag(1.0 / np.sqrt(evals)) @ evecs.T
    theta = theta_z @ W

    # Load questionnaire_scores.jsonl and aggregate y by persona_id/prompt_idx.
    score_path = pathlib.Path(output_dir) / "questionnaire_scores.jsonl"
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

    # Put y on a common scale by standardizing it.
    y_mean = y.mean(axis=0, keepdims=True)
    y_std = y.std(axis=0, keepdims=True)
    y_std = np.where(y_std < 1e-6, 1.0, y_std)
    y = (y - y_mean) / y_std

    result = compute_eig(
        x=x,
        theta=theta,
        y=y,
        k=k,
        cov_type=cov_type,
        alpha=alpha,
        cov_reg=cov_reg,
        eps=eps,
        eta=eta,
        volume_strategy=volume_strategy,
        weight_type=weight_type,
        weight_sigma=weight_sigma,
        winsorize_pct=winsorize_pct,
        weight_sqrt=True,
    )

    save_path = pathlib.Path(output_dir) / "eig_metric_result_weight_sqrt.json"
    a_list_rounded = np.array(result["A_list"]).round(4).tolist()
    out_json = {
        "EI_g": float(result["EI_g"]),
        "l_mean": float(result["l_mean"]),
        "VI": float(result["VI"]),
        "A_list": a_list_rounded,
    }
    with save_path.open("w", encoding="utf-8") as fout:
        json.dump(out_json, fout, ensure_ascii=False, indent=2)

    print(f"EI_g: {result['EI_g']:.6f}")
    print(f"l_mean: {result['l_mean']:.6f}")
    print(f"VI: {result['VI']:.6f}")
    print(f"Saved to {save_path}")


if __name__ == "__main__":
    main()
