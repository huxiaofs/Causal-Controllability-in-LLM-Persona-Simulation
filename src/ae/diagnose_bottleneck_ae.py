import json
import pathlib
import sys
from typing import Dict, Any

import numpy as np
import torch
import torch.nn as nn

ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from ae.train_bottleneck_ae import (
    BottleneckAE,
    load_cache,
    load_prompts,
    compute_embeddings,
    resolve_snapshot_path,
    compute_file_sha1,
)
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_cfg(path: pathlib.Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    if ss_tot <= 1e-12:
        return float("nan")
    return 1.0 - ss_res / ss_tot


def corr(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if np.std(y_true) < 1e-12 or np.std(y_pred) < 1e-12:
        return float("nan")
    return float(np.corrcoef(y_true, y_pred)[0, 1])


def main():
    cfg = load_cfg(pathlib.Path("config/ae.json"))
    prompts_file = pathlib.Path(cfg["prompts_file"])
    cache_path = pathlib.Path(cfg["embedding_cache"])

    # load or compute embeddings
    if cache_path.exists():
        embeddings, labels, meta = load_cache(cache_path)
        cache_hash = meta.get("prompts_sha1")
        prompts_hash = compute_file_sha1(prompts_file)
        if cache_hash != prompts_hash:
            print("[warn] embedding cache is stale vs prompts file")
    else:
        print("[warn] embedding cache not found, computing embeddings (may take time)")
        model_base = resolve_snapshot_path(pathlib.Path(cfg["model_path"]))
        tok_base = resolve_snapshot_path(pathlib.Path(cfg.get("tokenizer_path", cfg["model_path"])))
        tokenizer = AutoTokenizer.from_pretrained(
            str(tok_base), use_fast=True, legacy=False, local_files_only=True
        )
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id
        torch_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        device_map = "auto" if torch.cuda.is_available() else None
        llama = AutoModelForCausalLM.from_pretrained(
            str(model_base),
            torch_dtype=torch_dtype,
            device_map=device_map,
            local_files_only=True,
            output_hidden_states=True,
        )
        pairs = load_prompts(prompts_file)
        embeddings, labels = compute_embeddings(pairs, tokenizer, llama, cfg.get("max_seq_len", 256))

    # load AE checkpoint
    ckpt_path = pathlib.Path(cfg.get("ae_checkpoint", "output/ae_bottleneck.pt"))
    if not ckpt_path.exists():
        raise FileNotFoundError(f"AE checkpoint not found: {ckpt_path}")

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    ae = BottleneckAE(
        input_dim=embeddings.size(1),
        bottleneck_dim=cfg.get("bottleneck_dim", 6),
        dropout_p=float(cfg.get("dropout", 0.0)),
    ).to(device)
    ae.load_state_dict(torch.load(ckpt_path, map_location="cpu"))
    ae.eval()

    emb = embeddings.to(device)
    lab = labels.to(device)

    with torch.no_grad():
        recon, preds, z = ae(emb)

    recon_mse = nn.MSELoss()(recon, emb).item()
    sup_mse = nn.MSELoss()(preds, lab).item()

    z_np = z.cpu().numpy()
    preds_np = preds.cpu().numpy()
    labels_np = labels.cpu().numpy()

    z_var = z_np.var(axis=0)
    z_mean = z_np.mean(axis=0)

    print("[AE diagnostics]")
    print(f"recon_mse: {recon_mse:.6f}")
    print(f"sup_mse: {sup_mse:.6f}")
    print(f"z_mean: {np.round(z_mean, 4).tolist()}")
    print(f"z_var: {np.round(z_var, 6).tolist()}")

    dim_names = [
        "Honesty-Humility",
        "Emotionality",
        "Extraversion",
        "Agreeableness",
        "Conscientiousness",
        "Openness",
    ]
    for i, name in enumerate(dim_names):
        r = corr(labels_np[:, i], preds_np[:, i])
        r2 = r2_score(labels_np[:, i], preds_np[:, i])
        print(f"{name}: corr={r:.4f}, r2={r2:.4f}")

    if np.all(z_var < 1e-4):
        print("[warn] bottleneck appears collapsed (very low variance)")


if __name__ == "__main__":
    main()
