import argparse
import json
import pathlib
import sys
from typing import Dict, List, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# ensure src root on path
ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from ae.train_bottleneck_ae import (
    BottleneckAE,
    compute_embeddings,
    load_cache,
    load_prompts,
    resolve_snapshot_path,
    save_cache,
    compute_file_sha1,
)


def load_cfg(path: pathlib.Path) -> Dict:
    return json.loads(path.read_text(encoding="utf-8"))


def apply_output_suffix(cfg: Dict) -> Dict:
    suffix = str(cfg.get("output_suffix", "")).strip()
    if not suffix:
        return cfg
    base = str(cfg.get("output_dir", "output"))
    if not base.endswith(f"_{suffix}") and not base.endswith(f"/{suffix}"):
        base = f"{base}_{suffix}"
    cfg["output_dir"] = base
    for key in ("prompts_file", "embedding_cache", "ae_checkpoint"):
        path = cfg.get(key)
        if isinstance(path, str) and path.startswith("output/"):
            cfg[key] = path.replace("output/", f"{base}/", 1)
    return cfg


def normalize_model_paths(cfg: Dict) -> Dict:
    updated = dict(cfg)
    for key in ("model_path", "tokenizer_path"):
        raw = updated.get(key)
        if not raw:
            continue
        updated[key] = str(resolve_snapshot_path(pathlib.Path(raw)))
    return updated


def ensure_embeddings(cfg: Dict, tokenizer, llama) -> Tuple[torch.Tensor, torch.Tensor, List[Tuple[str, List[float]]]]:
    prompts_file = pathlib.Path(cfg["prompts_file"])
    cache_path = pathlib.Path(cfg["embedding_cache"])
    pairs = load_prompts(prompts_file)

    prompts_hash = compute_file_sha1(prompts_file)

    if cache_path.exists():
        embeddings, labels, meta = load_cache(cache_path)
        cache_hash = meta.get("prompts_sha1")
        if embeddings.size(0) != len(pairs) or cache_hash != prompts_hash:
            print(f"cache mismatch, recomputing embeddings: {embeddings.size(0)} != {len(pairs)}")
            embeddings, labels = compute_embeddings(pairs, tokenizer, llama, cfg.get("max_seq_len", 256))
            meta = {
                "prompts_sha1": prompts_hash,
                "num_prompts": len(pairs),
                "model_path": str(cfg["model_path"]),
                "tokenizer_path": str(cfg.get("tokenizer_path", cfg["model_path"])),
                "max_seq_len": int(cfg.get("max_seq_len", 256)),
            }
            save_cache(cache_path, embeddings, labels, meta)
            print(f"saved embeddings to {cache_path}")
        else:
            print(f"loaded cached embeddings from {cache_path}")
    else:
        embeddings, labels = compute_embeddings(pairs, tokenizer, llama, cfg.get("max_seq_len", 256))
        meta = {
            "prompts_sha1": prompts_hash,
            "num_prompts": len(pairs),
            "model_path": str(cfg["model_path"]),
            "tokenizer_path": str(cfg.get("tokenizer_path", cfg["model_path"])),
            "max_seq_len": int(cfg.get("max_seq_len", 256)),
        }
        save_cache(cache_path, embeddings, labels, meta)
        print(f"saved embeddings to {cache_path}")

    return embeddings, labels, pairs


def main(cfg_path: str | None = None):
    if cfg_path:
        cfg_path = pathlib.Path(cfg_path)
    else:
        default_cfg = pathlib.Path("config/ae_deepseek-r1-distill-qwen-7b.json")
        if not default_cfg.exists():
            default_cfg = pathlib.Path("config/ae.json")
        cfg_path = default_cfg
    cfg = apply_output_suffix(load_cfg(cfg_path))
    cfg = normalize_model_paths(cfg)
    torch.manual_seed(int(cfg.get("seed", 42)))

    model_base = pathlib.Path(cfg["model_path"])
    tok_base = pathlib.Path(cfg.get("tokenizer_path", cfg["model_path"]))

    tokenizer = AutoTokenizer.from_pretrained(
        str(tok_base),
        use_fast=True,
        legacy=False,
        local_files_only=True,
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

    embeddings, labels, pairs = ensure_embeddings(cfg, tokenizer, llama)

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    input_dim = embeddings.size(1)

    ae = BottleneckAE(input_dim=input_dim, bottleneck_dim=cfg.get("bottleneck_dim", 6))
    ckpt_path = pathlib.Path(cfg.get("ae_checkpoint", "output/ae_bottleneck.pt"))
    ae.load_state_dict(torch.load(ckpt_path, map_location="cpu"))
    ae = ae.to(device)
    ae.eval()

    target_dtype = next(ae.parameters()).dtype
    embeddings = embeddings.to(device, dtype=target_dtype)

    # Forward pass: obtain bottleneck z (theta) and the regression-head output.
    with torch.no_grad():
        _, preds, z = ae(embeddings)
    preds = preds.cpu().tolist()
    z_vecs = z.cpu().tolist()

    # Write one six-dimensional prediction per prompt as JSONL.
    output_dir = pathlib.Path(cfg.get("output_dir", "output"))
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "persona_hexaco_pred.jsonl"

    with out_path.open("w", encoding="utf-8") as f:
        for (text, label), pred, z_vec in zip(pairs, preds, z_vecs):
            record = {
                "prompt": text,
                "hexaco_label": label,
                "hexaco_pred": pred,
                "hexaco_z": z_vec,
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"saved predictions to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Infer HEXACO scores from persona prompts")
    parser.add_argument("--config", type=str, default=None, help="path to model config json")
    args = parser.parse_args()
    main(cfg_path=args.config)
