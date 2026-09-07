import argparse
import json
import pathlib
from typing import Dict, List, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_config(cfg_path: pathlib.Path) -> Dict:
    with cfg_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def apply_output_suffix(cfg: Dict) -> Dict:
    suffix = str(cfg.get("output_suffix", "")).strip()
    if not suffix:
        return cfg
    base = str(cfg.get("output_dir", "output"))
    if not base.endswith(f"_{suffix}") and not base.endswith(f"/{suffix}"):
        base = f"{base}_{suffix}"
    cfg["output_dir"] = base
    return cfg


def resolve_snapshot_path(base_path: pathlib.Path) -> pathlib.Path:
    """
    Prefer the newest snapshot under Hugging Face cache; fall back to the base directory.
    This allows using paths like models--meta-llama--Meta-Llama-3-8B-Instruct.
    """
    snapshot_root = base_path / "snapshots"
    if snapshot_root.exists():
        snapshots = sorted(snapshot_root.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True)
        for snap in snapshots:
            if (snap / "config.json").exists():
                return snap
    return base_path


def normalize_model_paths(cfg: Dict) -> Dict:
    updated = dict(cfg)
    for key in ("model_path", "tokenizer_path"):
        raw = updated.get(key)
        if not raw:
            continue
        updated[key] = str(resolve_snapshot_path(pathlib.Path(raw)))
    return updated


def format_persona(persona: Dict) -> str:
    return (
        f"Honesty-Humility (H): {persona['Honesty-Humility']}, "
        f"Emotionality (E): {persona['Emotionality']}, "
        f"Extraversion (X): {persona['Extraversion']}, "
        f"Agreeableness (A): {persona['Agreeableness']}, "
        f"Conscientiousness (C): {persona['Conscientiousness']}, "
        f"Openness (O): {persona['Openness']}"
    )


def build_messages(persona_desc: str) -> List[Dict[str, str]]:
    system_prompt = "You are an expert prompt engineer who creates persona-conditioning prompts for language models."
    user_prompt = (
        "Using the following six HEXACO trait scores, write three distinct persona prompts in English.\n"
        "Requirements:\n"
        "1) Keep each prompt under 60 words;\n"
        "2) Use different styles (formal, conversational, and scenario-based);\n"
        "3) Express extreme and boundary traits through concrete observable behavior;\n"
        "4) Number them #1, #2, and #3;\n"
        "5) Output only the three prompts.\n"
        f"Scores: {persona_desc}"
    )
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


def generate_prompts_for_persona(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    persona: Dict,
    cfg: Dict,
) -> List[str]:
    persona_desc = format_persona(persona)
    messages = build_messages(persona_desc)
    inputs = tokenizer.apply_chat_template(messages, return_tensors="pt")

    target_device = next(model.parameters()).device
    inputs = inputs.to(target_device)

    outputs = model.generate(
        input_ids=inputs,
        max_new_tokens=int(cfg.get("max_new_tokens", 128)),
        temperature=float(cfg.get("temperature", 0.8)),
        top_p=float(cfg.get("top_p", 0.9)),
        do_sample=True,
        num_return_sequences=3,
        pad_token_id=tokenizer.eos_token_id,
    )

    gen_tokens = outputs[:, inputs.shape[-1] :]
    texts = tokenizer.batch_decode(gen_tokens, skip_special_tokens=True)
    cleaned = [t.strip() for t in texts if t.strip()]
    return cleaned[:3]


def _read_existing_ids(output_file: pathlib.Path) -> List[int]:
    if not output_file.exists():
        return []
    ids: List[int] = []
    with output_file.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
                if "id" in row:
                    ids.append(int(row["id"]))
            except Exception:
                continue
    return ids


def main(max_personas: Optional[int] = None, cfg_path: Optional[str] = None):
    if cfg_path:
        cfg_path = pathlib.Path(cfg_path)
    else:
        default_cfg = pathlib.Path("config/deepseek-r1-distill-qwen-7b.json")
        if not default_cfg.exists():
            default_cfg = pathlib.Path("config/llama3.json")
        cfg_path = default_cfg
    cfg = apply_output_suffix(load_config(cfg_path))
    cfg = normalize_model_paths(cfg)

    persona_path = pathlib.Path(cfg["persona_file"])
    personas = json.loads(persona_path.read_text(encoding="utf-8"))
    if max_personas is not None:
        personas = personas[: max(0, int(max_personas))]

    output_dir = pathlib.Path(cfg.get("output_dir", "output"))
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / cfg.get("output_file", "persona_prompts.jsonl")

    if cfg.get("seed") is not None:
        torch.manual_seed(int(cfg["seed"]))

    model_base = pathlib.Path(cfg["model_path"])
    tok_base = pathlib.Path(cfg.get("tokenizer_path", cfg["model_path"]))

    tokenizer = AutoTokenizer.from_pretrained(tok_base, local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    torch_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        model_base,
        torch_dtype=torch_dtype,
        device_map="auto" if torch.cuda.is_available() else None,
        local_files_only=True,
    )

    existing_ids = _read_existing_ids(output_file)
    start_idx = max(existing_ids) + 1 if existing_ids else 0
    if start_idx >= len(personas):
        print(f"all prompts already generated: {start_idx}/{len(personas)}")
        return

    mode = "a" if output_file.exists() else "w"
    with output_file.open(mode, encoding="utf-8") as f:
        for idx in range(start_idx, len(personas)):
            persona = personas[idx]
            prompts = generate_prompts_for_persona(model, tokenizer, persona, cfg)
            record = {"id": idx, "persona": persona, "prompts": prompts}
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

            if (idx + 1) % 10 == 0 or (idx + 1) == len(personas):
                print(f"generated {idx + 1}/{len(personas)} personas")

    print(f"all prompts saved to {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate persona prompts")
    parser.add_argument("--config", type=str, default=None, help="path to model config json")
    parser.add_argument("--max-personas", type=int, default=None, help="limit number of personas")
    args = parser.parse_args()
    main(max_personas=args.max_personas, cfg_path=args.config)
