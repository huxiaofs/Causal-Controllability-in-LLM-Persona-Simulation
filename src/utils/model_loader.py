"""Shared model loading utility with optional 4-bit quantization support.

All pipeline scripts that need to load a HuggingFace CausalLM should use
``load_model_and_tokenizer`` so that quantization, dtype, and device-map
handling stay consistent.
"""
from __future__ import annotations

import pathlib
from typing import Any, Dict, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# PyTorch < 2.5 lacks Module.set_submodule, which transformers >= 5.x
# requires for bitsandbytes 4-bit quantization.  Patch a minimal fallback.
if not hasattr(torch.nn.Module, "set_submodule"):
    def _set_submodule(self, target: str, module: "torch.nn.Module") -> None:
        *parents, name = target.split(".")
        parent: "torch.nn.Module" = self
        for p in parents:
            parent = getattr(parent, p)
        setattr(parent, name, module)
    torch.nn.Module.set_submodule = _set_submodule


def resolve_snapshot_path(base_path: pathlib.Path) -> pathlib.Path:
    """Prefer the newest HF cache snapshot; fall back to *base_path*."""
    snap_root = base_path / "snapshots"
    if snap_root.exists():
        snaps = sorted(snap_root.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True)
        for s in snaps:
            if (s / "config.json").exists():
                return s
    return base_path


def normalize_model_paths(cfg: Dict[str, Any]) -> Dict[str, Any]:
    updated = dict(cfg)
    for key in ("model_path", "tokenizer_path"):
        raw = updated.get(key)
        if not raw:
            continue
        updated[key] = str(resolve_snapshot_path(pathlib.Path(raw)))
    return updated


def load_model_and_tokenizer(
    cfg: Dict[str, Any],
    *,
    output_hidden_states: bool = False,
) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    """Load tokenizer + CausalLM from *cfg*.

    Config keys read:
        model_path        (required) – HF id or local path
        tokenizer_path    (optional) – defaults to model_path
        quantization      (optional) – "4bit" to enable NF4 quantization
        max_seq_len       (optional) – not used here but kept for compatibility
    """
    model_base = resolve_snapshot_path(pathlib.Path(cfg["model_path"]))
    tok_base = resolve_snapshot_path(pathlib.Path(cfg.get("tokenizer_path", cfg["model_path"])))

    tokenizer = AutoTokenizer.from_pretrained(
        str(tok_base),
        use_fast=True,
        legacy=False,
        local_files_only=True,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    torch_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    if torch.cuda.is_available():
        device_map = {"": 0} if torch.cuda.device_count() == 1 else "auto"
    else:
        device_map = None

    quantization = str(cfg.get("quantization", "")).strip().lower()
    load_kwargs: Dict[str, Any] = dict(
        torch_dtype=torch_dtype,
        device_map=device_map,
        local_files_only=True,
    )
    if output_hidden_states:
        load_kwargs["output_hidden_states"] = True

    if quantization in ("4bit", "4-bit", "nf4"):
        try:
            from transformers import BitsAndBytesConfig
        except ImportError as exc:
            raise ImportError(
                "4-bit quantization requires bitsandbytes. "
                "Install with: pip install bitsandbytes accelerate"
            ) from exc
        # transformers 4.46 loads 4-bit weights incrementally, so a single
        # GPU suffices even for Mixtral-8x7B (~25GB in 4-bit on an 80GB A800).
        # Single-GPU inference avoids the crippling inter-GPU communication
        # overhead of multi-GPU dispatch during autoregressive generation.
        load_kwargs["device_map"] = {"": 0}
        load_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch_dtype,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
        # dtype is ignored when quantization_config is set; remove to avoid warning
        load_kwargs.pop("torch_dtype", None)

    model = AutoModelForCausalLM.from_pretrained(str(model_base), **load_kwargs)
    return model, tokenizer
