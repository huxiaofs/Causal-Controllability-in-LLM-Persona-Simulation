"""
Unified LLM client supporting OpenAI API + local HuggingFace models.

OpenAIClient: GPT-4o for persona description generation and judging.
LocalClient:  load any of the 4 local models for open-ended response generation.

All calls support seed/temperature/max_tokens; cache results to disk so we
can resume after interruption without re-running expensive inferences.
"""
from __future__ import annotations

import hashlib
import json
import os
import pathlib
import time
from dataclasses import dataclass
from typing import List, Optional

CACHE_ROOT = pathlib.Path(__file__).resolve().parent.parent.parent / "output" / "open_exp" / ".cache"


def _cache_key(prefix: str, model: str, messages: list, seed: int, temperature: float) -> str:
    payload = json.dumps(
        {"model": model, "messages": messages, "seed": seed, "t": temperature},
        ensure_ascii=False,
        sort_keys=True,
    )
    return f"{prefix}_{hashlib.sha256(payload.encode()).hexdigest()[:32]}"


def _cache_get(key: str) -> Optional[str]:
    p = CACHE_ROOT / f"{key}.txt"
    if p.exists():
        return p.read_text(encoding="utf-8")
    return None


def _cache_put(key: str, value: str) -> None:
    CACHE_ROOT.mkdir(parents=True, exist_ok=True)
    (CACHE_ROOT / f"{key}.txt").write_text(value, encoding="utf-8")


# ═══════════════════════════════════════════════════════════
#  OpenAI client (used for persona-desc generation & judging)
# ═══════════════════════════════════════════════════════════
@dataclass
class OpenAIClient:
    model: str = "gpt-4o"
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    max_retries: int = 5
    timeout: int = 120

    def __post_init__(self):
        from openai import OpenAI
        key = self.api_key or os.environ.get("OPENAI_API_KEY")
        if not key:
            raise RuntimeError("OPENAI_API_KEY not set")
        url = self.base_url or os.environ.get("OPENAI_BASE_URL")
        kwargs = {"api_key": key, "timeout": self.timeout}
        if url:
            kwargs["base_url"] = url
        self._client = OpenAI(**kwargs)

    def chat(
        self,
        messages: List[dict],
        seed: int = 0,
        temperature: float = 0.7,
        max_tokens: int = 1024,
        response_format: Optional[dict] = None,
        use_cache: bool = True,
    ) -> str:
        if use_cache:
            key = _cache_key("oai", self.model, messages, seed, temperature)
            cached = _cache_get(key)
            if cached is not None:
                return cached

        last_err = None
        for attempt in range(self.max_retries):
            try:
                kwargs = {
                    "model": self.model,
                    "messages": messages,
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                    "seed": seed,
                }
                if response_format:
                    kwargs["response_format"] = response_format
                resp = self._client.chat.completions.create(**kwargs)
                text = resp.choices[0].message.content
                if use_cache:
                    _cache_put(key, text)
                return text
            except Exception as e:
                last_err = e
                wait = 2 ** attempt
                print(f"  [openai retry {attempt+1}/{self.max_retries}] {type(e).__name__}: {e}; wait {wait}s")
                time.sleep(wait)
        raise RuntimeError(f"OpenAI failed after {self.max_retries} retries: {last_err}")


# ═══════════════════════════════════════════════════════════
#  Local HF model client (used for open-ended task generation)
# ═══════════════════════════════════════════════════════════
LOCAL_MODEL_PATHS = {
    "qwen2.5-7b": "Qwen/Qwen2.5-7B-Instruct",
    "deepseek-r1-distill-qwen-7b": "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
    "mistral-7b-instruct-v0.3": "mistralai/Mistral-7B-Instruct-v0.3",
    "llama3.1-8b": "meta-llama/Meta-Llama-3.1-8B-Instruct",
    "mixtral-8x7b": "mistralai/Mixtral-8x7B-Instruct-v0.1",
    "llama3.1-70b": "meta-llama/Meta-Llama-3.1-70B-Instruct",
    "qwen3-30b-a3b": "Qwen/Qwen3-30B-A3B",
}


def _resolve_snapshot(base: str) -> str:
    base_p = pathlib.Path(base)
    snap_root = base_p / "snapshots"
    if snap_root.exists():
        snaps = sorted(snap_root.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True)
        for snap in snaps:
            if (snap / "config.json").exists():
                return str(snap)
    return str(base_p)


def _local_files_only(model_path: str) -> bool:
    """Use offline weights only when model_path points to an existing local directory."""
    return pathlib.Path(model_path).exists()


@dataclass
class LocalClient:
    model_alias: str
    dtype: str = "bfloat16"
    device: str = "auto"

    def __post_init__(self):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        path = _resolve_snapshot(LOCAL_MODEL_PATHS[self.model_alias])
        offline = _local_files_only(path)
        self.tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=offline)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        # Use 4-bit quantization for large models that don't fit in a single
        # 80GB GPU in bfloat16 (70B ~140GB, 8x7B ~93GB BF16).
        needs_4bit = self.model_alias in ("mixtral-8x7b", "llama3.1-70b", "qwen3-30b-a3b")
        if needs_4bit and torch.cuda.is_available():
            from transformers import BitsAndBytesConfig
            torch_dtype = torch.bfloat16
            self.model = AutoModelForCausalLM.from_pretrained(
                path,
                torch_dtype=torch_dtype,
                device_map={"": 0},
                local_files_only=offline,
                quantization_config=BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch_dtype,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True,
                ),
            )
        else:
            torch_dtype = torch.bfloat16 if self.dtype == "bfloat16" else torch.float32
            self.model = AutoModelForCausalLM.from_pretrained(
                path,
                torch_dtype=torch_dtype,
                device_map=self.device if torch.cuda.is_available() else None,
                local_files_only=offline,
            )
        self.model.eval()
        self._torch = torch

    def chat(
        self,
        messages: List[dict],
        seed: int = 0,
        temperature: float = 0.7,
        max_tokens: int = 768,
        top_p: float = 0.9,
        use_cache: bool = True,
    ) -> str:
        if use_cache:
            key = _cache_key(f"local_{self.model_alias}", self.model_alias, messages, seed, temperature)
            cached = _cache_get(key)
            if cached is not None:
                return cached

        torch = self._torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        # Adapt messages for models whose chat template doesn't support
        # the system role (e.g. Mixtral).
        try:
            from src.utils.chat_compat import adapt_messages
        except ImportError:
            from utils.chat_compat import adapt_messages
        messages = adapt_messages(self.tokenizer, messages)

        input_ids = self.tokenizer.apply_chat_template(
            messages, return_tensors="pt", add_generation_prompt=True
        ).to(self.model.device)

        with torch.no_grad():
            out = self.model.generate(
                input_ids=input_ids,
                max_new_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id,
            )
        gen = out[0, input_ids.shape[-1]:]
        text = self.tokenizer.decode(gen, skip_special_tokens=True).strip()
        if use_cache:
            _cache_put(key, text)
        return text

    def unload(self):
        del self.model
        import gc
        gc.collect()
        if self._torch.cuda.is_available():
            self._torch.cuda.empty_cache()
