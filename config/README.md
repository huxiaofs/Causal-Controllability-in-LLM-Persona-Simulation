# Configuration

Each JSON file sets `model_path` and `tokenizer_path` to a Hugging Face model ID or a local directory.
Edit paths before running experiments on your machine.

| File | Model |
|------|-------|
| `qwen2.5-7b.json` | Qwen/Qwen2.5-7B-Instruct |
| `deepseek-r1-distill-qwen-7b.json` | deepseek-ai/DeepSeek-R1-Distill-Qwen-7B |
| `mistral-7b-instruct-v0.3.json` | mistralai/Mistral-7B-Instruct-v0.3 |
| `ae_*.json` | Same backbone + AE / EI_g pipeline settings |

Set `output_suffix` to isolate per-model artifacts (e.g. `output_qwen2.5-7b/`).
