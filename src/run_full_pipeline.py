import argparse
import json
import pathlib
import sys

from steps.generate_persona_prompts import main as gen_prompts
from steps.infer_persona_hexaco import main as infer_hexaco
from steps.run_questionnaire import main as run_questionnaire
from steps.score_questionnaire import main as score_questionnaire
from steps.eig_metric import main as eig_metric


def should_skip(path: pathlib.Path, force: bool) -> bool:
    return path.exists() and not force


def _load_cfg(path: str | None, fallback: dict) -> dict:
    if not path:
        return fallback
    cfg_path = pathlib.Path(path)
    if not cfg_path.exists():
        return fallback
    return json.loads(cfg_path.read_text(encoding="utf-8"))


def apply_output_suffix(cfg: dict) -> dict:
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


def main():
    parser = argparse.ArgumentParser(description="Run full HEXACO persona pipeline")
    parser.add_argument("--force", action="store_true", help="rerun all steps even if outputs exist")
    parser.add_argument("--max-personas", type=int, default=None, help="limit number of personas in pipeline")
    parser.add_argument(
        "--prompts-config",
        type=str,
        default=None,
        help="config for prompt generation (default: config/llama3.json)",
    )
    parser.add_argument(
        "--ae-config",
        type=str,
        default=None,
        help="config for AE/inference/questionnaire (default: config/ae.json)",
    )
    args = parser.parse_args()

    prompts_cfg = apply_output_suffix(_load_cfg(
        args.prompts_config,
        {"output_dir": "output", "output_file": "persona_prompts.jsonl"},
    ))
    ae_cfg = apply_output_suffix(_load_cfg(
        args.ae_config,
        {"output_dir": "output"},
    ))

    prompts_out = pathlib.Path(prompts_cfg.get("output_dir", "output")) / prompts_cfg.get(
        "output_file", "persona_prompts.jsonl"
    )
    ae_out_dir = pathlib.Path(ae_cfg.get("output_dir", "output"))

    steps = [
        (
            "prompts",
            lambda: gen_prompts(max_personas=args.max_personas, cfg_path=args.prompts_config),
            prompts_out,
        ),
        (
            "infer_hexaco",
            lambda: infer_hexaco(cfg_path=args.ae_config),
            ae_out_dir / "persona_hexaco_pred.jsonl",
        ),
        (
            "questionnaire",
            lambda: run_questionnaire(cfg_path=args.ae_config),
            ae_out_dir / "questionnaire_answers.jsonl",
        ),
        (
            "score",
            lambda: score_questionnaire(cfg_path=args.ae_config),
            ae_out_dir / "questionnaire_scores.jsonl",
        ),
        (
            "eig_metric",
            lambda: eig_metric(cfg_path=args.ae_config),
            ae_out_dir / "eig_metric_result.json",
        ),
    ]

    for name, fn, out_path in steps:
        if should_skip(out_path, args.force):
            print(f"[skip] {name}: {out_path} already exists")
            continue
        print(f"[run] {name} ...")
        try:
            fn()
        except SystemExit:
            # allow argparse in submodules to exit cleanly
            raise
        except Exception as e:
            print(f"[fail] {name}: {e}")
            sys.exit(1)
        print(f"[done] {name}")

    print("pipeline finished")


if __name__ == "__main__":
    main()
