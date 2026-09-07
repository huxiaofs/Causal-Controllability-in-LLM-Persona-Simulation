import argparse
import json
import pathlib
import sys
from collections import defaultdict
from typing import Dict, Iterable, List, Tuple, Union

ROOT = pathlib.Path(__file__).resolve().parent
PARENT = ROOT.parent
for p in (ROOT, PARENT):
    if str(p) not in sys.path:
        sys.path.append(str(p))

from steps.hexaco_scoring import score_hexaco


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


def load_answers(path: pathlib.Path) -> List[Dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


def group_by_prompt(answers: List[Dict]) -> Dict[Tuple[int, int], Dict[int, int]]:
    """Aggregate as {(persona_id, prompt_idx): {question_id: score}}."""
    grouped: Dict[Tuple[int, int], Dict[int, int]] = defaultdict(dict)
    for row in answers:
        pid = row["persona_id"]
        pidx = row["prompt_idx"]
        qid = row["question_id"]
        score = row.get("score")
        if score is None:
            continue
        grouped[(pid, pidx)][qid] = int(score)
    return grouped


def main(cfg_path: str | None = None):
    output_dir = "output"
    if cfg_path:
        cfg_file = pathlib.Path(cfg_path)
    else:
        cfg_file = pathlib.Path("config/deepseek-r1-distill-qwen-7b.json")
        if not cfg_file.exists():
            cfg_file = pathlib.Path("config/ae.json")
    if cfg_file.exists():
        cfg = apply_output_suffix(json.loads(cfg_file.read_text(encoding="utf-8")))
        output_dir = cfg.get("output_dir", output_dir)

    answers_path = pathlib.Path(output_dir) / "questionnaire_answers.jsonl"
    grouped = group_by_prompt(load_answers(answers_path))

    out_dir = pathlib.Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "questionnaire_scores.jsonl"

    with out_path.open("w", encoding="utf-8") as fout:
        for (pid, pidx), resp in grouped.items():
            try:
                scores = score_hexaco(resp, round_facets=2, round_dimensions=1, allow_missing=False)
            except Exception as e:
                scores = {"error": str(e)}
            record = {
                "persona_id": pid,
                "prompt_idx": pidx,
                "items_answered": len(resp),
                "scores": scores,
            }
            fout.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"saved scores to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Score HEXACO questionnaire")
    parser.add_argument("--config", type=str, default=None, help="path to model config json")
    args = parser.parse_args()
    main(cfg_path=args.config)
