"""
Step 3 — GPT-4o judge scores each response on RAS / BSS / RCS (1-5).

  RAS = Role Alignment Score: whether the response fits the persona
  BSS = Behavioral Specificity: whether the behavior is distinctive
  RCS = Reasoning Coherence: whether decision and rationale are coherent

Inputs:  output/open_exp/responses.jsonl
Outputs: output/open_exp/judgments.jsonl  (resume-safe)

The judge is given:
    - HEXACO scores (numeric, full info)
    - both Prompt A and Prompt B (so it knows both reference framings)
    - the task scenario
    - the response

Output is forced to JSON via response_format={"type":"json_object"}.

Usage:
    OPENAI_API_KEY=sk-... python -m src.open_exp.step3_judge
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from src.open_exp.llm_client import OpenAIClient

JUDGE_SYSTEM = """\
You are a strict persona-alignment judge. Given a HEXACO persona, an open-ended
task, and a model response, score these dimensions independently (integer 1-5;
5=excellent, 1=poor):

【RAS】Role Alignment Score — fit in content, wording, and values.
  Compare H/E/X/A/C/O, including extreme (≤2 or ≥4) and boundary (2.7-3.3) scores.
  - 5: all six traits are precise and independently inferable;
  - 4: major traits are accurate, with 1-2 ambiguous;
  - 3: broadly aligned, but several traits are missing or mildly conflicting;
  - 2: clearly inconsistent with the specification;
  - 1: no persona evidence or the opposite direction.

【BSS】Behavioral Specificity Score — concrete, identifiable, personalized details.
  - 5: highly personalized details, clearly unlike a neutral response;
  - 3: some distinctive details but still generic;
  - 1: fully templated and compatible with any persona.

【RCS】Reasoning Coherence Score — whether the decision and rationale are coherent.
  - 5: tightly self-consistent;
  - 3: broadly connected but with logical gaps;
  - 1: contradictory or entirely vague.

Output only JSON:
{
  "RAS": 1-5,
  "BSS": 1-5,
  "RCS": 1-5,
  "brief_reason": "brief reason in at most 30 words"
}
Do not output anything else."""


def build_judge_user(persona_meta: dict, task_meta: dict, response: str) -> str:
    h = persona_meta["hexaco"]
    score_str = ", ".join(f"{k}={h[k]:.2f}" for k in ["H","E","X","A","C","O"])
    return (
        f"=== Persona ===\n"
        f"HEXACO scores: {score_str}\n"
        f"Description: {persona_meta['prompt_B_natural']}\n\n"
        f"=== Task ===\n"
        f"{task_meta['scenario']}\n\n"
        f"=== Model response ===\n"
        f"{response}\n\n"
        f"=== Score (JSON only) ==="
    )


def load_done(out_path: pathlib.Path) -> set:
    if not out_path.exists():
        return set()
    keys = set()
    with out_path.open("r", encoding="utf-8") as f:
        for line in f:
            try:
                r = json.loads(line)
                keys.add((r["model"], r["persona"], r["prompt_type"], r["task_id"], r["seed"]))
            except Exception:
                continue
    return keys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--responses", default=str(ROOT / "output/open_exp/responses.jsonl"))
    ap.add_argument("--out", default=str(ROOT / "output/open_exp/judgments.jsonl"))
    ap.add_argument("--judge_model", default="gpt-4o")
    ap.add_argument("--limit", type=int, default=None, help="debug: only judge first N")
    args = ap.parse_args()

    personas = {p["id"]: p for p in json.loads(
        (ROOT / "output/open_exp/persona_prompts.json").read_text(encoding="utf-8"))["personas"]}
    tasks = {t["id"]: t for t in json.loads(
        (ROOT / "data/open_exp/tasks.json").read_text(encoding="utf-8"))["tasks"]}

    in_path = pathlib.Path(args.responses)
    out_path = pathlib.Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    done = load_done(out_path)
    records = [json.loads(l) for l in in_path.open("r", encoding="utf-8")]
    pending = [r for r in records
               if (r["model"], r["persona"], r["prompt_type"], r["task_id"], r["seed"]) not in done]
    if args.limit:
        pending = pending[:args.limit]
    print(f"Total responses: {len(records)}; already judged: {len(done)}; pending: {len(pending)}")

    client = OpenAIClient(model=args.judge_model, timeout=60)
    fout = out_path.open("a", encoding="utf-8")

    for i, r in enumerate(pending):
        p_meta = personas[r["persona"]]
        t_meta = tasks[r["task_id"]]
        try:
            raw = client.chat(
                [{"role": "system", "content": JUDGE_SYSTEM},
                 {"role": "user", "content": build_judge_user(p_meta, t_meta, r["response"])}],
                seed=42,
                temperature=0.0,
                max_tokens=200,
                response_format={"type": "json_object"},
            )
            j = json.loads(raw)
            for k in ["RAS", "BSS", "RCS"]:
                j[k] = int(j[k])
                assert 1 <= j[k] <= 5
            score = (j["RAS"] + j["BSS"] + j["RCS"]) / 3.0
        except Exception as e:
            print(f"  [err] {r['model']}/{r['persona']}/{r['prompt_type']}/{r['task_id']}/s{r['seed']}: {e}")
            continue

        out = {
            "model": r["model"],
            "persona": r["persona"],
            "prompt_type": r["prompt_type"],
            "task_id": r["task_id"],
            "task_category": r["task_category"],
            "seed": r["seed"],
            "RAS": j["RAS"], "BSS": j["BSS"], "RCS": j["RCS"],
            "score": score,
            "brief_reason": j.get("brief_reason", ""),
        }
        fout.write(json.dumps(out, ensure_ascii=False) + "\n")
        fout.flush()

        if (i+1) % 25 == 0 or i == len(pending)-1:
            print(f"  [{i+1}/{len(pending)}]")

    fout.close()
    print(f"\nDone → {out_path}")


if __name__ == "__main__":
    main()
