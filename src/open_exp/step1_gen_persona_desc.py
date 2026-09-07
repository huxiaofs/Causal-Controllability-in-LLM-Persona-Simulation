"""
Step 1 — Generate one unified natural-language persona description per persona via GPT-4.

This is the ONE thing all 4 local models share. Without it, each model would read
its own self-generated persona prompt → unfair cross-model comparison.

We also derive Prompt A (structured HEXACO) deterministically from the scores.
Both Prompt A and Prompt B are stored in `output/open_exp/persona_prompts.json`
and consumed by `step2_run_open_tasks.py`.

Usage:
    OPENAI_API_KEY=sk-... python -m src.open_exp.step1_gen_persona_desc
"""
from __future__ import annotations

import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from src.open_exp.llm_client import OpenAIClient

DIM_NAMES = {
    "H": "Honesty-Humility",
    "E": "Emotionality",
    "X": "Extraversion",
    "A": "Agreeableness",
    "C": "Conscientiousness",
    "O": "Openness",
}

# Prompt A — structured HEXACO (deterministic, language-model-agnostic)
PROMPT_A_TEMPLATE = (
    "Here is the persona specification (six HEXACO scores on a 1-5 scale; "
    "1=very low, 3=moderate, 5=very high):\n"
    "{lines}\n\n"
    "Answer the following questions as this persona. Your answers should naturally "
    "express extreme and boundary traits without repeating the scores or trait names."
)

# Prompt B — natural language (generated once per persona by GPT-4)
DESC_GEN_SYSTEM = (
    "You are an expert persona writer. Given six HEXACO scores on a 1-5 scale, "
    "write a 60-100 word English persona description. Requirements:\n"
    "1) Express every extreme or boundary trait through observable behavior;\n"
    "2) Do not mention numbers or trait names;\n"
    "3) Describe the person directly without commentary;\n"
    "4) Use second person (you)."
)


def build_prompt_a(hexaco: dict) -> str:
    lines = []
    for k in ["H", "E", "X", "A", "C", "O"]:
        v = hexaco[k]
        if v >= 4.0: tag = "high"
        elif v <= 2.0: tag = "low"
        elif 2.7 <= v <= 3.3: tag = "moderate (boundary)"
        else: tag = "moderate"
        lines.append(f"  - {DIM_NAMES[k]}: {v:.2f} ({tag})")
    return PROMPT_A_TEMPLATE.format(lines="\n".join(lines))


def build_desc_user_prompt(hexaco: dict) -> str:
    score_lines = "\n".join(
        f"  - {DIM_NAMES[k]}: {hexaco[k]:.2f}" for k in ["H","E","X","A","C","O"]
    )
    return f"HEXACO scores:\n{score_lines}\n\nWrite the persona description directly (60-100 words, second person)."


def main():
    personas = json.loads((ROOT / "data/open_exp/personas.json").read_text(encoding="utf-8"))["personas"]
    out_path = ROOT / "output/open_exp/persona_prompts.json"

    client = OpenAIClient(model="gpt-4o", timeout=60)

    result = {"personas": []}
    for p in personas:
        print(f"\n[{p['id']}] {p.get('region', '')}")
        prompt_a = build_prompt_a(p["hexaco"])
        desc_user = build_desc_user_prompt(p["hexaco"])
        prompt_b = client.chat(
            [
                {"role": "system", "content": DESC_GEN_SYSTEM},
                {"role": "user", "content": desc_user},
            ],
            seed=42,
            temperature=0.5,
            max_tokens=300,
        ).strip()
        print(f"  Prompt B: {prompt_b}")
        result["personas"].append({
            "id": p["id"],
            "category": p["category"],
            "region": p["region"],
            "global_idx": p["global_idx"],
            "hexaco": p["hexaco"],
            "prompt_A_structured": prompt_a,
            "prompt_B_natural": prompt_b,
        })

    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nSaved → {out_path}")


if __name__ == "__main__":
    main()
