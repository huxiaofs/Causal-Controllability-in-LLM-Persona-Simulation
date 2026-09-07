"""
Step 5 — Pairwise persona-discrimination judge for Behavioral Separability.

Goal:
    For each (model, task), pick one canonical response per persona, then for
    every persona pair (i, j) ask GPT-4o to map two anonymous responses back
    to the two persona descriptions.  The pairwise accuracy averaged over
    tasks is the model's Separability score.

Why this metric:
    Higher EI_g means the persona-conditioned representation theta is spread
    out more in the bottleneck space.  If EI_g is a real geometric handle on
    persona controllability, models with higher EI_g should produce more
    distinguishable open-ended behavior under different persona interventions.
    A pairwise mapping accuracy directly measures *behavioral separability*.

Selection rule for the canonical response:
    canonical = (prompt_type=="B", seed=0).  Prompt B is the natural-language
    persona painting, which is what we want to test --- A is structured and
    explicitly cites the dimensions, so it artificially inflates separability.
    seed=0 is the first sampled draw.  We optionally also run seed=1 (toggle
    --both_seeds) to give Stability a paired counterpart.

Output:
    output/open_exp/separability.jsonl  (one row per pairwise judgement)
    output/open_exp/separability.json   (aggregated)

Calls:
    Default:  4 models x 9 tasks x C(6,2)=15 pairs = 540 GPT-4o calls.
    Each pair is asked twice with response order swapped to neutralise
    position bias, so total calls = 1080.  Cost ~$5-10.

Usage:
    OPENAI_API_KEY=sk-... python -m src.open_exp.step5_pairwise_separability
    python -m src.open_exp.step5_pairwise_separability --both_seeds
    python -m src.open_exp.step5_pairwise_separability --models qwen2.5-7b
"""
from __future__ import annotations

import argparse
import itertools
import json
import pathlib
import random
import sys
from collections import defaultdict

ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from src.open_exp.llm_client import OpenAIClient

ALL_MODELS = [
    "qwen2.5-7b",
    "deepseek-r1-distill-qwen-7b",
    "mistral-7b-instruct-v0.3",
    "llama3.1-8b",
]

JUDGE_SYSTEM = """\
You are a strict persona-matching judge. Below are two HEXACO personas (X and Y)
and two responses to the same open-ended task (R1 and R2). Each response came
from one of the personas. Determine the mapping. Compare H/E/X/A/C/O, including
extreme (≤2 or ≥4) and boundary (2.7-3.3) scores. If the responses are
indistinguishable, make your best mapping and use low confidence.

Output only JSON:
{
  "r1": "X" or "Y",
  "r2": "X" or "Y",
  "confidence": 1-5,
  "brief": "brief reason in at most 30 words"
}
Do not output anything else."""


def build_judge_user(task_meta: dict,
                     persona_x: dict, persona_y: dict,
                     resp_first: str, resp_second: str) -> str:
    return (
        f"=== Task ===\n{task_meta['scenario']}\n\n"
        f"=== Persona X ===\n"
        f"HEXACO scores: {format_hexaco(persona_x['hexaco'])}\n"
        f"Description: {persona_x['prompt_B_natural']}\n\n"
        f"=== Persona Y ===\n"
        f"HEXACO scores: {format_hexaco(persona_y['hexaco'])}\n"
        f"Description: {persona_y['prompt_B_natural']}\n\n"
        f"=== Response R1 ===\n{resp_first}\n\n"
        f"=== Response R2 ===\n{resp_second}\n\n"
        f"=== Mapping (JSON only) ==="
    )


def format_hexaco(h: dict) -> str:
    return ", ".join(f"{k}={h[k]:.2f}" for k in ["H", "E", "X", "A", "C", "O"])


def load_done(out_path: pathlib.Path) -> set:
    if not out_path.exists():
        return set()
    keys = set()
    with out_path.open("r", encoding="utf-8") as f:
        for line in f:
            try:
                r = json.loads(line)
                keys.add((r["model"], r["task_id"], r["persona_x"],
                          r["persona_y"], r["seed"], r["swap"]))
            except Exception:
                continue
    return keys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--responses", default=str(ROOT / "output/open_exp/responses.jsonl"))
    ap.add_argument("--out", default=str(ROOT / "output/open_exp/separability.jsonl"))
    ap.add_argument("--judge_model", default="gpt-4o")
    ap.add_argument("--prompt_type", default="B",
                    help="A=structured, B=natural; B is the default.")
    ap.add_argument("--both_seeds", action="store_true",
                    help="Run for seed=0 AND seed=1; default seed=0 only.")
    ap.add_argument("--no_swap", action="store_true",
                    help="Skip the order-swap repeat (halves cost, reintroduces position bias).")
    ap.add_argument("--models", nargs="+", default=None)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    personas = json.loads(
        (ROOT / "output/open_exp/persona_prompts.json").read_text(encoding="utf-8"))["personas"]
    persona_by_id = {p["id"]: p for p in personas}
    tasks = json.loads(
        (ROOT / "data/open_exp/tasks.json").read_text(encoding="utf-8"))["tasks"]

    in_path = pathlib.Path(args.responses)
    out_path = pathlib.Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    responses = [json.loads(l) for l in in_path.open("r", encoding="utf-8")]

    # index: (model, persona, prompt_type, task_id, seed) -> response
    resp_idx = {}
    for r in responses:
        key = (r["model"], r["persona"], r["prompt_type"], r["task_id"], r["seed"])
        resp_idx[key] = r["response"]

    models_to_run = args.models or ALL_MODELS
    seeds_to_run = [0, 1] if args.both_seeds else [0]
    swaps_to_run = [0] if args.no_swap else [0, 1]

    persona_ids = [p["id"] for p in personas]
    pairs = list(itertools.combinations(persona_ids, 2))

    done = load_done(out_path)
    fout = out_path.open("a", encoding="utf-8")

    rng = random.Random(42)

    client = OpenAIClient(model=args.judge_model, timeout=60)

    plan = []
    for m in models_to_run:
        for t in tasks:
            for seed in seeds_to_run:
                for (pid_x, pid_y) in pairs:
                    for swap in swaps_to_run:
                        key = (m, t["id"], pid_x, pid_y, seed, swap)
                        if key in done:
                            continue
                        plan.append((m, t, seed, pid_x, pid_y, swap))
    if args.limit:
        plan = plan[:args.limit]
    print(f"pending pairwise judgements: {len(plan)} "
          f"(models={len(models_to_run)}, tasks={len(tasks)}, "
          f"pairs={len(pairs)}, seeds={len(seeds_to_run)}, swaps={len(swaps_to_run)})")

    n_done = 0
    for (m, t, seed, pid_x, pid_y, swap) in plan:
        rx = resp_idx.get((m, pid_x, args.prompt_type, t["id"], seed))
        ry = resp_idx.get((m, pid_y, args.prompt_type, t["id"], seed))
        if rx is None or ry is None:
            print(f"  [skip-missing] {m}/{t['id']}/{pid_x}-{pid_y}/seed={seed}")
            continue

        # swap=0  -> R1=rx (true=X), R2=ry (true=Y)
        # swap=1  -> R1=ry (true=Y), R2=rx (true=X)
        if swap == 0:
            r_first, r_second = rx, ry
            true_r1, true_r2 = "X", "Y"
        else:
            r_first, r_second = ry, rx
            true_r1, true_r2 = "Y", "X"

        try:
            raw = client.chat(
                [
                    {"role": "system", "content": JUDGE_SYSTEM},
                    {"role": "user", "content": build_judge_user(
                        t, persona_by_id[pid_x], persona_by_id[pid_y],
                        r_first, r_second)},
                ],
                seed=42,
                temperature=0.0,
                max_tokens=200,
                response_format={"type": "json_object"},
            )
            j = json.loads(raw)
            r1_pred = str(j["r1"]).upper().strip()
            r2_pred = str(j["r2"]).upper().strip()
            assert r1_pred in {"X", "Y"} and r2_pred in {"X", "Y"}
            confidence = int(j.get("confidence", 3))
        except Exception as e:
            print(f"  [err] {m}/{t['id']}/{pid_x}-{pid_y}/seed={seed}/swap={swap}: {e}")
            continue

        correct_r1 = (r1_pred == true_r1)
        correct_r2 = (r2_pred == true_r2)
        # pair-level "correct" requires BOTH responses correctly mapped (the
        # judge cannot get full credit by guessing one and picking the
        # complement, because we still demand the second mapping).  This is
        # the standard 2-alternative forced-choice criterion.
        consistent = (r1_pred != r2_pred)
        pair_correct = correct_r1 and correct_r2

        out = {
            "model": m,
            "task_id": t["id"],
            "task_category": t["category"],
            "persona_x": pid_x,
            "persona_y": pid_y,
            "seed": seed,
            "swap": swap,
            "prompt_type": args.prompt_type,
            "r1_pred": r1_pred,
            "r2_pred": r2_pred,
            "r1_correct": correct_r1,
            "r2_correct": correct_r2,
            "consistent": consistent,
            "pair_correct": pair_correct,
            "confidence": confidence,
            "brief": j.get("brief", ""),
        }
        fout.write(json.dumps(out, ensure_ascii=False) + "\n")
        fout.flush()

        n_done += 1
        if n_done % 25 == 0 or n_done == len(plan):
            print(f"  [{n_done}/{len(plan)}]  last={m}/{t['id']}/{pid_x}-{pid_y}")

    fout.close()
    print(f"\nDone -> {out_path}")
    aggregate_and_save(out_path)


def aggregate_and_save(jsonl_path: pathlib.Path) -> None:
    """Aggregate pair-level results into per-model and per-(model,task) scores."""
    rows = [json.loads(l) for l in jsonl_path.open("r", encoding="utf-8")]
    if not rows:
        print("[aggregate] no rows; skip.")
        return

    by_model = defaultdict(list)
    by_model_task = defaultdict(list)
    by_model_pair = defaultdict(list)
    consistent_by_model = defaultdict(list)
    confidence_by_model = defaultdict(list)

    for r in rows:
        by_model[r["model"]].append(r["pair_correct"])
        by_model_task[(r["model"], r["task_id"])].append(r["pair_correct"])
        by_model_pair[(r["model"], r["persona_x"], r["persona_y"])].append(r["pair_correct"])
        consistent_by_model[r["model"]].append(r["consistent"])
        confidence_by_model[r["model"]].append(r["confidence"])

    def mean(xs):
        return float(sum(xs) / len(xs)) if xs else float("nan")

    sep = {
        "per_model": {
            m: {
                "separability": mean(vs),
                "consistency": mean(consistent_by_model[m]),
                "judge_confidence_mean": mean(confidence_by_model[m]),
                "n_pairs": len(vs),
            }
            for m, vs in by_model.items()
        },
        "per_model_task": {
            f"{m}|{t}": {"separability": mean(vs), "n": len(vs)}
            for (m, t), vs in by_model_task.items()
        },
        "per_model_pair": {
            f"{m}|{px}-{py}": {"separability": mean(vs), "n": len(vs)}
            for (m, px, py), vs in by_model_pair.items()
        },
        "n_total_judgements": len(rows),
    }
    out_json = jsonl_path.with_suffix(".json")
    out_json.write_text(json.dumps(sep, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Aggregated -> {out_json}")
    print("\n=== Separability (pairwise accuracy) per model ===")
    for m, d in sorted(sep["per_model"].items(), key=lambda x: -x[1]["separability"]):
        print(f"  {m:35s}  sep={d['separability']:.3f}  cons={d['consistency']:.3f}  "
              f"conf={d['judge_confidence_mean']:.2f}  n={d['n_pairs']}")


if __name__ == "__main__":
    main()
