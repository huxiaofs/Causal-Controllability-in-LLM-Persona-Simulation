"""
Step 5b — N-way Persona-Identification judge for a continuous, non-saturating
Behavioral Separability score.

Why this exists:
    The pairwise judge in step5 is a 2-AFC task (chance = 0.5).  GPT-4o
    saturates at >= 0.93 on all four models, which (a) hides true
    differences between models and (b) makes the resulting Spearman
    correlation collapse to a handful of integer values when there are only
    n=4 models.  The N-way variant replaces 2-AFC with **6-AFC** (chance
    = 1/6 = 0.167), gives the judge the entire candidate set, and asks it to
    score each candidate from 0 to 5.  We can then read out three separability
    signals from the same call:

        top1_acc        = 1 if argmax(scores) == ground-truth persona, else 0
        margin          = score[true] - max(score[other_personas])
        prob_correct    = softmax(scores)[true]      (continuous in (0, 1))
        log_prob_correct= log of the above           (continuous in (-inf, 0])

    `prob_correct` and `log_prob_correct` are continuous, do not saturate,
    and produce well-separated per-model means even when the judge is sure
    about most cases.

Calls:
    4 models × 6 personas × 9 tasks × 2 seeds = 432 GPT-4o calls.  Cost
    roughly $3-5, 8-12 minutes.

Outputs:
    output/open_exp/identification.jsonl     (one row per judgement)
    output/open_exp/identification.json      (aggregated per model / task)

Usage:
    OPENAI_API_KEY=sk-... python -m src.open_exp.step5b_nway_identification
    python -m src.open_exp.step5b_nway_identification --responses output/open_exp/responses_cleaned.jsonl
"""
from __future__ import annotations

import argparse
import json
import math
import pathlib
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
You are a strict persona-matching judge. Below are one open-task response and six
candidate HEXACO personas (P1-P6). The response came from exactly one candidate.
Give each candidate a 0-5 match score (5=exact match, 0=no match), then select
one best_match. Compare H/E/X/A/C/O, including extreme (≤2 or ≥4) and boundary
(2.7-3.3) scores. Multiple candidates may receive high scores when ambiguous,
but best_match must be unique.

Output only JSON:
{
  "scores": {"P1": 0-5, "P2": 0-5, "P3": 0-5, "P4": 0-5, "P5": 0-5, "P6": 0-5},
  "best_match": "P1"|"P2"|"P3"|"P4"|"P5"|"P6",
  "confidence": 1-5,
  "brief": "brief reason in at most 30 words"
}
Do not output anything else."""


def format_hexaco(h: dict) -> str:
    return ", ".join(f"{k}={h[k]:.2f}" for k in ["H", "E", "X", "A", "C", "O"])


def build_judge_user(task_meta: dict, candidates: list[dict],
                     response: str) -> str:
    cand_block = []
    for i, p in enumerate(candidates, start=1):
        cand_block.append(
            f"P{i} (HEXACO: {format_hexaco(p['hexaco'])}):\n"
            f"  {p['prompt_B_natural']}"
        )
    return (
        f"=== Task ===\n{task_meta['scenario']}\n\n"
        f"=== Candidate personas ===\n" + "\n\n".join(cand_block) + "\n\n"
        f"=== Response to identify ===\n{response}\n\n"
        f"=== Score (JSON only) ==="
    )


def softmax_from_scores(scores_dict: dict, ids: list[str]) -> dict:
    # GPT-4o emits 0-5 integer scores; we treat them as logits *5 to spread
    # them out, then softmax.  Tau=2.0 keeps probabilities in a moderate
    # range and avoids one-hot collapse on perfectly-confident judgements.
    tau = 2.0
    vals = [float(scores_dict.get(p, 0)) for p in ids]
    m = max(vals)
    exps = [math.exp((v - m) / tau) for v in vals]
    s = sum(exps)
    if s <= 0:
        return {p: 1.0 / len(ids) for p in ids}
    return {p: exps[i] / s for i, p in enumerate(ids)}


def load_done(out_path: pathlib.Path) -> set:
    if not out_path.exists():
        return set()
    keys = set()
    with out_path.open("r", encoding="utf-8") as f:
        for line in f:
            try:
                r = json.loads(line)
                keys.add((r["model"], r["task_id"], r["true_persona"], r["seed"]))
            except Exception:
                continue
    return keys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--responses", default=str(ROOT / "output/open_exp/responses_cleaned.jsonl"))
    ap.add_argument("--out", default=str(ROOT / "output/open_exp/identification.jsonl"))
    ap.add_argument("--judge_model", default="gpt-4o")
    ap.add_argument("--prompt_type", default="B")
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1])
    ap.add_argument("--models", nargs="+", default=None)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    personas = json.loads(
        (ROOT / "output/open_exp/persona_prompts.json").read_text(encoding="utf-8"))["personas"]
    persona_by_id = {p["id"]: p for p in personas}
    persona_ids = [p["id"] for p in personas]
    n_pers = len(persona_ids)
    cand_slots = [f"P{i}" for i in range(1, n_pers + 1)]

    tasks = json.loads(
        (ROOT / "data/open_exp/tasks.json").read_text(encoding="utf-8"))["tasks"]

    in_path = pathlib.Path(args.responses)
    out_path = pathlib.Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    responses = [json.loads(l) for l in in_path.open("r", encoding="utf-8")]
    resp_idx = {}
    for r in responses:
        key = (r["model"], r["persona"], r["prompt_type"], r["task_id"], r["seed"])
        resp_idx[key] = r["response"]

    models_to_run = args.models or ALL_MODELS
    done = load_done(out_path)
    fout = out_path.open("a", encoding="utf-8")
    client = OpenAIClient(model=args.judge_model, timeout=60)

    # We *fix* the candidate ordering across all calls so the judge cannot
    # game positional cues differently per call.  Order is the canonical
    # personas.json order.
    candidate_list = [persona_by_id[pid] for pid in persona_ids]

    plan = []
    for m in models_to_run:
        for t in tasks:
            for seed in args.seeds:
                for true_pid in persona_ids:
                    key = (m, t["id"], true_pid, seed)
                    if key in done:
                        continue
                    plan.append((m, t, seed, true_pid))
    if args.limit:
        plan = plan[:args.limit]
    print(f"pending identifications: {len(plan)} "
          f"(models={len(models_to_run)}, tasks={len(tasks)}, "
          f"personas={n_pers}, seeds={len(args.seeds)})")

    n_done = 0
    for (m, t, seed, true_pid) in plan:
        rkey = (m, true_pid, args.prompt_type, t["id"], seed)
        resp = resp_idx.get(rkey)
        if resp is None:
            print(f"  [skip-missing] {rkey}")
            continue
        try:
            raw = client.chat(
                [
                    {"role": "system", "content": JUDGE_SYSTEM},
                    {"role": "user", "content": build_judge_user(
                        t, candidate_list, resp)},
                ],
                seed=42,
                temperature=0.0,
                max_tokens=300,
                response_format={"type": "json_object"},
            )
            j = json.loads(raw)
            scores = {k: int(v) for k, v in j["scores"].items()}
            best = str(j["best_match"]).upper().strip()
            confidence = int(j.get("confidence", 3))
            assert all(0 <= v <= 5 for v in scores.values())
            assert best in cand_slots
        except Exception as e:
            print(f"  [err] {m}/{t['id']}/{true_pid}/seed={seed}: {e}")
            continue

        true_slot = cand_slots[persona_ids.index(true_pid)]
        probs = softmax_from_scores(scores, cand_slots)
        p_correct = float(probs[true_slot])
        log_p = float(math.log(max(p_correct, 1e-9)))
        # margin = score(true) - max(score(other))
        score_true = float(scores[true_slot])
        score_other = max(float(scores[s]) for s in cand_slots if s != true_slot)
        margin = score_true - score_other
        top1_correct = (best == true_slot)
        # also record the rank of the true persona
        ranking = sorted(cand_slots, key=lambda s: -scores[s])
        true_rank = ranking.index(true_slot) + 1  # 1 = best

        out = {
            "model": m,
            "task_id": t["id"],
            "task_category": t["category"],
            "true_persona": true_pid,
            "seed": seed,
            "prompt_type": args.prompt_type,
            "scores": scores,
            "best_match": best,
            "best_match_pid": persona_ids[cand_slots.index(best)],
            "confidence": confidence,
            "p_correct": p_correct,
            "log_p_correct": log_p,
            "margin": margin,
            "top1_correct": top1_correct,
            "true_rank": true_rank,
            "brief": j.get("brief", ""),
        }
        fout.write(json.dumps(out, ensure_ascii=False) + "\n")
        fout.flush()
        n_done += 1
        if n_done % 25 == 0 or n_done == len(plan):
            print(f"  [{n_done}/{len(plan)}]  last={m}/{t['id']}/{true_pid}/s{seed}")

    fout.close()
    print(f"\nDone -> {out_path}")
    aggregate_and_save(out_path)


def aggregate_and_save(jsonl_path: pathlib.Path) -> None:
    rows = [json.loads(l) for l in jsonl_path.open("r", encoding="utf-8")]
    if not rows:
        print("[aggregate] no rows.")
        return

    by_model = defaultdict(list)
    by_model_task = defaultdict(list)
    by_model_pers = defaultdict(list)
    for r in rows:
        m = r["model"]
        by_model[m].append(r)
        by_model_task[(m, r["task_id"])].append(r)
        by_model_pers[(m, r["true_persona"])].append(r)

    def agg(rs):
        n = len(rs)
        if n == 0:
            return {"n": 0}
        top1 = sum(r["top1_correct"] for r in rs) / n
        p_corr = sum(r["p_correct"] for r in rs) / n
        log_p = sum(r["log_p_correct"] for r in rs) / n
        margin = sum(r["margin"] for r in rs) / n
        mean_rank = sum(r["true_rank"] for r in rs) / n
        mrr = sum(1.0 / r["true_rank"] for r in rs) / n  # mean reciprocal rank
        return {
            "n": n,
            "top1_acc": float(top1),
            "p_correct_mean": float(p_corr),
            "log_p_correct_mean": float(log_p),
            "margin_mean": float(margin),
            "mean_rank": float(mean_rank),
            "mrr": float(mrr),
        }

    sep = {
        "per_model": {m: agg(rs) for m, rs in by_model.items()},
        "per_model_task": {f"{m}|{t}": agg(rs) for (m, t), rs in by_model_task.items()},
        "per_model_persona": {f"{m}|{p}": agg(rs) for (m, p), rs in by_model_pers.items()},
        "n_total": len(rows),
    }
    out_json = jsonl_path.with_suffix(".json")
    out_json.write_text(json.dumps(sep, ensure_ascii=False, indent=2),
                        encoding="utf-8")
    print(f"Aggregated -> {out_json}")
    print("\n=== 6-way Persona Identification per model ===")
    for m, d in sorted(sep["per_model"].items(), key=lambda x: -x[1]["p_correct_mean"]):
        print(f"  {m:35s}  top1={d['top1_acc']:.3f}  "
              f"p_correct={d['p_correct_mean']:.3f}  "
              f"log_p={d['log_p_correct_mean']:+.3f}  "
              f"margin={d['margin_mean']:+.3f}  "
              f"mrr={d['mrr']:.3f}  n={d['n']}")


if __name__ == "__main__":
    main()
