"""
Step 2 — Run 4 local models × 6 personas × 2 prompt types × 9 tasks × 2 seeds = 864 generations.

For each (model, persona, prompt_type, task, seed):
    system = persona prompt
    user   = task scenario + decision/rationale template
    -> response

Outputs append to `output/open_exp/responses.jsonl`. Resume-safe: existing
(model, persona, prompt_type, task, seed) tuples are skipped.

To save GPU swap thrashing we iterate the OUTER loop over models (load → run all
sub-loops → unload) instead of switching models on every call.

Usage:
    python -m src.open_exp.step2_run_open_tasks                # all models
    python -m src.open_exp.step2_run_open_tasks --models qwen2.5-7b
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from src.open_exp.llm_client import LocalClient, LOCAL_MODEL_PATHS

ALL_MODELS = list(LOCAL_MODEL_PATHS.keys())
SEEDS = [0, 1]
TEMPERATURE = 0.7
MAX_TOKENS = 600

USER_TEMPLATE = (
    "{scenario}\n\n"
    "Respond in exactly two non-empty sections:\n"
    "Decision:\n"
    "(one or two sentences stating your choice or action)\n\n"
    "Rationale:\n"
    "(3-6 sentences explaining why, in terms of your persona)"
)


def load_done_keys(out_path: pathlib.Path) -> set:
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
    ap.add_argument("--models", nargs="+", default=None,
                    help="Subset of model aliases. Default: all 4.")
    ap.add_argument("--out", default=str(ROOT / "output/open_exp/responses.jsonl"))
    args = ap.parse_args()

    models_to_run = args.models or ALL_MODELS
    out_path = pathlib.Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    personas = json.loads((ROOT / "output/open_exp/persona_prompts.json").read_text(encoding="utf-8"))["personas"]
    tasks = json.loads((ROOT / "data/open_exp/tasks.json").read_text(encoding="utf-8"))["tasks"]

    done = load_done_keys(out_path)
    print(f"Already done: {len(done)} / {len(models_to_run)*len(personas)*2*len(tasks)*len(SEEDS)}")

    fout = out_path.open("a", encoding="utf-8")

    for m in models_to_run:
        if m not in LOCAL_MODEL_PATHS:
            print(f"[skip] unknown model: {m}")
            continue
        # check if any work remains for this model before paying load cost
        pending = [
            (p, pt, t, s)
            for p in personas
            for pt in ["A", "B"]
            for t in tasks
            for s in SEEDS
            if (m, p["id"], pt, t["id"], s) not in done
        ]
        if not pending:
            print(f"[{m}] all done, skip loading.")
            continue

        print(f"\n=== Loading {m} ({len(pending)} pending) ===")
        t0 = time.time()
        client = LocalClient(model_alias=m)
        print(f"  loaded in {time.time()-t0:.1f}s")

        for i, (p, pt, t, seed) in enumerate(pending):
            sys_prompt = p["prompt_A_structured"] if pt == "A" else p["prompt_B_natural"]
            user = USER_TEMPLATE.format(scenario=t["scenario"])
            t0 = time.time()
            try:
                resp = client.chat(
                    [{"role": "system", "content": sys_prompt},
                     {"role": "user", "content": user}],
                    seed=seed,
                    temperature=TEMPERATURE,
                    max_tokens=MAX_TOKENS,
                )
            except Exception as e:
                print(f"  [err] {p['id']}/{pt}/{t['id']}/seed={seed}: {e}")
                continue
            rec = {
                "model": m,
                "persona": p["id"],
                "prompt_type": pt,
                "task_id": t["id"],
                "task_category": t["category"],
                "seed": seed,
                "response": resp,
            }
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fout.flush()
            done.add((m, p["id"], pt, t["id"], seed))
            if (i+1) % 10 == 0 or i == len(pending)-1:
                print(f"  [{i+1}/{len(pending)}] {p['id']}/{pt}/{t['id']}/s{seed}  ({time.time()-t0:.1f}s)")

        client.unload()
        print(f"=== {m} done ===")

    fout.close()
    print(f"\nAll responses → {out_path}")


if __name__ == "__main__":
    main()
