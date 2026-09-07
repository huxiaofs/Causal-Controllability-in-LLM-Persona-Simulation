"""
Sanitize raw responses.jsonl so that the pairwise-separability judge sees a
fair, comparable input across all 4 models.

Three cleaning operations are applied:

1) **Strip reasoning traces.**  DeepSeek-R1-Distill-Qwen-7B is a reasoning
   model whose default output is `<think>...</think> answer`.  When the
   `</think>` tag is present, only the text after it is the user-facing
   answer; we drop everything up to and including `</think>`.  If the model
   forgot to close the tag (the response may begin with a reasoning trace and
   run out of `max_tokens` before
   emitting `</think>`), we apply a heuristic rule: drop everything before
   the first occurrence of 'Decision:' (the answer
   template marker), since the reasoning trace by construction precedes the
   structured answer.

2) **Mask self-references to persona vocabulary.**  Across all four models,
   we redact phrases that explicitly cite HEXACO dimension names or the
    explicit persona-reference phrases, because those phrases let the judge
   identify the persona via lexical leakage rather than behavioural cues.
   This is a *conservative* rewrite: only short token-level redactions, the
   surrounding sentence is left intact.

3) **Truncate empty responses.**  After (1) and (2), if the cleaned text is
   shorter than 30 characters, fall back to the original answer-marker
   substring.

Output:
    output/open_exp/responses_cleaned.jsonl

The cleaned file has the same schema as `responses.jsonl` plus a
`response_raw` field that preserves the original.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import re
import sys
from collections import Counter

ROOT = pathlib.Path(__file__).resolve().parent.parent.parent

THINK_OPEN_RE = re.compile(r"<think>", flags=re.IGNORECASE)
THINK_CLOSE_RE = re.compile(r"</think>", flags=re.IGNORECASE)
ANSWER_MARKERS = ["Decision:", "Decision\n", "**Decision**"]

# Lexical leakage patterns: HEXACO dimension names and explicit references.
HEXACO_DIM_TOKENS = [
    "Honesty-Humility", "Emotionality", "Extraversion", "Agreeableness",
    "Conscientiousness", "Openness",
    "HEXACO",
]
META_REF_TOKENS = [
    "my persona specification", "my persona profile", "my personality",
    "given my persona", "given my personality",
]


def strip_reasoning_trace(text: str) -> str:
    """Remove DeepSeek-style <think>...</think> reasoning prefix.

    Robust to (a) full open+close tags, (b) only-close tag, (c) only-open
    tag (missing close), (d) no tags at all but obvious answer-marker.
    """
    s = text

    # case A: explicit </think> closer (works whether <think> open is there or not)
    m_close = THINK_CLOSE_RE.search(s)
    if m_close is not None:
        s = s[m_close.end():].lstrip()
        return s

    # case B: only <think> open without close (model ran out of tokens or
    # forgot).  We strip the open marker; the rest is reasoning trace and
    # will be handled by case D below.
    m_open = THINK_OPEN_RE.search(s)
    if m_open is not None:
        s = s[m_open.end():].lstrip()

    # case D: text begins with a reasoning preamble — pure
    # reasoning trace.  Heuristic: cut at the first answer-template marker.
    for mk in ANSWER_MARKERS:
        idx = s.find(mk)
        if idx > 0:
            return s[idx:].strip()
    return s.strip()


def mask_lexical_leakage(text: str) -> str:
    """Redact tokens that name HEXACO dims or directly cite the persona meta.

    We replace each match with '[persona trait]' so the sentence still reads but
    the lexical signal is removed.
    """
    out = text
    for tok in HEXACO_DIM_TOKENS + META_REF_TOKENS:
        out = out.replace(tok, "[persona trait]")
    # collapse repeats
    out = re.sub(r"(\[persona trait\][,.;: ]*){2,}", "[persona trait], ",
                 out)
    return out


def sanitize_one(rec: dict) -> dict:
    raw = rec["response"]
    s = strip_reasoning_trace(raw)
    s = mask_lexical_leakage(s)
    if len(s) < 30:
        # fall back to the raw answer-marker tail
        for mk in ANSWER_MARKERS:
            idx = raw.find(mk)
            if idx >= 0:
                s_alt = mask_lexical_leakage(raw[idx:])
                if len(s_alt) >= 30:
                    s = s_alt
                    break
    out = dict(rec)
    out["response_raw"] = raw
    out["response"] = s
    out["sanitised"] = (s != raw)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_path",
                    default=str(ROOT / "output/open_exp/responses.jsonl"))
    ap.add_argument("--out", default=str(ROOT / "output/open_exp/responses_cleaned.jsonl"))
    args = ap.parse_args()

    in_path = pathlib.Path(args.in_path)
    out_path = pathlib.Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rows = [json.loads(l) for l in in_path.open("r", encoding="utf-8")]
    print(f"input rows: {len(rows)}")

    n_changed = Counter()
    n_total = Counter()
    n_short = 0
    cleaned = []
    avg_raw_len = Counter()
    avg_clean_len = Counter()
    cnt_per_model = Counter()
    for r in rows:
        cleaned_r = sanitize_one(r)
        cleaned.append(cleaned_r)
        n_total[r["model"]] += 1
        cnt_per_model[r["model"]] += 1
        if cleaned_r["sanitised"]:
            n_changed[r["model"]] += 1
        if len(cleaned_r["response"]) < 30:
            n_short += 1
        avg_raw_len[r["model"]] += len(r["response"])
        avg_clean_len[r["model"]] += len(cleaned_r["response"])

    with out_path.open("w", encoding="utf-8") as f:
        for r in cleaned:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"\nSanitisation summary:")
    print(f"{'Model':35s}  {'changed':>9s}  {'len_raw':>8s} -> {'len_clean':>9s}")
    for m, n in n_total.items():
        rl = avg_raw_len[m] / n
        cl = avg_clean_len[m] / n
        print(f"  {m:33s}  {n_changed[m]:>5d}/{n:<3d}  {rl:>8.0f} -> {cl:>9.0f}")
    print(f"\n  responses with cleaned length < 30 chars: {n_short}")
    print(f"\nSaved -> {out_path}")


if __name__ == "__main__":
    main()
