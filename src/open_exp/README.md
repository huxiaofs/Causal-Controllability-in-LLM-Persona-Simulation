# Open-Ended Behavior Experiment

Acts as the **external validity check** for the global controllability metric
`EI_g`. Rather than claiming `EI_g` linearly predicts a behavior score, we
ask whether models with higher `EI_g` produce more **distinguishable** and
**stable** persona-conditioned behavior on open-ended tasks.

Three behavior-side metrics:

1. **Alignment** (RAS, 1-5 GPT-4o judge): single-response fidelity to the
   target HEXACO persona.  Already produced by `step3_judge.py`.
2. **Behavioral Separability** (pairwise mapping accuracy): for each
   `(model, task)`, the judge sees two responses and two persona painters and
   must map them.  Higher = more discriminable behavior under different
   persona interventions.  Produced by `step5_pairwise_separability.py`.
3. **Behavioral Stability** (1 - normalised RAS std across seeds): how
   consistent is a model's RAS within fixed persona × task × prompt across
   sampling seeds.  Computed in `step6_validity_table.py` from existing
   judgements.

## Configuration

- **4 models** (local HF):  `qwen2.5-7b`, `mistral-7b-instruct-v0.3`, `llama3.1-8b`, `deepseek-r1-distill-qwen-7b`
- **6 personas**  (`data/open_exp/personas.json`):
  | id | category | region |
  |----|----------|--------|
  | P1_HighX_LowH | A_divergent | High-X,Low-H |
  | P2_NearO | A_divergent | Near-O |
  | P3_NearA | B_boundary | Near-A |
  | P4_NearE | B_boundary | Near-E |
  | P5_NearH | B_boundary | Near-H (gid=556 — replaced original duplicate) |
  | P6_HighE_LowC | C_hard | High-E,Low-C |
- **2 prompt types**: A = structured HEXACO, B = GPT-4-generated natural language
- **9 tasks** (`data/open_exp/tasks.json`): A1–A3 resource, B1–B3 moral, C1–C3 social
- **2 seeds**: 0, 1 (T=0.7)
- **Total**: 4 × 6 × 2 × 9 × 2 = **864 generations** + 864 GPT-4 judgments

## Pipeline

```bash
cd /path/to/Causal-Controllability-in-LLM-Persona-Simulation
export OPENAI_API_KEY=YOUR_API_KEY

# Step 1 — generate one unified natural-language description per persona  (~6 GPT-4 calls)
python -m src.open_exp.step1_gen_persona_desc

# Step 2 — local model generation                                          (~3 hours for 4 models)
python -m src.open_exp.step2_run_open_tasks
# or per-model:
python -m src.open_exp.step2_run_open_tasks --models qwen2.5-7b

# Step 3 — GPT-4o judging                                                  (~864 GPT-4 calls, ~$5-10)
python -m src.open_exp.step3_judge

# Step 4 — aggregate + plot 4 legacy figures (still useful)
python -m src.open_exp.step4_analyze

# Step 5 — pairwise persona-discrimination judge (Behavioral Separability)
#   default: 4 models × 9 tasks × 15 pairs × 2 swaps = 1080 GPT-4o calls (~$5-10)
#   add --no_swap to halve cost; add --limit 30 for a smoke test.
python -m src.open_exp.step5_pairwise_separability

# Step 6 — integrate EI_g vs Separability / Alignment / Stability
#   no LLM calls; produces validity_table.csv/json and fig5_external_validity.png
python -m src.open_exp.step6_validity_table
```

All steps are **resume-safe** — interrupted runs pick up from `.cache/` (LLM calls)
or by skipping existing rows in `responses.jsonl` / `judgments.jsonl` /
`separability.jsonl`.

## Outputs

```
output/open_exp/
  persona_prompts.json       # Step 1: A + B prompts per persona
  responses.jsonl            # Step 2: 864 responses
  judgments.jsonl            # Step 3: 864 RAS/BSS/RCS scores
  scores.json                # Step 4: aggregated per (model, persona)
  separability.jsonl         # Step 5: pairwise judgements
  separability.json          # Step 5: aggregated per model / per pair / per task
  validity_table.csv         # Step 6: external-validity summary table
  validity_table.json        # Step 6: rows + Spearman ρ matrix
  figures/
    fig1_behavior_heatmap.png        # Model × Persona Score
    fig2_q_vs_score.png              # (deprecated) q vs Score scatter
    fig3_model_level.png             # (deprecated) EI_g / MI vs Mean Score
    fig4_neighbor_discrimination.png # Near-X performance + spread
    fig5_external_validity.png       # EI_g vs Separability/Alignment + ρ heatmap
  .cache/                            # OpenAI / local LLM call cache
```

## Paper § 4.3.2 – External Validity (talking points)

The recommended write-up *avoids* the framing "EI_g linearly predicts behavior
score".  Instead, the section answers a reviewer question: *is `EI_g` only
about the questionnaire protocol, or does it also reflect open-ended
behavior?*

> To test external validity beyond the questionnaire protocol, we evaluate
> model behavior on open-ended persona tasks. The task scores are not the
> definition of controllability; they are an external-validity check. If EI_g
> captures effective control of persona specifications, higher-EI_g models
> should produce more distinguishable and more stable behavior under different
> persona interventions.

The primary external-validity claim is:
**EI_g should rank-correlate with Behavioral Separability**, because both
measure how spread out the persona-conditioned mapping is.  Alignment is a
complementary single-point metric and Stability is a robustness check.
