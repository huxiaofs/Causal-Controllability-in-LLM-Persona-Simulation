# Causal Controllability in LLM Persona Simulation

Official code and public data accompanying:

> **Beyond Output Consistency: Diagnosing Representation-Level Causal Controllability in LLM Persona Simulation**

This repository studies whether a persona specification actually controls a
language model's internal representation and observable behavior, rather than
merely producing text that looks persona-consistent. It implements the
representation-level control chain

```text
persona intervention x  →  internal state θ  →  observable behavior y
```

The main diagnostic is the geometric effective-information measure
`EI_g`, built from two local objects:

- **Intervention geometry** `g(θ)`: whether distinct persona specifications
induce distinguishable internal states;
- **Effect geometry** `h(θ)`: whether internal differences transmit to
questionnaire or open-ended behavior with low distortion.

The code includes controlled synthetic worlds, multi-model HEXACO evaluation,  
permutation and mutual-information baselines, and an optional open-ended  
external-validity pipeline.

## Repository layout

```text
.
├── config/                 # Public model/configuration templates
├── data/
│   ├── persona/            # LHS-sampled HEXACO intervention points
│   ├── questions/          # English HEXACO-60 questionnaire
│   └── open_exp/           # English open-ended tasks and selected personas
├── src/
│   ├── ae/                 # Bottleneck autoencoder and diagnostics
│   ├── open_exp/           # Optional open-ended external-validity pipeline
│   ├── steps/              # Persona generation, scoring, baselines, metrics
│   └── run_full_pipeline.py
├── LICENSE
├── README.md
└── requirements.txt
```

Generated responses, model checkpoints, embeddings, figures, result tables,
and cached API calls are intentionally not part of this public source tree.

## Installation

Python 3.10 or newer is required. A CUDA-capable GPU is recommended for local
LLM inference.

```bash
git clone https://github.com/huxiaofs/Causal-Controllability-in-LLM-Persona-Simulation.git
cd Causal-Controllability-in-LLM-Persona-Simulation
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The configuration files use Hugging Face model identifiers. Edit the model
configuration before running inference if the model is stored locally. Some
open-ended judging steps additionally require an OpenAI-compatible endpoint:

```bash
export OPENAI_API_KEY=YOUR_API_KEY
# Optional: export OPENAI_BASE_URL=...
```



## Public data


| Resource            | Location                                   | Description                                               |
| ------------------- | ------------------------------------------ | --------------------------------------------------------- |
| Persona grid        | `data/persona/hexaco_samples_1-5_500.json` | 500 Latin-hypercube HEXACO points on a 1–5 scale          |
| Questionnaire       | `data/questions/hexaco_60_en.json`         | English 60-item questionnaire and response-scale metadata |
| Open-ended personas | `data/open_exp/personas.json`              | Six selected personas for external-validity tasks         |
| Open-ended tasks    | `data/open_exp/tasks.json`                 | Nine English decision-making scenarios                    |


The English questionnaire and open-ended task bank are release-ready working
translations of the materials used in the experiments. Users should verify
wording and any applicable questionnaire licensing requirements before using
them for a new human study.

## Typical pipeline

Run commands from the repository root. The commands below write generated
artifacts to local `output/` directories, which are not included in the
repository.

### 1. Generate persona prompts

```bash
python -m src.steps.generate_persona_prompts \
  --config config/qwen2.5-7b.json
```



### 2. Train the bottleneck and extract `θ`

```bash
python -m src.ae.train_bottleneck_ae \
  --config config/ae_qwen2.5-7b.json
python -m src.steps.infer_persona_hexaco \
  --config config/ae_qwen2.5-7b.json
```



### 3. Run and score the English questionnaire

```bash
python -m src.steps.run_questionnaire \
  --config config/qwen2.5-7b.json
python -m src.steps.score_questionnaire \
  --config config/ae_qwen2.5-7b.json
```



### 4. Compute `EI_g`

```bash
python -m src.steps.eig_metric \
  --config config/ae_qwen2.5-7b.json
```



### 5. Optional baselines and open-ended validation

```bash
python -m src.steps.compute_baselines_all_models
python -m src.steps.exp2_permutation_test
python -m src.steps.exp3_discriminant_validity
```

The open-ended pipeline is documented in `src/open_exp/README.md`:

```bash
python -m src.open_exp.step1_gen_persona_desc
python -m src.open_exp.step2_run_open_tasks
python -m src.open_exp.step3_judge
python -m src.open_exp.step5_pairwise_separability
python -m src.open_exp.step6_validity_table
```

These steps require local model weights and, for judge calls, an
OpenAI-compatible API. They are external-validity checks; they do not redefine
`EI_g` itself.

## Reproducibility and scope

- Configuration files identify the backbone model and local output locations;
they do not ship model weights.
- The public tree contains source code and input assets, but no experimental
result files or cached generations.
- The questionnaire protocol uses English prompts and the five-point scale
`1 = strongly disagree` through `5 = strongly agree`.
- The open-ended judge protocol uses English decision/rationale prompts and
includes both pairwise and six-way identification utilities.



## Citation

If you use this repository, please cite:

```bibtex
@inproceedings{jiang2025causalcontrollability,
  title     = {Beyond Output Consistency: Diagnosing Representation-Level Causal Controllability in LLM Persona Simulation},
  author    = {Jiang, Chaoyong and Wang, Shengling and Chao, Ke and Hao, Tianyu and Jiang, Kunpeng and Yang, Zhaolin},
  year      = {2025},
  note      = {Code and datasets available at https://github.com/huxiaofs/Causal-Controllability-in-LLM-Persona-Simulation}
}
```



## License and contact

The code is released under the [MIT License](LICENSE). For questions, bug  
reports, or requests concerning the public resources, please open an issue in  
the [GitHub repository](https://github.com/huxiaofs/Causal-Controllability-in-LLM-Persona-Simulation/issues).