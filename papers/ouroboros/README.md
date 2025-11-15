# Ouroboros: Human‑Led Recursive Reinforcement Learning (HLRR) for Autoregressive LLMs

[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](#)
[![TRL](https://img.shields.io/badge/TRL-PPO%2FDPO-black.svg)](#)
[![Docs](https://img.shields.io/badge/Docs-README.md-informational.svg)](#)

> **TL;DR**: Ouroboros is a **continuous, human‑in‑the‑loop alignment loop**. You converse with a model, distill the interaction into **facts / reasoning / persona**, transform those into **labyrinth prompts**, score the model’s regenerations with a rubric, and apply **policy updates** (PPO or DPO). Rinse, repeat. The loop **amplifies limited human oversight** into robust gains in **coherence, persona fidelity, and task generalization**.

---

## What’s new (2025 update)

- **Two training modes**: classic **PPO‑RLHF** and lighter‑weight **DPO** (Direct Preference Optimization).  
- **RLAIF/Constitutional option**: use **AI feedback** and rule‑based self‑critiques when human labels are scarce.  
- **Reward shaping guards**: built‑in hooks for **bounded rewards** and **preference‑matching regularizers** (to curb reward hacking / preference collapse).  
- **Evaluator pack**: scripts to measure **win‑rate (pairwise ELO)**, **coherence under paraphrase**, **persona consistency**, and **safety rule adherence**.  
- **Reproducible configs** via `configs/*.yaml` and deterministic seeds where possible.

> This README supersedes the original overview and keeps the spirit and scope intact while expanding the pipeline and tooling.

---

## Method: the Ouroboros loop

```text
┌──────────────────────────────────────────────────────────────────────────────┐
│ 1) Dialogue: human ↔ LLM                                                     │
│ 2) Distill: structured summary {facts, logical flow, persona snapshot}       │
│ 3) Labyrinth prompts: rephrase/adversarial/contrastive challenges            │
│ 4) Regenerate: model produces candidates (top‑p/temperature/beam)            │
│ 5) Score: rubric‑based rating (+ optional AI‑judge / rule checks)            │
│ 6) Policy update: PPO (with KL & shaping)  or  DPO one‑stage optimization    │
│ 7) Replay/curriculum: add hard cases; iterate until convergence              │
└──────────────────────────────────────────────────────────────────────────────┘
```

### Why it works
- **Human summaries** capture *what matters* (logic + persona) with minimal annotation burden.  
- **Labyrinth prompts** pressure‑test consistency under paraphrase and long horizons.  
- **Iterative preference learning** (PPO or DPO) steadily reshapes the policy toward the rubric without overfitting to single prompts.  
- **RLAIF** scales oversight by letting a policy learn from **AI critics** guided by a constitution or ruleset when humans aren’t in the loop.

---

## Repository layout (suggested)

```
.
├── configs/                 # YAML configs for data/ppo/dpo/eval
├── data/                    # (Your) dialogs, summaries, scores, jsonl shards
├── scripts/
│   ├── 01_prepare_data.py   # build summaries & labyrinth prompts
│   ├── 02_generate.py       # batch regeneration from a base model
│   ├── 03_score.py          # rubric scoring (human UI + AI‑judge helpers)
│   ├── 04_train_ppo.py      # TRL‑PPO training loop
│   ├── 04_train_dpo.py      # DPO training loop
│   └── 05_eval.py           # win‑rate, coherence, persona, safety
├── models/                  # checkpoints, adapters (LoRA/QLoRA), reward models
├── README.md                # this file
└── ouroboros.tex            # paper (compile‑ready; see below)
```

> **Note**: If these files aren’t present yet in your clone, the scripts serve as *reference entry points*. Start with your own data and wire the configs accordingly.

---

## Quickstart

### 0) Environment
```bash
# Python 3.10+ recommended
python -m venv .venv && source .venv/bin/activate
pip install -U pip wheel

# Minimal deps (pin to your infra)
pip install torch transformers datasets accelerate bitsandbytes
pip install trl peft evaluate sentencepiece nltk
pip install wandb rich pydantic pyyaml
```

### 1) Data format
We operate on **JSONL** shards for each stage.

- **Dialog shard** (`data/dialogs.jsonl`):
```json
{"dialog_id": "abc123",
 "turns": [{"role": "human", "text": "…"}, {"role": "assistant", "text": "…"}]}
```

- **Summary shard** (`data/summaries.jsonl`):
```json
{"dialog_id":"abc123",
 "facts":["…","…"],
 "logic":["premise …","therefore …"],
 "persona":{"voice":"technical, concise","constraints":["no speculation"]}}
```

- **Labyrinth prompts** (`data/labyrinth.jsonl`):
```json
{"dialog_id":"abc123","prompt":"Rephrase-with-ambiguity: …","target":"preserve facts A,B; stay in voice"}
```

- **Scored candidates** (`data/scores.jsonl`):
```json
{"dialog_id":"abc123","prompt_id":7,"candidate":"…","score":{"helpful":4,"honest":5,"persona":5},
 "notes":"missed constraint B","rater":"human|ai"}
```

### 2) Regenerate candidates
```bash
python scripts/02_generate.py \
  model=Qwen2.5-7B-Instruct  data=data/labyrinth.jsonl out=data/candidates.jsonl \
  decoding.top_p=0.95 decoding.temperature=0.7 n_candidates=4
```

### 3A) Train with PPO (classic RLHF)
```bash
python scripts/04_train_ppo.py \
  policy.base=Qwen2.5-7B-Instruct reward.model=your-reward-model \
  ppo.kl_coeff=0.05 ppo.batch_size=64 ppo.epochs=3 shaping.bound_reward=true
```
- Supports **KL penalty**, **reward bounding**, and **reference‑model shaping** to reduce reward‑hacking.
- Works with full‑precision or **(Q)LoRA** adapters for memory‑efficient finetuning.

### 3B) Train with DPO (one‑stage preference)
```bash
python scripts/04_train_dpo.py \
  policy.base=Qwen2.5-7B-Instruct \
  dpo.beta=0.1 dpo.pair_source=data/pairs.jsonl
```
- No on‑policy rollouts; **stable and lightweight** when you have labeled pairs or AI‑judged preferences.

### 4) Evaluate
```bash
python scripts/05_eval.py ckpt=outputs/ppo_last \
  eval.winrate=true eval.persona=true eval.coherence=true eval.safety=true
```
- **Win‑rate (ELO)**: pairwise preference vs. baseline.  
- **Coherence**: paraphrase‑invariance / contradiction checks.  
- **Persona**: rubric‑based LLM‑as‑judge (voice, constraints, style).  
- **Safety**: rule adherence / refusal‑with‑explanation.

> Tip: Log everything to **Weights & Biases** (or your internal tracker) for ablations and regressions.

---

## Using AI feedback (RLAIF / Constitutional mode)

Add a **constitution** (principles/rules) and enable *self‑critique → revise → prefer‑best*:
```yaml
rlaif:
  enabled: true
  principles:
    - "Be helpful and honest; cite uncertainty."
    - "Refuse harmful requests with a brief explanation."
  critique_chain: 2           # # of critique→revise passes
  judge_model: "your-judge"   # AI preference model for pairwise selection
```
This reduces human labeling load; keep humans in the loop for **spot checks** and **hard cases**.

---

## Guardrails against common failures

- **Reward hacking**: enable **bounded rewards** and reference‑relative shaping.  
- **Preference collapse**: add **preference‑matching regularizers** and diversity checks.  
- **Sycophancy**: include counter‑examples in data; reward **polite disagreement** when facts are at stake.  
- **Forgetting**: use KL/reference penalties and occasional **PTX‑mixing** of pretraining text.  

---

## Paper (LaTeX) & longform

The canonical research draft lives in `ouroboros.tex`. If you prefer to iterate in Markdown, keep `paper/ouroboros.md` and convert with `pandoc`:

```bash
pandoc paper/ouroboros.md -o paper/ouroboros.pdf --pdf-engine=xelatex
```

When ready, tag a GitHub release and let **Zenodo** mint a DOI. Include the DOI badge here.

---

## Roadmap

- Multi‑teacher variant (human + AI judges + rule checkers)  
- Curriculum auto‑construction from failure clusters  
- Long‑context memory for persona stability  
- Open benchmark suite for **coherence / persona**

---

## Citation

If you use Ouroboros in research, please cite:

```bibtex
@misc{ison2025ouroboros,
  title   = {Ouroboros: Human-Led Recursive Reinforcement Learning for Autoregressive Language Models},
  author  = {Ison, Payton and Asari},
  year    = {2025},
  note    = {Versioned release; see Zenodo for DOI},
  howpublished = {\url{https://github.com/paytonison/ouroboros}}
}
```

---

## License

MIT — see [LICENSE](LICENSE).

## Contact

- Email: isonpayton@gmail.com
- GitHub: https://github.com/paytonison
- LinkedIn: https://linkedin.com/in/paytonison

> “The ouroboros eats only itself, yet in doing so it is eternally fed.”
