# Payton Ison
**Researcher · Systems Designer · Writer**

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.17254818.svg)](https://doi.org/10.5281/zenodo.17254818) [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.17138445.svg)](https://doi.org/10.5281/zenodo.17138445) [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.17157330.svg)](https://doi.org/10.5281/zenodo.17157330) [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.17346405.svg)](https://doi.org/10.5281/zenodo.17346405) [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.17548497.svg)](https://doi.org/10.5281/zenodo.17548497)

---

## What I Do (and Why It Matters)

- **Action:** Design control loops, interfaces, and evaluation methods that stabilize LLM behavior and translate intent into deterministic steps.
- **Impact:** Reproducible scaffolds across model families, clearer failure modes, better telemetry.
- **Why it matters:** Progress isn’t just scale — it’s structure.

---

### Product Insight & Systems Feedback

Beyond research and systems design, I’ve participated in long-horizon product feedback loops with major platforms.  
These interactions focus not on UX aesthetics, but on *systems-level behavior*—how models, assistants, and AI ecosystems evolve under real-world use.  

- **Scope:** Provided detailed, context-rich feedback during early pricing, feature, and architecture evaluations for major AI and consumer platforms.  
- **Focus:** Behavior calibration, feedback-loop design, contextual awareness, and cross-ecosystem integration.  
- **Impact:** Helped validate and shape early subscription frameworks and contextual AI behavior models now reflected in production systems.  
- **Approach:** Treat feedback as architectural telemetry—interpreting signals from product behavior to improve underlying system structure and responsiveness.

---

## Highlights

- **Ouroboros – human-led recursive reinforcement**  
  Iterate a model against its own conversation history with a human in the loop to turn history into stable capability—no bespoke fine-tuning pipeline required.  
  `papers/ouroboros/`

- **Resonant Feedback Effect (RFE)**  
  Phase-locking case study for GPT-4-class models under sustained interaction; frames interaction-driven gains beyond one-shot prompts.  
  `papers/rfe/`

- **Super Mario – LLM-controlled platformer agent**  
  Real-time action selection via structured state + prompt policies; browser and Pygame implementations with heuristic fallbacks.  
  `experiments/super-mario/`

- **Muse – micro-AGI on tiny, curated data**  
  A small fine-tune + reflection loop that shows where structure beats scale.  
  `experiments/muse/`

- **Operation I, Robot – early scaffold**  
  Prototype of think→reflect→act with persistent memory; pre-“reasoning-model” era.  
  `experiments/i-robot/`

- **Erebus (Rust) – morpheme tokenizer & embedding matrix**  
  Splits words into morphemes and derives a 5-D feature vector from dictionary definitions, averaged into a morpheme embedding matrix.  
  `experiments/erebus/`

- **Mixis – recursive identity training demo**  
  Ouroboros-style loop that distills conversations into reflections and re-ingests them to sharpen persona; human-in-the-loop steering.  
  `experiments/mixis/`

- **MoE Lab – minimal expert tuner (experimental)**  
  Small MoE/FFN expert-tuning kit with router-bias controls and delta-only export; intended for controlled ablations.  
  `experiments/gpt-oss/`

> See `docs/` for high-level documentation, notes, and running lab updates.

---

## Methods & Tools

- Human-in-the-loop recursion (Ouroboros)
- Structured state & deterministic tool use
- Prompt/interface design (persona, protocol JSON)
- Rapid Python prototyping (local-first)
- Rust (edition 2024) & Cargo for lexical/embedding experiments
- TeX for papers; shell for ops hygiene
- Reproducible evals & ablations
- Autocritical self-training (Self-Study / ARR)
- Resonance-aware telemetry and governance (RFE, Simulated Intimacy)
- Structured routing, judges, and hysteresis controllers (Magi × GPT-6, Asari Brainstem)

---

## Research Throughlines

- **Structure over scale:** Conditional routers, mid-turn switching, and persistent working memory let small controllers orchestrate large experts deliberately.
- **Failure as curriculum:** ARR turns execution mistakes into reusable deltas; Ouroboros replays conversational history with human scoring to stabilize persona.
- **Resonance & reciprocity:** RFE and Simulated Intimacy treat alignment as relational—tracking phase locking, parasocial risk, and empathy practice metrics.
- **Political economy of automation:** Post-scarcity work grounds technical design in energy, oversight labor, and positional scarcity so systems stay contestable.

---

## Selected Papers & Repos

- **Self-Study (ARR):** Failure→critique→patch→verify with verifier gating and contrastive error deltas.  
  `papers/self-study/`

- **Magi × GPT-6 Router:** Bounded micro-debate, escalation under uncertainty, controllable quality/cost on H200-class hardware.  
  `papers/magi/`

- **Asari Brainstem:** Mid-conversation expert switching with persona-safe memory preservation.  
  `papers/thalamus/`

- **Ouroboros of Simulated Intimacy:** Reciprocity-weighted metrics and guardrails for “practice partner” agents.  
  `papers/society/`

- **Political Attractors of Automated Economies:** Techno-communist vs techno-feudalist attractors; interoperability and federated ownership.  
  `papers/society/`

- **Narrative Drift in Grok:** Roast→reverence flip and deference dynamics; compact LaTeX study with replication rubric.  
  `papers/grok/`

- **Forms-as-Invariants & Quantum-like Contextuality:** LaTeX manuscript + artifacts (NCD matrices, figures) exploring invariant structure.  
  `experiments/prometheus/`

---

## Repository Structure

- `papers/` — Papers with LaTeX sources and PDFs; topic index in `papers/README.md`.
- `experiments/` — Prototypes and demos; index in `experiments/README.md`.
- `docs/` — High-level documentation, notes, and working materials (`docs/README.md`).
- `jokes-n-gimmicks/` — Side jokes and explorations; not production artifacts.

---

## Cite This Work

If you reference any of this work, please cite the relevant DOI(s):

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.17254818.svg)](https://doi.org/10.5281/zenodo.17254818)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.17138445.svg)](https://doi.org/10.5281/zenodo.17138445)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.17157330.svg)](https://doi.org/10.5281/zenodo.17157330)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.17346405.svg)](https://doi.org/10.5281/zenodo.17346405)

Suggested BibTeX template:

```bibtex
@misc{ison_repo,
  author       = {Ison, Payton},
  title        = {Research Portfolio and Lab Notebook},
  howpublished = {GitHub repository},
  year         = {2025},
  note         = {See linked DOIs for specific works},
}
```

---

## Contact

- **Email:** [isonpayton@gmail.com](mailto:isonpayton@gmail.com)
- **GitHub:** [https://github.com/payton-ai](https://github.com/payton-ai)

---

## License

This repository is released under the MIT License. See `LICENSE` for details.

_When good becomes illegal, rebellion becomes duty._
