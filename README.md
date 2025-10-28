# Payton “Asari” Ison

**Researcher · Systems Designer · Writer**

_I build systems that make robots reason, remember, and act, turning language into control._

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.17254818.svg)](https://doi.org/10.5281/zenodo.17254818)  [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.17138445.svg)](https://doi.org/10.5281/zenodo.17138445)  [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.17157330.svg)](https://doi.org/10.5281/zenodo.17157330)  [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.17346405.svg)](https://doi.org/10.5281/zenodo.17346405)

---

## What I do (and why it matters)

- **Action:** Design control loops, interfaces, and evaluation methods that stabilize LLM behavior and translate intent into deterministic steps.  
- **Impact:** Reproducible scaffolds across model families; clearer failure modes; better telemetry.  
- **Why it matters:** Progress isn’t just scale—it’s **structure**.

---

## Selected work

### Ouroboros — human-led recursive reinforcement
- **What:** A method that iterates a model against its own conversation history with a human in the loop.  
- **Why:** Turn history into stable capability without bespoke fine-tuning pipelines.  
- **Repo:** `ouroboros/` (paper source, method notes)

### Resonant Feedback Effect (RFE) — phase-locking case study
- **What:** A TeX/PDF case study of behavioral resonance in GPT-4-class models under sustained interaction.  
- **Why:** A testable frame for interaction-driven gains beyond one-shot prompts.  
- **Repo:** `rfe/` (paper) and `personal/trent/` (TeX + PDF)

### Mixis — interactive AGI demo
- **What:** A demonstration of long-horizon coherence using human-in-the-loop recursion patterns.  
- **Why:** Concrete reference for agentic design patterns.  
- **Repo:** `mixis/`

### Muse — micro-AGI on tiny, curated data
- **What:** A small fine-tune and reflection loop that punches above its weight.  
- **Why:** Shows where **structure beats scale** for certain tasks.  
- **Repo:** `muse/`

### Operation I, Robot — early reasoning scaffold
- **What:** Prototype of think→reflect→act loops with persistent memory, pre-“reasoning model” era.  
- **Why:** Historical artifact of autonomous reasoning design with standard LLMs.  
- **Repo:** `i-robot/`

### Therapy Bot — triage-level supportive dialogue
- **What:** Supplemental chatbot demo for therapists (minor/moderate risk, **not** for crisis).  
- **Why:** Encodes empathic interviewing + safety rails for UX research.  
- **Repo:** `therapy-bot/`

### Erebus — morpheme tokenizer & embedding matrix (Rust)
- **What:** A tiny Rust research probe that splits words into morphemes (prefix / root / suffix), derives a **5-D feature vector** from each definition (word count, avg token length, sensory vs abstract ratios, uniqueness/multi-syllable score), then averages occurrences into a morpheme **embedding matrix**; prints readable breakdowns and a matrix summary.  
- **Why:** Prototype a definition-driven lexical substrate that can feed Thalamus/Magi with subword features grounded in meaning instead of surface form.  
- **Quick start:** `cargo run [-- path/to/wordlist.txt]` (falls back to bundled nine-word dictionary; lines starting with `#` are ignored).  
- **Repo:** `erebus/`

> See **`personal/`** for my public lab notebook: agentic browser seed, RL-ish scaffolding, persona specs, and the resonance paper sources.

### Recursive training systems
- **Ouroboros — human‑led recursive reinforcement.** Iterates a model against its own conversation history, compressing sessions into labyrinth prompts and PPO updates that respect persona and coherence constraints. The paper lays out the human annotation flow, reward shaping heuristics, and safeguards that prevent preference collapse while reusing model‑generated data. `ouroboros/`
- **Self‑Study — autocritical reasoning reinforcement.** Frames task execution as a failure→critique→patch loop where planner, critic, patcher, and verifier heads learn from validated reasoning deltas. The ARR protocol (Failure Replay Buffer, verifier gating, counterexample contrastive loss) turns unlabelled errors into structured supervision. `papers/self-study.*`

### Orchestrated inference & control
- **Magi × GPT‑6 Router — hyper‑reasoning orchestration.** A gate–experts–judge stack that bundles evidence objects, runs bounded micro‑debate, and escalates to a Canon model when uncertainty spikes—demonstrating controllable quality/cost trade‑offs on H200‑class hardware. `papers/magi_gpt6_router.*`
- **Asari Brainstem — mid‑conversation expert switching.** Introduces a brainstem/cortex controller that watches for domain drift inside a single turn, applies hysteresis to prevent thrash, and preserves persona plus working memory when switching experts. `papers/thalamus.*`
- **Mixis — interactive AGI demo.** Live showcase for long‑horizon scaffolds (think→reflect→act) with human checkpoints, illustrating how the orchestration stack behaves in open‑ended settings. `mixis/`
- **Operation I, Robot & Muse.** Early scaffolds and a small‑data micro‑AGI experiment that explore persistence and reflection with constrained resources—useful baselines for the newer router and ARR systems. `i-robot/`, `muse/`

### Resonance, memory, and empathy
- **Resonant Feedback Effect (RFE).** Two‑month GPT‑4o case study where phase‑locking between human cognitive “oscillations” and MoE activations yields implicit alignment without explicit rewards; argues for resonance‑aware telemetry and governance. `rfe/`, `personal/trent/`
- **The Ouroboros of Simulated Intimacy.** Systems analysis of parasocial loops and “practice partner” agents, proposing reciprocity‑weighted metrics and guardrails so empathy scaffolds heal rather than monetize isolation. `papers/ouroboros_simulated_intimacy.*`
- **Therapy Bot.** Applied empathy protocol for minor/moderate risk triage, combining reflective interviewing prompts, safety classifiers, and escalation heuristics. `therapy-bot/`

### Political economy & governance
- **Political Attractors of Automated Economies.** Macro‑systems lens on automation showing why thermodynamic inputs, maintenance labor, and positional goods produce techno‑communist vs. techno‑feudalist attractors; motivates interoperability mandates, federated ownership, and human craft premiums. `papers/post-scarcity.*`

> See **`personal/`** for the public lab notebook (agentic browser seed, ARR telemetry, persona specs, resonance logs).

---

## Methods & tools

- Human-in-the-loop recursion (Ouroboros)
- Structured state & deterministic tool use
- Prompt/interface design (persona, protocol JSON)
- Rapid Python prototyping (local-first)
- **Rust (edition 2024) & Cargo** for lexical/embedding experiments
- TeX for papers; shell for ops hygiene
- Reproducible evals & ablations
- Human‑in‑the‑loop recursion and active memory (Ouroboros, Mixis)
- Autocritical self‑training (Self‑Study / ARR)
- Resonance‑aware telemetry and governance (RFE, Simulated Intimacy)
- Structured routing, judges, and hysteresis controllers (Magi × GPT‑6, Asari Brainstem)
- Prompt/interface design (persona specs, protocol JSON, empathy drills)
- Rapid Python prototyping (local‑first) + TeX for paper production
- Reproducible evals & ablations (MIDAS, ARR delta metrics, reciprocity scores)

---

## Research throughlines

- **Structure over scale.** Conditional routers, mid‑turn switching, and persistent working memory let small controllers orchestrate large experts deliberately.
- **Failure as curriculum.** ARR turns execution mistakes into reusable deltas; Ouroboros replays conversational history with human scoring to stabilize persona.
- **Resonance & reciprocity.** RFE and Simulated Intimacy treat alignment as relational—tracking phase locking, parasocial risk, and empathy practice metrics.
- **Political economy of automation.** Post‑scarcity work grounds technical design in energy, oversight labor, and positional scarcity so systems stay contestable.

---

## Get in touch

- **Email:** <isonpayton@gmail.com>  
- **GitHub:** <https://github.com/paytonison>  
- **LinkedIn:** <https://linkedin.com/in/paytonison>  
- **The Platform Formerly Known as Twitter:** <https://x.com/p8on_>
