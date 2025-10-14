# Payton “Asari” Ison

**Researcher · Systems Designer · Writer**

_I build human-in-the-loop systems that make language models reason, remember, and act—turning language into control._

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

---

## Methods & tools

- Human-in-the-loop recursion (Ouroboros)
- Structured state & deterministic tool use
- Prompt/interface design (persona, protocol JSON)
- Rapid Python prototyping (local-first)
- **Rust (edition 2024) & Cargo** for lexical/embedding experiments
- TeX for papers; shell for ops hygiene
- Reproducible evals & ablations

---

## Get in touch

- **Email:** isonpayton@gmail.com  
- **GitHub:** https://github.com/paytonison  
- **LinkedIn:** https://linkedin.com/in/paytonison  
- **The Platform Formerly Known as Twitter:** https://x.com/p8on_
