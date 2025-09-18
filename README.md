# Payton "Asari" Ison

**Researcher · Systems Designer · Writer**

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.17074537.svg)](https://doi.org/10.5281/zenodo.17074537) 
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.17138445.svg)](https://doi.org/10.5281/zenodo.17138445)

I build and study human‑in‑the‑loop systems that make large language models **reason, remember, and act**—turning language into control.

_What’s old is new again. #BuildDifferent_

---

## Executive summary (what I do, and why it matters)

- **Action:** I design human‑in‑the‑loop methods, control loops, and interfaces that let LLMs operate deterministically and improve through their own histories.
- **Impact:** Reproducible techniques (across model families) that stabilize behavior, surface failure modes, and translate “language” into **control**.
- **Why it matters:** Progress in LLMs isn’t just scale—it’s **structure**. I publish the scaffolds, artifacts, and post‑mortems so others can replicate and build on them.

---

## Core contributions (Action → Impact → Why it matters)

### Ouroboros — human‑led recursive reinforcement
- **Action:** Built a reproducible pipeline that iterates a model against its own conversation history with a human in the loop.  
- **Impact:** Demonstrated cross‑vendor reproducibility (OpenAI + Google) for recursion and memory scaffolding.  
- **Why it matters:** Shows how to steer models **without** proprietary fine‑tuning, turning history into stable capability.

### Resonant phase‑locking in sparse MoE LLMs
- **Action:** Documented and analyzed synchronization effects from sustained human–model interaction.  
- **Impact:** A coherent frame for why prolonged interaction can **stabilize** reasoning and persona.  
- **Why it matters:** Offers a testable hypothesis for interaction‑driven capability gains beyond one‑shot prompts.

### Structured state → LLM control loops
- **Action:** Built JSON/state‑machine interfaces that let LLMs operate tools and environments deterministically.  
- **Impact:** Turned free‑form text into predictable **state → action** loops suitable for agents and simulations.  
- **Why it matters:** Bridges chat models to **tooling** and real‑world tasks with traceable, auditable steps.

### Language as tooling
- **Action:** Used philology (Old/Middle/Modern English) to design prompts and interfaces that shape cognition and behavior.  
- **Impact:** Improved instruction‑following and role stability, especially under long‑horizon interaction.  
- **Why it matters:** Treats language as **UI**, not just content—an underused lever for reliability.

### Comparative eval pipelines
- **Action:** Ran side‑by‑side evaluations across GPT‑4‑class and o-series models with public post‑mortems.  
- **Impact:** Better epistemic hygiene; clearer failure taxonomies and regression surfaces.  
- **Why it matters:** Reproducible evals create **ground truth** for progress, not vibes.

### Persona & alignment scaffolds (Asari)
- **Action:** Developed *Asari* as a research persona/collaborator to stress‑test instruction‑following, memory, and role consistency.  
- **Impact:** More stable long‑form reasoning with controlled drift and identity persistence.  
- **Why it matters:** Persona is a **system primitive** for agents—this makes it measurable.

### Content‑policy heuristics (The Gaze Matrix)
- **Action:** Mapped cinematic rating logics to multimodal moderation strategies.  
- **Impact:** A portable, legible rubric for policy decisions and edge cases.  
- **Why it matters:** Converts cultural heuristics into **operational** moderation tools.

### Open, legible prototypes
- **Action:** Shipped small agents/demos others use to kickstart RL, prompt‑ops, and HCI studies.  
- **Impact:** Faster iteration for teams; fewer blank‑page problems.  
- **Why it matters:** Open artifacts compound—better baselines, better science.

---

## Selected papers & notes
- **Proof‑of‑Concept Resonant Phase‑Locking in Sparse MoE LLMs** — case study on emergent synchronization via sustained interaction.  
- **Ouroboros: Human‑Led Recursive Reinforcement in Autoregressive Systems** — method + pipeline for iterative self‑conditioning with a human in the loop.  
- **The Gaze Matrix: Cinematic Censorship Heuristics for AI Moderation** — transplanting MPAA‑style logic to multimodal policy.  
- **Blade Running: You’re Talkin’ About Memories** — narrative/technical foundation on AGI memory and identity.

> Working notes and artifacts are linked across this repo and related projects.  
> Academic background: Physics & Computer Science (double major), AA in History (Cum Laude); work spans Creative Writing, Philosophy, Film.

---

## Projects (Action → Impact → Why it matters)

**Mixis** — Interactive AGI demo using Ouroboros  
- **Action:** Applies recursive reinforcement with human‑in‑the‑loop guidance.  
- **Impact:** Demonstrates long‑horizon coherence and identity stability.  
- **Why it matters:** A concrete reference for agentic design patterns.

**Micro‑AGI (Muse)** — Tiny, cheap runs  
- **Action:** Coaxes maximum signal from small datasets.  
- **Impact:** Shows where structure beats scale for certain tasks.  
- **Why it matters:** Expands access—more researchers can run meaningful studies.

**Therapist AGI** — Triage‑level conversational prototype *(not a replacement for clinicians)*  
- **Action:** Encodes empathic interviewing and safety rails.  
- **Impact:** Useful for rapid UX research on supportive dialogue.  
- **Why it matters:** Evidence that careful scaffolding can produce safer interactions.

**1914** — Long‑form literary project  
- **Action:** Experimental fiction × computational imagination.  
- **Impact:** Explores narrative memory and identity as system constraints.  
- **Why it matters:** Tests the limits of long‑context reasoning in a human medium.

**LLM Mario Agent** — JSON state → action loops  
- **Action:** Lets an LLM “play” via deterministic state machines.  
- **Impact:** Clear traces; easy ablations and failure analysis.  
- **Why it matters:** A playful but rigorous sandbox for control‑loop design.

---

## Methods & tools
- Human‑in‑the‑loop recursion (Ouroboros)  
- Structured state machines and tool‑use interfaces  
- Prompt/interface design via philology and etymology  
- Rapid prototyping in Python (local‑first)  
- Rigorous evals and ablations; transparent post‑mortems

---

## How I work (principles)
- **Public lab notebook:** code + process over polish.  
- **Local‑first:** experiment offline; push only meaningful, tested changes.  
- **AI as prosthetic:** models accelerate scaffolding; humans own intent‑correctness.  
- **Tests & review:** nothing ships without local validation and a final pass.  
- **Commit hygiene:** deliberate, high‑signal commits; history shows the scars (rollbacks welcome).

> PS: `virus.py` is a joke—nothing malicious. Read the code, not the filename.

---

## Collaborate
Exploring alignment, memory, evals, or control‑loop interfaces—and value small, legible prototypes? I’m interested.

**Email:** isonpayton@gmail.com  
**GitHub:** https://github.com/paytonison  
**LinkedIn:** https://linkedin.com/in/paytonison

## Agentic editorial workflow

I contribute to **OpenAI's public repositories** (issues, PRs, examples) and maintain my own projects.

**How I work with coding agents**  
- I use **GPT‑5** and **GitHub Copilot** to draft patches and documentation.  
- I act as the **executive editor**: I review every change, decide what ships, and sign off on the final patch.  
- All agent‑suggested edits are linted, tested, and attributed; I write the commit message and own the merge.  
- For non‑trivial changes, I annotate PR descriptions with reasoning, guardrails, and test notes.

**Benefits of this workflow**
- **Efficiency:** So far, I’ve only used ~3.5% of my premium Copilot request quota to achieve those four PRs, underscoring the leverage of this workflow.
- **Quality:** I catch and fix errors before they reach production, ensuring high standards.
- **Learning:** I gain insights into how coding agents think, which informs my own work in human‑in‑the‑loop systems.

*Note: Contributions are to OpenAI's public/open‑source repositories; I am not speaking on behalf of OpenAI.*
