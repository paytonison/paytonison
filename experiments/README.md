# Experiments  

This directory contains prototypes, autonomous agents, and one-off studies. Each substantial experiment should include its own README with a clear goal, approach, and how to run it.  

## Index  

### Directory overview  
| Directory | Focus |
|-----------|-------|
| `agents-system-prompt/` | System-prompt scaffolds and atlas-agent prompt engineering trials. |
| `browser project/` | Browser automation sandboxes (note the space in the folder name when cd'ing). |
| `cnn/` | Convolutional model experiments for text/video hybrids. |
| `electra/` | ELECTRA-style discriminator studies plus Ouroboros reflection notes. |
| `erebus/` | Rust morpheme tokenizer + embedding builder (Cargo project). |
| `gpt-oss/` | Minimal MoE lab with router-bias controls. |
| `grok/` | Narrative drift simulations for Grok-class models. |
| `i-robot/` | Early think→reflect→act scaffolds with persistent memory. |
| `mixis/` | Recursive identity demos that distill multi-persona chat logs. |
| `muse/` | Micro-AGI experiments on curated, tiny datasets. |
| `philosobot/` | Long-horizon philosophizing agents and prompts. |
| `prometheus/` | Forms-as-invariants & quantum-like contextuality artifacts. |
| `q-network/` | Simple Q-learning controllers used for benchmarking. |
| `reflective-ai/` | Journaling/practice loops for resonance-aware feedback. |
| `recurrent-neural-network/` | Char-level language-model experiments. |
| `socratic-dialogue/` | Prompt datasets and scripts for Socratic teaching loops. |
| `super-mario/` | LLM-controlled platformer agent (browser + pygame). |
| `therapy-bot/` | Reflection-heavy dialogue agents with safety prompts. |
| `tutor/` | Minimal tutoring agents (`tutor_minimal.py`, `tutor_charlm.py`). |
| `turing-demo/` | JSON-configurable Turing machine simulator. |
| `two-stooges-and-an-idiot/` | Multi-agent comedy harness for stress-testing tool calls. |

`TIMELINE.md` in this directory tracks experiment start/stop dates across the above projects.  

### Standalone scripts and datasets  
These live at the root of `experiments/` and support multiple sub-projects:
- `aeon.py` / `aeon.json` – cyclical process controllers plus seed data.  
- `busy_beaver.py` – Busy Beaver search heuristics.  
- `calliope.py` – muse-inspired creative generator.  
- `file analyzer.py` – lightweight inspection utility for datasets/logs.  
- `hyperion.py` – macroscopic systems prototype.  
- `interlinked.txt` – concept graph notes.  
- `louvre.txt` – aesthetic framing memo for creative agents.  
- `mirror test.py` – self-recognition probe.  
- `oss.py` – helper for working with open-source mirrors.  
- `panacea.jsonl` – high-signal dataset for prompt bootstrapping.  
- `system optimizer.py` – brute-force tuning helper.  
- `tartarus.py` – deep-dive control experiment.  
- `the ghost.json` – structured data for “The Ghost” scenario.  
- `trent.tex` – LaTeX scratchpad for in-progress write-ups.  
- `turing.txt` / `turing_demo.py` – additional Turing machine references.  

### Contribution checklist  
1. Add a short README to every new sub-directory describing purpose, setup, and run instructions.  
2. Register the directory or script in the appropriate table/list above.  
3. Update `TIMELINE.md` with start/end dates and link back to any related paper or doc.  
