# Papers

Topic-focused folders with outlines, drafts, and references. Each subdirectory below contains its own LaTeX sources, figures, and (where applicable) compiled PDFs.

## Directory index
| Directory | Theme |
|-----------|-------|
| `agi/` | Full-length article on AGI control loops. |
| `andrew-ditch/` | Notes and drafts for the Andrew Ditch case study. |
| `attunement-game/` | Attunement game manuscript + Makefile for compilation. |
| `chasro/` | Coordinated heuristic safety research outline. |
| `csd/` | Control systems design paper (`main.tex`, `references.bib`). |
| `gpt-6/` | GPT-6 era router considerations and forecasts. |
| `grok/` | Narrative drift analysis for Grok-class models. |
| `logos-sophia/` | Logos & Sophia philosophical notes plus sourced PDFs. |
| `magi/` | Magi × GPT-6 router paper. |
| `mantissa/` | Mantissa manuscript and supporting README. |
| `operation-neptune/` | Operation Neptune strategic memo. |
| `ouroboros/` | Human-in-the-loop recursion framework. |
| `quantum-ai/` | Quantum-inspired contextuality treatise. |
| `rctp/` | Recursive control theory primer. |
| `rfe/` | Resonant Feedback Effect study. |
| `self-study/` | ARR/self-study methodology. |
| `society/` | Political economy & simulated intimacy manuscripts. |
| `thalamus/` | Asari Brainstem / thalamic routing work. |

## Working conventions
1. Each paper directory should include a `README.md` (or short note) describing scope, build steps, and publication status.  
2. Run `latexmk` or the provided `Makefile` inside the directory to regenerate PDFs; commit only the sources plus final PDF.  
3. When adding a new topic, update this index with a one-line summary and ensure references live beside the manuscript (`refs.bib`, `references.bib`, etc.).  
