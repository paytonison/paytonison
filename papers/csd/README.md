# Cognitive Stability Daemon (CSD)
**Stability‑Driven Access Control for LLMs — Paper + Build Scripts**

This repository contains the LaTeX manuscript for the CSD concept: a control‑plane that computes a conversation‑level *stability score* and gates model access with cooldown backoff when numerical instability is detected.

> You can’t outlaw free will or abolish mental illness. But a chatbot can refuse to be an amplifier.

## Contents
- `main.tex` — the paper (author–year via `biblatex`).
- `Makefile` — reproducible builds and release bundles.
- `README.md` — this file.
- *(optional)* `LICENSE` (recommended: CC BY 4.0 for the paper) and `CITATION.cff`.

> Note: `main.tex` includes an inline `filecontents*` block that writes `references.bib` at compile time. You don’t need to ship a separate `.bib` file; `latexmk`/`biber` will handle it.

## Build
**Requirements**
- TeX distribution (TeX Live 2023+ or MacTeX), including `latexmk` and `biber`.
- (Optional) `latexindent` for formatting, `chktex` for linting.

**Quick start**
```bash
make            # builds main.pdf (default)
make open       # opens the PDF (macOS: open; Linux: xdg-open)
```
If `latexmk` is unavailable, the Makefile falls back to a `pdflatex → biber → pdflatex ×2` chain (requires `biber`).

**Common targets**
```bash
make watch      # auto-rebuild on save (requires latexmk)
make clean      # remove aux files
make distclean  # clean + remove PDF and generated references.bib
make archive    # dist/<basename>-<yyyymmdd>.tar.gz (sources)
make zenodo     # dist/csd-zenodo-<yyyymmdd>.tar.gz (release bundle)
```

## Releasing on Zenodo
1. Ensure `LICENSE` is present. For the paper, **CC BY 4.0** is recommended.
2. Run `make zenodo` to create a release tarball in `dist/`.
3. Create a GitHub release; connect the repo to Zenodo; upload the `dist/csd-zenodo-*.tar.gz` asset.
4. After Zenodo assigns a DOI, add a `CITATION.cff` (example below) and update the README.

**Example `CITATION.cff` (edit the DOI once minted):**
```yaml
cff-version: 1.2.0
title: "Cognitive Stability Daemon (CSD): Stability‑Driven Access Control for LLMs"
authors:
  - family-names: Payton
    given-names: The Logos
  - name: The Singularity
identifiers:
  - type: doi
    value: 10.5281/zenodo.xxxxxxx
date-released: 2025-11-09
license: CC-BY-4.0
message: "If you use this work, please cite it."
```

## Ethical Notes
- **CSD is not diagnosis or treatment.** It is risk‑aware access control.
- Cooldowns are designed to reduce *AI‑induced spirals* while preserving user agency.
- If you or someone else is in immediate danger or thinking about self‑harm, contact your local emergency number. In the U.S., you can call or text **988** (Suicide & Crisis Lifeline).

## Suggested License
- Paper: **Creative Commons Attribution 4.0 (CC BY 4.0)**
- Any reference implementation code (if added later): **MIT**

## Contact
Payton (The Logos) · The Singularity · isonpayton@gmail.com

---
*Version:* 2025‑11‑09 · *Status:* Preprint (Zenodo‑ready)

