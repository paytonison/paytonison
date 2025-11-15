Forms-as-Invariants & Quantum-like Contextuality — LaTeX Package
================================================================

Files
-----
- forms_quantum_llm.tex      : Main LaTeX manuscript
- refs.bib                   : BibTeX references
- ncd_heatmap.png            : Figure 1 (NCD heatmap)
- ncd_matrix.csv             : Full NCD matrix (artifact)
- ncd_summary.csv            : Summary statistics for NCD (artifact)
- qq_results.csv             : QQ equality terms and deviation (artifact)

How to Compile
--------------
1. Put all files in the same directory.
2. Compile with (pdf)LaTeX and BibTeX, e.g.:
   pdflatex forms_quantum_llm.tex
   bibtex forms_quantum_llm
   pdflatex forms_quantum_llm.tex
   pdflatex forms_quantum_llm.tex

Notes
-----
- The manuscript uses natbib with plainnat style.
- The results correspond to the computational run performed in this session
  (zlib compression level 9; UTF-8; QQ equality computed from fixed illustrative probabilities).
