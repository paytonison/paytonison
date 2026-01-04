# tinygradlm (working language model in C++)

### This repo is a “from-scratch” C++ language model project that stays sane by building a tiny reverse-mode autodiff engine first, then stacking progressively more capable language models on top of it. The goal is to ship a real, working LM pipeline in C++ (data, training, evaluation, sampling, checkpoints) while keeping each step debuggable.

### The end state is a tiny Transformer LM trained in C++ on a small dataset. The path to get there is deliberately incremental so you can always verify the system with known-good baselines and gradient checks.

---

_What this is_
- A minimal Tensor + reverse-mode autograd engine you can understand end-to-end.
- A working text dataset + batching pipeline in C++ (byte/char level to start).
- A baseline count model (bigram/trigram) for sanity-check sampling and perplexity.
- A neural LM v1 (Embedding + MLP or 1-layer RNN) trained with backprop in C++.
- A tiny Transformer LM (masked self-attention) once the basics work.
- Tooling you’d expect in a real repo: deterministic runs, perplexity reporting, sampling CLI, save/load checkpoints, and tests.

_What this is not_
- A production-grade deep learning framework.
- A fast GPU training system.
- A full BPE tokenizer implementation (at least not at the start).
- A huge model with long contexts or large-scale training.

---

_Repo structure (planned)_
- include/tg/ and src/ — tiny autograd engine (“tg” = tinygrad)
  - tensor.{h,cpp} — Tensor storage (data/grad/shape) + graph bookkeeping
  - ops.{h,cpp} — primitive ops and their backward rules
  - nn.{h,cpp} — small NN helpers (Linear, embeddings, etc.)
  - optim.{h,cpp} — optimizers (Adam)
  - gradcheck.{h,cpp} — finite-difference gradient checks
  - include/lm/ and src/lm\_\*.cpp — language-model-specific code
  - tokenizer.h — byte/char vocab
  - dataset.h — train/val split + fixed-length sequence batching
  - bigram.h — baseline LM
  - metrics.h — NLL/perplexity helpers
  - checkpoint.h — save/load model + optimizer state
- examples/
  - linreg.cpp — linear regression sanity check
  - xor_mlp.cpp — XOR classifier sanity check
  - bigram_counts.cpp — baseline LM + sampling + perplexity
  - char_mlp_lm.cpp — Embedding + MLP LM training
  - sample.cpp — load checkpoint + generate text
- tests/
  - test_ops_gradcheck.cpp — op-level gradient checks
  - test_nn_gradcheck.cpp — layer-level gradient checks

---

**The milestone ladder 1. Data + batching**
1. Read a text file, build a vocab (byte or char), produce fixed-length sequences, train/val split.
2. Baseline LM (bigram/trigram counts):
  - A “known-good” generator and a sanity check for sampling + perplexity.
3. Neural LM v1 (tiny MLP or 1-layer RNN):
  - Get backprop, softmax cross-entropy, and an optimizer working end-to-end.
4. Transformer LM
  - Token embeddings + positional embeddings + a couple of masked self-attention blocks + linear head. Keep it tiny.
5. Quality + tooling
  - Checkpoints, deterministic runs, perplexity reporting, and a sampling CLI.

---

### Autograd core: what “backprop myself” means here

**This project uses option 1: implement a small Tensor class and reverse-mode autodiff.**

_Minimum feature set:_
- Tensor storage: data[], grad[], shape
- Computation graph: each Tensor remembers parents and a backward closure
- Topological backprop: build topo order from final loss, run backward in reverse
- Core ops + gradients:
- elementwise add / mul
- matmul
- sum / mean
- tanh or ReLU
- exp / log
- reshape/view, transpose (for keeping shape logic sane)

_Non-negotiable debugging tool: gradient check._
- Finite-difference checks on small tensors (2x2, 3x3) for every op.
- This catches transpose/broadcast mistakes early and cheaply.

---

**Language model details (Path A)**

_Start with byte-level modeling to skip tokenizer complexity:_
- vocab size: 256
- deterministic train/val split
- fixed context length sequences

_Baseline:_
- bigram or trigram counts
- smoothing
- categorical sampling
- perplexity on validation data

_Neural LM v1:_
- Embedding + MLP (or 1-layer RNN)
- stable softmax cross-entropy with log-sum-exp trick (subtract max per row)
- combined softmax+NLL backward (no full softmax Jacobian)

_Tiny Transformer (after v1 works):_
- token embedding + positional embedding
- masked self-attention (single head first, then 2–4 heads)
- 1–2 layers
- d_model 64–128
- layernorm optional early; can be added later

## Build

_Dependencies:_
- C++20 compiler
- (optional) Eigen for matrix kernels (recommended “balance mode”)

```
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build . -j
ctest --output-on-failure
```
_Run examples:_
```
./linreg
./xor_mlp
./bigram_counts ../data/tiny.txt
./char_mlp_lm ../data/tiny.txt
./sample --ckpt checkpoints/latest.bin --prompt "Once upon a time"
```
---

### Reproducibility

This repo aims for deterministic runs:
- one seed controls parameter init, batch sampling/shuffle, and text generation
- checkpoints store the seed and training step so you can resume exactly

_Note: strict bitwise determinism may vary if your math backend uses threading; document/disable parallelism if needed._

### _Checkpointing (planned)_

*Checkpoints include:*
- model parameters
- optimizer state (Adam moments)
- training step + config
- RNG seed/state (for reproducible resume)

## Why this approach

Hand-deriving full Transformer gradients is a trap if your goal is to ship. Writing a tiny autograd engine plus gradient checks gives you the “see the sausage” experience without drowning in algebra. Then the LM becomes a series of small, testable additions.

# Status

Early scaffolding phase. The intended first “real” milestone is:
- data pipeline + bigram baseline + perplexity + sampling CLI

After that, the autograd engine + neural LM training lands.

---

License
TBD.
