# Tutor Loop — Forward-Feedback Training (No Backprop Through the Student)

A tiny, dependency-light prototype of a **Tutor ⇄ Student** training loop that **improves a model without backpropagating through it**.
The *Student* is an LM-like network with a low-rank (LoRA-style) adapter.
The *Tutor* is a small network that observes simple error features from a forward pass and emits **directional coefficients** to nudge the Student’s adapter **with no gradients through the Student**.
Optionally, a 2-pass **SPSA** warm-up estimates a good sign for early steps.

This repo is intentionally clean: **no Hugging Face, no PEFT, no datasets**. Just PyTorch.

---

## Why

Backprop is expensive because it doubles compute/memory and forces tight coupling to GPU vendor stacks.
This project shows a pragmatic alternative:

* **Forward-only Student updates** (scale LoRA-B columns in place)
* **Small Tutor** learns to predict good update directions from cheap features
* **Accept/Reject** each update by measuring immediate loss improvement on the same batch (or a tiny probe)

Think **“Photoshop brushstrokes”** instead of photocopying gradients.

---

## What’s Here

* `tutor_minimal.py` — single-file demo that:

  * Builds a tiny Student (embedding + MLP + low-rank output adapter)
  * Builds a tiny Tutor (MLP producing rank-R coefficients)
  * Uses **SPSA** briefly to bootstrap a target direction
  * Applies Tutor coefficients to Student **without backprop**
  * **Accepts** the update only if loss improves; otherwise **reverts**
  * Prints loss deltas and acceptance rate as it learns

Default task is a toy next-token problem (mod-V), so you can see the loop work in seconds.

---

## Install

You only need PyTorch.

```bash
# CPU-only (works fine)
pip install torch

# Or CUDA build (example URL, pick for your CUDA)
pip install torch --extra-index-url https://download.pytorch.org/whl/cu121
```

No other deps required.

---

## Run

```bash
python tutor_minimal.py
```

You should see logs like:

```
[0000] base=3.9321 new=3.9027 Δ=0.0294 keep=1 acc_rate=1.00 tutor_loss=0.98
...
Final held-out loss: 3.10
```

Key signals:

* `Δ` mostly positive over time
* Acceptance rate stabilizes > 0.5 as the Tutor learns
* No NaNs; if you see them, reduce `step_size` or tighten `clip`

---

## How It Works (Short)

1. **Forward pass**: compute Student logits → derive **loss, entropy, margin** → pack into a 3-D feature vector.
2. **Tutor step**:

   * Early on, get a pseudo-target via **SPSA** (two forward passes with ±ε multiplicative noise on the low-rank columns); choose the direction that lowers loss.
   * Train Tutor to predict that direction from the feature vector (this *does* backprop through the Tutor only).
3. **Apply feedback**: Tutor emits a rank-R coefficient vector; we **multiply** the Student’s low-rank `B` columns by `(1 + lr * coeff)` (with tanh + clip + EMA smoothing).
4. **Guard & decide**: recompute loss; **keep** the update if loss improved by ≥ ε, otherwise **revert**.

This is a **local learning rule** guided by a learned critic, not global gradient descent through the full model.

---

## Tuning Knobs

Inside `tutor_minimal.py`:

* `R` — low-rank dimension (start 8; raise to 16 later)
* `apply_feedback(... step_size=0.02, clip=0.25, ema=0.9)` — stability controls

  * If it oscillates: lower `step_size` or `clip`
  * If it’s sluggish: increase `step_size` slightly (e.g., 0.03)
* **Acceptance threshold** `accept_eps=1e-4` — how picky you are about keeping updates
* **SPSA** warm-up:

  * `eps=0.03` — perturbation magnitude
  * Reduce warm-up length once Tutor stabilizes

---

## Roadmap (Fast, Practical Upgrades)

1. **Character-LM task**
   Replace the toy task with a char-LM over a `.txt`. Same loop, more language-like signal.

2. **Richer features**
   Feed the Tutor: `[loss, entropy, margin, perplexity_delta]` or a small histogram of token losses (e.g., 16 bins).

3. **Granularity**
   Add a second adapter (e.g., on a hidden MLP) and let the Tutor choose which adapter to nudge per step.

4. **Budget scheduler**
   Track LoRA-B norm growth; decay `step_size` when acceptance rate < 30%.

5. **Anchor step (optional)**
   Every N steps, do **one tiny true SGD step** on the Student to create a weak label for the Tutor (grad sign). Keeps the Tutor calibrated without paying the full backprop tax.

6. **Bigger models, still no HF**
   Load weights with `safetensors`, wrap only the `nn.Linear` layers you care about with your own low-rank factors, and reuse this loop.

---

## FAQ

**Q: Does the Student ever use backprop?**
No. The Student is updated by **forward-only, multiplicative nudges on low-rank columns**, guarded by accept/reject. Only the Tutor trains with standard backprop (it’s ~tiny).

**Q: Why does SPSA help?**
Cold-start. It gives you a cheap, 2-forward-pass estimate of a “good direction” before the Tutor learns to predict one.

**Q: Isn’t this unstable?**
It can be. That’s why we:

* clip/tanh coeffs,
* use EMA smoothing,
* cap LoRA-B norm growth,
* and **revert** updates that don’t improve loss.

**Q: Can this scale?**
Yes, in principle: parallelize Tutor evaluation per batch, shard adapters by layer, and keep Student forward-only. The expensive backward graph never exists.

---

## License

MIT (or your preferred permissive license). Do whatever you want; attribution appreciated.

---

## Citation (optional)

```
@misc{tutorloop2025,
  title  = {Tutor Loop: Forward-Feedback Training without Backpropagation through the Student},
  author = {Ison, Payton and Collaborators},
  year   = {2025},
  note   = {https://github.com/your/repo (prototype)}
}
```
