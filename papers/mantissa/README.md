# Mantissa Reserve: High‑Precision–First Inference for Phase‑Faithful Sparse MoE and Energy‑Aware Scheduling

This README explains the key ideas, results, and practical guidance from the paper “Mantissa Reserve: High‑Precision–First Inference for Phase‑Faithful Sparse Mixture‑of‑Experts and Energy‑Aware Scheduling.” It is written for engineers and researchers implementing low‑bit inference (FP8/FP16/bfloat16) on modern accelerators and for systems folks designing energy‑aware schedulers.


## 1) Problem: Low‑Bit Inference Can Break “Phase”

- Phase = directional structure and rankings induced by inner products (vector directions, sign patterns, ordering of router logits).
- Sparse Mixture‑of‑Experts (MoE) is very sensitive: tiny numeric perturbations can flip top‑k routing and corrupt expert aggregation.
- Precision reduction saves memory/bandwidth, but accumulation error (not multiplies) is the dominant source of phase drift in long reductions (e.g., GEMM inner sums, router logits, attention, expert combine).

Key numerical constant: For a reduction of length `n`, the classical accumulation constant is

- `u = 2^{-(t+1)}` where `t` is mantissa bits; and `γ_n ≈ (n·u)/(1 − n·u)` governs worst‑case relative error of a dot product.
- With an FP32 accumulator (`t = 23`), `γ_n` stays small for practical `n` (e.g., `n = 4096 → γ_n ≈ 2.44e−4`).
- With FP16/bfloat16/FP8 accumulation, `n·u` can exceed 1 for typical `n`, voiding stability guarantees.

Takeaway: Multiplying at low precision is usually fine; the main risk is low‑precision accumulation in long reductions.


## 2) Core Idea: Mantissa Reserve Accumulation (MRA)

A compute pattern that preserves phase without sacrificing most low‑bit FLOPs:

1) Multiply in low bit‑width (FP8/FP16/bfloat16) to save bandwidth/compute.
2) Accumulate in FP32 (a “mantissa reserve” of 23 bits).
3) Optionally carry a small FP32 “phase residual” (compensation) to the next phase‑critical barrier and re‑inject it to suppress cumulative drift.

Benefits:
- Restores a small `γ_n` in long reductions even when inputs are low‑bit.
- Aligns with hardware paths that already support FP32 accumulate for tensor ops.
- Residual carry (Kahan/Neumaier‑style) further mitigates cancellation with negligible overhead.

Pseudo‑kernel sketch (see paper Alg. 1): multiply low‑bit, accumulate FP32 with compensated summation, return `C + residual`.


## 3) Phase‑Fidelity Theory (What It Guarantees)

Two stability notions matter in sparse MoE:

- Angular stability: preserve directions (cosine similarity) through layers.
- Rank stability: preserve the ordering of router logits for top‑k routing.

Main results (informal):

- Layerwise phase drift: with FP32 accumulation, expected cosine misalignment after a linear layer is dominated by the small FP32 `γ_n` term; low‑bit multiplies contribute a smaller second‑order term.
- Top‑k stability theorem: if the router’s margin `m_k = s_(k) − s_(k+1)` exceeds `2·η`, where `η` is a bound on dot‑product error dominated by the FP32 accumulation constant, then the top‑k set is unchanged. FP32 accumulation dramatically increases the regime where this holds.

Intuition: The extra 23 mantissa bits collapse accumulation error so that typical router margins comfortably exceed perturbations.


## 4) PFQ: Phase‑Fidelity Quantization for MoE

Concentrate precision where phase is fragile and leave the rest low‑bit.

- Router logits: compute `⟨w_i, x⟩` with FP32 accumulate; choose top‑k before any down‑cast.
- Dispatch weights: compute softmax (or log‑sum‑exp) in FP32; downcast after normalization.
- Expert internals: run experts with low‑bit multiplies, FP32 accumulate inside each MLP.
- Expert combine: weighted sum across experts in FP32 accumulate.
- Normalization/attention: compute statistics (mean/var, softmax denominators) with FP32 accumulate.

This “precision at boundaries” preserves routing and aggregate direction while retaining low‑bit execution for most FLOPs.


## 5) PRISM: Precision‑Aware Scheduling Under Energy/SLO

A datacenter scheduling policy that activates the mantissa reserve (profiles with FP32 accumulate at phase‑critical sites) only when it provably helps.

- Inputs per job: deadline/SLO and a Phase Sensitivity Index (PSI) estimated from telemetry (router margins, attention entropy, calibration SNR, gradient proxies).
- Profiles: e.g., `{ FP8×/FP32+, bf16×/FP32+, FP16×/FP32+ }` where `×` denotes multiply precision and `+` denotes accumulator precision.
- Two‑stage heuristic:
  1) Phase‑first filtering: if `PSI > θ`, require a reserve‑capable profile.
  2) Energy‑aware bin packing: sort by laxity and assign the lowest‑energy profile that meets SLOs; aggressively co‑schedule low‑PSI jobs on low‑bit profiles.

Effect: Energy savings close to all‑low‑bit, but “snap” to FP32 accumulation where it protects phase.


## 6) Worked Numbers (Why FP32 Accumulate Matters)

For `n = 4096` (typical head or MLP width):

- FP32 accumulate: mantissa 23, `u ≈ 5.96e−8` → `γ_n ≈ 2.44e−4` (stable)
- FP16 accumulate: mantissa 10, `n·u ≈ 2.0` → no worst‑case guarantee
- bfloat16 accumulate: mantissa 7, `n·u ≈ 16.0` → no guarantee
- FP8 (E4M3) accumulate: mantissa 3, `n·u ≈ 256.0` → no guarantee

Even with FP8 multiplies, FP32 accumulation keeps router rankings stable under moderate margins; low‑bit accumulation cannot guarantee stability for typical widths.


## 7) Practical Guidance

Spend precision:
- Long reductions (GEMM inner sums) and any cross‑expert/expert‑combine reduce.
- Router logits (MoE), attention softmax denominators, and layernorm stats.

Save precision:
- Expert internal multiplies (use FP8/FP16/bf16 multiplies with FP32 accumulate).
- Activation storage between non‑critical layers.

Numerical hygiene:
- Prefer fused multiply‑add (FMA); use log‑sum‑exp; avoid subtractive cancellation.
- Enable compensated summation for exceptionally long reductions.

Deployment tips:
- Ensure kernels use FP32 accumulation paths on your target accelerators (vendor flags often exist).
- Log router margins and attention entropy to estimate PSI for PRISM.
- Downcast only after phase‑critical decisions (top‑k selection, normalization).


## 8) Limitations and Caveats

- Bounds are worst‑case and conservative; real models often have more structure.
- Platform details matter (denormals, subnormal flushing, rounding mode, tensor‑core paths). Validate on target hardware.
- Phase residual carry adds a small state; the best compression strategy is workload‑dependent.


## 9) Implementation Checklist

- GEMM/attention kernels: low‑bit multiplies, FP32 accumulate (+ optional Kahan/Neumaier compensation).
- Router path: FP32‑accumulated logits; top‑k executed before any downcast.
- Softmax/normalization: compute denominators in FP32 (use log‑sum‑exp), then downcast.
- Expert combine: FP32 accumulation across experts.
- Telemetry for PRISM: router margin histogram, attention entropy, calibration SNR (to derive PSI).
- Scheduler policy: reserve profiles when `PSI > θ`; otherwise assign lowest‑energy low‑bit profiles meeting SLOs.


## 10) Glossary

- Mantissa reserve: using a wider‑mantissa accumulator (FP32) while keeping multiplies low‑bit.
- Phase: directional and ranking information carried by inner products.
- `γ_n`: accumulation constant controlling worst‑case relative error for length‑`n` reductions.
- PSI: Phase Sensitivity Index used by PRISM to decide when the reserve helps.


## 11) Mapping to IEEE Formats

- FP8 (E4M3/E5M2): mantissa ≈ 2–3 bits (post‑implicit); unsafe for long accumulations.
- bfloat16: mantissa 7; multiply OK, accumulate not phase‑safe for typical `n`.
- FP16: mantissa 10; multiply OK, accumulate marginal for typical `n`.
- FP32: mantissa 23; preferred accumulator for phase‑critical reductions.


## 12) References (short)

- IEEE 754‑2019; Higham, “Accuracy and Stability of Numerical Algorithms”; Shazeer et al., Sparsely‑Gated MoE; Fedus et al., Switch Transformer; Micikevicius et al., Mixed Precision Training; Jouppi et al., TPU performance.

---

If you want, we can add example kernels/configs for specific accelerator backends showing FP32 accumulate toggles and a minimal PRISM‑style scheduler prototype.

