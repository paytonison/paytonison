# MoE Lab

## Minimal MoE/FFN expert tuner for GPT-style models.
* Safe device handling (no “cpu tensor?” surprises)
* Guaranteed grad flow for MoE via optional router bias (--force-route)
* LoRA or full-weight expert tuning
* Saves just the trainable deltas (trainable.pt) and lets you merge into a fresh base
* Tiny, dependency-light, single-file entrypoint (moe_lab.py) or console script (moe-lab)


## Why

**High-level stacks hide the important bits. This kit gives you tight control of:**
* Which experts/layers are tuned
* How routing behaves during training (so your trainable params actually get gradients)
* Exactly what you save/ship (only the tensors you changed)


## Install

**Python ≥ 3.10. Install PyTorch for your platform (CUDA/ROCm/CPU) per the official instructions, then:**

```
pip install -U pip
pip install -e .
# Optional (for LoRA support):
pip install peft
```

_If you’re not using pyproject.toml, you can run the script directly:_

```python moe_lab.py -h```


## Data format

Either plain text (one prompt per line) or JSONL with chat messages.

_Text_

```
Explain mixture-of-experts like I’m five.
Summarize this paragraph...
```

_JSONL_

```
{"messages":[{"role":"user","content":"Who are you?"}]}
{"messages":[{"role":"user","content":"Explain MoE experts."}]}
```


## Quick start

**Create a tiny stub so the command runs:**
```
  --oss-20b \
  --path runs/l18_e0_3_full/trainable.pt \
  --out merged/gpt-oss-20b_l18_e0_3
```

## (Optional) LoRA instead of full-weight

List LoRA targets first (should NOT include router)
```
moe-lab list-lora-targets --model openai/gpt-oss-20b --layer 18
```

## If targets exist:
```
  moe-lab train-expert \
  --model openai/gpt-oss-20b \
  --data data.jsonl \
  --method lora \
  --layer 18 --experts 0 3 \
  --steps 200 --batch-size 2 --lr 8e-5 \
  --out runs/l18_e0_3_lora
```


## Command reference

```train-expert```

- Fine-tunes MoE experts (or falls back to FFN in a layer).
	- ult bf16
	* --steps, --batch-size, --lr, --warmup, --max-len, --out

- Outputs
	* trainable.pt (full-weight path) or a LoRA adapter directory (PEFT).


```import-merge```

- Loads trainable.pt into a clean base model and optionally saves a merged checkpoint.
	- Args
	* --model, --path, --out (dir to write merged model)
 		* tensors are saved in trainable.pt to keep artifacts small and composable.

---

## Troubleshooting
* Pointer argument … cpu tensor?
	Inputs were on CPU while Triton ran a GPU kernel. This kit moves all fields to the embedding device automatically.
* element 0 of tensors does not require grad
	Router didn’t pick your unfrozen experts. Use --force-route or omit --experts to allow all experts in the layer.
* FileNotFoundError: trainable.pt after a run
	Training aborted before saving (usually due to the two issues above). Fix them and re-run.
* No LoRA targets found
	Your model uses fused/custom layers without obvious Linear names. The kit auto-falls back to full-weight; you can also inspect with list-lora-targets.


---

### CI

A minimal GitHub Actions workflow (.github/workflows/ci.yml) runs a sm

---

``list-lora-targets``
Prints discovered linear modules eligible for LoRA under expert/FFN paths (router excluded).
- Args
	* --model, --layer, --experts (filters)


## Design notes
* Device routing is automatic. Inputs are moved to the model’s embedding device; device_map="cuda" handles CUDA and ROCm (PyTorch aliases ROCm as “cuda”).
* Router bias (--force-route) bumps router biases for the selected experts at a layer so they win top-k; remove it for a brief router-only pass later to “breathe”.
* Grad guardrails. The trainer probes once to ensure loss.requires_grad before looping; if not, it applies bias or errors with a clear message.
* Minimal saves. Only trainable 
	Key args
	* --model (str): HF model id or local path
	*	--data (path): .txt or .jsonl (format above)
	*	--method (lora|full): adapters vs full-weight
	*	--layer (int): target layer index (optional; defaults to last if omitted)
	*	--experts (ints…): expert IDs to tune (omit to allow all experts in that layer)
	*	--force-route (flag): bias router so chosen experts get traffic (prevents “no grad” crashes)
	*	--dtype (bf16|fp16|fp32): defa
printf '{"messages":[{"role":"user","content":"who are you?"}]}\n' > data.jsonl
* Train experts (full-weight) with router bias
* Ensures the selected experts actually receive traffic → gradients flow.

**console script form (if installed):**
```
moe-lab train-expert \
  --model openai/gpt-oss-20b \
  --data data.jsonl \
  --method full \
  --layer 18 \
  --experts 0 3 \
  --force-route \
  --steps 200 --batch-size 2 --lr 8e-5 \
  --out runs/l18_e0_3_full
```

**or module form:**
``python moe_lab.py train-expert <...>``
This writes runs/l18_e0_3_full/trainable.pt.

**or merge the deltas into a fresh base**
``moe-lab import-merge \
  --model openai/gpt``

---

## Acknowledgements

Built to make expert specialization practical without fighting the framework. If this saves you an hour, it did its job.
* NVIDIA / CUDA: works out of the box.
* AMD / ROCm: PyTorch exposes ROCm as "cuda"; no changes needed here.
* Quantization: train in bf16/fp16; quantize post-hoc for inference.

---

### License / Citation

Distributed under MIT license.  
Citation: Payton Ison (isonpayton@icloud.com, https://github.com/paytonison)
