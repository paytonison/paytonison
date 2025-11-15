# MoE Lab

Minimal MoE/FFN expert tuner for GPT‑style models.

- Safe device handling for batches
- Optional router bias (`--force-route`) to guarantee gradients hit selected experts
- Full‑weight or LoRA tuning (if `peft` is installed)
- Saves only trainable deltas as `trainable.pt`, plus an import‑merge path

## Install

- Python 3.10+ and PyTorch installed for your platform
- Optional: `peft` for LoRA

You can run the script directly without installing a package:

```
python experiments/gpt-oss/moe_lab.py -h
```

## Data formats

Supports plain text (one prompt per line) or JSONL chat.

Text (`.txt`)
```
Explain mixture-of-experts like I’m five.
Summarize this paragraph...
```

JSONL (`.jsonl`)
```
{"messages":[{"role":"user","content":"Who are you?"}]}
{"messages":[{"role":"user","content":"Explain MoE experts."}]}
```

An example JSONL file is provided: `experiments/gpt-oss/examples/data.jsonl`.

## Quickstart

Train FFN fallback (full‑weight):
```
python experiments/gpt-oss/moe_lab.py train-expert \
  --model openai/gpt-oss-20b \
  --data experiments/gpt-oss/examples/data.jsonl \
  --method full \
  --layer 18 \
  --steps 20 --batch-size 2 --lr 8e-5 \
  --out runs/l18_ffn_full
```

Train experts with router bias (full‑weight):
```
python experiments/gpt-oss/moe_lab.py train-expert \
  --model openai/gpt-oss-20b \
  --data experiments/gpt-oss/examples/data.jsonl \
  --method full \
  --layer 18 --experts 0 3 --force-route \
  --steps 20 --batch-size 2 --lr 8e-5 \
  --out runs/l18_e0_3_full
```

List LoRA targets (if using adapters):
```
python experiments/gpt-oss/moe_lab.py list-lora-targets \
  --model openai/gpt-oss-20b --layer 18
```

Train with LoRA (if targets found):
```
python experiments/gpt-oss/moe_lab.py train-expert \
  --model openai/gpt-oss-20b \
  --data experiments/gpt-oss/examples/data.jsonl \
  --method lora \
  --layer 18 --experts 0 3 \
  --steps 20 --batch-size 2 --lr 8e-5 \
  --out runs/l18_e0_3_lora
```

Import and optionally save a merged checkpoint:
```
python experiments/gpt-oss/moe_lab.py import-merge \
  --model openai/gpt-oss-20b \
  --path runs/l18_e0_3_full/trainable.pt \
  --out merged/gpt-oss-20b_l18_e0_3
```

## Outputs

- Full‑weight: `out/trainable.pt` with only trainable tensors
- LoRA: adapter directory (saved via `save_pretrained`)

## Troubleshooting

- Loss has no grad: enable `--force-route` or widen selection (`--layer`, `--experts`).
- No LoRA targets: model uses fused/custom modules; fallback to `--method full`.
- Device issues: inputs are moved to the model’s device automatically; ensure your PyTorch install matches your hardware.

## License / Attribution

MIT. Built for controlled expert specialization and quick ablations.
