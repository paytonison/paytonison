#!/usr/bin/env python3
# moe_lab.py
# Minimal MoE lab kit for GPT-OSS-style MoE models.
# Commands:
#   - train-expert:   tune specific experts (LoRA or full)
#   - export-expert:  save expert-only weights/deltas
#   - import-expert:  load expert-only weights/deltas
#   - sharpen-router: router-only brief tune
#   - generate:       quick smoke test generation
#
# Assumptions:
#   * Single GPU is fine; multi-GPU will still work with device_map="cuda".
#   * Dataset is simple text (one prompt per line) or JSONL of {"messages":[...]}.
#   * LoRA requires `peft` installed; otherwise use --method full.
#
# Zero hand-holding: we force devices, dtypes, and sane defaults.

import os, sys, json, argparse, math, time, random
from typing import Iterable, Dict, Any, List, Optional

import torch
from torch.utils.data import Dataset, DataLoader

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    get_linear_schedule_with_warmup,
)

# -----------------------------
# Utilities
# -----------------------------

def log(s: str):
    print(f"[moe_lab] {s}", flush=True)

def load_model_and_tokenizer(model_name: str,
                             dtype: str = "bf16",
                             device_map: str = "cuda"):
    dtype_map = {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "fp32": torch.float32,
    }
    if dtype not in dtype_map:
        raise ValueError(f"Unsupported dtype: {dtype}")
    torch_dtype = dtype_map[dtype]

    tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        device_map=device_map,         # force onto accelerator(s)
        dtype=torch_dtype,             # modern arg; works with fp4 checkpoints too
    )

    # Ensure pad token exists for batching
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok, model

def embed_device(model) -> torch.device:
    return next(model.get_input_embeddings().parameters()).device

def to_model_input_device(model, batch):
    """Accepts dict or tensor, returns a dict on the correct device."""
    dev = embed_device(model)
    if isinstance(batch, dict):
        out = {}
        for k, v in batch.items():
            if hasattr(v, "to"):
                out[k] = v.to(dev)
            else:
                out[k] = v
        return out
    # tensor path (e.g., chat_template returned a pt tensor)
    return {
        "input_ids": batch.to(dev),
        "attention_mask": torch.ones_like(batch, dtype=torch.long, device=dev),
    }

def encode_messages(tok, messages: List[Dict[str, str]], model):
    # Robust path: render to string, then tokenize -> dict
    prompt = tok.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    enc = tok(prompt, return_tensors="pt", padding=False, truncation=True)
    return to_model_input_device(model, enc)

def iter_params(model, includes: List[str]) -> Iterable[torch.nn.Parameter]:
    for n, p in model.named_parameters():
        if any(s in n for s in includes):
            yield p

def freeze_all(model):
    for p in model.parameters():
        p.requires_grad_(False)

def set_trainable_experts(model, layer_idx: Optional[int], expert_ids: Optional[List[int]]):
    """Enable grad only for FFN expert modules that match criteria."""
    train_count = 0
    for n, p in model.named_parameters():
        train = False
        if ".experts." in n and (".ffn." in n or "mlp" in n or "gate" in n):
            train = True
            if layer_idx is not None and f".layers.{layer_idx}." not in n:
                train = False
            if expert_ids is not None:
                ok = any((f".expert_{eid}." in n) for eid in expert_ids)
                train = train and ok
        p.requires_grad_(train)
        train_count += int(p.requires_grad)
    log(f"Trainable params: {train_count}")
    return [p for p in model.parameters() if p.requires_grad]

def set_trainable_router_only(model):
    freeze_all(model)
    train_count = 0
    for n, p in model.named_parameters():
        if "router" in n:
            p.requires_grad_(True)
            train_count += 1
    log(f"Router trainable params: {train_count}")
    return [p for p in model.parameters() if p.requires_grad]

# -----------------------------
# LoRA (optional)
# -----------------------------

def try_enable_lora(model, target_filter: List[str], r: int = 16, alpha: int = 32, dropout: float = 0.0):
    try:
        from peft import LoraConfig, get_peft_model, TaskType
    except Exception as e:
        raise RuntimeError(
            "LoRA requested but `peft` is not installed. "
            "Install with: pip install peft"
        ) from e

    # Collect linear modules inside expert blocks
    target_modules = []
    for name, module in model.named_modules():
        if any(s in name for s in target_filter):
            # Heuristic: adapt linear-like layers
            if module.__class__.__name__.lower() in ("linear", "qlinear", "fusedlinear"):
                target_modules.append(name)

    if not target_modules:
        log("Warning: no target linear modules found for LoRA under specified filters; using broad match on '.experts.'")
        for name, module in model.named_modules():
            if ".experts." in name and module.__class__.__name__.lower() in ("linear", "qlinear", "fusedlinear"):
                target_modules.append(name)

    # Dedup
    target_modules = sorted(set(target_modules))
    log(f"LoRA targeting {len(target_modules)} linear modules.")
    cfg = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=r,
        lora_alpha=alpha,
        lora_dropout=dropout,
        target_modules=target_modules
    )
    from peft import get_peft_model
    lora_model = get_peft_model(model, cfg)
    lora_model.print_trainable_parameters()
    return lora_model

# -----------------------------
# Datasets
# -----------------------------

class TextOrJsonlDataset(Dataset):
    def __init__(self, path: str, tok, model, max_len: int = 4096):
        self.path = path
        self.lines = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                self.lines.append(line)
        self.tok = tok
        self.model = model
        self.max_len = max_len

    def __len__(self):
        return len(self.lines)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        line = self.lines[idx]
        # JSONL with messages
        if line.startswith("{") and "\"messages\"" in line:
            obj = json.loads(line)
            messages = obj["messages"]
            prompt = self.tok.apply_chat_template(messages, add_generation_prompt=False, tokenize=False)
            enc = self.tok(prompt, return_tensors="pt", truncation=True, max_length=self.max_len)
        else:
            enc = self.tok(line, return_tensors="pt", truncation=True, max_length=self.max_len)

        # Flatten batch dim
        enc = {k: v[0] for k, v in enc.items()}
        return enc

def collate_and_move(batch: List[Dict[str, torch.Tensor]], model):
    # Pad on CPU then move to model device
    keys = batch[0].keys()
    out = {}
    for k in keys:
        tensors = [b[k] for b in batch]
        out[k] = torch.nn.utils.rnn.pad_sequence(
            tensors, batch_first=True, padding_value=model.config.pad_token_id if hasattr(model.config, "pad_token_id") and model.config.pad_token_id is not None else 0
        )
    out = to_model_input_device(model, out)
    # LM labels
    out["labels"] = out["input_ids"].clone()
    return out

# -----------------------------
# Training loops
# -----------------------------

def train_loop(model, dataloader, steps: int, lr: float, warmup_ratio: float = 0.05, grad_clip: float = 1.0):
    model.train()
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr)
    total_steps = steps
    warmup_steps = int(total_steps * warmup_ratio)
    sched = get_linear_schedule_with_warmup(opt, warmup_steps, total_steps)

    step = 0
    losses = []
    t0 = time.time()
    for batch in dataloader:
        if step >= steps:
            break
        out = model(**batch)
        loss = out.loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, grad_clip)
        opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
        step += 1
        losses.append(loss.item())
        if step % 10 == 0:
            avg = sum(losses[-10:]) / min(len(losses), 10)
            log(f"step {step}/{steps}  loss={avg:.4f}")
    t1 = time.time()
    log(f"Done {step} steps in {t1-t0:.1f}s. Final loss ~ {losses[-1]:.4f}" if losses else "No steps run")
    return sum(losses)/len(losses) if losses else None

# -----------------------------
# Checkpoint I/O (expert-only)
# -----------------------------

def filter_state_dict_for_experts(sd: Dict[str, torch.Tensor],
                                  layer_idx: Optional[int],
                                  expert_ids: Optional[List[int]]):
    out = {}
    for k, v in sd.items():
        if ".experts." not in k:
            continue
        if layer_idx is not None and f".layers.{layer_idx}." not in k:
            continue
        if expert_ids is not None and not any((f".expert_{eid}." in k) for eid in expert_ids):
            continue
        out[k] = v
    return out

def save_expert_weights(model, save_dir: str, layer_idx: Optional[int], expert_ids: Optional[List[int]]):
    os.makedirs(save_dir, exist_ok=True)
    sd = model.state_dict()
    filt = filter_state_dict_for_experts(sd, layer_idx, expert_ids)
    if not filt:
        log("Warning: no expert weights matched selection; nothing saved.")
    path = os.path.join(save_dir, "experts.pt")
    torch.save(filt, path)
    log(f"Saved expert weights: {path} (keys={len(filt)})")

def load_expert_weights(model, load_path: str, strict: bool = False):
    filt = torch.load(load_path, map_location="cpu")
    missing, unexpected = model.load_state_dict(filt, strict=False)
    log(f"Loaded expert weights from {load_path}. Missing={len(missing)} Unexpected={len(unexpected)}")
    if strict and (missing or unexpected):
        raise RuntimeError("Strict load failed due to key mismatch.")

# -----------------------------
# Commands
# -----------------------------

def cmd_train_expert(args):
    tok, model = load_model_and_tokenizer(args.model, dtype=args.dtype, device_map="cuda")

    # Optional LoRA
    if args.method == "lora":
        model = try_enable_lora(model, target_filter=[".experts."], r=args.lora_r, alpha=args.lora_alpha, dropout=args.lora_dropout)

    if args.method == "full":
        freeze_all(model)
        _ = set_trainable_experts(model, args.layer, args.experts)

    ds = TextOrJsonlDataset(args.data, tok, model, max_len=args.max_len)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True,
                    collate_fn=lambda b: collate_and_move(b, model))

    train_loop(model, dl, steps=args.steps, lr=args.lr, warmup_ratio=args.warmup)

    # Save checkpoint/deltas
    os.makedirs(args.out, exist_ok=True)
    if args.method == "lora":
        try:
            from peft import PeftModel
            model.save_pretrained(args.out)
            log(f"Saved LoRA adapters to {args.out}")
        except Exception:
            log("LoRA save failed (is peft installed properly?). Saving full model as fallback.")
            model.save_pretrained(args.out)
    else:
        # Save only expert weights
        save_expert_weights(model, args.out, args.layer, args.experts)

def cmd_export_expert(args):
    _, model = load_model_and_tokenizer(args.model, dtype=args.dtype, device_map="cuda")
    save_expert_weights(model, args.out, args.layer, args.experts)

def cmd_import_expert(args):
    _, model = load_model_and_tokenizer(args.model, dtype=args.dtype, device_map="cuda")
    load_expert_weights(model, args.path, strict=args.strict)
    # Optionally save merged model
    if args.out:
        model.save_pretrained(args.out)
        log(f"Saved merged model to {args.out}")

def cmd_sharpen_router(args):
    tok, model = load_model_and_tokenizer(args.model, dtype=args.dtype, device_map="cuda")
    params = set_trainable_router_only(model)

    ds = TextOrJsonlDataset(args.data, tok, model, max_len=args.max_len)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True,
                    collate_fn=lambda b: collate_and_move(b, model))

    train_loop(model, dl, steps=args.steps, lr=args.lr, warmup_ratio=args.warmup)

    os.makedirs(args.out, exist_ok=True)
    # Save router-only weights
    router_sd = {k: v for k, v in model.state_dict().items() if "router" in k}
    torch.save(router_sd, os.path.join(args.out, "router.pt"))
    log(f"Saved router weights to {args.out}/router.pt")

def cmd_generate(args):
    tok, model = load_model_and_tokenizer(args.model, dtype=args.dtype, device_map="cuda")

    # Optional: load expert/router weights if provided
    if args.expert_path:
        load_expert_weights(model, args.expert_path, strict=False)
    if args.router_path:
        router_sd = torch.load(args.router_path, map_location="cpu")
        model.load_state_dict(router_sd, strict=False)

    if args.messages:
        messages = json.loads(args.messages)
        enc = encode_messages(tok, messages, model)
    else:
        # plain prompt
        enc = to_model_input_device(model, tok(args.prompt, return_tensors="pt"))

    with torch.inference_mode():
        out = model.generate(**enc, max_new_tokens=args.max_new_tokens, do_sample=not args.greedy, temperature=args.temperature)
    text = tok.decode(out[0], skip_special_tokens=True)
    print(text)

# -----------------------------
# Main / CLI
# -----------------------------

def build_parser():
    p = argparse.ArgumentParser("moe_lab")
    sub = p.add_subparsers(dest="cmd", required=True)

    common = dict(
        model = dict(type=str, default="openai/gpt-oss-20b"),
        dtype = dict(type=str, default="bf16", choices=["bf16","fp16","fp32"]),
        max_len = dict(type=int, default=4096),
    )

    # train-expert
    t = sub.add_parser("train-expert", help="Fine-tune selected experts (LoRA or full).")
    t.add_argument("--model", **common["model"]); t.add_argument("--dtype", **common["dtype"])
    t.add_argument("--data", type=str, required=True, help="Path to .txt or .jsonl of prompts/messages.")
    t.add_argument("--method", type=str, default="lora", choices=["lora","full"])
    t.add_argument("--layer", type=int, default=None, help="Restrict to a specific layer index.")
    t.add_argument("--experts", type=int, nargs="*", default=None, help="Expert IDs to tune (e.g., 0 3 7).")
    t.add_argument("--lora-r", type=int, default=16); t.add_argument("--lora-alpha", type=int, default=32)
    t.add_argument("--lora-dropout", type=float, default=0.0)
    t.add_argument("--batch-size", type=int, default=2)
    t.add_argument("--steps", type=int, default=200)
    t.add_argument("--lr", type=float, default=8e-5)
    t.add_argument("--warmup", type=float, default=0.05)
    t.add_argument("--max-len", **common["max_len"])
    t.add_argument("--out", type=str, required=True, help="Output dir for adapters or expert weights.")
    t.set_defaults(func=cmd_train_expert)

    # export-expert
    e = sub.add_parser("export-expert", help="Export expert-only weights to a .pt file.")
    e.add_argument("--model", **common["model"]); e.add_argument("--dtype", **common["dtype"])
    e.add_argument("--layer", type=int, default=None)
    e.add_argument("--experts", type=int, nargs="*", default=None)
    e.add_argument("--out", type=str, required=True)
    e.set_defaults(func=cmd_export_expert)

    # import-expert
    i = sub.add_parser("import-expert", help="Load expert-only weights into a base model; optionally save merged.")
    i.add_argument("--model", **common["model"]); i.add_argument("--dtype", **common["dtype"])
    i.add_argument("--path", type=str, required=True, help="Path to experts.pt")
    i.add_argument("--strict", action="store_true")
    i.add_argument("--out", type=str, default=None)
    i.set_defaults(func=cmd_import_expert)

    # sharpen-router
    r = sub.add_parser("sharpen-router", help="Router-only brief tune to tighten gating.")
    r.add_argument("--model", **common["model"]); r.add_argument("--dtype", **common["dtype"])
    r.add_argument("--data", type=str, required=True)
    r.add_argument("--batch-size", type=int, default=4)
    r.add_argument("--steps", type=int, default=200)
    r.add_argument("--lr", type=float, default=1e-4)
    r.add_argument("--warmup", type=float, default=0.05)
    r.add_argument("--max-len", **common["max_len"])
    r.add_argument("--out", type=str, required=True)
    r.set_defaults(func=cmd_sharpen_router)

    # generate
    g = sub.add_parser("generate", help="Quick generation test (optionally with imported experts/router).")
    g.add_argument("--model", **common["model"]); g.add_argument("--dtype", **common["dtype"])
    g.add_argument("--prompt", type=str, default=None)
    g.add_argument("--messages", type=str, default=None, help='JSON string of messages, e.g. \'[{"role":"user","content":"Hi"}]\'')
    g.add_argument("--expert-path", type=str, default=None)
    g.add_argument("--router-path", type=str, default=None)
    g.add_argument("--max-new-tokens", type=int, default=64)
    g.add_argument("--temperature", type=float, default=0.8)
    g.add_argument("--greedy", action="store_true")
    g.set_defaults(func=cmd_generate)

    return p

def main():
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)

if __name__ == "__main__":
    main()
