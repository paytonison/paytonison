#!/usr/bin/env python3

import os
import json
import argparse
from typing import List, Optional, Dict, Any

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    get_linear_schedule_with_warmup,
)


def log(msg: str) -> None:
    print(f"[moe-lab] {msg}", flush=True)


def to_device(batch: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    return {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}


def load_model_and_tokenizer(model_name: str, dtype: str = "bf16", device_map: str = "auto"):
    dtypes = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}
    torch_dtype = dtypes[dtype]
    tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        torch_dtype=torch_dtype,
        device_map=device_map,
    )
    return model, tok


class PromptDataset(Dataset):
    def __init__(self, path: str):
        self.items: List[Dict[str, Any]] = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                s = line.strip()
                if not s:
                    continue
                if s.startswith("{"):
                    try:
                        obj = json.loads(s)
                        if isinstance(obj, dict) and "messages" in obj:
                            self.items.append(obj)
                            continue
                    except Exception:
                        pass
                self.items.append({"text": s})

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.items[idx]


def render_item_for_model(item: Dict[str, Any]) -> str:
    if "text" in item:
        return str(item["text"]).strip()
    msgs = item.get("messages", [])
    lines = []
    for m in msgs:
        role = m.get("role", "user").strip()
        content = m.get("content", "").strip()
        if role == "system":
            lines.append(f"System: {content}")
        elif role == "assistant":
            lines.append(f"Assistant: {content}")
        else:
            lines.append(f"User: {content}")
    lines.append("Assistant:")
    return "\n".join(lines)


def make_collate(tok, model, max_len: int):
    def collate(batch: List[Dict[str, Any]]):
        texts = [render_item_for_model(b) for b in batch]
        enc = tok(texts, padding=True, truncation=True, max_length=max_len, return_tensors="pt")
        enc["labels"] = enc["input_ids"].clone()
        pad_id = tok.pad_token_id
        enc["labels"][enc["input_ids"] == pad_id] = -100
        device = next(model.parameters()).device
        return to_device(enc, device)

    return collate


FFN_KEYS = ("mlp", "ffn", "feed_forward")
LINEAR_KEYS = ("gate_proj", "up_proj", "down_proj", "w1", "w2", "w3")


def select_trainable_params(model, layer_idx: Optional[int], expert_ids: Optional[List[int]]):
    cnt = 0
    for name, p in model.named_parameters():
        low = name.lower()
        if layer_idx is not None and f".layers.{layer_idx}." not in low:
            continue
        if ("experts" in low or "expert" in low or ".moe" in low) and ("weight" in low or "bias" in low):
            if expert_ids is not None and not any(f".expert_{e}." in low or f".experts.{e}." in low for e in expert_ids):
                continue
            p.requires_grad_(True)
            cnt += 1
    if cnt > 0:
        log(f"Trainable tensors (experts): {cnt}")
        return [p for p in model.parameters() if p.requires_grad]

    target_layer = layer_idx
    if target_layer is None:
        last = -1
        for name, _ in model.named_parameters():
            if ".layers." in name:
                try:
                    idx = int(name.split(".layers.")[1].split(".")[0])
                    last = max(last, idx)
                except Exception:
                    pass
        target_layer = last if last >= 0 else None

    cnt = 0
    for name, p in model.named_parameters():
        low = name.lower()
        if target_layer is not None and f".layers.{target_layer}." not in low:
            continue
        if any(k in low for k in FFN_KEYS + LINEAR_KEYS):
            if "router" in low:
                continue
            p.requires_grad_(True)
            cnt += 1
    if cnt == 0:
        raise RuntimeError("No trainable tensors matched (neither experts nor FFN fallback).")
    log(f"Trainable tensors (FFN layer {target_layer}): {cnt}")
    return [p for p in model.parameters() if p.requires_grad]


def discover_lora_targets(model, layer_idx: Optional[int] = None, expert_ids: Optional[List[int]] = None) -> List[str]:
    targets: List[str] = []
    for name, mod in model.named_modules():
        low = name.lower()
        if "router" in low:
            continue
        if layer_idx is not None and f".layers.{layer_idx}." not in low:
            continue
        if expert_ids is not None and not any(f".expert_{e}." in low or f".experts.{e}." in low for e in expert_ids):
            if "expert" in low or "experts" in low:
                continue
        w = getattr(mod, "weight", None)
        if isinstance(w, torch.Tensor) and w.ndim == 2:
            if any(k in low for k in ("experts", "expert", "moe") + FFN_KEYS + LINEAR_KEYS):
                targets.append(name)
    return sorted(set(targets))


def enable_lora(model, r: int, alpha: int, dropout: float, targets: List[str]):
    try:
        from peft import LoraConfig, TaskType, get_peft_model
    except Exception as e:
        raise RuntimeError("peft is required for LoRA (--method lora)") from e
    cfg = LoraConfig(task_type=TaskType.CAUSAL_LM, r=r, lora_alpha=alpha, lora_dropout=dropout, target_modules=targets)
    model = get_peft_model(model, cfg)
    try:
        model.print_trainable_parameters()
    except Exception:
        pass
    log(f"LoRA enabled on {len(targets)} modules.")
    return model


def bias_router_toward_experts(model, layer_idx: Optional[int], expert_ids: List[int], boost: float = 12.0):
    modified = []
    for name, mod in model.named_modules():
        low = name.lower()
        if "router" not in low:
            continue
        if layer_idx is not None and f".layers.{layer_idx}." not in low:
            continue
        b = getattr(mod, "bias", None)
        if isinstance(b, torch.Tensor) and b.ndim == 1:
            with torch.no_grad():
                for eid in expert_ids:
                    if 0 <= eid < b.numel():
                        b[eid] += boost
                        modified.append((name, eid))
    if modified:
        head = ", ".join(f"{n}[{e}]" for n, e in modified[:6])
        log(f"Router bias boosted at {len(modified)} positions: {head}{' ...' if len(modified) > 6 else ''}")
    else:
        log("Router bias boost found no matching modules.")


def ensure_grads_flow(model, tok, layer_idx, force_route: bool, expert_ids: Optional[List[int]]):
    probe = tok("hello", return_tensors="pt")
    device = next(model.parameters()).device
    probe = to_device(probe, device)
    loss = model(**probe, labels=probe["input_ids"]).loss
    if loss.requires_grad:
        return
    if force_route and expert_ids:
        bias_router_toward_experts(model, layer_idx, expert_ids)
        loss = model(**probe, labels=probe["input_ids"]).loss
        if loss.requires_grad:
            log("Grad flow restored via router bias.")
            return
    raise RuntimeError("Loss has no grad; router likely skipped trainable modules. Enable --force-route or widen selection.")


def save_trainable(model, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    sd = model.state_dict()
    keys = [n for n, p in model.named_parameters() if p.requires_grad]
    filt = {k: sd[k] for k in keys if k in sd}
    path = os.path.join(out_dir, "trainable.pt")
    torch.save(filt, path)
    log(f"Saved trainable weights: {path} (tensors={len(filt)})")


def load_partial_weights(model, path: str):
    filt = torch.load(path, map_location="cpu")
    missing, unexpected = model.load_state_dict(filt, strict=False)
    log(f"Loaded {len(filt)} tensors from {path}. Missing={len(missing)} Unexpected={len(unexpected)}")


def train_expert(args):
    if not os.path.exists(args.data):
        raise FileNotFoundError(f"Data file not found: {args.data}")

    model, tok = load_model_and_tokenizer(args.model, dtype=args.dtype)

    used_lora = False
    if args.method == "lora":
        targets = discover_lora_targets(model, args.layer, args.experts)
        if targets:
            model = enable_lora(model, args.lora_r, args.lora_alpha, args.lora_dropout, targets)
            used_lora = True
        else:
            log("No LoRA targets found; falling back to full-weight training of FFN layer.")

    if not used_lora:
        select_trainable_params(model, args.layer, args.experts)

    if args.force_route and args.experts:
        bias_router_toward_experts(model, args.layer, args.experts)

    ensure_grads_flow(model, tok, args.layer, args.force_route, args.experts)

    ds = PromptDataset(args.data)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True, collate_fn=make_collate(tok, model, args.max_len))

    trainables = [p for p in model.parameters() if p.requires_grad]
    if not trainables:
        raise RuntimeError("No trainable parameters selected.")

    opt = torch.optim.AdamW(trainables, lr=args.lr)
    total_steps = max(args.steps, 1)
    sched = get_linear_schedule_with_warmup(opt, num_warmup_steps=int(args.warmup * total_steps), num_training_steps=total_steps)

    model.train()
    step = 0
    while step < total_steps:
        for batch in dl:
            out = model(**batch)
            loss = out.loss
            if not loss.requires_grad:
                raise RuntimeError("Loss has no grad during training; try --force-route or different targets.")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainables, 1.0)
            opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
            step += 1
            if step % 10 == 0:
                log(f"step {step}/{total_steps} loss={loss.detach().float().item():.4f}")
            if step >= total_steps:
                break

    if used_lora and hasattr(model, "save_pretrained"):
        os.makedirs(args.out, exist_ok=True)
        model.save_pretrained(args.out)
        log(f"Saved LoRA adapters to {args.out}")
    else:
        save_trainable(model, args.out)


def import_merge(args):
    model, tok = load_model_and_tokenizer(args.model, dtype=args.dtype)
    load_partial_weights(model, args.path)
    if args.out:
        os.makedirs(args.out, exist_ok=True)
        model.save_pretrained(args.out)
        tok.save_pretrained(args.out)
        log(f"Saved merged model to: {args.out}")


def cmd_list_lora_targets(args):
    model, _ = load_model_and_tokenizer(args.model, dtype=args.dtype)
    t = discover_lora_targets(model, args.layer, args.experts)
    for name in t:
        print(name)
    log(f"{len(t)} LoRA targets found.")


def build_parser():
    p = argparse.ArgumentParser("moe-lab")
    sub = p.add_subparsers(dest="cmd", required=True)

    common_model = dict(type=str, default="openai/gpt-oss-20b")
    common_dtype = dict(type=str, default="bf16", choices=["bf16", "fp16", "fp32"])
    common_len = dict(type=int, default=4096)

    t = sub.add_parser("train-expert", help="Fine-tune MoE experts or FFN in one layer.")
    t.add_argument("--model", **common_model)
    t.add_argument("--dtype", **common_dtype)
    t.add_argument("--data", type=str, required=True)
    t.add_argument("--method", type=str, choices=["full", "lora"], default="full")
    t.add_argument("--layer", type=int, default=None)
    t.add_argument("--experts", type=int, nargs="*", default=None)
    t.add_argument("--force-route", action="store_true", help="Bias router toward --experts at --layer")
    t.add_argument("--lora-r", type=int, default=16)
    t.add_argument("--lora-alpha", type=int, default=32)
    t.add_argument("--lora-dropout", type=float, default=0.0)
    t.add_argument("--batch-size", type=int, default=2)
    t.add_argument("--steps", type=int, default=200)
    t.add_argument("--lr", type=float, default=8e-5)
    t.add_argument("--warmup", type=float, default=0.05)
    t.add_argument("--max-len", **common_len)
    t.add_argument("--out", type=str, required=True)
    t.set_defaults(func=lambda a: train_expert(a))

    i = sub.add_parser("import-merge", help="Load trainable.pt into a base model; optionally save merged.")
    i.add_argument("--model", **common_model)
    i.add_argument("--dtype", **common_dtype)
    i.add_argument("--path", type=str, required=True)
    i.add_argument("--out", type=str, default=None)
    i.set_defaults(func=lambda a: import_merge(a))

    l = sub.add_parser("list-lora-targets", help="List candidate Linear modules for LoRA in a layer/experts.")
    l.add_argument("--model", **common_model)
    l.add_argument("--dtype", **common_dtype)
    l.add_argument("--layer", type=int, default=None)
    l.add_argument("--experts", type=int, nargs="*", default=None)
    l.set_defaults(func=cmd_list_lora_targets)

    return p


def main():
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()

