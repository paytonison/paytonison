#!/usr/bin/env python3
# what.py — MoE/FFN expert tuner with safe device handling + grad guarantees

import os, json, argparse, time
from typing import List, Dict, Optional
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM, get_linear_schedule_with_warmup
from transformers.tokenization_utils_base import BatchEncoding

# ---------------- utils ----------------

def log(msg: str): print(f"[kit] {msg}", flush=True)

def load_model_and_tokenizer(model_name: str, dtype: str = "bf16", device_map: str = "cuda"):
    dt = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[dtype]
    tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        device_map=device_map,
        dtype=dt,  # HF now prefers dtype= over torch_dtype=
    )
    if tok.pad_token is None:
        tok.pad_tet(Dataset):
    def __init__(self, path: str):
        self.items = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                s = line.strip()
                if not s: continue
                if s.startswith("{") and '"messages"' in s:
                    self.items.append(json.loads(s))  # expects {"messages":[...]}
                else:
                    self.items.append({"text": s})
    def __len__(seel_input_device(model, enc)
        # LM labels
        enc["labels"] = enc["input_ids"].clone()
        # mask pads with -100 so they don't contribute to loss
        if "attention_mask" in enc:
            pad_id = model.config.pad_token_id
            pad_mask = (enc["input_ids"] == pad_id)
            enc["labels"][pad_mask] = -100
        return enc
    return collate

# ---------------- selection (experts / FFN) ----------------

FF_KEYS = ("mlp", "fayers.{layer_idx}." not in n:
                continue
            if expert_ids is not None and not any(f".expert_{e}." in n for e in expert_ids):
                continue
            p.requires_grad_(True); cnt += 1
    if cnt > 0:
        log(f"Trainable tensors (experts): {cnt}")
        return [p for p in model.parameters() if p.requires_grad]

    # Fallback to FFN params in one layer (given or last discovered)
    tgt = layer_idx
    if tgt is None:           layer={tgt}): {cnt}")
    if cnt == 0:
        raise RuntimeError("No trainable tensors matched (neither experts nor FFN fallback).")
    return [p for p in model.parameters() if p.requires_grad]

# ---------------- LoRA (optional) ----------------

def discover_lora_targets(model, layer_idx=None, expert_ids=None):
    targets = []
    for name, mod in model.named_modules():
        low = name.lower()
        if "router" in low:  # don't LoRA the router by default
            continue
        if any(t in low for t in ("experts", "expert", "moe", "ffn", "mlp", "feed_forward")):
            w = getattr(mod, "weight", None)
            if isinstance(w, torch.Tensor) and w.ndim == 2:
                if layer_idx is not None aodules():
            if any(k in name for k in ("gate_proj", "up_proj", "down_proj", "w1", "w2", "w3")):
                w = getattr(mod, "weight", None)
                if isinstance(w, torch.Tensor) and w.ndim == 2:
                    if layer_idx is not None and f".layers.{layer_idx}." not in name: continue
                    if expert_ids is not None and not any(f".expert_{e}." in name for e in expert_ids): continue
                    targets.append(name)
    return sorted(set(targets))

def enable_lora_or_fallb   cfg = LoraConfig(task_type=TaskType.CAUSAL_LM, r=r, lora_alpha=alpha,
                     lora_dropout=dropout, target_modules=targets)
    model = get_peft_model(model, cfg)
    try: model.print_trainable_parameters()
    except: pass
    log(f"LoRA enabled on {len(targets)} modules.")
    return model, True

# ---------------- router bias (force-route) ----------------

def bias_router_toward_experts(model, layer_idx: Optional[int], expert_ids: List[int], boost: float = 12.0):
    """
    Nudges router Linear biases so specified experts win top-k. Works for many MoE impls
    where router is a Lexpert_ids:
                if 0 <= eid < b.numel():
                    b[eid] += boost
                    modified.append((name, eid))
    if modified:
        head = ", ".join(f"{n}[{e}]" for n, e in modified[:6])
        log(f"Router bias boosted at {len(modified)} positions: {head}{' ...' if len(modified) > 6 else ''}")
    else:
        log("Router bias boost found no matching modules (continuing without force-route).")

# ---------------- grad safety ----------------

def ensure_grads_flow(model, tok, layer_idx, force_route: bool, expert_ids: Optional[List[int]]):
    """       if force_route and expert_ids:
        bias_router_toward_experts(model, layer_idx, expert_ids)
        loss = model(**enc).loss
        if loss.requires_grad:
            log("Grad flow restored via router bias.")
            return
    # last-resort: ensure at least one always-used param has grad
    for n, p in model.named_parameters():
        if "lm_head" in n:
            p.requires_grad_(True)
            break
    loss = model(**enc).loss
  time.time()
    while step < steps:
        for batch in dl:
            out = model(**batch)
            loss = out.loss
            if not loss.requires_grad:
                raise RuntimeError("Loss has no grad during training (router likely skipped trainable modules). Enable --force-route or widen selection.")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, grad_clip)
            opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
            step += 1
            losses.append(loss.detach().float().item())
            if step % 10 == 0:
                log(f"step {() if p.requires_grad]
    filt = {k: sd[k] for k in keys if k in sd}
    path = os.path.join(out_dir, "trainable.pt")
    torch.save(filt, path)
    log(f"Saved trainable weights: {path} (tensors={len(filt)})")

def load_partial_weights(model, path: str):
    filt = torch.load(path, map_location="cpu")
    missing, unexpected = model.load_state_dict(filt, strict=False)
    log(f"Loaded {len(filt)} tensors from {path}. Missing={len(missing)} Unexpected={len(unexpected)}")
route and args.experts:
        bias_router_toward_experts(model, args.layer, args.experts)

    # Make sure we actually have grads before spinning the loop
    ensure_grads_flow(model, tok, args.layer, args.force_route, args.experts)

    ds = PromptDataset(args.data)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True,
                    collate_fn=make_collate(tok, model, args.max_len))
                args.model, dtype=args.dtype, device_map="cuda")
    t = discover_lora_targets(model, args.layer, args.experts)
    for name in t: print(name)
    log(f"{len(t)} LoRA targets found.")

# ---------------- cli ----------------

def build_parser():
    p = argparse.ArgumentParser("what")
    sub = p.add_subparsers(dest="cmd", required=True)

    common_model = dict(type=str, default="openai/gpt-oss-20b")
    common_dtype = dict(type=str, default="bf16", choices=["bf16","fp16","fp32"])
    common_len   = dict(type=int, default=4096)

    t = sub.add_parser("train-expert", hstore_true", help="Bias router toward --experts at --layer")
    t.add_argument("--lora-r", type=int, default=16)
    t.add_argument("--lora-alpha", type=int, default=32)
    t.add_argument("--lora-dropout", type=float, default=0.0)
    t.add_argument("--batch-size", type=int, default=2)
    t.add_argument("--steps", type=int, default=20le names.")
    l.add_argument("--model", **common_model); l.add_argument("--dtype", **common_dtype)
    l.add_argument("--layer", type=int, default=None)
    l.add_argument("--experts", type=int, nargs="*", default=None)
    l.set_defaults(func=cmd_list_lora_targets)

    return p

def main():
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)

if __name__ == "__main__":
    main()0)
    t.add_argument("--lr", type=float, default=8e-5)
    t.add_argument("--warmup", type=float, default=0.05)
    t.add_argument("--max-len", **common_len)
    t.add_argument("--out", type=str, required=True)
    t.set_defaults(func=cmd_train_expert)

    i = sub.add_parser("import-merge", help="Load a saved trainable.pt into a base model; optionally save.")
    i.add_argument("--model", **common_model); i.add_argument("--dtype", **common_dtype)
    i.add_argument("--path", type=str, required=True)
    i.add_argument("--out", type=str, default=None)
    i.set_defaults(func=cmd_import_merge)

    l = sub.add_parser("list-lora-targets", help="Print detected LoRA target moduelp="Tune experts or FFN fallback (LoRA if possible).")
    t.add_argument("--model", **common_model); t.add_argument("--dtype", **common_dtype)
    t.add_argument("--data", type=str, required=True)
    t.add_argument("--method", type=str, default="lora", choices=["lora","full"])
    t.add_argument("--layer", type=int, default=None)
    t.add_argument("--experts", type=int, nargs="*", default=None)
    t.add_argument("--force-route", action="
    train_loop(model, dl, steps=args.steps, lr=args.lr, warmup_ratio=args.warmup)

    if used_lora and hasattr(model, "save_pretrained"):
        model.save_pretrained(args.out); log(f"Saved LoRA adapters to {args.out}")
    else:
        save_trainable_weights(model, args.out)

def cmd_import_merge(args):
    _, model = load_model_and_tokenizer(args.model, dtype=args.dtype, device_map="cuda")
    load_partial_weights(model, args.path)
    if args.out:
        model.save_pretrained(args.out); log(f"Saved merged model to {args.out}")

def cmd_list_lora_targets(args):
    _, model = load_model_and_tokenizer(
# ---------------- commands ----------------

def cmd_train_expert(args):
    if not os.path.exists(args.data):
        raise FileNotFoundError(f"Data file not found: {args.data}")
    tok, model = load_model_and_tokenizer(args.model, dtype=args.dtype, device_map="cuda")

    model, used_lora = enable_lora_or_fallback(
        model, args.method, args.layer, args.experts, args.lora_r, args.lora_alpha, args.lora_dropout
    )
    if not used_lora:
        _ = set_trainable_experts_or_ffn(model, args.layer, args.experts)

    if args.force_step}/{steps} loss={sum(losses[-10:])/min(10,len(losses)):.4f}")
            if step >= steps: break
    log(f"Finished {steps} steps in {time.time()-t0:.1f}s")
    return sum(losses)/len(losses)

# ---------------- save/load ----------------

def save_trainable_weights(model, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    sd = model.state_dict()
    keys = [n for n, p in model.named_parameters  if not loss.requires_grad:
        raise RuntimeError("Gradients still not flowing. Try removing --experts, or set --force-route with valid IDs.")

# ---------------- training ----------------

def train_loop(model, dl, steps: int, lr: float, warmup_ratio: float = 0.05, grad_clip: float = 1.0):
    model.train()
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr)
    warmup = int(max(1, steps * warmup_ratio))
    sched = get_linear_schedule_with_warmup(opt, warmup, steps)
    step = 0; losses=[]
    t0 = 
    Probe one forward pass; if loss has no grad, try to bias router; if still no grad,
    finally make lm_head trainable to avoid crashes (and warn).
    """
    enc = tok("probe", return_tensors="pt")
    enc = to_model_input_device(model, enc)
    enc["labels"] = enc["input_ids"].clone()
    model.train()
    loss = model(**enc).loss
    if loss.requires_grad:
        return
    log("No grads flowing with current selection.")
  inear(..., out_features=n_experts) with a bias.
    """
    modified = []
    for name, mod in model.named_modules():
        low = name.lower()
        if "router" not in low: continue
        if layer_idx is not None and f".layers.{layer_idx}." not in name: continue
        b = getattr(mod, "bias", None)
        if not isinstance(b, torch.Tensor): continue
        with torch.no_grad():
            for eid in ack(model, method: str, layer_idx, expert_ids, r, alpha, dropout):
    if method != "lora":
        return model, False
    try:
        from peft import LoraConfig, get_peft_model, TaskType
    except Exception:
        log("PEFT not installed; falling back to FULL FFN/experts.")
        return model, False
    targets = discover_lora_targets(model, layer_idx, expert_ids)
    if not targets:
        log("No LoRA targets found; falling back to FULL FFN/experts.")
        return model, False
 nd f".layers.{layer_idx}." not in name: continue
                if expert_ids is not None and not any(f".expert_{e}." in name for e in expert_ids): continue
                targets.append(name)
    # fallback to common LLaMA names
    if not targets:
        for name, mod in model.named_m
        last = -1
        for n, _ in model.named_parameters():
            if ".layers." in n:
                try:
                    idx = int(n.split(".layers.")[1].split(".")[0])
                    last = max(last, idx)
                except: pass
        tgt = last if last >= 0 else None

    cnt = 0
    for n, p in model.named_parameters():
        if tgt is not None and f".layers.{tgt}." not in n:
            continue
        if any(k in n for k in FF_KEYS):
            p.requires_grad_(True); cnt += 1
    log(f"Trainable tensors (FFN fallback,fn", "feed_forward", "gate_proj", "up_proj", "down_proj",
           "w1", "w2", "w3", "fc_in", "fc_out", "dense_h_to_4h", "dense_4h_to_h")

def set_trainable_experts_or_ffn(model, layer_idx: Optional[int], expert_ids: Optional[List[int]]):
    """
    Prefer MoE experts; if nothing matched, fall back to FFN in chosen (or last) layer.
    """
    freeze_all(model)
    cnt = 0
    # Try experts first
    for n, p in model.named_parameters():
        if any(tok in n for tok in (".experts.", ".expert_", ".moe.")):
            if layer_idx is not None and f".llf): return len(self.items)
    def __getitem__(self, i): return self.items[i]

def make_collate(tok, model, max_len: int):
    def collate(batch: List[Dict]):
        texts = []
        for item in batch:
            if "messages" in item:
                prompt = tok.apply_chat_template(item["messages"], add_generation_prompt=False, tokenize=False)
                texts.append(prompt)
            else:
                texts.append(item["text"])
        enc = tok(texts, return_tensors="pt", padding=True, truncation=True, max_length=max_len)
        enc = to_mod
    if getattr(model.config, "pad_token_id", None) is None:
        model.config.pad_token_id = tok.pad_token_id
    return tok, model

def embed_device(model) -> torch.device:
    return next(model.get_input_embeddings().parameters()).device

def to_model_input_device(model, batch):
    dev = embed_device(model)
    # Dict-like (incl. BatchEncoding): move each field
    if isinstance(batch, (dict, BatchEncoding)):
        out = {}
        for k, v in batch.items():
            out[k] = v.to(dev) if hasattr(v, "to") else v
        return out
    # Raw tensor: wrap as input_ids + synthesize mask
    if torch.is_tensor(batch):
        return {
            "input_ids": batch.to(dev),
            "attention_mask": torch.ones_like(batch, dtype=torch.long, device=dev),
        }
    raise TypeError(f"Unsupported batch type for to_model_input_device: {type(batch)}")

def freeze_all(model):
    for p in model.parameters():
        p.requires_grad_(False)

# ---------------- data ----------------

class PromptDatasoken = tok.eos_token
