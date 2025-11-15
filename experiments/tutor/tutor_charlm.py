# Minimal character-level LM with Tutor→Student forward-feedback (no backprop through Student)
# Usage: python tutor_charlm.py --txt_path your_corpus.txt
import argparse, os, math, random, sys
try:
    import torch
    from torch import nn
    import torch.nn.functional as F
except Exception:
    print("PyTorch not found. Install:\n  pip install torch\n(or CUDA build from pytorch.org)")
    sys.exit(1)

# ----------------------- Args -----------------------
def get_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--txt_path", type=str, required=True, help="Path to a UTF-8 text file")
    ap.add_argument("--seq_len", type=int, default=128)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--rank", type=int, default=16, help="LoRA-like low rank")
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--lr_tutor", type=float, default=1e-3)
    ap.add_argument("--ff_step", type=float, default=0.02, help="forward-feedback step size")
    ap.add_argument("--clip", type=float, default=0.25, help="tanh/clip on coeffs")
    ap.add_argument("--ema", type=float, default=0.9)
    ap.add_argument("--accept_eps", type=float, default=1e-4)
    ap.add_argument("--spsa_warmup", type=int, default=400, help="steps to use SPSA targets")
    ap.add_argument("--val_tokens", type=int, default=4000)
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return ap.parse_args()

# ----------------------- Data -----------------------
def load_text(path):
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()

def build_vocab(text):
    chars = sorted(list(set(text)))
    stoi = {ch:i for i,ch in enumerate(chars)}
    itos = {i:ch for ch,i in stoi.items()}
    ids  = torch.tensor([stoi[ch] for ch in text], dtype=torch.long)
    return stoi, itos, ids

def make_batch(ids, seq_len, batch, device):
    # random contiguous chunks
    L = ids.size(0) - (seq_len + 1)
    ix = torch.randint(0, L, (batch,))
    x  = torch.stack([ids[i:i+seq_len] for i in ix]).to(device)
    y  = torch.stack([ids[i+1:i+seq_len+1] for i in ix]).to(device)
    return x, y

def split_train_val(ids, val_tokens):
    val_tokens = min(val_tokens, ids.numel() // 10)
    return ids[:-val_tokens], ids[-val_tokens:]

# ----------------------- Models -----------------------
class Student(nn.Module):
    """
    Char-LM-ish student: token embedding + 2-layer MLP + low-rank output adapter we can scale forward-only.
    """
    def __init__(self, vocab, hidden, rank):
        super().__init__()
        self.emb = nn.Embedding(vocab, hidden)
        self.mlp = nn.Sequential(
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU()
        )
        self.head = nn.Linear(hidden, vocab, bias=False)
        # Low-rank "adapter"
        self.lora_A = nn.Linear(hidden, rank, bias=False)
        self.lora_B = nn.Linear(rank,  vocab, bias=False)
        nn.init.normal_(self.lora_A.weight, std=0.02)
        nn.init.normal_(self.lora_B.weight, std=0.02)

    def forward(self, x):
        # x: [B,T]
        h = self.emb(x)           # [B,T,H]
        h = self.mlp(h)           # [B,T,H]
        base = self.head(h)       # [B,T,V]
        lora = self.lora_B(self.lora_A(h))
        return base + lora

class Tutor(nn.Module):
    """
    Tiny MLP that maps batch-level error features -> rank-dim coefficients.
    """
    def __init__(self, rank, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(4, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, rank)
        )
    def forward(self, feat):  # feat: [4] = [loss, entropy, margin, ppl_est]
        return self.net(feat)

# ----------------------- Metrics/Helpers -----------------------
def loss_entropy_margin(logits, y):
    B,T,V = logits.shape
    loss = F.cross_entropy(logits.reshape(B*T, V), y.reshape(B*T), reduction="mean")
    probs = logits.softmax(-1)
    entropy = -(probs * (probs.clamp_min(1e-12).log())).sum(-1).mean()
    top2 = torch.topk(logits, k=2, dim=-1).values
    margin = (top2[...,0] - top2[...,1]).mean()
    return loss, entropy, margin

@torch.no_grad()
def ppl_from_loss(loss):
    return float(torch.exp(loss))

@torch.no_grad()
def eval_ppl(student, ids_val, seq_len, device):
    student.eval()
    V = student.head.weight.size(0)
    losses = []
    with torch.no_grad():
        # stride across validation tokens
        stride = seq_len
        for i in range(0, ids_val.numel() - (seq_len+1), stride):
            x = ids_val[i:i+seq_len].unsqueeze(0).to(device)
            y = ids_val[i+1:i+seq_len+1].unsqueeze(0).to(device)
            logits = student(x)
            loss,_,_ = loss_entropy_margin(logits, y)
            losses.append(loss)
    return math.exp(torch.stack(losses).mean().item()) if losses else float("inf")

@torch.no_grad()
def apply_feedback(student, coeff, step_size=0.02, clip=0.25, ema_buf=None, ema=0.9):
    c = coeff.tanh().clamp(-clip, clip)  # [R]
    if ema_buf is not None:
        if ema_buf.shape != c.shape:
            ema_buf.copy_(c)
        else:
            ema_buf.mul_(ema).add_(c, alpha=1-ema)
        c = ema_buf
    B = student.lora_B.weight           # [V,R]
    scale = (1.0 + step_size * c).view(1, -1)
    base_norm = torch.linalg.norm(B)
    B.mul_(scale)
    new_norm = torch.linalg.norm(B)
    if new_norm > 2.0 * base_norm.clamp_min(1e-6):
        B.mul_((2.0 * base_norm) / new_norm)

# ----------------------- Train -----------------------
def main():
    args = get_args()
    DEVICE = args.device
    torch.manual_seed(0)

    text = load_text(args.txt_path)
    stoi, itos, ids = build_vocab(text)
    V = len(stoi)
    ids_tr, ids_val = split_train_val(ids, args.val_tokens)

    student = Student(V, args.hidden, args.rank).to(DEVICE).eval()
    tutor   = Tutor(args.rank).to(DEVICE).train()
    opt_tutor = torch.optim.AdamW(tutor.parameters(), lr=args.lr_tutor)
    ema_buf = torch.zeros(args.rank, device=DEVICE)

    # quick baseline ppl
    base_ppl = eval_ppl(student, ids_val, args.seq_len, DEVICE)
    print(f"Initial validation PPL: {base_ppl:.3f}  (vocab={V}, seq_len={args.seq_len})")

    accept, total = 0, 0
    for step in range(1, args.steps+1):
        x, y = make_batch(ids_tr, args.seq_len, args.batch, DEVICE)
        with torch.no_grad():
            logits0 = student(x)
            loss0, ent0, mar0 = loss_entropy_margin(logits0, y)
        feat = torch.tensor([loss0.item(), ent0.item(), mar0.item(), ppl_from_loss(loss0)], device=DEVICE)

        # Target direction: SPSA warm-up then switch to learned tutor
        if step <= args.spsa_warmup:
            eps = 0.03
            z = torch.randint(0,2,(args.rank,), device=DEVICE).float()*2-1
            with torch.no_grad():
                B = student.lora_B.weight
                # +eps
                B.mul_((1+eps*z).view(1,-1))
                lp,_,_ = loss_entropy_margin(student(x), y)
                B.mul_((1/(1+eps*z)).view(1,-1))
                # -eps
                B.mul_((1-eps*z).view(1,-1))
                ln,_,_ = loss_entropy_margin(student(x), y)
                B.mul_((1/(1-eps*z)).view(1,-1))
            target = (z if (ln < lp) else -z).detach()
            pred = tutor(feat)
            t_loss = F.mse_loss(pred, target)
            opt_tutor.zero_grad(); t_loss.backward(); opt_tutor.step()
            coeff = pred.detach()
        else:
            coeff = tutor(feat).detach()
            t_loss = torch.tensor(0.0, device=DEVICE)

        # Apply + accept/reject
        with torch.no_grad():
            B_backup = student.lora_B.weight.detach().clone()
            apply_feedback(student, coeff, step_size=args.ff_step, clip=args.clip, ema_buf=ema_buf, ema=args.ema)
            logits1 = student(x)
            loss1,_,_ = loss_entropy_margin(logits1, y)
            improve = (loss0 - loss1).item()
            keep = improve >= args.accept_eps
            if not keep:
                student.lora_B.weight.copy_(B_backup)
            accept += int(keep); total += 1

        if step % 50 == 0:
            acc_rate = accept / max(1,total)
            print(f"[{step:04d}] loss {loss0.item():.4f}->{loss1.item():.4f} Δ={improve:.5f} "
                  f"keep={int(keep)} acc={acc_rate:.2f} tutor_loss={t_loss.item():.4f}")
            accept, total = 0, 0

        if step % 300 == 0:
            ppl = eval_ppl(student, ids_val, args.seq_len, DEVICE)
            print(f"  ↳ val PPL: {ppl:.3f}")

    final_ppl = eval_ppl(student, ids_val, args.seq_len, DEVICE)
    print(f"Final validation PPL: {final_ppl:.3f}")

if __name__ == "__main__":
    main()

