# tutor_minimal.py — tiny forward-feedback demo without Transformers/PEFT
import sys, math, random
try:
    import torch
    from torch import nn
except Exception as e:
    print("PyTorch not found. Install with:\n  pip install torch --extra-index-url https://download.pytorch.org/whl/cu121\n(or just 'pip install torch' for CPU)")
    sys.exit(1)

torch.manual_seed(0)

# --- Tiny synthetic LM-ish task: predict next id in a modulo sequence ---
V = 50          # vocab
T = 16          # sequence length
B = 32          # batch
H = 128         # hidden
R = 8           # low-rank "LoRA-like" rank
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def make_batch(batch=B, T=T, V=V):
    x = torch.randint(0, V, (batch, T), device=DEVICE)
    y = (x + 1) % V  # next-token = (token+1) mod V
    return x, y

# --- Student: tiny MLP with a low-rank output adapter we can scale forward-only ---
class Student(nn.Module):
    def __init__(self, V, H, R):
        super().__init__()
        self.emb = nn.Embedding(V, H)
        self.mlp = nn.Sequential(nn.Linear(H, H), nn.ReLU(), nn.Linear(H, H), nn.ReLU())
        self.head = nn.Linear(H, V, bias=False)
        # "LoRA-B" style: extra low-rank factor at the head we can scale without backprop
        self.lora_A = nn.Linear(H, R, bias=False)
        self.lora_B = nn.Linear(R, V, bias=False)
        nn.init.normal_(self.lora_A.weight, std=0.02)
        nn.init.normal_(self.lora_B.weight, std=0.02)

    def forward(self, x):
        h = self.emb(x)                    # [B,T,H]
        h = self.mlp(h)                    # [B,T,H]
        base = self.head(h)                # [B,T,V]
        lora = self.lora_B(self.lora_A(h)) # [B,T,V]
        return base + lora

# --- Tutor: maps simple error features -> low-rank coeffs (R) we can apply by scaling lora_B ---
class Tutor(nn.Module):
    def __init__(self, R, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3, hidden), nn.ReLU(),
            nn.Linear(hidden, R)
        )

    def forward(self, feat):  # feat: [3] (loss, entropy, margin)
        return self.net(feat)  # [R]

def loss_entropy_margin(logits, y):
    # cross-entropy on all positions
    B,T,V = logits.shape
    loss = torch.nn.functional.cross_entropy(
        logits.reshape(B*T, V), y.reshape(B*T), reduction='mean'
    )
    probs = logits.softmax(-1)
    entropy = -(probs * (probs.clamp_min(1e-12).log())).sum(-1).mean()
    top2 = torch.topk(logits, k=2, dim=-1).values
    margin = (top2[...,0]-top2[...,1]).mean()
    return loss, entropy, margin

@torch.no_grad()
def apply_feedback(student, coeff, step_size=0.02, clip=0.3, ema_buf=None, ema=0.9):
    # scale columns of lora_B by (1 + lr * coeff)
    c = coeff.tanh().clamp(-clip, clip)   # [R]
    if ema_buf is not None:
        if ema_buf.shape != c.shape:
            ema_buf.copy_(c)
        else:
            ema_buf.mul_(ema).add_(c, alpha=1-ema)
        c = ema_buf
    B = student.lora_B.weight             # [V,R]
    scale = (1.0 + step_size * c).view(1,-1)
    base_norm = torch.linalg.norm(B)
    B.mul_(scale)
    # simple guardrail
    new_norm = torch.linalg.norm(B)
    if new_norm > 2.0 * base_norm.clamp_min(1e-6):
        B.mul_((2.0 * base_norm) / new_norm)

def main():
    student = Student(V,H,R).to(DEVICE).eval()
    tutor   = Tutor(R).to(DEVICE).train()
    opt_tutor = torch.optim.AdamW(tutor.parameters(), lr=1e-3)
    ema_buf = torch.zeros(R, device=DEVICE)

    steps = 600
    accept_eps = 1e-4
    accept, total = 0, 0

    for step in range(steps):
        x,y = make_batch()
        with torch.no_grad():
            logits0 = student(x)
            base_loss, ent, mar = loss_entropy_margin(logits0, y)
        feat = torch.tensor([base_loss.item(), ent.item(), mar.item()], device=DEVICE)

        # Train tutor a tiny bit to point in good direction (uses label from SPSA 2x forward)
        # SPSA to get a pseudo-sign without backprop through student
        eps = 0.03
        z = torch.randint(0,2,(R,), device=DEVICE).float()*2-1

        # +eps
        with torch.no_grad():
            B = student.lora_B.weight
            B.mul_((1+eps*z).view(1,-1))
            logits_pos = student(x)
            loss_pos,_,_ = loss_entropy_margin(logits_pos,y)
            # revert
            B.mul_((1/(1+eps*z)).view(1,-1))
            # -eps
            B.mul_((1-eps*z).view(1,-1))
            logits_neg = student(x)
            loss_neg,_,_ = loss_entropy_margin(logits_neg,y)
            B.mul_((1/(1-eps*z)).view(1,-1))

        target = (z if (loss_neg < loss_pos) else -z).detach()  # desired direction
        pred = tutor(feat)                                      # [R]
        tutor_loss = torch.nn.functional.mse_loss(pred, target)
        opt_tutor.zero_grad()
        tutor_loss.backward()
        opt_tutor.step()

        # Apply tutor’s coeff and keep only if improves
        with torch.no_grad():
            # snapshot
            B_backup = student.lora_B.weight.detach().clone()
            apply_feedback(student, pred.detach(), step_size=0.02, clip=0.25, ema_buf=ema_buf)

            logits1 = student(x)
            new_loss,_,_ = loss_entropy_margin(logits1, y)
            improve = (base_loss - new_loss).item()
            keep = improve >= accept_eps
            if not keep:
                student.lora_B.weight.copy_(B_backup)
            accept += int(keep); total += 1

        if step % 20 == 0:
            print(f"[{step:04d}] base={base_loss.item():.4f} new={new_loss.item():.4f} Δ={improve:.5f} "
                  f"keep={int(keep)} acc_rate={(accept/max(1,total)):.2f} tutor_loss={tutor_loss.item():.4f}")
            accept, total = 0, 0

    # quick held-out check
    with torch.no_grad():
        x,y = make_batch(batch=256)
        logits = student(x)
        val_loss,_,_ = loss_entropy_margin(logits,y)
        print(f"Final held-out loss: {val_loss.item():.4f}")

if __name__ == "__main__":
    main()

