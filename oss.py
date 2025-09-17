import math, torch
import torch.nn as nn
import torch.nn.functional as F

class RMSNorm(nn.Module):
    def __init__(self, d, eps=1e-5): super().__init__(); self.w = nn.Parameter(torch.ones(d)); self.eps=eps
    def forward(self, x): return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True)+self.eps) * self.w

def rope(q, k, base=10000.0):
    # rotary embeddings (even/odd interleave)
    d = q.shape[-1]; half = d//2
    freqs = torch.arange(half, device=q.device) / half
    theta = 1.0 / (base ** freqs)                   # [half]
    t = torch.arange(q.shape[-2], device=q.device)  # seq
    ang = torch.einsum('t,h->t h', t, theta)        # [T,half]
    cos, sin = ang.cos()[None,None,:,:], ang.sin()[None,None,:,:]  # [1,1,T,half]
    def rot(x):
        x1, x2 = x[..., :half], x[..., half:]
        return torch.cat([x1*cos - x2*sin, x1*sin + x2*cos], dim=-1)
    return rot(q), rot(k)

class SelfAttn(nn.Module):
    def __init__(self, d, nheads):
        super().__init__()
        self.nh = nheads; self.dh = d//nheads
        self.qkv = nn.Linear(d, 3*d, bias=False); self.o = nn.Linear(d, d, bias=False)
    def forward(self, x):
        B,T,D = x.shape
        q,k,v = self.qkv(x).chunk(3, dim=-1)
        q = q.view(B,T,self.nh,self.dh).transpose(1,2)  # [B,H,T,dh]
        k = k.view(B,T,self.nh,self.dh).transpose(1,2)
        v = v.view(B,T,self.nh,self.dh).transpose(1,2)
        q,k = rope(q,k)
        att = (q @ k.transpose(-2,-1)) / math.sqrt(self.dh)
        mask = torch.triu(torch.full((T,T), -float('inf'), device=x.device), 1)
        att = att + mask                           # causal
        w = att.softmax(-1)
        y = (w @ v).transpose(1,2).contiguous().view(B,T,D)
        return self.o(y)

class MLP(nn.Module):
    def __init__(self, d, mult=4):
        super().__init__()
        self.w1 = nn.Linear(d, mult*d, bias=False)
        self.wg = nn.Linear(d, mult*d, bias=False)  # gate
        self.w2 = nn.Linear(mult*d, d, bias=False)
    def forward(self, x):
        return self.w2(F.silu(self.wg(x)) * self.w1(x))  # SWiGLU-ish

class Block(nn.Module):
    def __init__(self, d, nheads):
        super().__init__(); self.n1=RMSNorm(d); self.attn=SelfAttn(d,nheads); self.n2=RMSNorm(d); self.mlp=MLP(d)
    def forward(self, x): x = x + self.attn(self.n1(x)); x = x + self.mlp(self.n2(x)); return x

class TinyGPT(nn.Module):
    def __init__(self, vocab, d=512, nheads=8, nlayers=6):
        super().__init__()
        self.emb = nn.Embedding(vocab, d); self.pos = nn.Embedding(4096, d)
        self.blocks = nn.ModuleList([Block(d,nheads) for _ in range(nlayers)])
        self.norm = RMSNorm(d); self.lm = nn.Linear(d, vocab, bias=False)
    def forward(self, idx):
        B,T = idx.shape
        x = self.emb(idx) + self.pos(torch.arange(T, device=idx.device))[None,:]
        for b in self.blocks: x = b(x)
        return self.lm(self.norm(x))  # [B,T,V]

@torch.no_grad()
def generate(model, idx, max_new=50, temperature=1.0, top_p=1.0, eos=None):
    for _ in range(max_new):
        logits = model(idx)[:, -1, :] / max(temperature, 1e-6)
        if top_p < 1.0:  # nucleus sampling
            probs = logits.softmax(-1)
            sorted_probs, sorted_idx = torch.sort(probs, descending=True)
            cum = torch.cumsum(sorted_probs, dim=-1)
            keep = cum <= top_p
            keep[...,0] = True
            thresh = sorted_probs[~keep].min() if (~keep).any() else 0.0
            logits[probs < thresh] = -float('inf')
        next_id = torch.multinomial(logits.softmax(-1), num_samples=1)
        idx = torch.cat([idx, next_id], dim=1)
        if eos is not None and next_id.item() == eos: break
    return idx

if __name__ == "__main__":
    # toy demo: random weights, fake vocab, greedy decode from token 1
    torch.manual_seed(0)
    V = 32000
    m = TinyGPT(V, d=256, nheads=8, nlayers=4).eval().to("cuda" if torch.cuda.is_available() else "cpu")
    x = torch.tensor([[1]], device=next(m.parameters()).device)
    out = generate(m, x, max_new=10, temperature=0.8, top_p=0.9)
    print(out.tolist())
