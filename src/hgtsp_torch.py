"""PROPOSED MODEL - Hybrid Graph Transformer Skill Prediction (HGTSP).  Requires PyTorch.

Architecture (one training sample = one time step t, ALL skills at once):
  1. Temporal Transformer encoder per skill: tokens = [own demand_t, graph-neighbour demand_t, sin/cos(month)]
     + learned positional embedding -> 2-layer TransformerEncoder -> attention-pooling (learned query)
     => skill state h_s  (+ temporal attention weights alpha_t  -> explainability)
  2. Skill embedding (initialised from SBERT/TF-IDF vectors, fine-tuned) + static graph features
     (PageRank, weighted degree, clustering) are fused into h_s through a gated linear unit.
  3. Graph Attention layer over the skill knowledge graph (mask = KG co-occurrence adjacency):
     h'_s = sum_j softmax_j(LeakyReLU(a^T[W h_s || W h_j])) W h_j   (+ attention weights beta_sj -> explainability)
  4. Head: [h_s || h'_s] -> MLP -> predicted normalised demand at t+H.
  5. Attention-based ranking: skills are ranked by predicted growth, with beta/alpha used as explanations.
"""
import numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as F
from config import LOOKBACK, HORIZON, SEED

class GAT(nn.Module):
    def __init__(s, d):
        super().__init__(); s.W = nn.Linear(d, d, bias=False); s.a_src = nn.Linear(d, 1, bias=False); s.a_dst = nn.Linear(d, 1, bias=False)
    def forward(s, h, mask):                          # h: B,S,d ; mask: S,S (bool)
        Wh = s.W(h)
        e = F.leaky_relu(s.a_src(Wh) + s.a_dst(Wh).transpose(1, 2), 0.2)   # B,S,S
        e = e.masked_fill(~mask.unsqueeze(0), -1e9)
        beta = torch.softmax(e, -1)
        return F.elu(beta @ Wh), beta

class HGTSP(nn.Module):
    def __init__(s, n_skills, g_dim, L=LOOKBACK, d=48, heads=4, emb_init=None):
        super().__init__()
        s.L, s.d = L, d
        s.inp = nn.Linear(4, d); s.pos = nn.Parameter(torch.randn(1, L, d) * 0.02)
        s.enc = nn.TransformerEncoder(nn.TransformerEncoderLayer(d, heads, 2 * d, 0.1, batch_first=True, norm_first=True), 2)
        s.q = nn.Parameter(torch.randn(d) * 0.1)
        s.emb = nn.Embedding(n_skills, d)
        if emb_init is not None: s.emb.weight.data.copy_(emb_init)
        s.gproj = nn.Linear(g_dim, d); s.gate = nn.Linear(2 * d, 2 * d)
        s.fuse = nn.Linear(2 * d, d)
        s.gat = GAT(d)
        s.head = nn.Sequential(nn.Linear(2 * d, d), nn.GELU(), nn.Dropout(0.1), nn.Linear(d, 1))
    def forward(s, x, xn, moy_sin, moy_cos, gfeat, mask):
        B, S, L = x.shape
        tok = torch.stack([x, xn, moy_sin.expand_as(x), moy_cos.expand_as(x)], -1).reshape(B * S, L, 4)
        z = s.enc(s.inp(tok) + s.pos)                                      # BS,L,d
        alpha = torch.softmax((z @ s.q) / np.sqrt(s.d), -1)               # temporal attention
        h = (alpha.unsqueeze(-1) * z).sum(1).reshape(B, S, s.d)
        stat = torch.cat([s.emb.weight, s.gproj(gfeat)], -1).unsqueeze(0).expand(B, -1, -1)
        g = torch.sigmoid(s.gate(stat)); h = s.fuse(g * stat) + h           # gated fusion of static info
        h2, beta = s.gat(h, mask)
        y = s.head(torch.cat([h, h2], -1)).squeeze(-1)                      # B,S
        return y, alpha.reshape(B, S, L), beta

def _tensors(share, A, L, H, ts):
    Y = share.values.T.astype(np.float32); S, T = Y.shape
    nbr = A @ Y
    X, XN, Tg, SC, MS, MC = [], [], [], [], [], []
    for t in ts:
        w = Y[:, t - L + 1:t + 1]; wn = nbr[:, t - L + 1:t + 1]; sc = w.mean(1, keepdims=True) + 1e-3
        X.append(w / sc); XN.append(wn / (wn.mean(1, keepdims=True) + 1e-3)); SC.append(sc[:, 0])
        Tg.append(Y[:, t + H] / sc[:, 0] if t + H < T else np.zeros(S, np.float32))
        mo = share.index[min(t + H, T - 1)].month
        MS.append(np.sin(2 * np.pi * mo / 12)); MC.append(np.cos(2 * np.pi * mo / 12))
    f = lambda a: torch.tensor(np.array(a), dtype=torch.float32)
    return f(X), f(XN), f(Tg), f(MS).view(-1, 1, 1), f(MC).view(-1, 1, 1)

def build_model(share, A, gfeat, emb=None):
    S = share.shape[1]
    g = torch.tensor(((gfeat - gfeat.mean()) / (gfeat.std() + 1e-6)).fillna(0).values, dtype=torch.float32)
    mask = torch.tensor((A > 0) | (A.T > 0) | np.eye(S, dtype=bool))
    return HGTSP(S, g.shape[1], emb_init=None if emb is None else torch.tensor(emb, dtype=torch.float32)), g, mask

def fit_hgtsp(share, A, gfeat, tr, te, split, epochs=120, lr=2e-3, emb=None):
    """Returns normalised predictions aligned with rows of `te` (caller rescales)."""
    torch.manual_seed(SEED); np.random.seed(SEED)
    L, H = LOOKBACK, HORIZON
    model, g, mask = build_model(share, A, gfeat, emb)
    t_tr = sorted(tr.t.unique()); t_te = sorted(te.t.unique())
    X, XN, Tg, MS, MC = _tensors(share, A, L, H, t_tr)
    opt = torch.optim.AdamW(model.parameters(), lr, weight_decay=1e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    n_val = max(2, len(t_tr) // 6); best, bad, best_state = 1e9, 0, None
    for ep in range(epochs):
        model.train(); perm = torch.randperm(len(t_tr) - n_val)
        for i in range(0, len(perm), 8):
            b = perm[i:i + 8]; opt.zero_grad()
            y, _, _ = model(X[b], XN[b], MS[b], MC[b], g, mask)
            F.huber_loss(y, Tg[b]).backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        sched.step(); model.eval()
        with torch.no_grad():
            yv, _, _ = model(X[-n_val:], XN[-n_val:], MS[-n_val:], MC[-n_val:], g, mask)
            vl = F.l1_loss(yv, Tg[-n_val:]).item()
        if vl < best: best, bad, best_state = vl, 0, {k: v.clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad > 20: break
    model.load_state_dict(best_state); model.eval()
    Xt, XNt, _, MSt, MCt = _tensors(share, A, L, H, t_te)
    with torch.no_grad(): P, alpha, beta = model(Xt, XNt, MSt, MCt, g, mask)
    P = P.numpy(); tpos = {t: i for i, t in enumerate(t_te)}
    fit_hgtsp.last = dict(model=model, alpha=alpha.numpy(), beta=beta.numpy(), t_te=t_te)   # expose attention for XAI
    return np.array([P[tpos[t], s] for s, t in zip(te.s, te.t)])
