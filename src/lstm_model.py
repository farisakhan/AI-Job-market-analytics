"""LSTM baseline (needs PyTorch)."""
import numpy as np, torch, torch.nn as nn
from config import SEED

class _Net(nn.Module):
    def __init__(s, h=48):
        super().__init__(); s.l = nn.LSTM(1, h, 2, batch_first=True, dropout=0.1); s.o = nn.Linear(h, 1)
    def forward(s, x): return s.o(s.l(x.unsqueeze(-1))[0][:, -1]).squeeze(-1)

def fit_lstm(Xtr, ytr, Xte, epochs=150, lr=2e-3):
    torch.manual_seed(SEED); net = _Net(); opt = torch.optim.Adam(net.parameters(), lr)
    X = torch.tensor(Xtr, dtype=torch.float32); y = torch.tensor(ytr, dtype=torch.float32)
    for _ in range(epochs):
        perm = torch.randperm(len(X))
        for i in range(0, len(X), 128):
            b = perm[i:i + 128]; opt.zero_grad(); loss = nn.functional.huber_loss(net(X[b]), y[b]); loss.backward(); opt.step()
    with torch.no_grad(): return net(torch.tensor(Xte, dtype=torch.float32)).numpy()
