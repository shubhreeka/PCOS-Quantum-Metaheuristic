"""Fusion MLP over quantum + classical features, and its training routine (shared by steps 3 and 4)."""
import numpy as np
import torch
import torch.nn as nn

EPOCHS = 60


class HybridPCOSClassifier(nn.Module):
    def __init__(self, quantum_dim=256, classical_dim=20, num_classes=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(quantum_dim + classical_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, num_classes),
        )

    def forward(self, q_feats, c_feats):
        return self.net(torch.cat((q_feats, c_feats), dim=1))


def fit_hybrid(Xq, Xc, y, tr, seed, epochs=EPOCHS):
    """Train on rows `tr`. Returns (model in eval mode, normalisation stats)."""
    torch.manual_seed(seed)
    stats = {"mu_q": Xq[tr].mean(0), "sd_q": Xq[tr].std(0) + 1e-6,
             "mu_c": Xc[tr].mean(0), "sd_c": Xc[tr].std(0) + 1e-6}
    q, c = normalise(Xq, Xc, stats)
    t = torch.tensor(y, dtype=torch.long)
    model = HybridPCOSClassifier(quantum_dim=q.shape[1], classical_dim=c.shape[1])
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-3)
    idx = torch.tensor(np.where(tr)[0])
    for _ in range(epochs):
        model.train()
        for b in idx[torch.randperm(len(idx))].split(64):
            if len(b) < 2:  # BatchNorm needs at least 2 samples
                continue
            opt.zero_grad()
            nn.functional.cross_entropy(model(q[b], c[b]), t[b]).backward()
            opt.step()
    model.eval()
    return model, stats


def normalise(Xq, Xc, stats):
    q = torch.tensor((Xq - stats["mu_q"]) / stats["sd_q"], dtype=torch.float32)
    c = torch.tensor((Xc - stats["mu_c"]) / stats["sd_c"], dtype=torch.float32)
    return q, c


def predict_hybrid(model, stats, Xq, Xc):
    q, c = normalise(Xq, Xc, stats)
    with torch.no_grad():
        return torch.softmax(model(q, c), 1)[:, 1].numpy()
