import torch
import torch.nn as nn

class HybridPCOSClassifier(nn.Module):
    def __init__(self, quantum_dim=64, classical_dim=256, num_classes=2):
        super(HybridPCOSClassifier, self).__init__()
        self.in_dim = quantum_dim + classical_dim
        self.net = nn.Sequential(
            nn.Linear(self.in_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, num_classes)
        )

    def forward(self, q_feats, c_feats):
        fused = torch.cat((q_feats, c_feats), dim=1)
        return self.net(fused)