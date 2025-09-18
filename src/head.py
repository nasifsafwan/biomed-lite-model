import torch
import torch.nn as nn


class ClassifierHead(nn.Module):
    """
    Global Average Pooling → Linear → Dropout → Linear
    Converts backbone feature maps into class logits.
    """

    def __init__(self, in_channels, num_classes, hidden_dim=512, dropout=0.2):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)  # shrink H×W to 1×1
        self.fc1 = nn.Linear(in_channels, hidden_dim)
        self.drop = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, num_classes)
        self.act = nn.SiLU()

    def forward(self, x):
        x = self.pool(x).flatten(1)     # [B, C, H, W] → [B, C]
        x = self.drop(self.act(self.fc1(x)))
        return self.fc2(x)              # [B, num_classes]
