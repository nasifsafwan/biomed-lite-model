import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from src.backbone import SimpleBackbone
from src.head import ClassifierHead


class BiomedNet(nn.Module):
    """
    Full model: Backbone + Classifier Head
    """
    def __init__(self, in_channels=3, num_classes=7,
                 channels=[32, 64, 128, 256], num_blocks=[2, 2, 3, 2]):
        super().__init__()
        self.backbone = SimpleBackbone(in_channels, channels, num_blocks)
        self.head = ClassifierHead(self.backbone.out_channels, num_classes)

    def forward(self, x):
        feats = self.backbone(x)
        logits = self.head(feats)
        return logits


def train_one_epoch(model, loader, criterion, optimizer, device):
    """
    One training epoch
    """
    model.train()
    total_loss = 0.0
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)

        optimizer.zero_grad()
        logits = model(imgs)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * imgs.size(0)

    return total_loss / len(loader.dataset)


def evaluate(model, loader, criterion, device):
    """
    Simple evaluation loop
    """
    model.eval()
    total_loss, correct, n = 0.0, 0, 0
    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            logits = model(imgs)
            loss = criterion(logits, labels)
            total_loss += loss.item() * imgs.size(0)

            preds = logits.argmax(1)
            correct += (preds == labels).sum().item()
            n += labels.size(0)

    return total_loss / n, correct / n


def build_model(num_classes=7):
    """
    Helper to quickly build the model
    """
    return BiomedNet(num_classes=num_classes)
