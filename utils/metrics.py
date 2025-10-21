"""
Evaluation metrics: accuracy, perturbation norms, etc.
"""
import torch
import numpy as np

def accuracy(model, dataloader, device='cpu'):
    model.to(device)
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return correct / total if total > 0 else 0.0

def l_infty_norm(orig, adv):
    # orig and adv are tensors
    return (adv - orig).abs().view(orig.size(0), -1).max(dim=1)[0].cpu().numpy()
