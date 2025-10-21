import torch
import torch.nn as nn

def fgsm_attack(model, images, labels, eps=0.03, device='cpu'):
    model.eval()
    images = images.to(device).detach().requires_grad_(True)
    labels = labels.to(device)
    outputs = model(images)
    loss = nn.CrossEntropyLoss()(outputs, labels)
    model.zero_grad()
    loss.backward()
    data_grad = images.grad.data
    perturbed = images + eps * data_grad.sign()
    perturbed = torch.clamp(perturbed, 0, 1)
    return perturbed
