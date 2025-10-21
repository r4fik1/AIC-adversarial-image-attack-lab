import torch
import torch.nn as nn

def pgd_attack(model, images, labels, eps=0.03, alpha=0.007, iters=10, device='cpu'):
    images = images.to(device)
    labels = labels.to(device)
    ori_images = images.clone().detach()

    for i in range(iters):
        images.requires_grad = True
        outputs = model(images)
        loss = nn.CrossEntropyLoss()(outputs, labels)
        model.zero_grad()
        loss.backward()
        grad = images.grad.data
        images = images + alpha * grad.sign()
        eta = torch.clamp(images - ori_images, min=-eps, max=eps)
        images = torch.clamp(ori_images + eta, 0, 1).detach()

    return images
