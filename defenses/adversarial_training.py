import torch
import torch.nn as nn

def adversarial_train_epoch(model, dataloader, optimizer, attack_fn, eps, device='cpu'):
    model.train()
    total_loss = 0.0
    for images, labels in dataloader:
        images, labels = images.to(device), labels.to(device)
        # create adversarial examples on-the-fly
        adv_images = attack_fn(model, images, labels, eps=eps, device=device)
        outputs = model(adv_images)
        loss = nn.CrossEntropyLoss()(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * images.size(0)
    return total_loss / len(dataloader.dataset)
