import torch
import torchvision.models as models
import torch.nn as nn

def load_resnet18(num_classes=10, pretrained=False, device='cpu'):
    model = models.resnet18(pretrained=pretrained)
    # adapt final layer
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model.to(device)
