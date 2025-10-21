import torch
from torchvision import transforms
from PIL import Image

to_tensor = transforms.ToTensor()
to_pil = transforms.ToPILImage()

def preprocess_pil(img_pil, size=(32,32)):
    # img_pil: PIL Image
    transform = transforms.Compose([
        transforms.Resize(size),
        transforms.ToTensor()
    ])
    return transform(img_pil).unsqueeze(0)  # batch dim

def postprocess_tensor(img_tensor):
    img_tensor = img_tensor.detach().cpu().squeeze(0)
    img_tensor = torch.clamp(img_tensor, 0, 1)
    return to_pil(img_tensor)
