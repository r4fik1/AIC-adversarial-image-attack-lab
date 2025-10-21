"""
Small visualization helpers: convert tensors <-> PIL, plotting utility.
"""
import numpy as np
from PIL import Image
import torchvision.transforms as T
import matplotlib.pyplot as plt

to_pil = T.ToPILImage()
to_tensor = T.ToTensor()

def tensor_to_pil(img_tensor):
    # expects [C,H,W] in 0..1
    return to_pil(img_tensor.cpu())

def pil_to_tensor(pil_img):
    return to_tensor(pil_img)

def show_comparison(orig_tensor, adv_tensor, figsize=(6,3), titles=('Original','Adversarial')):
    orig = tensor_to_pil(orig_tensor)
    adv = tensor_to_pil(adv_tensor)
    fig, axes = plt.subplots(1,2, figsize=figsize)
    axes[0].imshow(orig)
    axes[0].set_title(titles[0])
    axes[0].axis('off')
    axes[1].imshow(adv)
    axes[1].set_title(titles[1])
    axes[1].axis('off')
    plt.tight_layout()
    return fig
