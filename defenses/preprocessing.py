import torch
import torchvision.transforms as T

def jpeg_compress_decompress(img_tensor, quality=75):
    # img_tensor: 1xCxHxW tensor (0..1)
    from io import BytesIO
    from PIL import Image
    import torchvision

    pil = T.ToPILImage()(img_tensor.squeeze(0).cpu())
    buff = BytesIO()
    pil.save(buff, format='JPEG', quality=quality)
    buff.seek(0)
    pil2 = Image.open(buff).convert('RGB')
    return T.ToTensor()(pil2).unsqueeze(0)
