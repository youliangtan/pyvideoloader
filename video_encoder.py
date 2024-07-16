"""
Test code for
https://github.com/TencentARC/Open-MAGVIT2?tab=readme-ov-file

Checkpoints: https://huggingface.co/TencentARC/Open-MAGVIT2/tree/main
"""

import torch
from PIL import Image
import numpy as np
from omegaconf import OmegaConf

from taming.models.lfqgan import VQModel

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def load_config(config_path, display=False):
    config = OmegaConf.load(config_path)
    if display:
        print(OmegaConf.to_yaml(config))
    return config


def load_vqgan_new(config, ckpt_path=None, is_gumbel=False):
    model = VQModel(**config.model.init_args)
    if ckpt_path is not None:
        sd = torch.load(ckpt_path, map_location="cpu")["state_dict"]
        model.load_state_dict(sd, strict=False)
    return model.eval()


def encode_decode_image(model, image_path, image_size=256):
    img = Image.open(image_path).convert('RGB')
    img = img.resize((image_size, image_size), Image.LANCZOS)
    img_tensor = torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 255.0
    img_tensor = img_tensor.unsqueeze(0) * 2 - 1  # Normalize to [-1, 1]
    img_tensor = img_tensor.to(DEVICE)

    img_size = img_tensor.element_size() * img_tensor.nelement()
    print("Image tensor shape:", img_tensor.shape)
    print("Image size in bytes:", img_size)

    # Encoding and decoding
    with torch.no_grad():
        quant, diff, indices, _ = model.encode(img_tensor)
        reconstructed_img = model.decode(quant)

        quant_size = quant.element_size() * quant.nelement()
        print("Quant shape:", quant.shape)
        print("Quant size in bytes:", quant_size)

        print(" => COMPRESSION FACTOR = ", round(img_size / quant_size, 2))

    # Convert back to PIL Image
    reconstructed_img = reconstructed_img.squeeze(0).permute(1, 2, 0).cpu().numpy()
    reconstructed_img = (reconstructed_img + 1) / 2 * 255  # Denormalize
    reconstructed_img = Image.fromarray(reconstructed_img.astype(np.uint8))
    return reconstructed_img


if __name__ == "__main__":
    config_path = 'configs/imagenet_lfqgan_256_B.yaml'
    ckpt_path = 'imagenet_256_B.ckpt'
    image_path = 'image.png'

    config_model = load_config(config_path, display=False)
    model = load_vqgan_new(config_model, ckpt_path=ckpt_path).to(DEVICE)

    reconstructed_image = encode_decode_image(model, image_path)
    reconstructed_image.show()
