#!/usr/bin/env python
# coding: utf-8

import numpy as np
import os
from tqdm import tqdm
import argparse

import torch
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision import transforms

from diffusers.models import AutoencoderKL

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


def parse_args():
    parser = argparse.ArgumentParser(description="Extract latent features from images")
    parser.add_argument("--dataset_dir", type=str, required=True, help="Path to the image dataset directory")
    parser.add_argument("--latent_save_dir", type=str, required=True, help="Path to save extracted latent features")
    parser.add_argument("--image_size", type=int, default=256, help="Size of input images")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for processing")
    parser.add_argument("--gpu_index", type=int, default=0, help="GPU device index")
    return parser.parse_args()


def main(args):
    use_cuda = torch.cuda.is_available()
    device = torch.device(args.gpu_index if use_cuda else "cpu")

    assert args.image_size % 8 == 0, "Image size must be divisible by 8 (for the VAE encoder)."

    os.makedirs(args.latent_save_dir, exist_ok=True)

    vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-ema").to(device)

    transform = transforms.Compose([
        transforms.Resize(args.image_size),
        transforms.CenterCrop(args.image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
    ])

    dataset = ImageFolder(args.dataset_dir, transform=transform)
    data_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=8, pin_memory=True)

    img_index = 0
    with torch.no_grad():
        for x, y in tqdm(data_loader, leave=False):
            x = x.to(device)
            with torch.cuda.amp.autocast():
                latent_features = vae.encode(x).latent_dist.sample().mul_(0.18215)
                latent_features = latent_features.detach().cpu()  # (bs, 4, image_size//8, image_size//8)

            for latent in latent_features.split(1, 0):
                np.save(os.path.join(args.latent_save_dir, f'{img_index}.npy'), latent.squeeze(0).numpy())
                img_index += 1


if __name__ == "__main__":
    args = parse_args()
    main(args)
