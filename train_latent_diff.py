import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torchvision.utils as vutils

import os
from tqdm import trange, tqdm
from collections import defaultdict
import argparse

from diffusers.models import AutoencoderKL

import Helpers as hf
from Unet import Unet
from datasets import LatentDataset

parser = argparse.ArgumentParser(description="Training Params")
parser.add_argument("--model_name", "-mn", type=str, required=True, help="Experiment save name")
parser.add_argument("--dataset_root", "-dr", type=str, required=True, help="Dataset root dir")
parser.add_argument("--save_dir", "-sd", type=str, default=".", help="Root dir for saving model and data")
parser.add_argument("--nepoch", type=int, default=2000, help="Number of training epochs")
parser.add_argument("--batch_size", "-bs", type=int, default=32, help="Training batch size")
parser.add_argument("--latent_size", '-las', type=int, default=32, help="Input latent Height/Width")
parser.add_argument("--ch_multi", '-w', type=int, default=64, help="Channel width multiplier")
parser.add_argument("--block_widths", '-bw', type=int, nargs='+', default=(1, 2, 4, 8), help="Channel multiplier for the input of each block")
parser.add_argument("--device_index", type=int, default=0, help="GPU device index")
parser.add_argument("--save_interval", '-si', type=int, default=256, help="Number of iteration per save")
parser.add_argument("--accum_steps", '-as', type=int, default=1, help="Number of gradient accumulation steps")
parser.add_argument("--num_steps", '-ns', type=int, default=100, help="Number of training diffusion steps")
parser.add_argument("--lr", type=float, default=2e-5, help="Learning rate")
parser.add_argument("--noise_sigma", type=float, default=1.0, help="Sigma of the sampled noise vector for generation")
parser.add_argument("--load_checkpoint", '-cp', action='store_true', help="Load from checkpoint")
args = parser.parse_args()

device = torch.device(args.device_index if torch.cuda.is_available() else "cpu")

os.makedirs(os.path.join(args.save_dir, "Models"), exist_ok=True)
os.makedirs(os.path.join(args.save_dir, "Results"), exist_ok=True)

data_set = LatentDataset(latent_dir=args.dataset_root)
data_loader = DataLoader(data_set, batch_size=args.batch_size, shuffle=True, num_workers=4)

diffusion_net = Unet(channels=4,  # Assuming 4 channels for latent space
                     img_size=args.latent_size,
                     out_dim=4,
                     dim=args.ch_multi,
                     dim_mults=args.block_widths).to(device)

optimizer = optim.Adam(diffusion_net.parameters(), lr=args.lr)
lr_schedule = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.nepoch, eta_min=0)

vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-ema").to(device)

save_file_name = f"{args.model_name}_{args.latent_size}"
checkpoint_path = os.path.join(args.save_dir, "Models", f"{save_file_name}.pt")

if args.load_checkpoint:
    if os.path.isfile(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        diffusion_net.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        lr_schedule.load_state_dict(checkpoint["lr_schedule_state_dict"])
        start_epoch = checkpoint["epoch"]
        data_logger = defaultdict(lambda: [], checkpoint["data_logger"])
        print("-Checkpoint loaded!")
    else:
        raise ValueError("Checkpoint does NOT exist -> check model name or save directory")
else:
    if os.path.isfile(checkpoint_path):
        raise ValueError("Checkpoint exists -> add -cp flag to use this checkpoint")
    else:
        print("-Starting from scratch")
        start_epoch = 0
        data_logger = defaultdict(lambda: [])

scaler = torch.cuda.amp.GradScaler()
alphas = torch.flip(hf.cosine_alphas_bar(args.num_steps), (0,)).to(device)

for epoch in trange(start_epoch, args.nepoch, leave=False):
    diffusion_net.train()
    mean_loss = 0

    for i, latents in enumerate(tqdm(data_loader, leave=False)):
        current_iter = i + epoch * len(data_loader)
        latents = latents.to(device)
        bs, c, h, w = latents.shape

        with torch.cuda.amp.autocast():
            index = torch.randint(args.num_steps, (bs,), device=device)
            alpha = alphas[index].reshape(bs, 1, 1, 1)
            random_sample = torch.randn_like(latents)
            noise_latents = alpha.sqrt() * latents + (1 - alpha).sqrt() * random_sample

            model_output = diffusion_net(noise_latents, index)
            loss = F.l1_loss(model_output, latents)
            mean_loss += loss.item()

        scaler.scale(loss/args.accum_steps).backward()

        if (i + 1) % args.accum_steps == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

            data_logger["loss"].append(mean_loss/args.accum_steps)
            mean_loss = 0

        if (current_iter + 1) % (args.save_interval * args.accum_steps) == 0:
            diffusion_net.eval()
            with torch.no_grad():
                diff_img, _ = hf.cold_diffuse(diffusion_model=diffusion_net,
                                              batch_size=args.batch_size,
                                              total_steps=args.num_steps,
                                              device=device,
                                              input_size=args.latent_size,
                                              input_channels=4,
                                              noise_sigma=args.noise_sigma)
                fake_sample = vae.decode(diff_img / 0.18215).sample

                vutils.save_image(fake_sample.cpu().float(),
                                  f"{args.save_dir}/Results/{args.model_name}_{args.latent_size}_images.png",
                                  normalize=True)

            torch.save({
                'epoch': epoch + 1,
                'data_logger': dict(data_logger),
                'model_state_dict': diffusion_net.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'lr_schedule_state_dict': lr_schedule.state_dict(),
            }, checkpoint_path)

            diffusion_net.train()

    lr_schedule.step()
