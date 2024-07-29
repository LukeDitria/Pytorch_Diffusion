import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.datasets as Datasets
import torchvision.transforms as transforms
import torch.nn.functional as F
import torchvision.utils as vutils

import os
from tqdm import trange, tqdm
from collections import defaultdict
import argparse

import Helpers as hf
from Unet import Unet

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

parser = argparse.ArgumentParser(description="Training Params")
parser.add_argument("--model_name", "-mn", help="Experiment save name", type=str, required=True)
parser.add_argument("--dataset_root", "-dr", help="Dataset root dir", type=str, required=True)
parser.add_argument("--save_dir", "-sd", help="Root dir for saving model and data", type=str, default=".")
parser.add_argument("--conditional_input", "-cin", help="Diffusion conditional input", type=str, default="none")
parser.add_argument("--nepoch", help="Number of training epochs", type=int, default=2000)
parser.add_argument("--batch_size", "-bs", help="Training batch size", type=int, default=32)
parser.add_argument("--image_size", '-ims', help="Input image size", type=int, default=64)
parser.add_argument("--ch_multi", '-w', help="Channel width multiplier", type=int, default=64)
parser.add_argument("--block_widths", '-bw', help="Channel multiplier for the input of each block", type=int, nargs='+', default=(1, 2, 4, 8))
parser.add_argument("--device_index", help="GPU device index", type=int, default=0)
parser.add_argument("--save_interval", '-si', help="Number of iteration per save", type=int, default=256)
parser.add_argument("--accum_steps", '-as', help="Number of gradient accumulation steps", type=int, default=1)
parser.add_argument("--num_steps", '-ns', help="number of training diffusion steps", type=int, default=100)
parser.add_argument("--conditional_dim", "-cdim", help="Dimension of conditional input ", type=int, default=100)
parser.add_argument("--lr", help="Learning rate", type=float, default=2e-5)
parser.add_argument("--noise_sigma", help="Sigma of the sampled noise vector for generation", type=float, default=1.0)
parser.add_argument("--load_checkpoint", '-cp', action='store_true', help="Load from checkpoint")
args = parser.parse_args()

device = torch.device(args.device_index if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.Resize(args.image_size),
    transforms.CenterCrop(args.image_size),
    transforms.ToTensor(),
    transforms.Normalize(0.5, 0.5)
])

data_set = Datasets.ImageFolder(root=args.dataset_root, transform=transform)
data_loader = DataLoader(data_set, batch_size=args.batch_size, shuffle=True, num_workers=4)

test_images, _ = next(iter(data_loader))

diffusion_net = Unet(
    channels=test_images.shape[1],
    img_size=args.image_size,
    out_dim=test_images.shape[1],
    dim=args.ch_multi,
    dim_mults=args.block_widths,
    conditional_in=args.conditional_input,
    conditional_dim=args.conditional_dim
).to(device)

optimizer = optim.Adam(diffusion_net.parameters(), lr=args.lr)
lr_schedule = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.nepoch, eta_min=0)
scaler = torch.cuda.amp.GradScaler()

num_model_params = sum(p.numel() for p in diffusion_net.parameters())
print(f"-This Model Has {num_model_params} (Approximately {num_model_params//1e6} Million) Parameters!")

os.makedirs(args.save_dir + "/Models", exist_ok=True)
os.makedirs(args.save_dir + "/Results", exist_ok=True)

save_file_name = f"{args.model_name}_{args.image_size}"
checkpoint_path = f"{args.save_dir}/Models/{save_file_name}.pt"

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
        raise ValueError("Warning: Checkpoint does NOT exist -> check model name or save directory")
else:
    if os.path.isfile(checkpoint_path):
        raise ValueError("Warning: Checkpoint exists -> add -cp flag to use this checkpoint")
    start_epoch = 0
    data_logger = defaultdict(lambda: [])

alphas = torch.flip(hf.cosine_alphas_bar(args.num_steps), (0,)).to(device)

for epoch in trange(start_epoch, args.nepoch, leave=False):
    diffusion_net.train()
    mean_loss = 0
    for i, (images, labels) in enumerate(tqdm(data_loader, leave=False)):
        current_iter = i + epoch * len(data_loader)
        images = images.to(device)
        bs, c, h, w = images.shape

        with torch.cuda.amp.autocast():
            index = torch.randint(args.num_steps, (bs,), device=device)
            alpha = alphas[index].reshape(bs, 1, 1, 1)
            random_sample = torch.randn_like(images)
            noise_image = alpha.sqrt() * images + (1 - alpha).sqrt() * random_sample

            model_output = diffusion_net(noise_image, index,
                                         cond_input=labels if args.conditional_input != "none" else None)
            loss = F.l1_loss(model_output, images)
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
                class_indx = torch.randint(args.conditional_dim,
                                           (args.batch_size,),
                                           device=device) if args.conditional_input == "class" else None

                diff_img, _ = hf.cold_diffuse(
                    diffusion_model=diffusion_net,
                    batch_size=args.batch_size,
                    total_steps=args.num_steps,
                    device=device,
                    input_size=args.image_size,
                    noise_sigma=args.noise_sigma,
                    class_indx=class_indx
                )

                vutils.save_image(diff_img.cpu().float(),
                                  f"{args.save_dir}/Results/{args.model_name}_{args.image_size}_images.png",
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