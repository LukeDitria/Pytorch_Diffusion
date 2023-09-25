import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
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

parser = argparse.ArgumentParser(description="Training Params")
# string args
parser.add_argument("--model_name", "-mn", help="Experiment save name", type=str, required=True)
parser.add_argument("--dataset_root", "-dr", help="Dataset root dir", type=str, required=True)

parser.add_argument("--save_dir", "-sd", help="Root dir for saving model and data", type=str, default=".")
parser.add_argument("--conditional_input", "-cin", help="Diffusion conditional input", type=str, default="none")

# int args
parser.add_argument("--nepoch", help="Number of training epochs", type=int, default=2000)
parser.add_argument("--batch_size", "-bs", help="Training batch size", type=int, default=32)
parser.add_argument("--image_size", '-ims', help="Input image size", type=int, default=64)
parser.add_argument("--ch_multi", '-w', help="Channel width multiplier", type=int, default=64)
parser.add_argument("--block_widths", '-bw', help="Channel multiplier for the input of each block",
                    type=int, nargs='+', default=(1, 2, 4, 8))

parser.add_argument("--device_index", help="GPU device index", type=int, default=0)
parser.add_argument("--save_interval", '-si', help="Number of iteration per save", type=int, default=256)
parser.add_argument("--accum_steps", '-as', help="Number of gradient accumulation steps", type=int, default=1)
parser.add_argument("--num_steps", '-ns', help="number of training diffusion steps", type=int, default=100)

parser.add_argument("--conditional_dim", "-cdim", help="Dimension of conditional input ", type=int, default=100)

# float args
parser.add_argument("--lr", help="Learning rate", type=float, default=2e-5)
parser.add_argument("--noise_sigma", help="Sigma of the sampled noise vector for generation", type=float, default=1.0)

# bool args
parser.add_argument("--load_checkpoint", '-cp', action='store_true', help="Load from checkpoint")

args = parser.parse_args()

use_cuda = torch.cuda.is_available()
device = torch.device(args.device_index if use_cuda else "cpu")
print("")

# Create dataloaders
# This code assumes there is no pre-defined test/train split and will create one for you
print("-Target Image Size %dx%d" % (args.image_size, args.image_size))
transform = transforms.Compose([transforms.Resize(args.image_size),
                                transforms.CenterCrop(args.image_size),
                                transforms.ToTensor(),
                                transforms.Normalize(0.5, 0.5)])

data_set = Datasets.ImageFolder(root=args.dataset_root, transform=transform)
data_loader = DataLoader(data_set, batch_size=args.batch_size, shuffle=True, num_workers=4)

# Get a test image batch from the test_loader to visualise the reconstruction quality etc
dataiter = iter(data_loader)
test_images, _ = next(dataiter)

# Create Diffusion network.
diffusion_net = Unet(channels=test_images.shape[1],
                     img_size=args.image_size,
                     out_dim=test_images.shape[1],
                     dim=args.ch_multi,
                     dim_mults=args.block_widths,
                     conditional_in=args.conditional_input,
                     conditional_dim=args.conditional_dim).to(device)

# Setup optimizer
optimizer = optim.Adam(diffusion_net.parameters(), lr=args.lr)

# AMP Scaler
scaler = torch.cuda.amp.GradScaler()

# Let's see how many Parameters our Model has!
num_model_params = 0
for param in diffusion_net.parameters():
    num_model_params += param.flatten().shape[0]

print("-This Model Has %d (Approximately %d Million) Parameters!" % (num_model_params, num_model_params//1e6))

# Create the save directory if it does not exist
if not os.path.isdir(args.save_dir + "/Models"):
    os.makedirs(args.save_dir + "/Models")
if not os.path.isdir(args.save_dir + "/Results"):
    os.makedirs(args.save_dir + "/Results")

# Checks if a checkpoint has been specified to load, if it has, it loads the checkpoint
# If no checkpoint is specified, it checks if a checkpoint already exists and raises an error if
# it does to prevent accidental overwriting. If no checkpoint exists, it starts from scratch.
save_file_name = args.model_name + "_" + str(args.image_size)
if args.load_checkpoint:
    if os.path.isfile(args.save_dir + "/Models/" + save_file_name + ".pt"):
        checkpoint = torch.load(args.save_dir + "/Models/" + save_file_name + ".pt",
                                map_location="cpu")
        print("-Checkpoint loaded!")
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        diffusion_net.load_state_dict(checkpoint['model_state_dict'])

        if not optimizer.param_groups[0]["lr"] == args.lr:
            print("-Updating lr!")
            optimizer.param_groups[0]["lr"] = args.lr

        start_epoch = checkpoint["epoch"]
        data_logger = defaultdict(lambda: [], checkpoint["data_logger"])
    else:
        raise ValueError("Warning Checkpoint does NOT exist -> check model name or save directory")
else:
    # If checkpoint does exist raise an error to prevent accidental overwriting
    if os.path.isfile(args.save_dir + "/Models/" + save_file_name + ".pt"):
        raise ValueError("Warning Checkpoint exists -> add -cp flag to use this checkpoint")
    else:
        print("-Starting from scratch")
        start_epoch = 0
        # Loss and metrics logger
        data_logger = defaultdict(lambda: [])
print("")

alphas = torch.flip(hf.cosine_alphas_bar(args.num_steps), (0, )).to(device)
mean_loss = 0

# Start training loop
for epoch in trange(start_epoch, args.nepoch, leave=False):
    diffusion_net.train()
    for i, (images, labels) in enumerate(tqdm(data_loader, leave=False)):
        current_iter = i + epoch * len(data_loader)
        images = images.to(device)
        bs, c, h, w = images.shape

        # We will train with mixed precision!
        with torch.cuda.amp.autocast():
            with torch.no_grad():
                # Randomly sample batch of alphas
                index = torch.randint(args.num_steps, (bs,), device=device)
                alpha = alphas[index].reshape(bs, 1, 1, 1)
                random_sample = torch.randn_like(images)
                noise_image = alpha.sqrt() * images + (1 - alpha).sqrt() * random_sample

            if args.conditional_input == "none":
                model_output = diffusion_net(noise_image, index)
            else:
                model_output = diffusion_net(noise_image, index, cond_input=labels)

            # Predict the original image
            loss = F.l1_loss(model_output, images)
            mean_loss += loss.item()

        scaler.scale(loss/args.accum_steps).backward()

        if (i + 1) % args.accum_steps == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

            # Log mean loss every update step
            data_logger["loss"].append(mean_loss/args.accum_steps)
            mean_loss = 0

        # Save results and a checkpoint at regular intervals
        if (current_iter + 1) % (args.save_interval * args.accum_steps) == 0:
            diffusion_net.eval()
            with torch.no_grad():
                with torch.cuda.amp.autocast():
                    if args.conditional_input == "class":
                        class_indx = torch.randint(args.conditional_dim, (args.batch_size,), device=device)
                    else:
                        class_indx = None

                    diff_img, init_img = hf.cold_diffuse(diffusion_model=diffusion_net,
                                                         batch_size=args.batch_size,
                                                         total_steps=args.num_steps,
                                                         device=device,
                                                         image_size=args.image_size,
                                                         noise_sigma=args.noise_sigma,
                                                         class_indx=class_indx)

                    vutils.save_image(diff_img.cpu().float(),
                                      "%s/%s/%s_%d_images.png" % (args.save_dir,
                                                                "Results",
                                                                args.model_name,
                                                                args.image_size),
                                      normalize=True)

                # Save a checkpoint
                torch.save({
                            'epoch': epoch + 1,
                            'data_logger': dict(data_logger),
                            'model_state_dict': diffusion_net.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                             }, args.save_dir + "/Models/" + save_file_name + ".pt")

                # Set the model back into training mode!!
                diffusion_net.train()
