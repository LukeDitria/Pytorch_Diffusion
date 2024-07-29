import torch
import copy
from tqdm import trange


def noise_from_x0(curr_img, img_pred, alpha):
    return (curr_img - alpha.sqrt() * img_pred)/((1 - alpha).sqrt() + 1e-4)


def cosine_alphas_bar(timesteps, s=0.008):
    steps = timesteps + 1
    x = torch.linspace(0, steps, steps)
    alphas_bar = torch.cos(((x / steps) + s) / (1 + s) * torch.pi * 0.5) ** 2
    alphas_bar = alphas_bar / alphas_bar[0]
    return alphas_bar


def cold_diffuse(diffusion_model, batch_size, total_steps, device, input_channels=3, input_size=32,
                 no_p_bar=True, noise_sigma=1, class_indx=None):

    diffusion_model.eval()
    random_image_sample = noise_sigma * torch.randn(batch_size, input_channels, input_size, input_size, device=device)
    sample_in = copy.deepcopy(random_image_sample)

    alphas = torch.flip(cosine_alphas_bar(total_steps), (0,)).to(device)

    for i in trange(total_steps - 1, disable=no_p_bar):
        index = i * torch.ones(batch_size, device=device)

        if class_indx is not None:
            img_output = diffusion_model(sample_in, index, cond_input=class_indx)
        else:
            img_output = diffusion_model(sample_in, index)

        noise = noise_from_x0(sample_in, img_output, alphas[i])
        x0 = img_output

        rep1 = alphas[i].sqrt() * x0 + (1 - alphas[i]).sqrt() * noise
        rep2 = alphas[i + 1].sqrt() * x0 + (1 - alphas[i + 1]).sqrt() * noise

        sample_in += rep2 - rep1

    index = (total_steps - 1) * torch.ones(batch_size, device=device)
    img_output = diffusion_model(sample_in, index)

    return img_output, random_image_sample
