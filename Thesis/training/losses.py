import torch
import torch.nn.functional as F

def vae_loss(x, x_hat, mu, logvar, beta):
    recon = 0.5 * torch.mean((x - x_hat) ** 2) + \
            0.5 * torch.mean(torch.abs(x - x_hat))

    kl = -0.5 * torch.sum(
        1 + logvar - mu.pow(2) - logvar.exp(),
        dim=1
    )  # (batch,)

    kl = torch.mean(kl)  # batch mean

    total = recon + beta * kl
    return total, recon.detach(), kl.detach()
