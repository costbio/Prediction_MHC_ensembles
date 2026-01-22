import torch
import torch.nn.functional as F

def vae_loss(x, x_hat, mu, logvar, beta):
    recon = torch.mean((x - x_hat) ** 2)

    kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
    kl = torch.mean(kl)

    total = recon + beta * kl
    return total, recon.detach(), kl.detach()