import torch

def vae_loss(x, x_hat, mu, logvar, beta):
    # reconstruction loss (MSE)
    recon = torch.mean((x - x_hat) ** 2)
    # KL term approximates the latent distribution , to the standard normal prior.
    kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
    kl = torch.mean(kl)
    
    total = recon + beta * kl
    return total, recon.detach(), kl.detach()

