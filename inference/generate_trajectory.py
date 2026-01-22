import torch

@torch.no_grad()
def generate_trajectory(
    model,
    n_samples,
    latent_dim,
    device="cuda",
    seed=42
):
    torch.manual_seed(seed)
    model.eval()

    z = torch.randn(n_samples, latent_dim, device=device)
    x = model.decode(z)

    return x.cpu()

