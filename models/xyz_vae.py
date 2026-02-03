import torch
import torch.nn as nn
import torch.nn.functional as F

class FeatureAttention(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.scale = dim ** -0.5

    def forward(self, x):
        # x: [batch, dim]
        q = self.q(x)
        k = self.k(x)
        v = self.v(x)

        # feature-wise attention
        attn = torch.matmul(q.unsqueeze(2), k.unsqueeze(1)) * self.scale
        attn = torch.softmax(attn, dim=-1)

        out = torch.matmul(attn, v.unsqueeze(2)).squeeze(2)
        return out

class Sampling(nn.Module):
    def forward(self, mu, logvar):
        logvar = torch.clamp(logvar, -6, 2)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std


def mlp_block(dim, dropout=0.1):
    return nn.Sequential(
        nn.Linear(dim, dim),
        nn.GELU(),
        nn.Dropout(dropout),
        nn.Linear(dim, dim),
        nn.GELU()
    )
class XYZVAE(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super().__init__()

        # ---------- Encoder ----------
        self.enc1 = nn.Linear(input_dim, 512)
        self.enc2 = nn.Linear(512, 256)
        self.enc3 = nn.Linear(256, 128)

        self.res1 = mlp_block(512)
        self.res2 = mlp_block(256)
        self.res3 = mlp_block(128)

        self.norm1 = nn.LayerNorm(512)
        self.norm2 = nn.LayerNorm(256)
        self.norm3 = nn.LayerNorm(128)

        # ATTENTION 
        self.attn = FeatureAttention(128)

        self.z_mean = nn.Linear(128, latent_dim)
        self.z_logvar = nn.Linear(128, latent_dim)

        self.sampling = Sampling()

        # ---------- Decoder ----------
        self.dec1 = nn.Linear(latent_dim, 128)
        self.dec2 = nn.Linear(128, 256)
        self.dec_out = nn.Linear(256, input_dim)

        self.dnorm1 = nn.LayerNorm(128)
        self.dnorm2 = nn.LayerNorm(256)

    def encode(self, x):
        x = F.gelu(self.enc1(x))
        x = self.norm1(x + self.res1(x))

        x = F.gelu(self.enc2(x))
        x = self.norm2(x + self.res2(x))

        x = F.gelu(self.enc3(x))
        x = self.norm3(x + self.res3(x))

        # ATTENTION + RESIDUAL
        x = x + self.attn(x)

        mu = self.z_mean(x)
        logvar = self.z_logvar(x)

        return mu, logvar

    def decode(self, z):
        y = F.gelu(self.dec1(z))
        y = self.dnorm1(y)

        y = F.gelu(self.dec2(y))
        y = self.dnorm2(y)

        return self.dec_out(y)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.sampling(mu, logvar)
        x_hat = self.decode(z)
        return x_hat, mu, logvar
