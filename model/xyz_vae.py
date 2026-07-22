import torch
import torch.nn as nn

'''
Input: flattened backbone xyz coordinates
Encoder: xyz -> hidden representation -> μ, logσ²
Sampling: z = μ + εσ
Decoder: z → reconstructed xyz coordinates
'''

class FeatureAttention(nn.Module):
    '''
    This attention block is feature-wise attention on the 128-dimensional 
    feature representation at the end of the encoder. 
    The aim is learn relationships between feature dimensions before latent variables.
    '''
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
    
# reparameterization trick
class Sampling(nn.Module):
    def forward(self, mu, logvar):
        # stability of variance
        logvar = torch.clamp(logvar, -6, 2)
        std = torch.exp(0.5 * logvar)
        # noise
        eps = torch.randn_like(std)
        return mu + eps * std

# residual block to more stability training and more feature capacity
def mlp_block(dim, dropout=0.1):
    return nn.Sequential(
        nn.Linear(dim, dim),
        nn.GELU(),
        nn.Dropout(dropout),
        nn.Linear(dim, dim),
        nn.GELU()
    )
class XYZVAE(nn.Module):
    def __init__(
        self,
        input_dim,
        latent_dim=8,
        encoder_dims=(512, 256, 128),
        decoder_dims=(128, 256, 512),
        use_residual=True,
        use_attention=True,
        use_layernorm=True,
        dropout=0.1,
        activation="gelu",
    ):
        super().__init__()

        self.latent_dim = latent_dim
        self.use_residual = use_residual
        self.use_attention = use_attention
        self.use_layernorm = use_layernorm

        act = nn.GELU if activation == "gelu" else nn.ReLU

        # encoder
        dims = [input_dim] + list(encoder_dims)
        self.encoder_layers = nn.ModuleList([
            nn.Linear(dims[i], dims[i+1]) for i in range(len(dims)-1)
        ])

        self.encoder_norms = nn.ModuleList([
            nn.LayerNorm(d) for d in encoder_dims
        ])

        self.encoder_res = nn.ModuleList([
            mlp_block(d, dropout=dropout) for d in encoder_dims
        ])

        self.act = act()

        last_dim = encoder_dims[-1]

        self.attn = FeatureAttention(last_dim) if use_attention else None

        self.z_mean = nn.Linear(last_dim, latent_dim)
        self.z_logvar = nn.Linear(last_dim, latent_dim)
        self.sampling = Sampling()

        # decoder
        dec_dims = [latent_dim] + list(decoder_dims)
        self.decoder_layers = nn.ModuleList([
            nn.Linear(dec_dims[i], dec_dims[i+1]) for i in range(len(dec_dims)-1)
        ])

        self.decoder_norms = nn.ModuleList([
            nn.LayerNorm(d) for d in decoder_dims
        ])

        self.dec_out = nn.Linear(decoder_dims[-1], input_dim)

    def encode(self, x):
        for i, layer in enumerate(self.encoder_layers):
            x = self.act(layer(x))

            if self.use_residual:
                x = x + self.encoder_res[i](x)

            if self.use_layernorm:
                x = self.encoder_norms[i](x)

        if self.use_attention:
            x = x + self.attn(x)

        mu = self.z_mean(x)
        logvar = self.z_logvar(x)
        return mu, logvar

    def decode(self, z):
        y = z
        for i, layer in enumerate(self.decoder_layers):
            y = self.act(layer(y))
            y = self.decoder_norms[i](y)

        return self.dec_out(y)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.sampling(mu, logvar)
        x_hat = self.decode(z)
        return x_hat, mu, logvar
