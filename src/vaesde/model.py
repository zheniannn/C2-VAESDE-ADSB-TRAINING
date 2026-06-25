"""Sequence VAE: flat MLP encoder-decoder with reparameterisation trick."""

import torch
import torch.nn as nn
from vaesde.constants import SEQ_LEN, N_FEAT, FLAT_DIM, LATENT_DIM


class SequenceVAE(nn.Module):
    """Flat MLP VAE over fixed-length (30, 4) ENU trajectory windows."""

    def __init__(self, flat_dim: int = FLAT_DIM, latent_dim: int = LATENT_DIM):
        super().__init__()
        self.flat_dim   = flat_dim
        self.latent_dim = latent_dim

        self.encoder = nn.Sequential(
            nn.Linear(flat_dim, 256), nn.ReLU(),
            nn.Linear(256, 128),      nn.ReLU(),
        )
        self.fc_mu     = nn.Linear(128, latent_dim)
        self.fc_logvar = nn.Linear(128, latent_dim)
        self.decoder   = nn.Sequential(
            nn.Linear(latent_dim, 128), nn.ReLU(),
            nn.Linear(128, 256),        nn.ReLU(),
            nn.Linear(256, flat_dim),
        )

    def encode(self, x: torch.Tensor):
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_logvar(h)

    @staticmethod
    def reparameterise(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        return mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor):
        b = x.size(0)
        x_flat = x.view(b, self.flat_dim)
        mu, logvar = self.encode(x_flat)
        z = self.reparameterise(mu, logvar)
        return self.decoder(z).view(b, SEQ_LEN, N_FEAT), mu, logvar
