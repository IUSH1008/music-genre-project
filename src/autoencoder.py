
import torch
import torch.nn as nn


class MusicAutoencoder(nn.Module):

    def __init__(self, input_dim=57, latent_dim=32):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 48),
            nn.ReLU(),

            nn.Linear(48, 32),
            nn.ReLU(),

            nn.Linear(32, latent_dim)
        )

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 32),
            nn.ReLU(),

            nn.Linear(32, 48),
            nn.ReLU(),

            nn.Linear(48, input_dim)
        )

    def forward(self, x):
        latent = self.encoder(x)
        reconstructed = self.decoder(latent)
        return reconstructed

    def extract_latent(self, x):
        return self.encoder(x)


