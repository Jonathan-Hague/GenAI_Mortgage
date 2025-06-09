import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class VAE(nn.Module):
    def __init__(self, input_channels=3, latent_dim=128, hidden_dims=[32, 64, 128, 256]):
        super(VAE, self).__init__()
        
        # Encoder
        modules = []
        in_channels = input_channels
        
        for h_dim in hidden_dims:
            modules.extend([
                nn.Conv2d(in_channels, h_dim, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(h_dim),
                nn.LeakyReLU()
            ])
            in_channels = h_dim
            
        self.encoder = nn.Sequential(*modules)
        
        # Calculate the size of the feature map
        self.feature_size = self._get_conv_output_size(input_channels)
        
        # Latent space
        self.fc_mu = nn.Linear(self.feature_size, latent_dim)
        self.fc_var = nn.Linear(self.feature_size, latent_dim)
        
        # Decoder
        modules = []
        self.decoder_input = nn.Linear(latent_dim, self.feature_size)
        
        hidden_dims.reverse()
        
        for i in range(len(hidden_dims) - 1):
            modules.extend([
                nn.ConvTranspose2d(hidden_dims[i], hidden_dims[i + 1],
                                 kernel_size=3, stride=2, padding=1, output_padding=1),
                nn.BatchNorm2d(hidden_dims[i + 1]),
                nn.LeakyReLU()
            ])
            
        modules.extend([
            nn.ConvTranspose2d(hidden_dims[-1], input_channels,
                             kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()
        ])
        
        self.decoder = nn.Sequential(*modules)
        
    def _get_conv_output_size(self, input_channels):
        # Helper function to calculate the size of the feature map
        x = torch.randn(1, input_channels, 64, 64)
        x = self.encoder(x)
        return int(np.prod(x.size()))
        
    def encode(self, x):
        x = self.encoder(x)
        x = x.view(x.size(0), -1)
        mu = self.fc_mu(x)
        log_var = self.fc_var(x)
        return mu, log_var
        
    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std
        
    def decode(self, z):
        z = self.decoder_input(z)
        z = z.view(z.size(0), -1, 4, 4)  # Adjust size based on your architecture
        return self.decoder(z)
        
    def forward(self, x):
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        return self.decode(z), mu, log_var
        
    def loss_function(self, recon_x, x, mu, log_var):
        # Reconstruction loss
        recon_loss = F.binary_cross_entropy(recon_x, x, reduction='sum')
        
        # KL divergence
        kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
        
        return recon_loss + kl_loss, recon_loss, kl_loss 