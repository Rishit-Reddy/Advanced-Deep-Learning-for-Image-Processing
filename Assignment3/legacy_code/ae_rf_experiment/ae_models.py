import torch
import torch.nn as nn

class ResBlock(nn.Module):
    """Residual Block: Skip connections stay WITHIN the encoder/decoder."""
    def __init__(self, in_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(in_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_ch, in_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(in_ch)
        )
    def forward(self, x):
        return torch.relu(x + self.conv(x))

class ResidualAE(nn.Module):
    def __init__(self):
        super().__init__()
        # Encoder: 128x128x6 -> 8x8x256
        self.encoder = nn.Sequential(
            nn.Conv2d(6, 64, 3, stride=2, padding=1), # 64x64
            nn.BatchNorm2d(64),
            nn.ReLU(),
            ResBlock(64),
            
            nn.Conv2d(64, 128, 3, stride=2, padding=1), # 32x32
            nn.BatchNorm2d(128),
            nn.ReLU(),
            ResBlock(128),
            
            nn.Conv2d(128, 256, 3, stride=2, padding=1), # 16x16
            nn.BatchNorm2d(256),
            nn.ReLU(),
            ResBlock(256),
            
            nn.Conv2d(256, 256, 3, stride=2, padding=1), # 8x8
            nn.BatchNorm2d(256),
            nn.ReLU(),
            ResBlock(256)
        )
        
        # Decoder (Symmetrical but NO cross-skip connections)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 3, stride=2, padding=1, output_padding=1), # 16x16
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1), # 32x32
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1), # 64x64
            nn.ReLU(),
            nn.ConvTranspose2d(32, 6, 3, stride=2, padding=1, output_padding=1), # 128x128
            nn.Sigmoid()
        )
        
        self.gap = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        latent = self.encoder(x)
        reconstructed = self.decoder(latent)
        
        # Features for Random Forest: GAP on the final encoder layer
        feat = self.gap(latent).view(latent.size(0), -1) 
        return reconstructed, feat
