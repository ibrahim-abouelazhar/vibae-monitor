import torch
import torch.nn as nn

class Dense2DAutoencoder(nn.Module):
    def __init__(self, height=128, width=32):
        super(Dense2DAutoencoder, self).__init__()
        self.height = height
        self.width = width
        self.flat_dim = height * width
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(self.flat_dim, 1024),
            nn.ReLU(True),
            nn.Linear(1024, 256),
            nn.ReLU(True),
            nn.Linear(256, 64),
            nn.ReLU(True)  # Bottleneck representation of size 64
        )
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(64, 256),
            nn.ReLU(True),
            nn.Linear(256, 1024),
            nn.ReLU(True),
            nn.Linear(1024, self.flat_dim),
            nn.Sigmoid()  # Magnitude values scaled in [0, 1]
        )

    def forward(self, x):
        # Input shape: (batch_size, 1, 128, 32)
        batch_size = x.size(0)
        # Flatten to (batch_size, 4096)
        x_flat = x.view(batch_size, -1)
        
        encoded = self.encoder(x_flat)
        decoded = self.decoder(encoded)
        
        # Reshape back to (batch_size, 1, 128, 32)
        return decoded.view(batch_size, 1, self.height, self.width)


class CNN2DAutoencoder(nn.Module):
    def __init__(self):
        super(CNN2DAutoencoder, self).__init__()
        # Input shape: (batch_size, 1, 128, 32)
        
        # Encoder
        # Output shapes:
        # Conv1: 1 -> 16 | 128x32 -> 64x16
        # Conv2: 16 -> 32 | 64x16 -> 32x8
        # Conv3: 32 -> 64 | 32x8 -> 16x4
        # Conv4: 64 -> 128 | 16x4 -> 8x2
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(True),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(True),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(True),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(True)
        )
        
        # Decoder
        # Output shapes:
        # ConvT1: 128 -> 64 | 8x2 -> 16x4
        # ConvT2: 64 -> 32 | 16x4 -> 32x8
        # ConvT3: 32 -> 16 | 32x8 -> 64x16
        # ConvT4: 16 -> 1 | 64x16 -> 128x32
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(True),
            nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(True),
            nn.ConvTranspose2d(in_channels=32, out_channels=16, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(True),
            nn.ConvTranspose2d(in_channels=16, out_channels=1, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # Input shape: (batch_size, 1, 128, 32)
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

def get_model(model_type="cnn", height=128, width=32):
    if model_type.lower() == "cnn":
        return CNN2DAutoencoder()
    elif model_type.lower() == "dense":
        return Dense2DAutoencoder(height, width)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
