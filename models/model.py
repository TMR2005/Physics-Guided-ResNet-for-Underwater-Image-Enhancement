import torch
import torch.nn as nn

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1)
        )

    def forward(self, x):
        return x + self.block(x)

class PhysicsGuidedUNet(nn.Module):
    def __init__(self):
        super().__init__()

        # 7-channel input
        self.enc1 = nn.Sequential(
            nn.Conv2d(7, 64, 3, padding=1),
            nn.ReLU(),
            ResidualBlock(64)
        )

        self.enc2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.ReLU(),
            ResidualBlock(128)
        )

        self.enc3 = nn.Sequential(
            nn.Conv2d(128, 256, 3, stride=2, padding=1),
            nn.ReLU(),
            ResidualBlock(256)
        )

        self.dec2 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 2, stride=2),
            nn.ReLU()
        )

        self.dec1 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 2, stride=2),
            nn.ReLU()
        )

        self.out = nn.Conv2d(64, 3, 1)

    def forward(self, x):
        # split inputs
        raw = x[:, :3, :, :]
        J_phys = x[:, 3:6, :, :]

        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)

        d2 = self.dec2(e3)
        d1 = self.dec1(d2)

        residual = self.out(d1)

        # ---- Residual learning ----
        out = J_phys + residual

        return torch.clamp(out, 0, 1)