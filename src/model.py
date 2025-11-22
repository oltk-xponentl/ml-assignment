import torch
import torch.nn as nn


class EncoderBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class DecoderBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels, out_channels, kernel_size=2, stride=2
            ),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class SODModel(nn.Module):
    """Simple encoder decoder network for salient object detection."""

    def __init__(self, in_channels: int = 3, base_channels: int = 32):
        super().__init__()
        self.enc1 = EncoderBlock(in_channels, base_channels)            
        self.enc2 = EncoderBlock(base_channels, base_channels * 2)      
        self.enc3 = EncoderBlock(base_channels * 2, base_channels * 4)  

        self.bottleneck = nn.Sequential(
            nn.Conv2d(base_channels * 4, base_channels * 8, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

        self.dec1 = DecoderBlock(base_channels * 8, base_channels * 4)  
        self.dec2 = DecoderBlock(base_channels * 4, base_channels * 2)  
        self.dec3 = DecoderBlock(base_channels * 2, base_channels)      

        self.out_conv = nn.Conv2d(base_channels, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input x shape, [B, 3, H, W] typically H=W=128
        x = self.enc1(x)  # [B, C, H/2, W/2]
        x = self.enc2(x)  # [B, 2C, H/4, W/4]
        x = self.enc3(x)  # [B, 4C, H/8, W/8]

        x = self.bottleneck(x)  # [B, 8C, H/8, W/8]

        x = self.dec1(x)  # [B, 4C, H/4, W/4]
        x = self.dec2(x)  # [B, 2C, H/2, W/2]
        x = self.dec3(x)  # [B, C, H, W]

        logits = self.out_conv(x)  # [B, 1, H, W]
        return logits


def create_model(in_channels: int = 3, base_channels: int = 32) -> SODModel:
    return SODModel(in_channels=in_channels, base_channels=base_channels)
