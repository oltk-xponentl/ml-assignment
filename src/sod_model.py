import torch
import torch.nn as nn

class EncoderBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, use_bn: bool = False):
        super().__init__()
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        ]
        if use_bn:
            layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.ReLU(inplace=True))
        
        layers.append(nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1))
        if use_bn:
            layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.ReLU(inplace=True))
        
        layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)

class DecoderBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, use_bn: bool = False):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        
        layers = []
        layers.append(nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1))
        if use_bn:
            layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.ReLU(inplace=True))
        self.conv_block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        x = self.conv_block(x)
        return x

class SODModel(nn.Module):
    def __init__(self, in_channels: int = 3, base_channels: int = 32, use_bn: bool = False):
        super().__init__()
        # Encoder
        self.enc1 = EncoderBlock(in_channels, base_channels, use_bn)
        self.enc2 = EncoderBlock(base_channels, base_channels * 2, use_bn)
        self.enc3 = EncoderBlock(base_channels * 2, base_channels * 4, use_bn)

        # Bottleneck
        bn_layers = [nn.Conv2d(base_channels * 4, base_channels * 8, kernel_size=3, padding=1)]
        if use_bn:
            bn_layers.append(nn.BatchNorm2d(base_channels * 8))
        bn_layers.append(nn.ReLU(inplace=True))
        self.bottleneck = nn.Sequential(*bn_layers)

        # Decoder
        self.dec1 = DecoderBlock(base_channels * 8, base_channels * 4, use_bn)
        self.dec2 = DecoderBlock(base_channels * 4, base_channels * 2, use_bn)
        self.dec3 = DecoderBlock(base_channels * 2, base_channels, use_bn)

        self.out_conv = nn.Conv2d(base_channels, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.enc1(x)
        x = self.enc2(x)
        x = self.enc3(x)
        x = self.bottleneck(x)
        x = self.dec1(x)
        x = self.dec2(x)
        x = self.dec3(x)
        return self.out_conv(x)

def create_model(in_channels: int = 3, base_channels: int = 32, use_bn: bool = False) -> SODModel:
    return SODModel(in_channels=in_channels, base_channels=base_channels, use_bn=use_bn)