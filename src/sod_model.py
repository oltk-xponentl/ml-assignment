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
        
        # Split pooling to allow Skip Connections (v4)
        self.conv_block = nn.Sequential(*layers)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor):
        x_stage = self.conv_block(x)
        x_pooled = self.pool(x_stage)
        return x_pooled, x_stage


class DecoderBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, use_bn: bool = False, use_skip: bool = False):
        super().__init__()
        self.use_skip = use_skip
        
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        
        conv_input = out_channels * 2 if use_skip else out_channels
        
        layers = []
        layers.append(nn.Conv2d(conv_input, out_channels, kernel_size=3, padding=1))
        if use_bn:
            layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.ReLU(inplace=True))
        
        # Add 2nd Conv only if using Skips (v4 U-Net style)
        if use_skip:
            layers.append(nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1))
            if use_bn:
                layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.ReLU(inplace=True))
        
        self.conv_block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor, skip: torch.Tensor = None) -> torch.Tensor:
        x = self.up(x)
        
        if self.use_skip and skip is not None:
            if x.shape != skip.shape:
                x = torch.nn.functional.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=True)
            x = torch.cat([x, skip], dim=1)
            
        x = self.conv_block(x)
        return x


class SODModel(nn.Module):
    def __init__(self, in_channels: int = 3, base_channels: int = 32, use_bn: bool = False, use_skip: bool = False):
        super().__init__()
        self.use_skip = use_skip
        
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
        self.dec1 = DecoderBlock(base_channels * 8, base_channels * 4, use_bn, use_skip)
        self.dec2 = DecoderBlock(base_channels * 4, base_channels * 2, use_bn, use_skip)
        self.dec3 = DecoderBlock(base_channels * 2, base_channels, use_bn, use_skip)

        self.out_conv = nn.Conv2d(base_channels, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder
        x, x_enc1 = self.enc1(x)
        x, x_enc2 = self.enc2(x)
        x, x_enc3 = self.enc3(x)

        # Bottleneck
        x = self.bottleneck(x)

        # Decoder
        if self.use_skip:
            x = self.dec1(x, skip=x_enc3)
            x = self.dec2(x, skip=x_enc2)
            x = self.dec3(x, skip=x_enc1)
        else:
            x = self.dec1(x)
            x = self.dec2(x)
            x = self.dec3(x)

        return self.out_conv(x)

    def load_weights(self, checkpoint_path, device):
        """Helper to load weights handling legacy (v1-v3) vs new (v4+) keys."""
        state_dict = torch.load(checkpoint_path, map_location=device)
        
        if "model_state" in state_dict:
            state_dict = state_dict["model_state"]

        new_state_dict = {}
        for k, v in state_dict.items():
            new_k = k
            
            # Rename Legacy Encoder keys (block -> conv_block)
            if ".block." in k and "enc" in k:
                new_k = k.replace(".block.", ".conv_block.")
            
            # Rename Legacy Decoder keys
            if "dec" in k and ".block." in k:
                if ".block.0." in k:
                    new_k = k.replace(".block.0.", ".up.")
                elif ".block.2." in k:
                    new_k = k.replace(".block.2.", ".conv_block.0.")
            
            new_state_dict[new_k] = v

        self.load_state_dict(new_state_dict)


def create_model(in_channels: int = 3, base_channels: int = 32, use_bn: bool = False, use_skip: bool = False) -> SODModel:
    return SODModel(in_channels=in_channels, base_channels=base_channels, use_bn=use_bn, use_skip=use_skip)