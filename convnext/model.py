import torch
import torch.nn as nn
from .block import ConvNeXtBlock
from .layers import LayerNorm2d

# 1. Stem
class Stem(nn.Module):
    def __init__(self, in_channels=3, out_channels=96, small_input=False):
        super().__init__()
        if small_input:
            # CIFAR-10: don't downsample aggressively
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        else:
            # ImageNet: patchify with 4x4 stride-4
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=4)
        self.norm = LayerNorm2d(out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        return x

# 2. Downsampling Layer
class DownsampleLayer(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.norm = LayerNorm2d(in_dim)
        self.conv = nn.Conv2d(in_dim, out_dim, kernel_size=2, stride=2)

    def forward(self, x):
        x = self.norm(x)
        x = self.conv(x)
        return x

# 3. Stage
class Stage(nn.Module):
    def __init__(self, dim, depth, drop_path_rates, layer_scale_init=1e-6):
        super().__init__()
        self.blocks = nn.Sequential(*[
            ConvNeXtBlock(
                dim=dim,
                drop_path=drop_path_rates[i],
                layer_scale_init=layer_scale_init
            )
            for i in range(depth)
        ])

    def forward(self, x):
        return self.blocks(x)

# 4. Full ConvNeXt Model
class ConvNeXt(nn.Module):
    def __init__(
        self,
        in_channels=3,
        num_classes=1000,
        depths=(3, 3, 9, 3),
        dims=(96, 192, 384, 768),
        drop_path_rate=0.1,
        layer_scale_init=1e-6,
        small_input=False,
    ):
        super().__init__()

        # Stem
        self.stem = Stem(in_channels, dims[0], small_input=small_input)

        # Stochastic depth schedule
        total_blocks = sum(depths)
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, total_blocks)]

        # Stages + Downsampling
        self.stages      = nn.ModuleList()
        self.downsamples = nn.ModuleList()

        cur = 0
        for i in range(4):
            stage = Stage(
                dim=dims[i],
                depth=depths[i],
                drop_path_rates=dp_rates[cur : cur + depths[i]],
                layer_scale_init=layer_scale_init,
            )
            self.stages.append(stage)
            cur += depths[i]

            if i < 3:
                self.downsamples.append(DownsampleLayer(dims[i], dims[i+1]))

        # Classification head
        self.head_norm = nn.LayerNorm(dims[-1])
        self.head      = nn.Linear(dims[-1], num_classes)

        # Weight init
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        # Stem
        x = self.stem(x)

        # Stages + downsampling
        for i, stage in enumerate(self.stages):
            x = stage(x)
            if i < 3:
                x = self.downsamples[i](x)

        # Global average pooling
        x = x.mean([-2, -1])       # (B, C, H, W) → (B, C)
        x = self.head_norm(x)
        x = self.head(x)
        return x

# 5. Variant Factories
def convnext_tiny(num_classes=1000, small_input=False):
    return ConvNeXt(depths=(3,3,9,3),   dims=(96,192,384,768),    num_classes=num_classes, small_input=small_input)

def convnext_small(num_classes=1000, small_input=False):
    return ConvNeXt(depths=(3,3,27,3),  dims=(96,192,384,768),    num_classes=num_classes, small_input=small_input)

def convnext_base(num_classes=1000, small_input=False):
    return ConvNeXt(depths=(3,3,27,3),  dims=(128,256,512,1024),  num_classes=num_classes, small_input=small_input)

def convnext_large(num_classes=1000, small_input=False):
    return ConvNeXt(depths=(3,3,27,3),  dims=(192,384,768,1536),  num_classes=num_classes, small_input=small_input)

def convnext_xlarge(num_classes=1000, small_input=False):
    return ConvNeXt(depths=(3,3,27,3),  dims=(256,512,1024,2048), num_classes=num_classes, small_input=small_input)

# 6. Quick Test
if __name__ == "__main__":
    # Test ImageNet input
    model = convnext_tiny(num_classes=1000)
    x = torch.randn(2, 3, 224, 224)
    out = model(x)
    params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"ImageNet | Input: {x.shape} → Output: {out.shape} | Params: {params:.1f}M")

    # Test CIFAR-10 input
    model_cifar = convnext_tiny(num_classes=10, small_input=True)
    x_cifar = torch.randn(2, 3, 32, 32)
    out_cifar = model_cifar(x_cifar)
    print(f"CIFAR-10 | Input: {x_cifar.shape} → Output: {out_cifar.shape}")