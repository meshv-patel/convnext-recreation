import torch
import torch.nn as nn
from .layers import DropPath, LayerNorm2d

class ConvNeXtBlock(nn.Module):
    def __init__(self, dim, drop_path=0., layer_scale_init=1e-6):
        super().__init__()

        # Depthwise conv — spatial mixing
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)

        # Norm
        self.norm = nn.LayerNorm(dim)

        # MLP — channel mixing (4x expansion)
        self.mlp_expand  = nn.Linear(dim, 4 * dim)
        self.act = nn.GELU()
        self.mlp_contract = nn.Linear(4 * dim, dim)

        # Layer scale
        self.gamma = nn.Parameter(layer_scale_init * torch.ones(dim))

        # Stochastic depth
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        residual = x

        # Spatial mixing
        x = self.dwconv(x)              # (B, C, H, W)

        # Switch to channel-last
        x = x.permute(0, 2, 3, 1)      # (B, H, W, C)
        x = self.norm(x)

        # MLP
        x = self.mlp_expand(x)
        x = self.act(x)
        x = self.mlp_contract(x)

        # Layer scale
        x = self.gamma * x

        # Back to channel-first
        x = x.permute(0, 3, 1, 2)      # (B, C, H, W)

        # Residual connection
        return residual + self.drop_path(x)