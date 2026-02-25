---
phase: 1
plan: 2
wave: 1
---

# Plan 1.2: ConvNeXt Block

## Objective
Implement the ConvNeXt block — the core building block of the architecture. Must match timm naming exactly for weight compatibility.

## Context
- .gsd/SPEC.md
- .gsd/DECISIONS.md (ADR-04: match timm naming)
- convnext/layers.py (DropPath, LayerNorm2d — from Plan 1.1)

## timm Naming Reference
Block attributes MUST be named exactly:
- `conv_dw` — depthwise 7×7 Conv2d
- `norm` — nn.LayerNorm (channels-last)
- `mlp.fc1` — first 1×1 linear (expansion)
- `mlp.fc2` — second 1×1 linear (projection)
- `gamma` — nn.Parameter for layer scale
- Uses DropPath from layers.py

## Tasks

<task type="auto">
  <name>Implement ConvNeXtBlock</name>
  <files>
    convnext/block.py (NEW)
  </files>
  <action>
    Create `convnext/block.py` with:

    **Mlp** (helper class, or inline as nn.Module):
    - Two linear layers: `fc1` (in → hidden) and `fc2` (hidden → out)
    - GELU activation between them
    - Optional dropout
    - Named `fc1` and `fc2` to match timm

    **ConvNeXtBlock(nn.Module)**:
    Constructor args: `dim: int`, `drop_path: float = 0.0`, `layer_scale_init_value: float = 1e-6`

    Architecture (forward pass):
    1. `self.conv_dw = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)` — depthwise conv
    2. `x = x.permute(0, 2, 3, 1)` — (B,C,H,W) → (B,H,W,C) for LayerNorm
    3. `self.norm = nn.LayerNorm(dim)` — standard LayerNorm on last dim
    4. `self.mlp = Mlp(in_features=dim, hidden_features=4*dim)` — inverted bottleneck (4× expansion)
    5. Layer scale: `self.gamma = nn.Parameter(layer_scale_init_value * torch.ones(dim))` if `layer_scale_init_value > 0`
    6. `x = x.permute(0, 3, 1, 2)` — back to (B,C,H,W)
    7. `self.drop_path = DropPath(drop_path)` — stochastic depth on residual
    8. Residual connection: `x = input + self.drop_path(x)`

    CRITICAL naming rules:
    - Attribute `conv_dw` (NOT `dwconv` or `depthwise_conv`)
    - Attribute `norm` (NOT `layer_norm` or `ln`)
    - MLP sub-attributes `fc1`, `fc2` (NOT `linear1`, `linear2`)
    - Attribute `gamma` as nn.Parameter (NOT inside a module)

    DO NOT:
    - Use expansion_ratio as a parameter — hardcode 4× as in the paper
    - Forget the residual connection
    - Use LayerNorm2d here — the block uses standard nn.LayerNorm after permuting to channels-last
  </action>
  <verify>
    python -c "
from convnext.block import ConvNeXtBlock
import torch

block = ConvNeXtBlock(dim=96, drop_path=0.1, layer_scale_init_value=1e-6)

# Shape test
x = torch.randn(2, 96, 56, 56)
out = block(x)
assert out.shape == (2, 96, 56, 56), f'Block shape wrong: {out.shape}'

# Naming test
sd = block.state_dict()
required_keys = ['conv_dw.weight', 'conv_dw.bias', 'norm.weight', 'norm.bias', 'mlp.fc1.weight', 'mlp.fc1.bias', 'mlp.fc2.weight', 'mlp.fc2.bias', 'gamma']
for key in required_keys:
    assert key in sd, f'Missing key: {key}'

# Residual test: zero init gamma should make block near-identity
block_zero = ConvNeXtBlock(dim=96, layer_scale_init_value=1e-6)
block_zero.eval()
x = torch.randn(2, 96, 8, 8)
out = block_zero(x)
diff = (out - x).abs().max().item()
assert diff < 0.1, f'Near-identity failed, max diff: {diff}'

print('Plan 1.2 verified ✓')
"
  </verify>
  <done>
    - ConvNeXtBlock produces correct output shape (B,C,H,W) → (B,C,H,W)
    - All state_dict keys match timm naming exactly
    - Layer scale with small init value makes block near-identity (residual learning)
  </done>
</task>

## Success Criteria
- [ ] ConvNeXtBlock preserves spatial dimensions
- [ ] State dict keys match timm naming: conv_dw, norm, mlp.fc1, mlp.fc2, gamma
- [ ] Layer scale initialized to 1e-6 (near-identity at init)
