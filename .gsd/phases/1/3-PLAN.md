---
phase: 1
plan: 3
wave: 2
---

# Plan 1.3: Full Model & Variant Factories

## Objective
Assemble the full ConvNeXt model with stem, stages (downsample + blocks), classification head, and factory functions for all 5 variants. Depends on Plans 1.1 and 1.2.

## Context
- .gsd/SPEC.md
- .gsd/DECISIONS.md (ADR-01: CIFAR stem, ADR-04: timm naming)
- convnext/layers.py (DropPath, LayerNorm2d)
- convnext/block.py (ConvNeXtBlock)

## Variant Configs

| Variant | depths | dims | drop_path_rate |
|---------|--------|------|---------------|
| Tiny | [3,3,9,3] | [96,192,384,768] | 0.1 |
| Small | [3,3,27,3] | [96,192,384,768] | 0.4 |
| Base | [3,3,27,3] | [128,256,512,1024] | 0.5 |
| Large | [3,3,27,3] | [192,384,768,1536] | 0.5 |
| XLarge | [3,3,27,3] | [256,512,1024,2048] | 0.5 |

## Tasks

<task type="auto">
  <name>Implement ConvNeXt model with stages and stem</name>
  <files>
    convnext/model.py (NEW)
  </files>
  <action>
    Create `convnext/model.py` with:

    **ConvNeXtStage(nn.Module)**:
    - `downsample`: nn.Sequential of [LayerNorm2d(in_dim), nn.Conv2d(in_dim, out_dim, 2, stride=2)]
      - For stage 0: downsample is None (stem handles initial processing)
    - `blocks`: nn.Sequential of ConvNeXtBlock instances
    - timm naming: `self.downsample` and `self.blocks`

    **ConvNeXt(nn.Module)**:
    Constructor args: `in_chans=3, num_classes=1000, depths=[3,3,9,3], dims=[96,192,384,768], drop_path_rate=0.0, layer_scale_init_value=1e-6, use_cifar_stem=False`

    Components (timm naming):
    1. `self.stem`: nn.Sequential
       - Default (ImageNet): [nn.Conv2d(in_chans, dims[0], 4, stride=4), LayerNorm2d(dims[0])]
       - CIFAR (`use_cifar_stem=True`): [nn.Conv2d(in_chans, dims[0], 3, stride=1, padding=1), LayerNorm2d(dims[0])]
       - Indexed as `stem.0` (conv) and `stem.1` (norm)

    2. `self.stages`: nn.Sequential of 4 ConvNeXtStage
       - Stage 0: no downsample, depths[0] blocks with dims[0]
       - Stages 1-3: downsample from dims[i-1] → dims[i], then depths[i] blocks
       - Stochastic depth: linearly increase drop_path_rate across ALL blocks
         - `dp_rates = torch.linspace(0, drop_path_rate, sum(depths))`
         - Each block gets its own rate from this schedule
       - Stage i downsample naming: `stages.{i}.downsample.0` (norm), `stages.{i}.downsample.1` (conv)
       - Stage i blocks naming: `stages.{i}.blocks.{j}.*`

    3. `self.head`: nn.Module with:
       - `self.head.norm = nn.LayerNorm(dims[-1])` — final norm
       - `self.head.fc = nn.Linear(dims[-1], num_classes)` — classifier
       - Create a small Head module or use a ModuleDict

    Forward pass:
    1. `x = self.stem(x)` — (B,3,H,W) → (B,dims[0],H/4,W/4) or (B,dims[0],H,W) for CIFAR
    2. `x = self.stages(x)` — pass through all 4 stages
    3. `x = x.mean([-2, -1])` — global average pooling (B,C,H,W) → (B,C)
    4. `x = self.head.norm(x)` — final layer norm
    5. `x = self.head.fc(x)` — classification

    Weight initialization:
    - Apply `trunc_normal_(weight, std=0.02)` to all Linear and Conv2d weights
    - Zero-init all biases

    DO NOT:
    - Forget to handle stage 0 specially (no downsample)
    - Use global average pooling as a module — use `x.mean([-2, -1])` inline
    - Miss the stochastic depth linear schedule across all blocks
  </action>
  <verify>
    python -c "
from convnext.model import ConvNeXt
import torch

# ImageNet Tiny
model = ConvNeXt(num_classes=1000, depths=[3,3,9,3], dims=[96,192,384,768])
x = torch.randn(2, 3, 224, 224)
out = model(x)
assert out.shape == (2, 1000), f'ImageNet shape: {out.shape}'

# CIFAR Tiny
model_c = ConvNeXt(num_classes=10, depths=[3,3,9,3], dims=[96,192,384,768], use_cifar_stem=True)
x_c = torch.randn(2, 3, 32, 32)
out_c = model_c(x_c)
assert out_c.shape == (2, 10), f'CIFAR shape: {out_c.shape}'

# Key naming
sd = model.state_dict()
checks = ['stem.0.weight', 'stem.1.weight', 'stages.0.blocks.0.conv_dw.weight',
          'stages.1.downsample.0.weight', 'stages.1.downsample.1.weight',
          'head.norm.weight', 'head.fc.weight']
for k in checks:
    assert k in sd, f'Missing: {k}'

print('Model structure verified ✓')
"
  </verify>
  <done>
    - ConvNeXt forward pass works for 224×224 and 32×32 inputs
    - State dict keys match timm naming for stem, stages, downsample, blocks, head
    - CIFAR stem uses 3×3 stride-1 conv
  </done>
</task>

<task type="auto">
  <name>Add variant factory functions and update package exports</name>
  <files>
    convnext/model.py (MODIFY — append factory functions)
    convnext/__init__.py (MODIFY — update exports)
  </files>
  <action>
    1. Add factory functions to bottom of `convnext/model.py`:
       ```python
       def convnext_tiny(num_classes=1000, **kwargs):
           return ConvNeXt(depths=[3,3,9,3], dims=[96,192,384,768],
                           drop_path_rate=0.1, num_classes=num_classes, **kwargs)

       def convnext_small(num_classes=1000, **kwargs):
           return ConvNeXt(depths=[3,3,27,3], dims=[96,192,384,768],
                           drop_path_rate=0.4, num_classes=num_classes, **kwargs)

       def convnext_base(num_classes=1000, **kwargs):
           return ConvNeXt(depths=[3,3,27,3], dims=[128,256,512,1024],
                           drop_path_rate=0.5, num_classes=num_classes, **kwargs)

       def convnext_large(num_classes=1000, **kwargs):
           return ConvNeXt(depths=[3,3,27,3], dims=[192,384,768,1536],
                           drop_path_rate=0.5, num_classes=num_classes, **kwargs)

       def convnext_xlarge(num_classes=1000, **kwargs):
           return ConvNeXt(depths=[3,3,27,3], dims=[256,512,1024,2048],
                           drop_path_rate=0.5, num_classes=num_classes, **kwargs)
       ```

    2. Update `convnext/__init__.py`:
       ```python
       from .model import ConvNeXt, convnext_tiny, convnext_small, convnext_base, convnext_large, convnext_xlarge
       from .block import ConvNeXtBlock
       from .layers import DropPath, LayerNorm2d
       ```
  </action>
  <verify>
    python -c "
from convnext import convnext_tiny, convnext_small, convnext_base, convnext_large, convnext_xlarge
import torch

expected_params = {
    'convnext_tiny': (25e6, 32e6),
    'convnext_small': (45e6, 55e6),
    'convnext_base': (83e6, 93e6),
    'convnext_large': (190e6, 205e6),
    'convnext_xlarge': (340e6, 360e6),
}

for name, fn in [('convnext_tiny', convnext_tiny), ('convnext_small', convnext_small),
                  ('convnext_base', convnext_base), ('convnext_large', convnext_large),
                  ('convnext_xlarge', convnext_xlarge)]:
    model = fn(num_classes=1000)
    params = sum(p.numel() for p in model.parameters())
    lo, hi = expected_params[name]
    assert lo < params < hi, f'{name}: {params/1e6:.1f}M not in range'
    out = model(torch.randn(1, 3, 224, 224))
    assert out.shape == (1, 1000), f'{name} output shape wrong'
    print(f'{name}: {params/1e6:.2f}M ✓')

print('All variants verified ✓')
"
  </verify>
  <done>
    - All 5 factory functions create models with correct param counts (within 10% of paper)
    - All variants produce (1, 1000) output for 224×224 input
    - Package exports work: `from convnext import convnext_tiny` etc.
  </done>
</task>

## Success Criteria
- [ ] ConvNeXt model forward pass works for both 224×224 and 32×32 inputs
- [ ] All 5 variants instantiate with param counts within 10% of paper values
- [ ] State dict keys match timm naming convention
- [ ] CIFAR stem preserves spatial resolution (no aggressive downsampling)
- [ ] Factory functions importable from `convnext` package
