---
phase: 1
plan: 1
wave: 1
---

# Plan 1.1: Core Layers

## Objective
Create the foundational utility layers (`DropPath`, `LayerNorm2d`) and the package structure that all other model components depend on.

## Context
- .gsd/SPEC.md
- .gsd/DECISIONS.md (ADR-03: stochastic depth included)

## Tasks

<task type="auto">
  <name>Create package structure and core layers</name>
  <files>
    convnext/__init__.py (NEW)
    convnext/layers.py (NEW)
  </files>
  <action>
    1. Create `convnext/__init__.py`:
       - Import and export `ConvNeXt` and factory functions (will be available after Plan 1.3)
       - For now, just create the file with a docstring and placeholder imports

    2. Create `convnext/layers.py` with two classes:

       **DropPath** (stochastic depth):
       - Constructor takes `drop_prob: float = 0.0`
       - `forward(x)`: During training, randomly drops the entire input tensor with probability `drop_prob` (returns zeros). Scale surviving samples by `1 / (1 - drop_prob)`. During eval, returns input unchanged.
       - Implementation: generate random tensor of shape `(batch_size, 1, 1, ..., 1)`, apply Bernoulli threshold, multiply input

       **LayerNorm2d** (channels-first LayerNorm):
       - Subclass `nn.LayerNorm`
       - `forward(x)`: Permute `(B, C, H, W)` → `(B, H, W, C)`, apply standard LayerNorm, permute back to `(B, C, H, W)`
       - This is needed because Conv2d outputs channels-first but LayerNorm expects channels-last

    DO NOT:
    - Use `nn.functional.drop_path` — implement from scratch for learning
    - Use any external dependencies beyond PyTorch
  </action>
  <verify>
    python -c "
from convnext.layers import DropPath, LayerNorm2d
import torch

# DropPath test
dp = DropPath(0.5)
x = torch.ones(4, 3, 8, 8)
dp.train()
out_train = dp(x)
dp.eval()
out_eval = dp(x)
assert out_eval.equal(x), 'DropPath eval should be identity'
assert out_train.shape == x.shape, 'DropPath should preserve shape'

# LayerNorm2d test
ln = LayerNorm2d(64)
x = torch.randn(2, 64, 8, 8)
out = ln(x)
assert out.shape == (2, 64, 8, 8), f'LayerNorm2d shape wrong: {out.shape}'

print('Plan 1.1 verified ✓')
"
  </verify>
  <done>
    - DropPath correctly drops during training and passes through during eval
    - LayerNorm2d accepts (B,C,H,W) and outputs (B,C,H,W)
    - Both classes importable from convnext.layers
  </done>
</task>

## Success Criteria
- [ ] `convnext/` package importable
- [ ] DropPath passes shape and behavior tests
- [ ] LayerNorm2d passes shape test on channels-first input
