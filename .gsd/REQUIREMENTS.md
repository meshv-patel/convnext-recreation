# REQUIREMENTS.md

## Format

| ID | Requirement | Source | Status |
|----|-------------|--------|--------|
| REQ-01 | ConvNeXt block: depthwise conv (7×7) → LayerNorm → 1×1 conv → GELU → 1×1 conv with residual connection | SPEC goal 1 | Pending |
| REQ-02 | Patchify stem: 4×4 stride-4 conv + LayerNorm | SPEC goal 1 | Pending |
| REQ-03 | Downsampling layers: LayerNorm + 2×2 stride-2 conv between stages | SPEC goal 1 | Pending |
| REQ-04 | Classification head: global average pool → LayerNorm → linear | SPEC goal 1 | Pending |
| REQ-05 | All 5 variants (T/S/B/L/XL) with correct channel dims and block counts | SPEC goal 1 | Pending |
| REQ-06 | CIFAR-10 data loading with train/val split | SPEC goal 2 | Pending |
| REQ-07 | Data augmentations: RandAugment, Mixup, CutMix, Random Erasing | SPEC goal 2 | Pending |
| REQ-08 | Configurable transforms for different input resolutions | SPEC goal 2 | Pending |
| REQ-09 | AdamW optimizer with weight decay | SPEC goal 3 | Pending |
| REQ-10 | Cosine annealing LR scheduler with warmup | SPEC goal 3 | Pending |
| REQ-11 | Label smoothing cross-entropy loss | SPEC goal 3 | Pending |
| REQ-12 | Checkpoint saving/loading (model + optimizer + scheduler state) | SPEC goal 3 | Pending |
| REQ-13 | Training metrics logging (loss, accuracy, LR per epoch) | SPEC goal 3 | Pending |
| REQ-14 | Unit tests: output shapes for all variants at multiple resolutions | SPEC goal 4 | Pending |
| REQ-15 | Unit tests: parameter count validation against paper | SPEC goal 4 | Pending |
| REQ-16 | Training smoke test: overfit a small batch | SPEC goal 4 | Pending |
| REQ-17 | Pretrained weight loading from timm | SPEC goal 4 | Pending |
