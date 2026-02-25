# ROADMAP.md

> **Current Phase**: Not started
> **Milestone**: v1.0

## Must-Haves (from SPEC)

- [ ] ConvNeXt-Tiny model with correct architecture
- [ ] CIFAR-10 data pipeline with augmentations
- [ ] Training loop that converges on CIFAR-10
- [ ] Unit tests validating shapes and parameter counts
- [ ] Pretrained weight compatibility with timm

## Phases

### Phase 1: Model Architecture
**Status**: ⬜ Not Started
**Objective**: Implement the complete ConvNeXt model architecture — blocks, stem, downsampling, stages, classification head — for all 5 variants.
**Requirements**: REQ-01, REQ-02, REQ-03, REQ-04, REQ-05

**Key deliverables:**
- `convnext/model.py` — ConvNeXt block, stage, and full model
- All variants (T/S/B/L/XL) instantiate with correct param counts
- Forward pass works for both 32×32 and 224×224 inputs

---

### Phase 2: Data Pipeline
**Status**: ⬜ Not Started
**Objective**: Build CIFAR-10 data loading with training augmentations, structured for easy ImageNet-1K swap.
**Requirements**: REQ-06, REQ-07, REQ-08

**Key deliverables:**
- `convnext/data.py` — Dataset loading, transforms, dataloaders
- Train/val split with proper augmentation pipeline
- Configurable for different resolutions and datasets

---

### Phase 3: Training Loop
**Status**: ⬜ Not Started
**Objective**: Implement the full training pipeline with AdamW, cosine scheduler, checkpointing, and logging.
**Requirements**: REQ-09, REQ-10, REQ-11, REQ-12, REQ-13

**Key deliverables:**
- `train.py` — Main training script
- `convnext/utils.py` — Checkpoint, logging, metrics utilities
- Working training on CIFAR-10 that converges

---

### Phase 4: Testing & Verification
**Status**: ⬜ Not Started
**Objective**: Validate the implementation with unit tests, smoke tests, and pretrained weight loading.
**Requirements**: REQ-14, REQ-15, REQ-16, REQ-17

**Key deliverables:**
- `tests/test_model.py` — Architecture unit tests
- `tests/test_training.py` — Training smoke tests
- `scripts/load_pretrained.py` — Weight loading verification
- All tests pass ✓
