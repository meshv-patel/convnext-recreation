# SPEC.md — Project Specification

> **Status**: `FINALIZED`

## Vision

A clean, modular, from-scratch PyTorch implementation of the ConvNeXt architecture from "A ConvNet for the 2020s" (Liu et al., 2022). This is a learning and portfolio project prioritizing readable, well-documented code over benchmark reproduction. The implementation starts with the Tiny variant on CIFAR-10 for rapid iteration, with architecture designed to scale to all variants and ImageNet-1K.

## Goals

1. **Implement ConvNeXt architecture** — Faithful reproduction of the ConvNeXt block design (depthwise conv → LayerNorm → 1×1 conv → GELU → 1×1 conv), stem, downsampling layers, and classification head for all standard variants (Tiny, Small, Base, Large, XLarge).
2. **Build a flexible data pipeline** — CIFAR-10 data loading with augmentations (RandAugment, Mixup, CutMix, Random Erasing), structured to easily swap in ImageNet-1K later.
3. **Training loop with modern recipe** — AdamW optimizer, cosine annealing LR scheduler, label smoothing, and proper logging/checkpointing.
4. **Verification and testing** — Unit tests for model architecture (output shapes, parameter counts), training smoke tests, and optional weight-loading validation against timm/official checkpoints.

## Non-Goals (Out of Scope)

- ConvNeXt V2 improvements (GRN, FCMAE)
- Isotropic ConvNeXt variants
- Multi-GPU / distributed training (DDP, FSDP)
- ONNX export or deployment pipelines
- Hitting exact paper benchmark numbers (reasonable proximity is fine)
- EMA (Exponential Moving Average) — can be added later

## Users

This project is for the author's own learning and portfolio demonstration. Secondary audience: other ML practitioners studying ConvNeXt internals.

## Constraints

- **Tech stack**: Python 3.10+, PyTorch 2.x, torchvision
- **Hardware**: Single GPU training (consumer hardware)
- **Dataset**: CIFAR-10 primary, ImageNet-1K aspirational
- **Code quality**: Clean, modular, well-commented, PEP 8 compliant
- **Weight compatibility**: Model architecture must be compatible with loading pretrained weights from timm or official Meta checkpoints

## Success Criteria

- [ ] ConvNeXt-Tiny forward pass produces correct output shape for both CIFAR-10 (32×32) and ImageNet (224×224) inputs
- [ ] All 5 variants (T/S/B/L/XL) instantiate with correct parameter counts matching the paper
- [ ] Training on CIFAR-10 converges and achieves reasonable accuracy (>90%)
- [ ] Code is modular: model, data, training are independent modules
- [ ] Pretrained weights from timm load successfully into the model
- [ ] All unit tests pass
