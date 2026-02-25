# DECISIONS.md — Architecture Decision Records

## Phase 1 Decisions

**Date:** 2026-02-25

### ADR-01: CIFAR-10 Stem Strategy
- **Chose:** Alternate small-input stem (3×3 stride-1 conv) alongside the standard patchify stem
- **Reason:** CIFAR-10 images are 32×32; the standard 4×4 stride-4 stem would collapse spatial dims too aggressively (to 8×8). An alternate stem preserves resolution for small inputs while keeping ImageNet compatibility via the default stem. Controlled by constructor parameter.
- **Rejected:** Upsampling CIFAR-10 to 224×224 in the data pipeline — wasteful and slower training.

### ADR-02: File Structure
- **Chose:** Multi-file structure (`convnext/block.py`, `convnext/layers.py`, `convnext/model.py`)
- **Reason:** Cleaner separation of concerns, easier to test and extend each component independently. More modular for a portfolio project.
- **Rejected:** Single `model.py` — would grow to 300+ lines and mix concerns.

### ADR-03: Stochastic Depth
- **Chose:** Include stochastic depth (drop path) in Phase 1
- **Reason:** Architecturally important regularization technique used in the paper. Harder to add cleanly as a retrofit.

### ADR-04: Parameter Naming Convention
- **Chose:** Match timm's parameter naming exactly from the start
- **Reason:** Avoids needing a parameter mapping dictionary when loading pretrained weights in Phase 4 (REQ-17). One-time effort now prevents recurring friction.
