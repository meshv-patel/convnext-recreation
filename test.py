import torch
from convnext import convnext_tiny

# Test 1: ImageNet 
model = convnext_tiny(num_classes=1000)
x = torch.randn(2, 3, 224, 224)

with torch.no_grad():
    out = model(x)

params = sum(p.numel() for p in model.parameters()) / 1e6
print(f"ImageNet | Input: {x.shape} → Output: {out.shape} | Params: {params:.1f}M")

# Test 2: CIFAR-10 
model_cifar = convnext_tiny(num_classes=10, small_input=True)
x_cifar = torch.randn(2, 3, 32, 32)

with torch.no_grad():
    out_cifar = model_cifar(x_cifar)

print(f"CIFAR-10 | Input: {x_cifar.shape} → Output: {out_cifar.shape}")

# Test 3: Param counts 
print(f"\nVariant Param Counts")
from convnext import convnext_small, convnext_base, convnext_large, convnext_xlarge

variants = {
    "Tiny":   convnext_tiny(num_classes=1000),
    "Small":  convnext_small(num_classes=1000),
    "Base":   convnext_base(num_classes=1000),
    "Large":  convnext_large(num_classes=1000),
    "XLarge": convnext_xlarge(num_classes=1000),
}

for name, m in variants.items():
    p = sum(x.numel() for x in m.parameters()) / 1e6
    print(f"ConvNeXt-{name}: {p:.1f}M params")