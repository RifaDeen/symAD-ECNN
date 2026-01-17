# SymAD-ECNN: Architecture Details - Model Comparison & Results

## 🏆 **FINAL RESULTS (January 2026)**

### Performance Comparison Table

| Model | Parameters | AUROC | AUPRC | Specificity | FP | Status |
|-------|------------|-------|-------|-------------|-----|--------|
| Baseline AE | ~8M | N/A | N/A | N/A | N/A | ❌ Failed |
| CNN-AE Small | ~8M | 0.7617 | 0.8255 | 56.42% | 1,590 | ✅ |
| CNN-AE Large | ~11M | 0.7803 | 0.8461 | 58.52% | 1,515 | ✅ |
| ECNN Buggy | ~11M | 0.7035 | 0.7716 | 47.86% | 1,904 | ⚠️ |
| **ECNN Optimized** | ~11M | **0.8109** | **0.8813** | **58.54%** | **1,514** | 🏆 |

### Key Insights

1. **Equivariance Beats Capacity**: +3.06% AUROC vs Large CNN-AE (same params)
2. **Architecture Bug Cost**: -7.74% AUROC from naive channel repetition
3. **Baseline AE Failure**: Fully-connected architecture couldn't handle 128×128 spatial data
4. **Thesis Validated**: ✅ "Structure > Capacity" - geometric inductive bias wins

---

## 🔧 Critical ECNN Bug & Fix

### The Bug (07_ecnn_autoencoder.ipynb)

```python
# BOTTLENECK - destroys equivariance
self.group_pool = e2nn.GroupPooling(self.feat_type_512)
self.flat_dim = 128 * 4 * 4  # 512 channels → 128 invariant

# DECODER - naive repetition (BUG!)
def forward(self, x):
    # ... encoder ...
    z = self.group_pool(bottleneck)  # ← Makes invariant
    flat = z.tensor.view(z.size(0), -1)
    decoded_flat = self.fc_decode(flat)
    decoded_features = decoded_flat.view(-1, 128, 4, 4)
    
    # 🔴 BUG: Just duplicates 128 channels 4 times!
    x_recon = e2nn.GeometricTensor(
        decoded_features.repeat(1, 4, 1, 1),  # ← DESTROYS EQUIVARIANCE
        self.feat_type_512
    )
```

**Result**: AUROC 0.7035 (catastrophic failure)

### The Fix (08_ecnn_optimized.ipynb)

```python
# BOTTLENECK - stays equivariant!
self.bottleneck = e2nn.R2Conv(self.feat_type_512, self.feat_type_512, 3, 1)
self.bottleneck_bn = e2nn.InnerBatchNorm(self.feat_type_512)

# DECODER - proper equivariant reconstruction
def forward(self, x):
    # ... encoder ...
    bottleneck = self.bottleneck(e4)  # ← Stays equivariant
    bottleneck = self.bottleneck_bn(bottleneck)
    
    # ✅ FIX: Proper upsampling with GeometricTensor throughout
    d4 = F.interpolate(bottleneck.tensor, scale_factor=2, mode='bilinear')
    d4 = self.dec4(e2nn.GeometricTensor(d4, self.feat_type_512))
    # ... continues with proper field types ...
    
# GroupPooling only for latent extraction (analysis), not reconstruction
def get_latent(self, x):
    # ... encode ...
    invariant = self.group_pool(bottleneck)  # ← Only used here
    return self.fc_latent(invariant.tensor.view(invariant.size(0), -1))
```

**Result**: AUROC 0.8109 (+7.74% improvement!)

---

## 📐 Detailed Model Architectures (Proposal Alignment)

**Reference**: This document provides layer-by-layer implementation details for all models, enabling reproducibility (NFR4) and benchmarking (FR8).

---

## Model 1: Baseline Autoencoder (Fully Connected)

### Architecture Diagram

```
INPUT: 128×128×1 = 16,384 pixels
         ↓
     [Flatten]
         ↓
    16,384 neurons
         ↓
┌─────────────────┐
│    ENCODER      │
├─────────────────┤
│ Dense(512)      │ ← ReLU + BatchNorm
│ Dense(256)      │ ← ReLU + BatchNorm + Dropout(0.2)
│ Dense(128)      │ ← LATENT SPACE (bottleneck)
└─────────────────┘
         ↓
┌─────────────────┐
│    DECODER      │
├─────────────────┤
│ Dense(256)      │ ← ReLU + BatchNorm
│ Dense(512)      │ ← ReLU + BatchNorm + Dropout(0.2)
│ Dense(16,384)   │ ← Sigmoid activation
└─────────────────┘
         ↓
    [Reshape]
         ↓
OUTPUT: 128×128×1
```

### Layer Details

| Layer | Type | Input Shape | Output Shape | Parameters | Activation |
|-------|------|-------------|--------------|------------|------------|
| Input | - | (128, 128, 1) | (128, 128, 1) | 0 | - |
| Flatten | Reshape | (128, 128, 1) | (16384,) | 0 | - |
| Dense1 | Fully Connected | (16384,) | (512,) | 8,388,608 | ReLU |
| BatchNorm1 | Normalization | (512,) | (512,) | 1,024 | - |
| Dense2 | Fully Connected | (512,) | (256,) | 131,072 | ReLU |
| BatchNorm2 | Normalization | (256,) | (256,) | 512 | - |
| Dropout1 | Regularization | (256,) | (256,) | 0 | - |
| Dense3 (Latent) | Fully Connected | (256,) | (128,) | 32,768 | ReLU |
| Dense4 | Fully Connected | (128,) | (256,) | 32,768 | ReLU |
| BatchNorm3 | Normalization | (256,) | (256,) | 512 | - |
| Dense5 | Fully Connected | (256,) | (512,) | 131,072 | ReLU |
| BatchNorm4 | Normalization | (512,) | (512,) | 1,024 | - |
| Dropout2 | Regularization | (512,) | (512,) | 0 | - |
| Dense6 | Fully Connected | (512,) | (16384,) | 8,388,608 | Sigmoid |
| Reshape | Reshape | (16384,) | (128, 128, 1) | 0 | - |

**Total Parameters**: ~8.5 Million  
**Trainable Parameters**: ~8.5 Million  
**Expected Performance**: AUROC 0.75-0.80 (baseline reference for FR8)

### Characteristics

**Advantages**:
- ✅ Simple architecture, easy to implement and debug
- ✅ Fast training (~20-30 minutes on Colab GPU - NFR3)
- ✅ Good baseline for comparative benchmarking (FR8)
- ✅ Interpretable latent space (128 dimensions)

**Disadvantages (Literature Review Findings - Table 11)**:
- ❌ No spatial awareness or geometric understanding
- ❌ High parameter count for simple architecture
- ❌ Loses 2D spatial structure information
- ❌ Poor at capturing local anatomical patterns
- ❌ Not rotation or translation invariant

**Use Case**: Establish minimum performance threshold for anomaly detection research

---

## Model 2: CNN-Autoencoder

### Architecture Diagram

```
INPUT: 128×128×1
         ↓
┌─────────────────────────────────────────┐
│              ENCODER                     │
├─────────────────────────────────────────┤
│ Conv2D(32, 3×3) → 128×128×32           │ ← ReLU + BatchNorm
│      ↓ MaxPool(2×2)                     │
│ Conv2D(64, 3×3) → 64×64×64             │ ← ReLU + BatchNorm
│      ↓ MaxPool(2×2)                     │
│ Conv2D(128, 3×3) → 32×32×128           │ ← ReLU + BatchNorm
│      ↓ MaxPool(2×2)                     │
│ Conv2D(256, 3×3) → 16×16×256           │ ← ReLU + BatchNorm
│      ↓ MaxPool(2×2)                     │
│           8×8×256                        │
│      ↓ Flatten                          │
│          16,384                          │
│      ↓ Dense                            │
│          256 ← LATENT SPACE             │
└─────────────────────────────────────────┘
         ↓
┌─────────────────────────────────────────┐
│              DECODER                     │
├─────────────────────────────────────────┤
│ Dense(16,384) → 16,384                 │ ← ReLU
│      ↓ Reshape                          │
│           8×8×256                        │
│      ↓ Conv2DTranspose(256, 3×3)       │ ← ReLU + BatchNorm
│           16×16×256                      │
│      ↓ Conv2DTranspose(128, 3×3)       │ ← ReLU + BatchNorm
│           32×32×128                      │
│      ↓ Conv2DTranspose(64, 3×3)        │ ← ReLU + BatchNorm
│           64×64×64                       │
│      ↓ Conv2DTranspose(32, 3×3)        │ ← ReLU + BatchNorm
│           128×128×32                     │
│      ↓ Conv2D(1, 3×3)                  │ ← Sigmoid
│           128×128×1                      │
└─────────────────────────────────────────┘
         ↓
OUTPUT: 128×128×1
```

### Layer Details

#### Encoder
| Layer | Type | Filters | Kernel | Stride | Output Shape | Parameters |
|-------|------|---------|--------|--------|--------------|------------|
| Input | - | - | - | - | (128, 128, 1) | 0 |
| Conv1 | Conv2D | 32 | 3×3 | 1 | (128, 128, 32) | 320 |
| BN1 | BatchNorm | - | - | - | (128, 128, 32) | 128 |
| Pool1 | MaxPool2D | - | 2×2 | 2 | (64, 64, 32) | 0 |
| Conv2 | Conv2D | 64 | 3×3 | 1 | (64, 64, 64) | 18,496 |
| BN2 | BatchNorm | - | - | - | (64, 64, 64) | 256 |
| Pool2 | MaxPool2D | - | 2×2 | 2 | (32, 32, 64) | 0 |
| Conv3 | Conv2D | 128 | 3×3 | 1 | (32, 32, 128) | 73,856 |
| BN3 | BatchNorm | - | - | - | (32, 32, 128) | 512 |
| Pool3 | MaxPool2D | - | 2×2 | 2 | (16, 16, 128) | 0 |
| Conv4 | Conv2D | 256 | 3×3 | 1 | (16, 16, 256) | 295,168 |
| BN4 | BatchNorm | - | - | - | (16, 16, 256) | 1,024 |
| Pool4 | MaxPool2D | - | 2×2 | 2 | (8, 8, 256) | 0 |
| Flatten | - | - | - | - | (16,384) | 0 |
| Dense | Fully Connected | - | - | - | (256) | 4,194,560 |

#### Decoder
| Layer | Type | Filters | Kernel | Stride | Output Shape | Parameters |
|-------|------|---------|--------|--------|--------------|------------|
| Dense | Fully Connected | - | - | - | (16,384) | 4,194,560 |
| Reshape | - | - | - | - | (8, 8, 256) | 0 |
| ConvT1 | Conv2DTranspose | 256 | 3×3 | 2 | (16, 16, 256) | 590,080 |
| BN5 | BatchNorm | - | - | - | (16, 16, 256) | 1,024 |
| ConvT2 | Conv2DTranspose | 128 | 3×3 | 2 | (32, 32, 128) | 295,040 |
| BN6 | BatchNorm | - | - | - | (32, 32, 128) | 512 |
| ConvT3 | Conv2DTranspose | 64 | 3×3 | 2 | (64, 64, 64) | 73,792 |
| BN7 | BatchNorm | - | - | - | (64, 64, 64) | 256 |
| ConvT4 | Conv2DTranspose | 32 | 3×3 | 2 | (128, 128, 32) | 18,464 |
| BN8 | BatchNorm | - | - | - | (128, 128, 32) | 128 |
| Conv5 | Conv2D | 1 | 3×3 | 1 | (128, 128, 1) | 289 |

**Total Parameters**: ~9.7 Million  
**Trainable Parameters**: ~9.7 Million

### Characteristics

**Advantages**:
- ✅ Spatial feature extraction
- ✅ Hierarchical representation learning
- ✅ Fewer parameters than baseline
- ✅ Better for image data
- ✅ Captures local and global patterns

**Disadvantages**:
- ❌ Not rotation invariant
- ❌ Requires data augmentation
- ❌ May overfit to training orientations
- ❌ Higher false positives on rotated inputs

**Use Case**: Standard deep learning approach for image anomaly detection

---

## Model 3: E(n)-Equivariant CNN-Autoencoder ⭐ MAIN MODEL

### Architecture Diagram

```
INPUT: 128×128×1
         ↓
┌─────────────────────────────────────────────────────────┐
│              E(2)-EQUIVARIANT ENCODER                    │
├─────────────────────────────────────────────────────────┤
│ R2Conv(32, C4) → 128×128×32                            │ ← Regular repr.
│      ↓ MaxPool(2×2)                                     │
│ R2Conv(64, C4) → 64×64×64                              │ ← Regular repr.
│      ↓ MaxPool(2×2)                                     │
│ R2Conv(128, C4) → 32×32×128                            │ ← Regular repr.
│      ↓ MaxPool(2×2)                                     │
│ R2Conv(256, C4) → 16×16×256                            │ ← Regular repr.
│      ↓ MaxPool(2×2)                                     │
│           8×8×256                                        │
│      ↓ GroupPooling (Invariant)                        │
│           8×8×256                                        │
│      ↓ Flatten                                          │
│          16,384                                          │
│      ↓ Dense                                            │
│          256 ← LATENT SPACE (Invariant)                │
└─────────────────────────────────────────────────────────┘
         ↓
┌─────────────────────────────────────────────────────────┐
│              E(2)-EQUIVARIANT DECODER                    │
├─────────────────────────────────────────────────────────┤
│ Dense(16,384) → 16,384                                 │
│      ↓ Reshape                                          │
│           8×8×256                                        │
│      ↓ R2ConvTranspose(256, C4)                        │ ← Regular repr.
│           16×16×256                                      │
│      ↓ R2ConvTranspose(128, C4)                        │ ← Regular repr.
│           32×32×128                                      │
│      ↓ R2ConvTranspose(64, C4)                         │ ← Regular repr.
│           64×64×64                                       │
│      ↓ R2ConvTranspose(32, C4)                         │ ← Regular repr.
│           128×128×32                                     │
│      ↓ R2Conv(1, Trivial)                              │ ← Sigmoid
│           128×128×1                                      │
└─────────────────────────────────────────────────────────┘
         ↓
OUTPUT: 128×128×1
```

### Understanding E(2)-Equivariant Layers

#### What is E(2)?
- **E(2)**: Euclidean group in 2D (rotations + translations)
- **C4**: Cyclic group with 4 rotations (0°, 90°, 180°, 270°)
- **Regular Representation**: Feature maps that transform with input

#### R2Conv vs Standard Conv2D

**Standard Conv2D**:
```
Input: [B, C_in, H, W]
Kernel: [C_out, C_in, K, K]
Output: [B, C_out, H', W']
```

**R2Conv (E(2)-Equivariant)**:
```
Input: [B, C_in, H, W]
Kernel: [C_out, C_in, K, K, |G|]  ← |G| = 4 for C4 group
Output: [B, C_out, H', W']

Where kernel has 4 rotated versions (0°, 90°, 180°, 270°)
```

#### Group Pooling (Making Invariant)
```
Input: [B, C, H, W, 4]  ← 4 rotated feature maps
GroupPooling: max/mean over group dimension
Output: [B, C, H, W]  ← Rotation-invariant features
```

### Layer Details

#### Encoder (E(2)-Equivariant)
| Layer | Type | Filters | Group | Output Shape | Equivariant? |
|-------|------|---------|-------|--------------|--------------|
| Input | - | - | - | (128, 128, 1) | - |
| R2Conv1 | E(2)-Conv | 32 | C4 | (128, 128, 32) | ✓ Rotation |
| BN1 | IIDBatchNorm | - | - | (128, 128, 32) | ✓ |
| ReLU1 | PointwiseNonLinearity | - | - | (128, 128, 32) | ✓ |
| Pool1 | PointwiseMaxPool | - | - | (64, 64, 32) | ✓ |
| R2Conv2 | E(2)-Conv | 64 | C4 | (64, 64, 64) | ✓ Rotation |
| BN2 | IIDBatchNorm | - | - | (64, 64, 64) | ✓ |
| ReLU2 | PointwiseNonLinearity | - | - | (64, 64, 64) | ✓ |
| Pool2 | PointwiseMaxPool | - | - | (32, 32, 64) | ✓ |
| R2Conv3 | E(2)-Conv | 128 | C4 | (32, 32, 128) | ✓ Rotation |
| BN3 | IIDBatchNorm | - | - | (32, 32, 128) | ✓ |
| ReLU3 | PointwiseNonLinearity | - | - | (32, 32, 128) | ✓ |
| Pool3 | PointwiseMaxPool | - | - | (16, 16, 128) | ✓ |
| R2Conv4 | E(2)-Conv | 256 | C4 | (16, 16, 256) | ✓ Rotation |
| BN4 | IIDBatchNorm | - | - | (16, 16, 256) | ✓ |
| ReLU4 | PointwiseNonLinearity | - | - | (16, 16, 256) | ✓ |
| Pool4 | PointwiseMaxPool | - | - | (8, 8, 256) | ✓ |
| GroupPool | GroupPooling | - | C4 | (8, 8, 256) | ✗ Invariant |
| Flatten | - | - | - | (16,384) | ✗ |
| Dense | Fully Connected | 256 | - | (256) | ✗ |

**Total Parameters**: ~10.5 Million  
**Trainable Parameters**: ~10.5 Million

### Characteristics

**Advantages**:
- ✅ **Rotation Equivariant**: Features rotate with input
- ✅ **Translation Equivariant**: Built into convolutions
- ✅ **No Data Augmentation**: Handles rotations internally
- ✅ **Reduced Overfitting**: Symmetry constraints
- ✅ **Lower False Positives**: Invariant to orientation
- ✅ **Better Generalization**: Works on unseen rotations
- ✅ **Physically Meaningful**: Respects image symmetries

**Disadvantages**:
- ❌ Slightly more complex implementation
- ❌ Requires e2cnn library
- ❌ ~8% more parameters than standard CNN (due to group structure)
- ❌ Slightly slower forward pass (group operations)

**Use Case**: Production-ready anomaly detection with robust performance

---

## Comparison Summary

| Feature | Baseline | CNN-AE | ECNN-AE |
|---------|----------|--------|---------|
| **Parameters** | ~17M | ~9.7M | ~10.5M |
| **Spatial Awareness** | ❌ | ✅ | ✅ |
| **Rotation Invariance** | ❌ | ❌ | ✅ |
| **Data Augmentation** | Required | Required | Not Required |
| **False Positive Rate** | High | Medium | Low |
| **Training Speed** | Fast | Medium | Medium-Slow |
| **Inference Speed** | Fast | Fast | Medium |
| **Memory Usage** | High | Medium | Medium |
| **Generalization** | Poor | Good | Excellent |
| **Best For** | Baseline | Standard images | Medical imaging |

---

## Mathematical Formulation

### Standard Autoencoder Loss
```
L = MSE(x, x̂) + λ * KL(q(z|x) || p(z))

Where:
- x: Input image
- x̂: Reconstructed image
- z: Latent representation
- λ: Regularization weight
```

### E(2)-Equivariant Constraint
```
For any rotation ρ ∈ E(2):
f(ρ(x)) = ρ(f(x))

Where:
- f: Neural network function
- ρ: Rotation transformation
- x: Input image
```

### Anomaly Score
```
A(x) = ||x - Decoder(Encoder(x))||²

High A(x) → Anomaly (tumor)
Low A(x) → Normal tissue
```

---

## Implementation Notes

### PyTorch Implementation Structure

```python
# Baseline Autoencoder
class BaselineAE(nn.Module):
    def __init__(self):
        self.encoder = nn.Sequential(...)
        self.decoder = nn.Sequential(...)
    
    def forward(self, x):
        z = self.encoder(x.view(-1, 16384))
        x_recon = self.decoder(z).view(-1, 1, 128, 128)
        return x_recon

# CNN Autoencoder
class CNNAE(nn.Module):
    def __init__(self):
        self.encoder = nn.Sequential(...)
        self.decoder = nn.Sequential(...)
    
    def forward(self, x):
        z = self.encoder(x)
        x_recon = self.decoder(z)
        return x_recon

# E(2)-Equivariant Autoencoder
from e2cnn import gspaces, nn as e2nn

class ECNNAE(nn.Module):
    def __init__(self):
        self.r2_act = gspaces.Rot2dOnR2(N=4)  # C4 group
        self.encoder = self._build_encoder()
        self.decoder = self._build_decoder()
    
    def forward(self, x):
        # Wrap input in geometric tensor
        x_g = e2nn.GeometricTensor(x, self.in_type)
        z = self.encoder(x_g)
        x_recon = self.decoder(z)
        return x_recon.tensor  # Extract tensor
```

---

## Visualization of Equivariance

```
Input Image (0°)          Rotated Input (90°)
     ┌────┐                    ┌────┐
     │ 🧠 │                    │ 🧠 │
     │    │                    │    │ ↻
     └────┘                    └────┘

Standard CNN:                E(2)-Equivariant CNN:
Features: [f1, f2, f3]       Features: [f1, f2, f3]
     ↓                            ↓
Different features!          Rotated features!
(Not equivariant)            (Equivariant ✓)

Result:                      Result:
May fail to detect           Correctly detects
rotated tumor                rotated tumor
```

---

**Next**: See `TRAINING_PIPELINE.md` for training details and `EQUIVARIANCE_EXPLAINED.md` for theory.
