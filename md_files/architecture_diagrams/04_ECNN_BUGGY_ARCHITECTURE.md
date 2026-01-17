# ECNN Autoencoder (Buggy) Architecture

**Model**: ECNN-AE Buggy (Naive Channel Repetition)  
**Status**: ⚠️ Bug Fixed (Documented)  
**Performance**: AUROC 0.7035 (-7.68% vs Large CNN-AE)  
**Parameters**: ~11M (same as Large CNN-AE & ECNN Optimized)  
**Type**: E(2)-Equivariant Convolutional Autoencoder (BROKEN DECODER)

---

## Purpose

**Bug Documentation**: This model shows what happens when equivariant architecture is implemented incorrectly.

**Critical Bug**: Decoder uses **naive channel repetition** instead of proper equivariant expansion.

**Impact**: 
- -7.68% AUROC vs Large CNN-AE (0.7035 vs 0.7803)
- ECNN should beat CNN-AE, but buggy version performs WORSE
- Proves equivariance must be implemented correctly throughout network

---

## Architecture Overview

```
INPUT: 128×128×1
         ↓
┌──────────────────────────────────────────┐
│     ENCODER (EQUIVARIANT ✅)              │
├──────────────────────────────────────────┤
│ R2Conv(trivial→64ch, k=7, s=2, p=3)      │ 64×64×64
│ ↓ InnerBatchNorm + ReLU                  │
│ ↓ PointwiseMaxPool(2)                    │ 32×32×64
│                                           │
│ R2Conv(64→128, k=3, s=1, p=1)            │ 32×32×128
│ ↓ InnerBatchNorm + ReLU                  │
│ ↓ PointwiseMaxPool(2)                    │ 16×16×128
│                                           │
│ R2Conv(128→256, k=3, s=1, p=1)           │ 16×16×256
│ ↓ InnerBatchNorm + ReLU                  │
│ ↓ PointwiseMaxPool(2)                    │ 8×8×256
│                                           │
│ R2Conv(256→512, k=3, s=1, p=1)           │ 8×8×512
│ ↓ InnerBatchNorm + ReLU                  │
│ ↓ PointwiseMaxPool(2)                    │ 4×4×512
└──────────────────────────────────────────┘
         ↓
    [GroupPooling: 512ch → 128ch]
    [Flatten: 128×4×4 = 2,048]
         ↓
┌──────────────────────────────────────────┐
│       LATENT SPACE (STANDARD)             │
├──────────────────────────────────────────┤
│ Linear(2,048 → 512)                      │ 512-dim
│                                           │
│ Linear(512 → 2,048)                      │
│ ↓ Reshape(128, 4, 4)                     │
│ ↓ .repeat(1, 4, 1, 1) [BUG! ❌]          │ 4×4×512
└──────────────────────────────────────────┘
         ↓
┌──────────────────────────────────────────┐
│     DECODER (BROKEN! ❌)                  │
├──────────────────────────────────────────┤
│ 🚨 NAIVE CHANNEL REPETITION               │
│    decoded_features.repeat(1, 4, 1, 1)   │
│    → Copies channels without rotation     │
│    → Breaks equivariance!                 │
│                                           │
│ Interpolate(scale=2) → 8×8               │ 8×8×512
│ ↓ R2Conv(512→512, k=3, p=1) + BN + ReLU │
│                                           │
│ Interpolate(scale=2) → 16×16             │ 16×16×512
│ ↓ R2Conv(512→256, k=3, p=1) + BN + ReLU │
│                                           │
│ Interpolate(scale=2) → 32×32             │ 32×32×256
│ ↓ R2Conv(256→128, k=3, p=1) + BN + ReLU │
│                                           │
│ Interpolate(scale=2) → 64×64             │ 64×64×128
│ ↓ R2Conv(128→64, k=3, p=1) + BN + ReLU  │
│                                           │
│ Interpolate(scale=2) → 128×128           │ 128×128×64
│ ↓ R2Conv(64→trivial, k=3, p=1)          │ 128×128×1
│ ↓ Sigmoid                                 │
└──────────────────────────────────────────┘
         ↓
OUTPUT: 128×128×1
```

---

## The Critical Bug Explained

### Problematic Code (Line 338-341)

```python
# Decode
decoded_flat = self.fc_decode(z)
# Reshape to (Batch, 128, 4, 4)
decoded_features = decoded_flat.view(-1, 128, 4, 4)

# 🚨 BUG: Naive channel repetition
x_recon = e2nn.GeometricTensor(decoded_features.repeat(1, 4, 1, 1), self.feat_type_512)
```

### What Happens

1. **GroupPooling (Encoder Output)**:
   - Input: 512 channels (128 fields × 4 rotations)
   - Output: 128 channels (max-pool over 4 rotations per field)
   - **Correct**: Reduces channels while preserving rotation information

2. **Fully-Connected Layers**:
   - Flatten: 128×4×4 = 2,048
   - Encode: Linear(2,048 → 512)
   - Decode: Linear(512 → 2,048)
   - Reshape: (Batch, 128, 4, 4)
   - **Correct**: Standard autoencoder bottleneck

3. **Naive Repetition (THE BUG)**:
   ```python
   decoded_features.repeat(1, 4, 1, 1)  # ❌ WRONG!
   # Shape: (Batch, 128, 4, 4) → (Batch, 512, 4, 4)
   # Copies each channel 4 times: [a, a, a, a, b, b, b, b, ...]
   ```

   **Why It's Wrong**:
   - Decoder expects **equivariant features**: [a₀, a₉₀, a₁₈₀, a₂₇₀, b₀, b₉₀, ...]
   - Bug gives **naive copies**: [a, a, a, a, b, b, b, b, ...]
   - **Breaks equivariance**: All 4 channels identical → no rotation information!

4. **Consequence**:
   - Decoder R2Conv layers expect rotated versions of each feature
   - Instead, receive 4 identical copies
   - Network can't reconstruct with rotation awareness
   - Performance collapses (-7.68% AUROC)

---

## Why This Breaks Equivariance

### Theory: Equivariant Decoder Requires Rotated Features

For a **regular representation** (C4 group), each feature field has 4 channels:
```
Field i: [f_i(0°), f_i(90°), f_i(180°), f_i(270°)]
```

**Correct Decoder Input** (128 fields × 4 rotations = 512 channels):
```
[f₀(0°), f₀(90°), f₀(180°), f₀(270°),   ← Field 0 rotated
 f₁(0°), f₁(90°), f₁(180°), f₁(270°),   ← Field 1 rotated
 ...
 f₁₂₇(0°), f₁₂₇(90°), f₁₂₇(180°), f₁₂₇(270°)]  ← Field 127 rotated
```

**Buggy Decoder Input** (naive repetition):
```
[f₀, f₀, f₀, f₀,   ← Same feature 4 times! ❌
 f₁, f₁, f₁, f₁,   ← Same feature 4 times! ❌
 ...
 f₁₂₇, f₁₂₇, f₁₂₇, f₁₂₇]  ← Same feature 4 times! ❌
```

### Impact on Reconstruction

**R2Conv Expects**:
- Input: Equivariant tensor with rotated versions
- Operation: Convolution that respects rotations
- Output: Correctly rotated features

**R2Conv Receives (Buggy)**:
- Input: 4 identical copies of each feature (no rotation info)
- Operation: Convolution sees same pattern 4 times
- Output: Reconstructed image ignores rotation → poor quality

### Mathematical Illustration

**Correct Equivariant Decoding**:
```
g * f(x) = f(g⁻¹ * x)  ← Rotation commutes with operations
```

**Buggy Decoding**:
```
g * f(x) ≠ f(g⁻¹ * x)  ← Rotation broken by naive repetition!
```

**Result**: Decoder loses rotation equivariance → can't handle rotated tumors → higher error.

---

## Detailed Layer-by-Layer Breakdown

### Encoder (CORRECT ✅)

#### Input Field Type
```python
self.r2_act = gspaces.Rot2dOnR2(N=4)  # C4 group
self.in_type = e2nn.FieldType(self.r2_act, [self.r2_act.trivial_repr])
# 1 channel (grayscale), trivial representation (no rotation)
```

#### Feature Field Types
```python
# Regular representation: Each feature has 4 rotations (0°, 90°, 180°, 270°)
self.feat_type_64  = e2nn.FieldType(self.r2_act, 16*[self.r2_act.regular_repr])  # 16 fields × 4 = 64 ch
self.feat_type_128 = e2nn.FieldType(self.r2_act, 32*[self.r2_act.regular_repr])  # 32 fields × 4 = 128 ch
self.feat_type_256 = e2nn.FieldType(self.r2_act, 64*[self.r2_act.regular_repr])  # 64 fields × 4 = 256 ch
self.feat_type_512 = e2nn.FieldType(self.r2_act, 128*[self.r2_act.regular_repr]) # 128 fields × 4 = 512 ch
```

#### Block 1: 128×128 → 32×32
```python
e2nn.R2Conv(self.in_type, self.feat_type_64, kernel_size=7, padding=3, stride=2)
e2nn.InnerBatchNorm(self.feat_type_64)
e2nn.ReLU(self.feat_type_64)
e2nn.PointwiseMaxPool(self.feat_type_64, 2)
```
- **Input**: 128×128×1 (trivial)
- **After R2Conv (s=2)**: 64×64×64 (16 fields)
- **After MaxPool**: 32×32×64
- **Parameters**: 1×16×4×7×7 + 64 = **3,200**
- **Equivariance**: ✅ R2Conv respects C4 rotations

#### Block 2: 32×32 → 16×16
```python
e2nn.R2Conv(self.feat_type_64, self.feat_type_128, kernel_size=3, padding=1, stride=1)
e2nn.InnerBatchNorm(self.feat_type_128)
e2nn.ReLU(self.feat_type_128)
e2nn.PointwiseMaxPool(self.feat_type_128, 2)
```
- **Input**: 32×32×64 (16 fields)
- **Output**: 16×16×128 (32 fields)
- **Parameters**: 16×32×4×4×3×3 = **73,728**
- **Equivariance**: ✅

#### Block 3: 16×16 → 8×8
```python
e2nn.R2Conv(self.feat_type_128, self.feat_type_256, kernel_size=3, padding=1, stride=1)
e2nn.InnerBatchNorm(self.feat_type_256)
e2nn.ReLU(self.feat_type_256)
e2nn.PointwiseMaxPool(self.feat_type_256, 2)
```
- **Input**: 16×16×128 (32 fields)
- **Output**: 8×8×256 (64 fields)
- **Parameters**: 32×64×4×4×3×3 = **294,912**
- **Equivariance**: ✅

#### Block 4: 8×8 → 4×4
```python
e2nn.R2Conv(self.feat_type_256, self.feat_type_512, kernel_size=3, padding=1, stride=1)
e2nn.InnerBatchNorm(self.feat_type_512)
e2nn.ReLU(self.feat_type_512)
e2nn.PointwiseMaxPool(self.feat_type_512, 2)
```
- **Input**: 8×8×256 (64 fields)
- **Output**: 4×4×512 (128 fields)
- **Parameters**: 64×128×4×4×3×3 = **1,179,648**
- **Equivariance**: ✅

---

### Bottleneck (PARTIALLY CORRECT ⚠️)

#### GroupPooling
```python
self.group_pool = e2nn.GroupPooling(self.feat_type_512)
```
- **Input**: 4×4×512 (128 fields × 4 rotations)
- **Output**: 4×4×128 (128 channels, max over 4 rotations per field)
- **Operation**: `max(f₀°, f₉₀°, f₁₈₀°, f₂₇₀°)` for each field
- **Purpose**: Achieve **rotation invariance** (max-pool removes rotation dimension)
- **Status**: ✅ **CORRECT** (standard equivariant pooling)

#### Fully-Connected Encode
```python
self.flat_dim = 128 * 4 * 4 = 2,048
self.fc_encode = nn.Linear(2,048, 512)
```
- **Input**: 2,048 (flattened)
- **Output**: 512-dim latent vector
- **Compression**: 4×
- **Parameters**: 2,048 × 512 + 512 = **1,048,576**
- **Status**: ✅ **CORRECT** (standard bottleneck)

#### Fully-Connected Decode
```python
self.fc_decode = nn.Linear(512, 2,048)
```
- **Input**: 512-dim latent
- **Output**: 2,048 (to reshape to 128×4×4)
- **Parameters**: 512 × 2,048 + 2,048 = **1,050,624**
- **Status**: ✅ **CORRECT** (expansion)

#### Reshape + Repetition (BUG LOCATION ❌)
```python
decoded_features = decoded_flat.view(-1, 128, 4, 4)
x_recon = e2nn.GeometricTensor(
    decoded_features.repeat(1, 4, 1, 1),  # ❌ BUG!
    self.feat_type_512
)
```
- **Shape Change**: (Batch, 128, 4, 4) → (Batch, 512, 4, 4)
- **Problem**: `.repeat(1, 4, 1, 1)` naively copies channels
- **Should Be**: Expand 128 channels into 128 fields × 4 rotations with proper rotation structure
- **Impact**: Decoder receives non-equivariant input → breaks reconstruction

---

### Decoder (BROKEN BY BUG ❌)

#### Block 1: 4×4 → 8×8
```python
nn.functional.interpolate(x_recon.tensor, scale_factor=2, mode='bilinear')
self.upconv1(e2nn.GeometricTensor(x_recon, self.feat_type_512))
# upconv1 = R2Conv(512→512, k=3, p=1) + InnerBatchNorm + ReLU
```
- **Input**: 4×4×512 (BUGGY: 4 identical copies per field)
- **Output**: 8×8×512
- **Parameters**: 128×128×4×4×3×3 = **2,359,296**
- **Status**: ❌ **INCORRECT INPUT** (receives non-rotated features)

#### Block 2: 8×8 → 16×16
```python
nn.functional.interpolate(scale_factor=2, mode='bilinear')
self.upconv2(...)  # R2Conv(512→256, k=3, p=1) + BN + ReLU
```
- **Input**: 8×8×512
- **Output**: 16×16×256
- **Parameters**: 128×64×4×4×3×3 = **1,179,648**
- **Status**: ❌ **PROPAGATES BUG** (equivariant layers can't fix non-equivariant input)

#### Block 3: 16×16 → 32×32
```python
nn.functional.interpolate(scale_factor=2, mode='bilinear')
self.upconv3(...)  # R2Conv(256→128, k=3, p=1) + BN + ReLU
```
- **Output**: 32×32×128
- **Parameters**: 64×32×4×4×3×3 = **294,912**
- **Status**: ❌

#### Block 4: 32×32 → 64×64
```python
nn.functional.interpolate(scale_factor=2, mode='bilinear')
self.upconv4(...)  # R2Conv(128→64, k=3, p=1) + BN + ReLU
```
- **Output**: 64×64×64
- **Parameters**: 32×16×4×4×3×3 = **73,728**
- **Status**: ❌

#### Block 5: 64×64 → 128×128
```python
nn.functional.interpolate(scale_factor=2, mode='bilinear')
self.final_conv(...)  # R2Conv(64→trivial, k=3, p=1)
self.sigmoid(...)
```
- **Output**: 128×128×1
- **Parameters**: 16×1×4×3×3 + 1 = **577**
- **Status**: ❌ **FINAL OUTPUT DEGRADED**

---

## Parameter Breakdown

| Component | Parameters | % of Total |
|-----------|------------|------------|
| **Encoder R2Conv** | 1.55M | 14.1% |
| **InnerBatchNorm** | ~5K | 0.05% |
| **GroupPooling** | 0 | 0% |
| **FC Encode** | 1.05M | 9.5% |
| **FC Decode** | 1.05M | 9.5% |
| **Decoder R2Conv** | 3.91M | 35.5% |
| **Decoder BatchNorm** | ~5K | 0.05% |
| **TOTAL** | **~11.0M** | 100% |

**Match**: Parameter count matches Large CNN-AE (~11M) and ECNN Optimized (~11M).

---

## Training Configuration

```python
Loss:      CombinedLoss(alpha=0.84) = 0.84×MSE + 0.16×(1-SSIM)
Optimizer: Adam(lr=1e-3, weight_decay=1e-5)
Scheduler: ReduceLROnPlateau(factor=0.5, patience=5)
Batch:     64
Epochs:    50 (early stopping patience=7)
Mixed:     FP16 + GradScaler
```

### Training Dynamics
- **Convergence**: Epoch 48 (slower than CNN-AE Large at epoch 45)
- **Training Time**: ~6 hours (slower due to e2cnn overhead)
- **Memory Usage**: 5.8 GB VRAM (higher due to GeometricTensor overhead)
- **Stability**: Training loss plateaus early → bug prevents learning

---

## Performance Analysis (POOR ⚠️)

### Quantitative Results
```
AUROC:       0.7035 ❌ (-7.68% vs Large CNN-AE)
AUPRC:       0.7842 (-6.19% vs Large CNN-AE)
Accuracy:    69.34% (-7.90%)
Precision:   72.18% (-6.11%)
Recall:      84.26% (-5.21%)
Specificity: 47.86% (-10.66%) 🔴 WORST
F1-Score:    0.7774 (-5.76%)

Confusion Matrix:
  TP: 6,570 | TN: 1,748
  FP: 1,904 | FN: 1,223

False Positives: 1,904 🔴 WORST (+389 vs Large CNN-AE)
```

### vs Large CNN-AE (Control)
```
Model              | AUROC  | Spec   | FP    | Status
-------------------|--------|--------|-------|--------
Large CNN-AE (11M) | 0.7803 | 58.52% | 1,515 | ✅ Control
ECNN Buggy (11M)   | 0.7035 | 47.86% | 1,904 | ❌ BUG
-------------------|--------|--------|-------|--------
Difference         | -7.68% | -10.66%| +389  | WORSE!
```

**Conclusion**: Bug causes ECNN to perform **worse** than standard CNN despite equivariance!

---

## Error Distribution

```
Normal (IXI) Mean Error:   0.0035 ± 0.0021 (vs 0.0027 Large CNN)
Anomaly (BraTS) Mean Error: 0.0052 ± 0.0029 (vs 0.0046 Large CNN)

Separation: 1.49× (worse discrimination than Large CNN's 1.70×)
```

**Insight**: Bug increases reconstruction error for BOTH normal and anomalous images.

---

## Bug Impact Analysis

### 1. **Reconstruction Quality Degradation**
```
MSE (Normal):
- Large CNN-AE: 0.0027
- ECNN Buggy:   0.0035 (+29.6% worse)

MSE (Anomaly):
- Large CNN-AE: 0.0046
- ECNN Buggy:   0.0052 (+13.0% worse)
```
**Why**: Decoder can't leverage rotation information → reconstructions blurry.

### 2. **False Positive Explosion**
```
False Positives:
- Large CNN-AE: 1,515 (41.48% of normals)
- ECNN Buggy:   1,904 (52.14% of normals) +389 FP!
```
**Why**: Poor reconstruction of normals → high anomaly scores for healthy brains.

### 3. **Specificity Collapse**
```
Specificity:
- Large CNN-AE: 58.52%
- ECNN Buggy:   47.86% (-10.66%)
```
**Why**: Can't distinguish normal variations from anomalies.

### 4. **Training Instability**
```
Training Loss Plateau:
- Large CNN-AE: Smooth convergence, early stop epoch 45
- ECNN Buggy:   Loss plateau at epoch 30, continues to 48 (overfitting)
```
**Why**: Bug creates information bottleneck → network can't improve.

---

## The Fix (ECNN Optimized)

### Problem Identified
```python
# BUGGY (07_ecnn_autoencoder.ipynb, line 341):
x_recon = e2nn.GeometricTensor(
    decoded_features.repeat(1, 4, 1, 1),  # ❌ Naive repetition
    self.feat_type_512
)
```

### Solution Implemented
```python
# FIXED (08_ecnn_optimized.ipynb):
# Option 1: Keep bottleneck wider (no GroupPooling)
# Skip GroupPooling, keep full 512 channels (128 fields × 4 rotations)
# Decoder receives proper equivariant tensor

# Option 2: Use EquivariantExpansion
class EquivariantExpansion(nn.Module):
    def expand(self, pooled_features):
        # Expand 128 channels back to 512 by:
        # 1. Replicate each feature 4 times
        # 2. Apply learnable rotation matrices
        # 3. Create proper equivariant structure
        return expanded_equivariant_tensor
```

**ECNN Optimized** uses **Option 1** (no GroupPooling) → full equivariant bottleneck.

### Result (ECNN Optimized)
```
AUROC:       0.8109 (+10.74% vs Buggy)
Specificity: 58.54% (+10.68%)
FP:          1,514 (-390, -20.5%)
```

**Validation**: Fix restores equivariance → ECNN beats CNN-AE by +3.06% AUROC!

---

## Lessons Learned

### 1. **Equivariance Must Be End-to-End**
- Encoder equivariant + Decoder non-equivariant = **BROKEN**
- All layers must respect group structure
- **Naive tensor operations break equivariance**

### 2. **GroupPooling Creates Irreversible Loss**
- GroupPooling: 512 ch → 128 ch (max over rotations)
- **Cannot reverse**: Naive repetition doesn't recreate rotation info
- **Solutions**:
  - Avoid GroupPooling in bottleneck (keep wide)
  - Use learnable equivariant expansion

### 3. **Debugging Equivariant Networks is Hard**
- Bug not obvious: Code runs without errors
- Loss converges (just poorly)
- **Only performance comparison reveals bug**
- **Recommendation**: Test equivariance property explicitly

### 4. **Performance Degradation Can Be Severe**
- -7.68% AUROC from single bug
- -10.66% specificity
- +389 false positives
- **Takeaway**: Correct equivariance implementation is critical

### 5. **Test Equivariance Explicitly**
```python
def test_equivariance(model, image, rotation):
    # Rotate then encode
    rotated = rotate(image, rotation)
    features_1 = model.encode(rotated)
    
    # Encode then rotate features
    features_2 = rotate_features(model.encode(image), rotation)
    
    # Should be equal!
    assert torch.allclose(features_1, features_2)
```
**If test fails**: Equivariance broken somewhere in network.

---

## Comparison: Buggy vs Optimized ECNN

| Metric | Buggy (11M) | Optimized (11M) | Improvement |
|--------|-------------|-----------------|-------------|
| **AUROC** | 0.7035 | 0.8109 | **+10.74%** |
| **Specificity** | 47.86% | 58.54% | **+10.68%** |
| **False Positives** | 1,904 | 1,514 | **-390 (-20.5%)** |
| **Training Time** | 6h | 5h | -16.7% (faster!) |
| **MSE (Normal)** | 0.0035 | 0.0024 | -31.4% (better) |
| **Equivariance** | ❌ Broken | ✅ Correct | FIXED |

**Same Parameters, Correct Structure → Massive Improvement**

---

## Role in Project

**Critical Documentation**: Shows what NOT to do when implementing equivariant networks.

**Debugging Value**:
- Identified bug through performance comparison with Large CNN-AE
- Buggy ECNN worse than standard CNN → proved equivariance broken
- Fixed architecture → ECNN Optimized beats CNN-AE (+3.06%)

**Thesis Contribution**:
- **Negative Result**: Documents common pitfall in ECNN implementation
- **Architectural Understanding**: Proves correct equivariance implementation matters
- **Comparative Analysis**: Isolates decoder bug impact (-7.68% AUROC)

**Educational Value**: Anyone implementing ECNNs can learn from this mistake.

---

## Conclusion

**Verdict**: ❌ **ECNN Buggy is a cautionary tale** - equivariance must be implemented correctly throughout the network.

**Bug**: Naive channel repetition in decoder breaks equivariance property.

**Impact**: -7.68% AUROC, -10.66% specificity, +389 false positives vs Large CNN-AE.

**Fix**: ECNN Optimized uses wider bottleneck (no GroupPooling) → proper equivariant decoder → +10.74% AUROC improvement.

**Key Takeaway**: **Architectural correctness > Parameter count**. Same 11M params, different structure:
- Buggy implementation: 0.7035 AUROC (WORST)
- Correct implementation: 0.8109 AUROC (BEST)

**Thesis Value**: Demonstrates importance of architectural rigor in equivariant deep learning.

**Final Ranking** (with bug documented):
1. ✅ ECNN Optimized: 0.8109 AUROC (11M, equivariant, FIXED)
2. ✅ Large CNN-AE: 0.7803 AUROC (11M, standard control)
3. ✅ Small CNN-AE: 0.7617 AUROC (8M, baseline)
4. ❌ **ECNN Buggy: 0.7035 AUROC (11M, equivariant, BROKEN)** ← This model
5. ❌ Baseline AE: Failed (OOM)

**Never repeat this bug!** ✅ Always verify equivariance end-to-end.
