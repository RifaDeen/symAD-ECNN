# Preprocessing Fix: Bicubic Interpolation for Sharp Images

## 🔍 Problem Discovered

**Date**: January 4, 2026  
**Issue**: Blurriness in preprocessed images  
**Root Cause**: Default linear interpolation (`order=1`) in `resize()` function

### Symptoms:
- BraTS test cell showed noticeably blurry images compared to original
- Loss of tumor boundary detail
- Smoothed texture that could impact anomaly detection

## ✅ Solution Applied

**Fix**: Use bicubic interpolation (`order=3`) with anti-aliasing

### Updated Resize Parameters:

```python
resized = resize(
    arr,
    (128, 128),
    order=3,              # Bicubic (sharper than default linear)
    mode='reflect',       # Reflect padding at edges
    anti_aliasing=True,   # Prevent jagged artifacts  
    preserve_range=True   # Maintain [0,1] normalization
)
```

## 📊 Comparison of Interpolation Methods

| Method | Order | Quality | Speed | Use Case |
|--------|-------|---------|-------|----------|
| **Nearest** | `order=0` | Blocky | Fastest | Preview only |
| **Linear** | `order=1` | Blurry ❌ | Fast | NOT for medical imaging |
| **Bicubic** | `order=3` | Sharp ✅ | Medium | **RECOMMENDED** |
| **Bicubic No AA** | `order=3, anti_aliasing=False` | Crispest but jagged | Medium | Check for artifacts first |

## 🎯 Why Bicubic (order=3)?

1. **Preserves details**: Tumor boundaries remain sharp
2. **Medical imaging standard**: Commonly used in radiology
3. **Balanced**: Sharp without excessive artifacts
4. **Proven**: Both IXI and BraTS now use this method

### Technical Explanation:

- **Linear interpolation (order=1)**: Averages between pixels → smooths details
- **Bicubic interpolation (order=3)**: Uses 4×4 pixel neighborhood → preserves edges
- **Anti-aliasing**: Prevents stair-stepping artifacts when downscaling

## 📝 Files Updated

### 1. BraTS Preprocessing Script
**File**: `legacy/brats_preprocessing_complete.py` (moved to legacy/)  
**Section 11**: Resize function updated with `order=3`

```python
# Before (WRONG - causes blurriness):
resized = resize(arr, (128, 128), mode='reflect', anti_aliasing=True)

# After (CORRECT - sharp images):
resized = resize(arr, (128, 128), order=3, mode='reflect', anti_aliasing=True, preserve_range=True)
```

### 2. IXI Preprocessing Notebook
**File**: `notebooks/data_preprocessing/ixi_t1_preprocessing.ipynb`  
**Status**: ✅ Already correct (updated earlier)  
**Lines**: 365-385

```python
slice_resized = resize(
    slice_img,
    IMG_SIZE,
    order=3,           # Bicubic (sharper)
    mode='reflect',
    anti_aliasing=True,
    preserve_range=True
)
```

### 3. BraTS Test Cell
**File**: `notebooks/data_preprocessing/brats2021_t1_preprocessing.ipynb`  
**Cell**: Last test cell (visual comparison)  
**Purpose**: Compare linear vs bicubic vs no-AA

## 🚀 Fresh Preprocessing Required

### Why Reprocess?

- **Existing BraTS data**: Processed with `order=1` (linear) → blurry ❌
- **Existing IXI data**: Already has `order=3` (bicubic) → sharp ✅
- **Mismatch**: Different interpolation methods between datasets
- **Solution**: Reprocess BraTS with `order=3` to match IXI

### Workflow:

1. **Backup old data**:
   ```powershell
   powershell -ExecutionPolicy Bypass -File cleanup_and_prep_reprocessing.ps1
   ```

2. **BraTS local preprocessing** (~60-70 min):
   ```powershell
   python notebooks/brats_preprocessing_complete.py
   ```
   - ✅ RAS orientation correction
   - ✅ Bicubic interpolation (`order=3`)
   - ✅ Creates ~50,000 slices
   - ✅ ZIP file for upload

3. **Upload to Google Drive**:
   - Find: `data/brats2021_processed/brats2021_filtered_RAS_4slices_*.zip`
   - Upload to: `MyDrive/symAD-ECNN/data/`

4. **IXI Colab preprocessing** (~3-5 hours, can run parallel):
   - Open: `notebooks/ixi_t1_preprocessing.ipynb` in Colab
   - Enable GPU (T4 GPU)
   - Run all cells
   - Already has correct `order=3` setting ✅

5. **Retrain CNN-AE**:
   - Both datasets now have:
     - ✅ RAS orientation
     - ✅ Bicubic interpolation
     - ✅ 128×128 size
     - ✅ [0,1] normalization
   - Expected AUROC: 0.75-0.90 (realistic)

## 🔬 Visual Verification

The test cell in BraTS notebook shows 3 methods side-by-side:

```
┌─────────────┬─────────────┐
│  Original   │  Linear     │
│  (High Res) │  (Blurry)   │
├─────────────┼─────────────┤
│  Bicubic    │  No AA      │
│  (Sharp ✅) │  (Crispest) │
└─────────────┴─────────────┘
```

**Expected result**: Bottom-left (Bicubic) should be noticeably sharper than top-right (Linear)

## 📚 Technical References

### Interpolation Theory:
- **Linear (order=1)**: $f(x) = a_0 + a_1x$ (line between points)
- **Bicubic (order=3)**: $f(x) = a_0 + a_1x + a_2x^2 + a_3x^3$ (curve between points)

### Medical Imaging Standards:
- DICOM viewers typically use bicubic or higher
- Research papers commonly use bicubic for consistency
- Linear interpolation avoided in medical analysis

### scikit-image Documentation:
- `order=0`: Nearest neighbor
- `order=1`: Bi-linear (default, but causes blur)
- `order=2`: Bi-quadratic
- `order=3`: Bi-cubic (recommended for quality)
- `order=4`: Bi-quartic
- `order=5`: Bi-quintic

## ⚠️ Important Notes

1. **Don't omit `order` parameter**: Defaults to `order=1` (linear) → blurry!
2. **Keep `anti_aliasing=True`**: Prevents jagged edges
3. **Include `preserve_range=True`**: Maintains [0,1] normalization
4. **Use `mode='reflect'`**: Better edge handling than 'constant'

## 🎯 Expected Outcomes

### Before Fix:
- ❌ AUROC = 1.0 (learning format differences)
- ❌ Blurry BraTS images
- ❌ Sharp IXI images
- ❌ Mismatch between datasets

### After Fix:
- ✅ AUROC = 0.75-0.90 (realistic performance)
- ✅ Sharp BraTS images
- ✅ Sharp IXI images  
- ✅ Consistent preprocessing

## 📝 Checklist

- [x] Identified blurriness issue
- [x] Root cause: Linear interpolation (order=1)
- [x] Solution: Bicubic interpolation (order=3)
- [x] Updated BraTS script
- [x] Verified IXI notebook (already correct)
- [x] Created test cell for visual comparison
- [x] Documented fix
- [ ] Run cleanup script
- [ ] Reprocess BraTS locally
- [ ] Upload to Drive
- [ ] Reprocess IXI in Colab
- [ ] Retrain CNN-AE
- [ ] Verify realistic AUROC

## 🔗 Related Files

- `legacy/brats_preprocessing_complete.py` - Complete BraTS pipeline (moved to legacy/)
- `notebooks/data_preprocessing/ixi_t1_preprocessing.ipynb` - IXI pipeline (Colab)
- `notebooks/data_preprocessing/brats2021_t1_preprocessing.ipynb` - BraTS notebook with test cell
- `cleanup_and_prep_reprocessing.ps1` - Backup old data
- `md_files/ORIENTATION_FIX_SUMMARY.md` - RAS orientation fix
- `md_files/PREPROCESSING_FIXES.md` - This file

---

**Date Created**: January 4, 2026  
**Last Updated**: January 4, 2026  
**Status**: Ready for fresh preprocessing ✅
