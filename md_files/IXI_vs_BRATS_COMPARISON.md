# IXI vs BraTS Processing Comparison

## Side-by-Side Pipeline Comparison

| Step | IXI (Colab) | BraTS (Local) | Purpose |
|------|-------------|---------------|---------|
| **1. Data Source** | Kaggle download | Already downloaded locally | Get raw data |
| **2. File Type** | `*mwp1*.nii` (GM maps) | `*_t1.nii.gz` (T1 MRI) | Input format |
| **3. Load Volume** | `nib.load()` | `nib.load()` | Same |
| **4. Extract Slices** | `vol[:,:,s]` | `vol[:,:,s]` | Same |
| **5. Normalize** | `(x - min) / (max - min)` | `(x - min) / (max - min)` | Same |
| **6. Filter Empty** | `np.count_nonzero() > 0.12` | `np.count_nonzero() > 0.12` | Same |
| **7. Filter Low Info** | `mean > 0.1` | `mean > 0.1` | Same |
| **8. Resize** | `resize(arr, (128,128))` | `resize(arr, (128,128))` | Same |
| **9. Save** | `.npy` files | `.npy` files | Same |
| **10. Output** | ~2000-3000 slices | ~1000-2000 slices | Training vs Testing |

---

## Code Comparison

### IXI (from preprocessing_ixi.ipynb)

```python
# Cell: Process 3D NIfTI Volumes into Normalized 2D Slices
def normalize(x):
    x = x.astype(np.float32)
    if x.max() - x.min() < 1e-6:
        return np.zeros_like(x)
    return (x - x.min()) / (x.max() - x.min())

idx = 0
for root, _, files in os.walk(RAW):
    for f in files:
        if "mwp1" in f and f.endswith(".nii"):  # GM maps
            vol = nib.load(full_path).get_fdata()
            Z = vol.shape[2]
            for s in range(Z):
                slice_ = vol[:,:,s]
                
                # skip empty slices
                if np.count_nonzero(slice_) / slice_.size < 0.12:
                    continue
                
                norm = normalize(slice_)
                np.save(f"{OUT}/slice_{idx:06d}.npy", norm)
                idx += 1
```

```python
# Cell: Filter and Persist 2D Slices
for f in tqdm(files, desc="Filtering slices"):
    arr = np.load(os.path.join(input_path, f))
    if arr.mean() > 0.1:   # discard empty slices
        np.save(output_file_path, arr)
        kept += 1
```

```python
# Cell: Resizing slices to standard 128 x 128
resized = resize(arr, (128, 128), mode="reflect", anti_aliasing=True)
np.save(dst_path, resized)
```

---

### BraTS (from brats2021_t1_preprocessing.ipynb)

```python
# Cell 8: Preprocessing Functions
def normalize(x):
    """
    Normalize array to [0, 1] range (same as IXI preprocessing)
    """
    x = x.astype(np.float32)
    if x.max() - x.min() < 1e-6:
        return np.zeros_like(x)
    return (x - x.min()) / (x.max() - x.min())

def is_valid_slice(slice_array, min_nonzero_ratio=0.12, min_mean=0.1):
    """
    Check if a slice contains meaningful information (same criteria as IXI)
    """
    nonzero_ratio = np.count_nonzero(slice_array) / slice_array.size
    if nonzero_ratio < min_nonzero_ratio:
        return False
    
    normalized = normalize(slice_array)
    if normalized.mean() < min_mean:
        return False
    
    return True
```

```python
# Cell 9: STEP 1: Extract and Normalize 2D Slices
for t1_file in tqdm(t1_files, desc="Processing T1 volumes"):
    nii_img = nib.load(t1_file)
    vol = nii_img.get_fdata()
    
    Z = vol.shape[2]
    for s in range(Z):
        slice_2d = vol[:, :, s]
        
        # Skip empty or low-information slices
        if not is_valid_slice(slice_2d, min_nonzero_ratio=0.12, min_mean=0.1):
            skipped_empty_slices += 1
            continue
        
        # Normalize to [0, 1]
        normalized_slice = normalize(slice_2d)
        
        # Save as .npy file
        np.save(f"{t1_raw_folder}/slice_{slice_idx:06d}.npy", normalized_slice)
        slice_idx += 1
```

```python
# Cell 11: STEP 2: Additional Filtering (Mean > 0.1)
for f in tqdm(raw_files, desc="Filtering slices"):
    arr = np.load(os.path.join(t1_raw_folder, f))
    if arr.mean() > 0.1:  # Same threshold as IXI
        np.save(output_file_path, arr)
        kept_count += 1
```

```python
# Cell 14: STEP 3: Resize to 128x128
arr = np.load(src_path)
resized = resize(arr, (128, 128), mode='reflect', anti_aliasing=True)
np.save(dst_path, resized)
```

---

## Key Takeaways

### ✅ What's IDENTICAL:
1. **Normalization function** - Exact same code
2. **Filtering criteria** - Same thresholds (0.12, 0.1)
3. **Resize parameters** - Same dimensions (128×128) and method
4. **Output format** - Both save as `.npy` files
5. **Data range** - Both normalized to [0, 1]

### ⚠️ What's DIFFERENT:
1. **Input files** - IXI uses GM maps, BraTS uses T1 MRI
2. **Processing location** - IXI in Colab, BraTS locally
3. **Volume characteristics** - Different anatomical features
4. **Use case** - IXI for training (normal), BraTS for testing (abnormal)

### 🎯 Why This Matters:
- **Consistency** ensures model can handle both datasets
- **Same preprocessing** means fair comparison
- **Same dimensions** means no model architecture changes needed
- **Same normalization** means same input distribution

---

## Data Validation Checklist

Before using in model, verify both datasets have:

- [ ] Shape: `(128, 128)` ✓
- [ ] Data type: `float32` or `float64` ✓
- [ ] Value range: `[0.0, 1.0]` ✓
- [ ] No NaN or Inf values ✓
- [ ] File format: `.npy` ✓
- [ ] Mean > 0.1 (filtered) ✓
- [ ] Non-zero ratio > 0.12 ✓

---

## Expected Dataset Statistics

### IXI (Training)
```
Total slices: ~2000-3000
Mean pixel value: 0.25-0.35
Std pixel value: 0.15-0.25
Shape: (128, 128)
Data type: float32/float64
Range: [0.0, 1.0]
```

### BraTS (Testing)
```
Total slices: ~1000-2000
Mean pixel value: 0.20-0.40
Std pixel value: 0.15-0.25
Shape: (128, 128)
Data type: float32/float64
Range: [0.0, 1.0]
```

---

## Visual Comparison

### IXI Slices (Normal Brain)
- Smooth grey matter distribution
- Symmetric appearance
- No abnormal structures
- Consistent intensity patterns

### BraTS Slices (Tumor Brain)
- Irregular intensity regions (tumors)
- Asymmetric appearance
- Abnormal bright/dark spots
- Distorted tissue patterns

### What Model Will Learn:
1. **Training on IXI**: Learn normal brain appearance
2. **Testing on BraTS**: Detect abnormal patterns (tumors)
3. **Anomaly Detection**: High reconstruction error = tumor region

---

## File Naming Convention

### IXI
```
slice_000000.npy
slice_000001.npy
slice_000002.npy
...
slice_002543.npy
```

### BraTS
```
slice_000000.npy
slice_000001.npy
slice_000002.npy
...
slice_001234.npy
```

**Note**: Sequential numbering, 6-digit zero-padded, same format for both

---

## Summary

The BraTS preprocessing notebook (`brats2021_t1_preprocessing.ipynb`) is essentially a **local adaptation** of your IXI preprocessing workflow. The core algorithms, parameters, and output format are **identical**, ensuring that both datasets are compatible with your anomaly detection model.

The only differences are:
- **Where** it runs (Colab vs Local)
- **What** files it processes (GM maps vs T1 MRI)
- **Why** we use it (Training vs Testing)

Everything else is the same! 🎯
