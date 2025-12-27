# IXI Dataset - Complete Information

## Dataset Overview

**Source**: [Kaggle - Preprocessed OASIS, Epilepsy, and IXI Dataset](https://www.kaggle.com/datasets/hamedamin/preprocessed-oasis-and-epilepsy-and-ixi)  
**License**: Creative Commons Attribution-ShareAlike 3.0 Unported License  
**Original Source**: IXI Dataset (publicly available T1-weighted MR scans)

## Dataset Description

This dataset consists of **preprocessed T1-weighted MRI images** from the IXI dataset, focused on gray matter (GM) and white matter (WM) segmentation.

###Preprocessing Tools Used

- **CAT12 Toolbox** for SPM (MATLAB)
- **SPM (Statistical Parametric Mapping)** for registration, segmentation, and Jacobian mapping

### Preprocessing Steps Applied

1. **Skull Stripping**: Brain extracted from surrounding skull and non-brain tissues
2. **Registration to MNI Standard**: Each subject's MRI registered to Montreal Neurological Institute (MNI) standard template for spatial normalization
3. **Tissue Segmentation**: Gray matter and white matter regions segmented from MRI scans
4. **Jacobian Mapping**: Maps showing local volume differences between original and normalized images

---

## File Types Available

| File Pattern | Description | Used in Project |
|--------------|-------------|-----------------|
| `mwp1*.nii` | **Gray Matter (GM)** segmented maps | ✅ **YES** - Primary data source |
| `mwp2*.nii` | **White Matter (WM)** segmented maps | ❌ No |
| `wj*.nii` | **Jacobian maps** (local volume changes after registration) | ❌ No |
| `wm*.nii` | **Registered and skull-stripped MRI** images | ❌ No |

---

## Your Processing Pipeline

You downloaded and processed this dataset using `preprocessing_ixi.ipynb`:

### Step 1: Download from Kaggle
```python
!kaggle datasets download -d hamedamin/preprocessed-oasis-and-epilepsy-and-ixi
```
- Downloaded preprocessed dataset to `/content/drive/MyDrive/symAD-ECNN/data/raw_ixi`

### Step 2: Extract GM Slices
```python
# Selected only Gray Matter maps (mwp1*.nii files)
# Extracted 2D slices from 3D volumes
# Normalized each slice to [0, 1] range
# Skipped empty slices (non-zero ratio < 0.12)
```
- **Output**: ~16,000+ 2D slices saved as `.npy` files
- **Location**: `/content/drive/MyDrive/symAD-ECNN/data/processed_ixi/`

### Step 3: Filter Empty Slices
```python
# Removed slices with mean pixel value < 0.1
# Keeps only brain-containing slices
```
- **Output**: ~15,000+ filtered slices
- **Location**: `/content/drive/MyDrive/symAD-ECNN/data/processed_ixi/filtered_ixi/`

### Step 4: Resize to Standard Size
```python
# Resized all slices to 128×128 pixels
# Used skimage.transform.resize with anti-aliasing
```
- **Output**: ~16,771 slices at 128×128
- **Location**: `/content/drive/MyDrive/symAD-ECNN/data/processed_ixi/resized_ixi/`

### Step 5: Train/Val Split
```python
# 90% training (~15,094 files)
# 10% validation (~1,677 files)
# Random split with seed=42 for reproducibility
```
- **Output**: 
  - Train: `/content/drive/MyDrive/symAD-ECNN/data/processed_ixi/train/`
  - Val: `/content/drive/MyDrive/symAD-ECNN/data/processed_ixi/val/`

---

## Final Dataset Statistics

| Metric | Value |
|--------|-------|
| **Original Files** | mwp1*.nii (Gray Matter maps) |
| **Total Subjects** | ~600 healthy subjects |
| **Total 2D Slices** | ~16,771 slices |
| **Training Set** | ~15,094 slices (90%) |
| **Validation Set** | ~1,677 slices (10%) |
| **Image Size** | 128 × 128 pixels |
| **Value Range** | [0, 1] (normalized) |
| **Data Type** | float32 |
| **File Format** | .npy (NumPy arrays) |
| **Total Size** | ~2-3 GB (processed) |

---

## Why Only Gray Matter (GM)?

You selected only **Gray Matter maps** (`mwp1*.nii`) because:

1. **Tissue Contrast**: GM provides better anatomical structure visibility
2. **Anomaly Detection**: Brain tumors often affect GM regions
3. **Standard Practice**: Many brain imaging studies focus on GM analysis
4. **Computational Efficiency**: Single tissue type reduces complexity

---

## Data Quality Checks

Your preprocessing included multiple quality checks:

1. **Integrity Check**: Loaded all NIfTI files to verify no corruption
2. **Slice Visualization**: Visualized sample slices at each processing stage
3. **Value Range**: Verified normalization (all values between 0 and 1)
4. **Empty Slice Filtering**: Removed background/empty slices
5. **Shape Consistency**: All slices resized to uniform 128×128

---

## Google Drive Structure

```
/content/drive/MyDrive/symAD-ECNN/
├── data/
│   ├── raw_ixi/                      # Original downloaded data
│   │   ├── OASIS/                    # (ignored)
│   │   ├── Epilepsy/                 # (ignored)
│   │   └── IXI/                      # Used!
│   │       └── *.nii files (mwp1, mwp2, wj, wm)
│   │
│   └── processed_ixi/
│       ├── slice_*.npy               # Initial extracted slices
│       ├── slice_*.png               # Preview images
│       │
│       ├── filtered_ixi/             # After filtering (mean > 0.1)
│       │   └── slice_*.npy
│       │
│       ├── resized_ixi/              # After resize to 128×128
│       │   └── slice_*.npy           # ~16,771 files
│       │
│       ├── train/                    # 90% train split
│       │   └── slice_*.npy           # ~15,094 files
│       │
│       └── val/                      # 10% validation split
│           └── slice_*.npy           # ~1,677 files
```

---

## Use Cases in Your Project

### 1. Autoencoder Training
- **Data**: IXI training set (15,094 normal brain slices)
- **Purpose**: Learn to reconstruct normal brain anatomy
- **Models**: Baseline AE, CNN-AE, ECNN-AE

### 2. Validation During Training
- **Data**: IXI validation set (1,677 normal brain slices)
- **Purpose**: Monitor overfitting, tune hyperparameters
- **Metrics**: Reconstruction loss, MS-SSIM

### 3. Anomaly Detection Testing
- **Training Data**: IXI (normal brains)
- **Testing Data**: BraTS (tumor brains)
- **Hypothesis**: High reconstruction error on tumorous regions = anomaly detected

---

## Key Preprocessing Decisions

| Decision | Rationale |
|----------|-----------|
| Use GM maps only | Better contrast, standard practice |
| Filter empty slices (mean > 0.1) | Remove background, save space |
| Resize to 128×128 | Balance between detail and computation |
| Normalize to [0,1] | Standard for neural networks |
| 90/10 split | Standard train/val ratio |
| Seed=42 | Reproducible splits |
| Save as .npy | Fast loading, compatible with PyTorch |

---

## Alignment with Proposal

**From Proposal (Table 6 - Dataset Specifications)**:

| Proposal | Your Implementation | ✅ Status |
|----------|---------------------|-----------|
| IXI Dataset | Used preprocessed IXI from Kaggle | ✅ |
| ~600 subjects | Processed all available subjects | ✅ |
| T1-weighted MRI | Gray Matter maps (derived from T1) | ✅ |
| 90:10 train/val split | Implemented with seed=42 | ✅ |
| Preprocessing required | CAT12 + your pipeline (extract, filter, resize) | ✅ |
| Normal brain scans | Healthy subjects only (no tumors) | ✅ |

---

## Citation

If publishing, cite:

**Kaggle Dataset**:
```
Hamed Amin. (2024). Preprocessed OASIS, Epilepsy, and IXI Dataset. 
Kaggle. https://www.kaggle.com/datasets/hamedamin/preprocessed-oasis-and-epilepsy-and-ixi
```

**Original IXI Dataset**:
```
IXI Dataset. Information eXtraction from Images (IXI). 
Brain Development. https://brain-development.org/ixi-dataset/
```

**Preprocessing Tools**:
```
CAT12 Toolbox: http://dbm.neuro.uni-jena.de/cat/
SPM: https://www.fil.ion.ucl.ac.uk/spm/
```

---

## References

1. **IXI Dataset**: https://brain-development.org/ixi-dataset/
2. **CAT12 Toolbox**: http://dbm.neuro.uni-jena.de/cat/
3. **SPM Software**: https://www.fil.ion.ucl.ac.uk/spm/
4. **MNI Standard Space**: https://www.bic.mni.mcgill.ca/ServicesAtlases/ICBM152NLin2009
5. **Kaggle Dataset**: https://www.kaggle.com/datasets/hamedamin/preprocessed-oasis-and-epilepsy-and-ixi

---

## Summary

✅ **IXI Dataset**: Preprocessed T1 GM maps from healthy subjects  
✅ **Processing**: Extract → Normalize → Filter → Resize → Split  
✅ **Result**: 15,094 train + 1,677 val slices at 128×128  
✅ **Purpose**: Train autoencoders on normal brain patterns  
✅ **Format**: .npy files in Google Drive, ready for training  

Your preprocessing pipeline successfully transformed raw 3D MRI volumes into a clean, normalized 2D dataset optimized for deep learning! 🎉
