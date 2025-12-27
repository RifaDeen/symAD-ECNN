# Quick Action Plan - BraTS Preprocessing & Drive Upload

## 🎯 Current Status

✅ **IXI Data**: Fully preprocessed and split in Google Drive
- Train: 15,094 slices
- Val: 1,677 slices
- Location: `/content/drive/MyDrive/symAD-ECNN/data/processed_ixi/train/` and `/val/`

⚠️ **BraTS Data**: Raw data on local PC, needs preprocessing
- Location: `C:\Users\rifad\symAD-ECNN\data\brats2021\`
- ~100 patient folders with T1 MRI scans
- Needs: Extract → Normalize → Filter → Resize → Upload

---

## 📋 Next Steps

### Step 1: Run BraTS Preprocessing Locally ⏰ 45-75 minutes

1. **Open notebook** in VS Code:
   ```bash
   cd C:\Users\rifad\symAD-ECNN\notebooks
   code brats2021_t1_preprocessing.ipynb
   ```

2. **Verify Python environment** (your .venv):
   ```powershell
   .\.venv\Scripts\Activate.ps1
   pip list | Select-String "nibabel|scikit"
   # If missing: pip install nibabel scikit-image scipy
   ```

3. **Run cells in order**:
   - Import libraries
   - Set paths (should be `C:/Users/rifad/symAD-ECNN/...`)
   - Extract T1 slices (⏰ 20-30 min)
   - Filter slices (⏰ 5-10 min)
   - Resize to 128×128 (⏰ 10-20 min)
   - Create ZIP for upload (⏰ 5-10 min)

4. **Verify output**:
   ```powershell
   cd C:\Users\rifad\symAD-ECNN\data\brats2021_processed
   Get-ChildItem *.npy | Measure-Object
   # Expected: 1000-2000 .npy files
   ```

### Step 2: Upload to Google Drive ⏰ 10-30 minutes

**Option A: Upload ZIP** (Recommended - Fastest)
1. Go to https://drive.google.com
2. Navigate to `MyDrive/symAD-ECNN/data/`
3. Upload `brats2021_processed.zip` (~200-300 MB)
4. Extract in Colab (see code below)

**Option B: Google Drive Desktop** (Easiest if installed)
1. Open Drive desktop app
2. Copy folder:
   ```
   C:\Users\rifad\symAD-ECNN\data\brats2021_processed
   → G:\My Drive\symAD-ECNN\data\
   ```
3. Wait for sync

**Extract ZIP in Colab** (if using Option A):
```python
from google.colab import drive
drive.mount('/content/drive')

import zipfile
import os

zip_path = '/content/drive/MyDrive/symAD-ECNN/data/brats2021_processed.zip'
extract_to = '/content/drive/MyDrive/symAD-ECNN/data/brats2021_processed'

with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall(extract_to)

print(f"✓ Extracted {len(os.listdir(extract_to))} files")
```

### Step 3: Verify in Colab

```python
from google.colab import drive
import os
import numpy as np
drive.mount('/content/drive')

# Check paths
base = '/content/drive/MyDrive/symAD-ECNN/data'
ixi_train = f'{base}/processed_ixi/train'
ixi_val = f'{base}/processed_ixi/val'
brats_test = f'{base}/brats2021_processed'

# Count files
print(f"IXI Train: {len(os.listdir(ixi_train))} files")
print(f"IXI Val: {len(os.listdir(ixi_val))} files")
print(f"BraTS Test: {len(os.listdir(brats_test))} files")

# Check sample
sample = np.load(os.path.join(brats_test, os.listdir(brats_test)[0]))
print(f"\nSample shape: {sample.shape}")
print(f"Sample range: [{sample.min():.3f}, {sample.max():.3f}]")

# Expected output:
# IXI Train: 15094 files
# IXI Val: 1677 files
# BraTS Test: 1000-2000 files
# Sample shape: (128, 128)
# Sample range: [0.000, 1.000]
```

### Step 4: Update Training Notebooks

Add at start of each training notebook (`01`, `02`, `03`):

```python
from google.colab import drive
drive.mount('/content/drive')

import os

# Data paths
BASE = '/content/drive/MyDrive/symAD-ECNN'
IXI_TRAIN_PATH = f'{BASE}/data/processed_ixi/train'
IXI_VAL_PATH = f'{BASE}/data/processed_ixi/val'
BRATS_TEST_PATH = f'{BASE}/data/brats2021_processed'  # NEW!

# Model save path
MODEL_SAVE_PATH = f'{BASE}/models/saved_models'
os.makedirs(MODEL_SAVE_PATH, exist_ok=True)

# Results path
RESULTS_PATH = f'{BASE}/results'
os.makedirs(RESULTS_PATH, exist_ok=True)

print("✓ Paths configured")
print(f"  Train: {len(os.listdir(IXI_TRAIN_PATH))} samples")
print(f"  Val: {len(os.listdir(IXI_VAL_PATH))} samples")
print(f"  Test (BraTS): {len(os.listdir(BRATS_TEST_PATH))} samples")
```

---

## 📁 Final Drive Structure

After completing all steps:

```
/content/drive/MyDrive/symAD-ECNN/
├── data/
│   ├── processed_ixi/
│   │   ├── train/                    ✅ 15,094 files
│   │   └── val/                      ✅ 1,677 files
│   │
│   └── brats2021_processed/          ⚠️ 1,000-2,000 files (after upload)
│       ├── brats_slice_000000.npy
│       ├── brats_slice_000001.npy
│       └── ...
│
├── models/
│   └── saved_models/                 (created during training)
│
└── results/                          (created during training)
    ├── figures/
    └── metrics/
```

---

## ⏱️ Time Estimate

| Task | Time | Can Do While... |
|------|------|-----------------|
| BraTS preprocessing (local) | 45-75 min | Other work |
| Upload ZIP to Drive | 10-20 min | Automatic |
| Extract in Colab | 2-5 min | Watching |
| Verify setup | 2 min | Quick check |
| **Total** | **~1-2 hours** | Mostly hands-off |

---

## ✅ Success Checklist

After completing all steps, you should have:

- [  ] BraTS preprocessing completed locally
- [  ] ~1,000-2,000 .npy files generated
- [  ] All files 128×128 shape, [0,1] range
- [  ] ZIP file created (if using Option A)
- [  ] Files uploaded to Google Drive
- [  ] Drive structure verified in Colab
- [  ] Training notebooks updated with BraTS path
- [  ] Sample visualization works

---

## 🚀 Ready to Train!

Once BraTS data is uploaded and verified:

1. ✅ IXI split complete (train 90%, val 10%)
2. ✅ BraTS preprocessed (test data ready)
3. ⏭️ **Start training** models in Colab:
   - `01_baseline_autoencoder.ipynb`
   - `02_cnn_autoencoder.ipynb`
   - `03_ecnn_autoencoder.ipynb`

4. ⏭️ **Evaluate on BraTS** (anomaly detection!)
   - High reconstruction error on tumors = detected
   - Calculate AUROC, AUPRC metrics
   - Generate anomaly heatmaps

---

## 📚 Documentation References

- **BraTS Processing**: `BRATS_PREPROCESSING_GUIDE.md`
- **IXI Dataset Info**: `IXI_DATASET_INFO.md`
- **Drive Setup**: `GOOGLE_DRIVE_CURRENT_STATUS.md`
- **Training Guide**: `TRAINING_PIPELINE.md`

---

## 🆘 Troubleshooting

### "Notebook cells fail - module not found"
→ Install dependencies: `pip install nibabel scikit-image scipy`

### "Can't find T1 files"
→ Check BraTS folder structure, should have `*_t1.nii.gz` files

### "Upload too slow"
→ Use Google Drive Desktop app or upload during off-peak hours

### "ZIP extract fails in Colab"
→ Check ZIP file size <500MB, try re-uploading

---

## 💡 Quick Commands

```powershell
# Activate environment
.\.venv\Scripts\Activate.ps1

# Check BraTS data
Get-ChildItem C:\Users\rifad\symAD-ECNN\data\brats2021 -Directory | Measure-Object

# Check processed output
Get-ChildItem C:\Users\rifad\symAD-ECNN\data\brats2021_processed\*.npy | Measure-Object

# Commit changes
git add .
git commit -m "Add BraTS preprocessing and IXI dataset documentation"
git push
```

---

**Current Priority**: Run BraTS preprocessing locally → Upload to Drive → Ready for training! 🎯
