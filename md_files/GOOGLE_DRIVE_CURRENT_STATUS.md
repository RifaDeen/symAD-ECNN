# Google Drive - Current Status & Setup Guide

## 📊 Current Status (Dec 2025)

### ✅ Already in Your Google Drive

```
Google Drive/
└── MyDrive/
    └── symAD-ECNN/                      # Your project folder
        ├── data/
        │   ├── raw_ixi/                 # Raw IXI dataset from Kaggle
        │   │   └── (mwp1*, mwp2*, wj*, wm* files)
        │   └── processed_ixi/
        │       ├── (initial extracted slices)
        │       ├── filtered_ixi/        # Filtered slices (mean > 0.1)
        │       └── resized_ixi/         # Resized 128x128 (~16,771 files)
        ├── notebooks/
        │   └── preprocessing_ixi.ipynb  # Your preprocessing notebook
        └── [other folders]
```

### 📝 IXI Dataset Details

**Source**: [Kaggle - Preprocessed OASIS, Epilepsy, and IXI Dataset](https://www.kaggle.com/datasets/hamedamin/preprocessed-oasis-and-epilepsy-and-ixi)

**Preprocessing Done**: CAT12 Toolbox (SPM/MATLAB)
- Skull stripping
- Registration to MNI standard space
- Tissue segmentation (GM/WM)
- Jacobian mapping

**Files Available**:
- `mwp1*.nii` - Gray Matter (GM) maps ✅ **You used these!**
- `mwp2*.nii` - White Matter (WM) maps
- `wj*.nii` - Jacobian maps (local volume changes)
- `wm*.nii` - Registered skull-stripped MRI

**Your Processing**:
- Extracted GM maps (mwp1) only
- Converted 3D volumes → 2D slices
- Normalized to [0,1] range
- Filtered empty slices (mean > 0.1)
- Resized to 128x128
- Result: ~16,771 .npy files ready for training

### ⚠️ Need to Do

1. **Split IXI data** into train/val (90/10)
   - Currently all in `resized_ixi/` folder
   - Need to split into `processed_ixi/train/` and `processed_ixi/val/`
   - ✅ Use Step 8 in `preprocessing_ixi.ipynb` (already added!)

2. **Process BraTS 2021 data**
   - Currently on local: `C:\Users\rifad\symAD-ECNN\data\brats2021\`
   - Run `brats2021_t1_preprocessing.ipynb` locally or in Colab
   - Extract T1 slices, normalize, filter, resize
   - Upload processed files to Drive

3. **Upload BraTS processed data** to Drive
   - Size: ~200-300 MB (processed .npy files)
   - Destination: `symAD-ECNN/data/brats2021_processed/`

---

## 🎯 Recommended Google Drive Structure

Organize your Drive like this for easy Colab access:

```
Google Drive/
└── MyDrive/
    └── symAD-ECNN/                      # Rename your folder to this
        │
        ├── data/                        # All datasets
        │   │
        │   ├── processed_ixi/           # ✅ Move ixi_resized here
        │   │   ├── train/               # 90% of data
        │   │   │   ├── slice_0000.npy
        │   │   │   ├── slice_0001.npy
        │   │   │   └── ... (~15,000 files)
        │   │   │
        │   │   └── val/                 # 10% of data
        │   │       ├── slice_0000.npy
        │   │       └── ... (~1,700 files)
        │   │
        │   ├── ixi_raw/                 # ✅ Optional: Keep for reference
        │   ├── ixi_filtered/            # ✅ Optional: Keep for reference
        │   │
        │   └── brats2021/               # ⚠️ Upload from local
        │       ├── BraTS2021_00000/
        │       │   ├── BraTS2021_00000_t1.nii.gz
        │       │   ├── BraTS2021_00000_seg.nii.gz
        │       │   └── ...
        │       ├── BraTS2021_00002/
        │       └── ... (100 folders)
        │
        ├── notebooks/                   # Optional: Store notebooks here
        │   └── brats2021_t1_preprocessing.ipynb  # ✅ Move here
        │
        ├── models/                      # Model checkpoints (created during training)
        │   └── saved_models/
        │
        └── results/                     # Training results (created during training)
            ├── figures/
            └── metrics/
```

---

## 🚀 Step-by-Step Setup

### Step 0: Split IXI Data (REQUIRED FIRST!)

Your IXI data is currently all in one folder. You need to split it into train/val before training.

**Option A: Run Notebook in Colab (Recommended)**

1. Upload `ixi_train_val_split.ipynb` to your Drive or open from GitHub
2. Open in Colab
3. Update `SOURCE_FOLDER` path in Cell 2 to point to your `ixi_resized` folder
4. Run all cells
5. Wait 5-10 minutes for split to complete
6. Verify split was successful (last cell shows summary)

**Option B: Run Python Script**

If you prefer a script:
```python
# See: legacy/ixi_train_val_split.py
# Upload to Colab and run
```

**What this does:**
- Reads all `.npy` files from your `ixi_resized` folder (~16,771 files)
- Randomly splits: 90% train (~15,094 files), 10% val (~1,677 files)  
- Creates: `processed_ixi/train/` and `processed_ixi/val/`
- Uses seed=42 for reproducibility
- Choice to COPY (safer) or MOVE (saves space)

### Step 1: Organize Existing IXI Data

**Option A: Manual (via Drive Web Interface)**
1. Go to https://drive.google.com
2. Find your current project folder
3. Rename it to `symAD-ECNN` (right-click → Rename)
4. Create folder structure:
   - Create `data` folder
   - Create `data/processed_ixi` folder
   - Move `ixi_resized` contents into `data/processed_ixi/`
   - Optionally move `ixi_raw` and `ixi_filtered` into `data/`

**Option B: In Colab (Recommended - Faster)**
```python
# Run this in a Colab notebook to organize your Drive

from google.colab import drive
drive.mount('/content/drive')

import os
import shutil

# Define paths (UPDATE these to match your current folder names!)
base = '/content/drive/MyDrive'
old_folder = f'{base}/[YOUR_CURRENT_FOLDER_NAME]'  # Update this!
new_folder = f'{base}/symAD-ECNN'

# Option 1: Rename folder
if os.path.exists(old_folder):
    os.rename(old_folder, new_folder)
    print(f"✓ Renamed folder to symAD-ECNN")

# Create directory structure
os.makedirs(f'{new_folder}/data/processed_ixi/train', exist_ok=True)
os.makedirs(f'{new_folder}/data/processed_ixi/val', exist_ok=True)
os.makedirs(f'{new_folder}/models/saved_models', exist_ok=True)
os.makedirs(f'{new_folder}/results/figures', exist_ok=True)
print("✓ Created folder structure")

# Move ixi_resized to processed_ixi (if needed)
# Note: Adjust paths based on your current structure
```

### Step 2: Upload BraTS 2021 Data

**Option A: Upload via Google Drive Desktop App (Recommended)**

1. **Install Google Drive for Desktop** (if not installed)
   - Download: https://www.google.com/drive/download/
   - Sign in with your Google account
   - Syncs Drive as a local folder

2. **Copy BraTS data**
   ```powershell
   # In PowerShell
   # Find your Drive sync folder (usually):
   cd "G:\My Drive\symAD-ECNN\data"  # Or wherever Drive syncs
   
   # Copy BraTS data
   xcopy "C:\Users\rifad\symAD-ECNN\data\brats2021" ".\brats2021" /E /I /Y
   
   # This will auto-upload to Drive
   ```

**Option B: Upload via Web Interface (Slower)**

1. Go to https://drive.google.com
2. Navigate to `symAD-ECNN/data/`
3. Click "New" → "Folder upload"
4. Select `C:\Users\rifad\symAD-ECNN\data\brats2021`
5. Wait for upload (~1-2 GB, may take 30-60 minutes)

**Option C: Upload from Colab (If data is small)**

```python
# Only use this for small files
from google.colab import files
uploaded = files.upload()  # Not practical for 1-2 GB!
```

### Step 3: Verify Structure in Colab

```python
# Run in Colab to verify everything is set up correctly

from google.colab import drive
drive.mount('/content/drive')

import os

# Check structure
base = '/content/drive/MyDrive/symAD-ECNN'

print("Checking folder structure...")
print()

# Check IXI data
ixi_train = f'{base}/data/processed_ixi/train'
ixi_val = f'{base}/data/processed_ixi/val'

if os.path.exists(ixi_train):
    train_count = len([f for f in os.listdir(ixi_train) if f.endswith('.npy')])
    print(f"✓ IXI Training data: {train_count} files")
else:
    print("✗ IXI Training data not found!")

if os.path.exists(ixi_val):
    val_count = len([f for f in os.listdir(ixi_val) if f.endswith('.npy')])
    print(f"✓ IXI Validation data: {val_count} files")
else:
    print("✗ IXI Validation data not found!")

# Check BraTS data
brats_path = f'{base}/data/brats2021'
if os.path.exists(brats_path):
    brats_count = len([d for d in os.listdir(brats_path) if d.startswith('BraTS')])
    print(f"✓ BraTS data: {brats_count} patient folders")
else:
    print("✗ BraTS data not found! Please upload from local.")

print()
print("Expected counts:")
print("  - IXI Training: ~15,000 files")
print("  - IXI Validation: ~1,700 files")
print("  - BraTS: ~100 patient folders")
```

---

## 📝 Using Your Data in Training Notebooks

Once organized, access your data in Colab like this:

### At Start of Each Notebook

```python
# Cell 1: Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

# Cell 2: Set data paths
import os

# Base project path
PROJECT_ROOT = '/content/drive/MyDrive/symAD-ECNN'

# Data paths
IXI_TRAIN_PATH = f'{PROJECT_ROOT}/data/processed_ixi/train'
IXI_VAL_PATH = f'{PROJECT_ROOT}/data/processed_ixi/val'
BRATS_PATH = f'{PROJECT_ROOT}/data/brats2021'

# Model save path
MODEL_SAVE_PATH = f'{PROJECT_ROOT}/models/saved_models'
os.makedirs(MODEL_SAVE_PATH, exist_ok=True)

# Results path
RESULTS_PATH = f'{PROJECT_ROOT}/results'
os.makedirs(f'{RESULTS_PATH}/figures', exist_ok=True)

print("✓ All paths configured")
print(f"  IXI Train: {IXI_TRAIN_PATH}")
print(f"  IXI Val: {IXI_VAL_PATH}")
print(f"  BraTS: {BRATS_PATH}")
```

### Loading IXI Data

```python
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

class IXIDataset(Dataset):
    def __init__(self, data_path):
        self.files = [f for f in os.listdir(data_path) if f.endswith('.npy')]
        self.data_path = data_path
    
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        file_path = os.path.join(self.data_path, self.files[idx])
        image = np.load(file_path)
        image = torch.FloatTensor(image).unsqueeze(0)  # Add channel dimension
        return image

# Create datasets
train_dataset = IXIDataset(IXI_TRAIN_PATH)
val_dataset = IXIDataset(IXI_VAL_PATH)

# Create dataloaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

print(f"Train samples: {len(train_dataset)}")
print(f"Val samples: {len(val_dataset)}")
```

---

## 💾 Storage Usage

Your estimated Google Drive usage:

| Dataset | Size | Status |
|---------|------|--------|
| IXI Raw | ~5 GB | ✅ In Drive (optional, can delete) |
| IXI Filtered | ~3 GB | ✅ In Drive (optional, can delete) |
| IXI Resized/Preprocessed | ~2 GB | ✅ In Drive (NEEDED) |
| BraTS 2021 | ~1-2 GB | ⚠️ Need to upload |
| Models (after training) | ~200 MB | Will be created |
| Results | ~50 MB | Will be created |
| **Total** | **~3-4 GB** | (after cleanup) |

**Free Google Drive**: 15 GB - You have plenty of space! ✅

### Optional: Clean Up to Save Space

If you need more space, you can delete the raw/filtered IXI data:
- Keep: `processed_ixi/` (needed for training)
- Optional: Delete `ixi_raw/` and `ixi_filtered/` (saves ~8 GB)

---

## 🔍 Troubleshooting

### Issue: "Can't find data in Colab"

**Check mount path:**
```python
# Run this in Colab
import os
print(os.listdir('/content/drive/MyDrive'))
# Should show 'symAD-ECNN' folder
```

**Solution**: Make sure Drive is mounted and folder names match exactly (case-sensitive!)

### Issue: "Permission denied"

**Solution**: 
```python
# Re-mount Drive with write permission
from google.colab import drive
drive.mount('/content/drive', force_remount=True)
```

### Issue: "Upload is too slow"

**Solutions**:
1. Use Google Drive Desktop app instead of web upload
2. Upload during off-peak hours
3. Compress data first: `tar -czf brats2021.tar.gz brats2021/`
4. Upload compressed file, extract in Colab

### Issue: "Out of storage"

**Solutions**:
1. Delete `ixi_raw` and `ixi_filtered` (keep only `processed_ixi`)
2. Upgrade to Google One (100 GB for $1.99/month)
3. Use Google Colab's local storage temporarily

---

## 🎯 Quick Start Checklist

- [ ] Rename Drive folder to `symAD-ECNN`
- [ ] Create `data/processed_ixi/` structure
- [ ] Move/organize IXI preprocessed data
- [ ] Upload BraTS 2021 data (~1-2 GB)
- [ ] Create `models/saved_models/` folder
- [ ] Create `results/figures/` folder
- [ ] Test data access in Colab (run verification script above)
- [ ] Update notebook paths to use Drive paths
- [ ] (Optional) Delete raw IXI data to save space

---

## 📚 Related Documentation

- **GITHUB_COLAB_SETUP.md** - How to open notebooks from GitHub in Colab
- **VSCODE_COLAB_CONNECTION.md** - VS Code and Colab integration
- **TRAINING_PIPELINE.md** - Complete training workflow

---

## ✅ Next Steps

After setting up Drive:

1. **Update your notebooks** with Drive mount code
2. **Push updated notebooks** to GitHub
3. **Open notebooks in Colab** from GitHub
4. **Start training** with free GPU!

See **TRAINING_PIPELINE.md** for detailed training instructions.
