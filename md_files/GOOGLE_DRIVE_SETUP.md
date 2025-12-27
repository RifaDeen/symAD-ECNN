# Google Drive Data Management Guide

## Your Current Setup

You already have a Google Drive folder with:
- ✅ IXI preprocessed data (from your preprocessing work)
- ✅ BraTS2021 data
- ✅ Other project files

This guide shows you how to organize and access this data in Colab.

---

## Recommended Google Drive Structure

```
Google Drive/
└── MyDrive/
    └── symAD-ECNN/                          # Main project folder
        ├── data/                            # All datasets here
        │   ├── processed_ixi/               # Your preprocessed IXI data
        │   │   ├── train/                   # Training slices (90%)
        │   │   │   ├── slice_0000.npy
        │   │   │   ├── slice_0001.npy
        │   │   │   └── ...
        │   │   └── val/                     # Validation slices (10%)
        │   │       ├── slice_0000.npy
        │   │       └── ...
        │   │
        │   └── brats2021/                   # BraTS test data
        │       ├── BraTS2021_00000/
        │       ├── BraTS2021_00002/
        │       └── ...
        │
        ├── models/                          # Saved model checkpoints
        │   └── saved_models/
        │       ├── baseline_ae_final.pth
        │       ├── cnn_ae_final.pth
        │       └── ecnn_ae_final.pth
        │
        └── results/                         # Training results
            ├── baseline_results.json
            ├── cnn_results.json
            ├── ecnn_results.json
            └── figures/
                ├── baseline_roc.png
                └── ...
```

---

## Setup Steps

### Step 1: Organize Your Existing Drive Folder

If your current Drive structure is different, you have two options:

#### Option A: Keep Your Current Structure (Easiest)
Just note down your current paths, e.g.:
```
/content/drive/MyDrive/IXI_Preprocessed/train/
/content/drive/MyDrive/IXI_Preprocessed/val/
/content/drive/MyDrive/BraTS2021/
```

We'll use these exact paths in the notebooks.

#### Option B: Reorganize (Recommended)
Move/copy your data to match the recommended structure:
1. In Google Drive web interface
2. Create `symAD-ECNN` folder in MyDrive
3. Create `data` subfolder
4. Move your preprocessed IXI data into `data/processed_ixi/`
5. Move BraTS data into `data/brats2021/`

---

## Step 2: Update Notebooks to Use Your Drive Paths

### Add Mount Cell (First Cell in Every Notebook)

Add this as the **first code cell** in each notebook:

```python
# ================================
# Mount Google Drive
# ================================
from google.colab import drive
import os

# Mount Drive
drive.mount('/content/drive')

# Set your data paths (UPDATE THESE to match YOUR folder structure)
DRIVE_ROOT = '/content/drive/MyDrive/symAD-ECNN'
DATA_ROOT = os.path.join(DRIVE_ROOT, 'data')

# IXI Data (preprocessed)
IXI_TRAIN_PATH = os.path.join(DATA_ROOT, 'processed_ixi/train')
IXI_VAL_PATH = os.path.join(DATA_ROOT, 'processed_ixi/val')

# BraTS Data (test)
BRATS_PATH = os.path.join(DATA_ROOT, 'brats2021')

# Model save path
MODEL_SAVE_PATH = os.path.join(DRIVE_ROOT, 'models/saved_models')
os.makedirs(MODEL_SAVE_PATH, exist_ok=True)

# Results save path
RESULTS_PATH = os.path.join(DRIVE_ROOT, 'results')
os.makedirs(RESULTS_PATH, exist_ok=True)

print("✓ Drive mounted successfully!")
print(f"✓ IXI Train: {IXI_TRAIN_PATH}")
print(f"✓ IXI Val: {IXI_VAL_PATH}")
print(f"✓ BraTS: {BRATS_PATH}")
print(f"✓ Models will be saved to: {MODEL_SAVE_PATH}")
print(f"✓ Results will be saved to: {RESULTS_PATH}")

# Verify paths exist
if not os.path.exists(IXI_TRAIN_PATH):
    print(f"⚠️ WARNING: {IXI_TRAIN_PATH} not found!")
    print("Please update DRIVE_ROOT to match your folder structure")
```

### Verification Cell (Second Cell)

```python
# ================================
# Verify Data Availability
# ================================
import glob

# Check IXI training data
train_files = glob.glob(os.path.join(IXI_TRAIN_PATH, '*.npy'))
val_files = glob.glob(os.path.join(IXI_VAL_PATH, '*.npy'))
brats_dirs = glob.glob(os.path.join(BRATS_PATH, 'BraTS2021_*'))

print("Data Summary:")
print(f"  IXI Training slices: {len(train_files)}")
print(f"  IXI Validation slices: {len(val_files)}")
print(f"  BraTS subjects: {len(brats_dirs)}")
print()

if len(train_files) == 0:
    print("❌ ERROR: No training data found!")
    print("Please check your Drive folder structure")
else:
    print("✓ All data found!")
    
    # Show sample files
    print("\nSample training files:")
    for f in train_files[:3]:
        print(f"  {os.path.basename(f)}")
```

---

## Step 3: Update Data Loading Code

### Current Code (Local Paths)
```python
# Old code - won't work in Colab
train_path = './data/processed_ixi/train'
```

### Updated Code (Drive Paths)
```python
# New code - works in Colab
train_path = IXI_TRAIN_PATH  # Already defined in mount cell
```

### Example: Update Dataset Class

```python
class IXIDataset(Dataset):
    def __init__(self, data_path, transform=None):
        """
        Args:
            data_path: Path to folder with .npy files (from Drive)
        """
        self.data_path = data_path
        self.transform = transform
        
        # Load all .npy files
        self.files = sorted(glob.glob(os.path.join(data_path, '*.npy')))
        
        if len(self.files) == 0:
            raise ValueError(f"No .npy files found in {data_path}")
        
        print(f"Loaded {len(self.files)} slices from {data_path}")
    
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        # Load from Drive
        img = np.load(self.files[idx])
        
        if self.transform:
            img = self.transform(img)
        
        return img, img  # Autoencoder: input=target

# Usage
train_dataset = IXIDataset(IXI_TRAIN_PATH)
val_dataset = IXIDataset(IXI_VAL_PATH)
```

---

## Step 4: Save Models and Results to Drive

### Save Model Checkpoints

```python
# During training - save to Drive (not local /content)
def save_checkpoint(model, epoch, loss, filename):
    checkpoint_path = os.path.join(MODEL_SAVE_PATH, filename)
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'loss': loss,
    }, checkpoint_path)
    print(f"✓ Checkpoint saved: {checkpoint_path}")

# Save final model
save_checkpoint(model, epoch=50, loss=final_loss, 
                filename='baseline_ae_final.pth')
```

### Save Results JSON

```python
import json

# Save results to Drive
results = {
    'model': 'Baseline AE',
    'train_loss': train_losses,
    'val_loss': val_losses,
    'auroc': 0.85,
    'auprc': 0.78,
    'training_time': '25 minutes'
}

results_path = os.path.join(RESULTS_PATH, 'baseline_results.json')
with open(results_path, 'w') as f:
    json.dump(results, f, indent=2)

print(f"✓ Results saved: {results_path}")
```

### Save Figures

```python
import matplotlib.pyplot as plt

# Plot and save to Drive
plt.figure(figsize=(10, 6))
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Val Loss')
plt.legend()
plt.title('Training Progress')

# Save to Drive
fig_path = os.path.join(RESULTS_PATH, 'figures', 'baseline_training.png')
os.makedirs(os.path.dirname(fig_path), exist_ok=True)
plt.savefig(fig_path, dpi=300, bbox_inches='tight')
print(f"✓ Figure saved: {fig_path}")
```

---

## Your Current Drive Folder

Based on your preprocessing work, you likely have something like:

```
Google Drive/
└── MyDrive/
    ├── IXI_Preprocessed/          # Your current folder
    │   ├── train/
    │   └── val/
    └── BraTS2021/                 # Your BraTS data
```

### Option 1: Use As-Is (Quick Start)

Update the mount cell to match YOUR paths:

```python
# Use YOUR existing paths
DRIVE_ROOT = '/content/drive/MyDrive'
IXI_TRAIN_PATH = '/content/drive/MyDrive/IXI_Preprocessed/train'
IXI_VAL_PATH = '/content/drive/MyDrive/IXI_Preprocessed/val'
BRATS_PATH = '/content/drive/MyDrive/BraTS2021'

# Create folders for outputs
MODEL_SAVE_PATH = '/content/drive/MyDrive/symAD-ECNN_models'
RESULTS_PATH = '/content/drive/MyDrive/symAD-ECNN_results'
os.makedirs(MODEL_SAVE_PATH, exist_ok=True)
os.makedirs(RESULTS_PATH, exist_ok=True)
```

### Option 2: Reorganize (Better Organization)

Move your existing folders:
1. Create `symAD-ECNN` folder in MyDrive
2. Create `data` subfolder
3. **Copy** (don't move yet) `IXI_Preprocessed` → `symAD-ECNN/data/processed_ixi`
4. **Copy** `BraTS2021` → `symAD-ECNN/data/brats2021`
5. Test notebooks work with new structure
6. Delete old folders once confirmed

---

## Empty Folders in Your Local Project

Your local `data/` folder should stay mostly empty because:

### What Should Be in Local `data/` Folder:
```
data/
├── .gitkeep                    # Empty file to track folder in Git
└── README.md                   # Explains where actual data is
```

### What Should NOT Be in Local `data/` Folder:
- ❌ `.npy` files (too large, in Drive)
- ❌ `.nii` files (too large, in Drive)
- ❌ Raw images (in Drive)

### Create Placeholder Files

```bash
# In PowerShell
cd C:\Users\rifad\symAD-ECNN\data

# Create .gitkeep to track empty folder
New-Item -ItemType File -Path ".gitkeep"

# Create README
@"
# Data Directory

This folder is for data files. Actual data is stored in Google Drive.

## Google Drive Structure

All training data is in Google Drive:
- IXI preprocessed: `/MyDrive/symAD-ECNN/data/processed_ixi/`
- BraTS test data: `/MyDrive/symAD-ECNN/data/brats2021/`

## Why Not in Git?

Data files are too large for GitHub (100MB limit per file).
Total dataset size: ~2-3GB

## Access in Colab

Mount Drive in notebooks:
```python
from google.colab import drive
drive.mount('/content/drive')
```

See GOOGLE_DRIVE_SETUP.md for details.
"@ | Out-File -FilePath "README.md" -Encoding UTF8
```

---

## Checklist

### ✅ Before First Colab Training Session

- [ ] Organize Google Drive with `symAD-ECNN` folder structure
- [ ] Verify IXI preprocessed data is in Drive (`train/` and `val/`)
- [ ] Verify BraTS data is in Drive
- [ ] Update notebook mount cells with YOUR Drive paths
- [ ] Test mount and data loading in Colab
- [ ] Run verification cell to check file counts
- [ ] Ensure models will save to Drive (not local `/content`)

### ✅ In Your Local Git Repo

- [ ] Keep `data/` folder with just `.gitkeep` and `README.md`
- [ ] `.gitignore` already excludes `*.npy`, `*.nii`, `*.pth`
- [ ] Don't commit large data files
- [ ] Commit notebook changes after adding Drive mount cells

---

## Quick Test in Colab

Run this in a new Colab notebook to verify your setup:

```python
# Test Drive access
from google.colab import drive
import os
import glob

drive.mount('/content/drive')

# UPDATE THIS to your actual path
YOUR_IXI_PATH = '/content/drive/MyDrive/IXI_Preprocessed/train'

# Check
files = glob.glob(os.path.join(YOUR_IXI_PATH, '*.npy'))
print(f"Found {len(files)} .npy files")

if len(files) > 0:
    print("✓ SUCCESS! Drive data is accessible")
    print(f"Sample files: {[os.path.basename(f) for f in files[:3]]}")
else:
    print("❌ ERROR: No files found")
    print("Check your path!")
```

---

## Summary

**Local Computer (VS Code)**:
- Code (`.py`, `.ipynb` files)
- Documentation (`.md` files)
- Empty `data/` folder (just placeholders)

**Google Drive**:
- All actual data (~2-3GB)
- Trained models (`.pth` files)
- Results (`.json`, `.png` files)

**GitHub**:
- Code and notebooks
- Documentation
- No large data files

**Colab**:
- Connects to Drive for data
- Trains models with GPU
- Saves results back to Drive

This separation keeps GitHub clean and gives you unlimited storage in Drive!
