# Fresh Preprocessing - Quick Start Guide

## 🎯 Objective

Reprocess **both** IXI and BraTS datasets with corrected settings:
- ✅ RAS orientation correction (`nib.as_closest_canonical()`)
- ✅ Bicubic interpolation (`order=3`) for sharp images
- ✅ Consistent preprocessing for both datasets

---

## 📋 Complete Workflow

### Step 1: Cleanup Old Data (~2 min)

```powershell
powershell -ExecutionPolicy Bypass -File cleanup_and_prep_reprocessing.ps1
```

**What it does**:
- Backs up local BraTS processed data → `brats2021_processed_OLD_timestamp`
- Provides Colab commands to backup Drive data

---

### Step 2: BraTS Local Preprocessing (~60-70 min)

**Option A: Run Python Script (Recommended)**
```powershell
cd c:\Users\rifad\symAD-ECNN
python notebooks/brats_preprocessing_complete.py
```

**Option B: Run Notebook**
- Open: `notebooks/brats2021_t1_preprocessing.ipynb`
- Note: Notebook is missing Sections 9-11 (use script instead)

**Output**:
- `data/brats2021_processed/raw_slices/` - ~97,882 slices with RAS orientation
- `data/brats2021_processed/filtered/` - ~49,888 middle slices
- `data/brats2021_processed/resized/` - ~49,888 slices at 128×128 (bicubic)
- `data/brats2021_processed/filtered_eval/` - ~5,000 slices (4 per patient)
- `data/brats2021_processed/brats2021_filtered_RAS_4slices_*.zip` - ZIP for upload

**Critical Settings Applied**:
```python
# RAS orientation
nii_img = nib.as_closest_canonical(nii_img)

# Bicubic resize (sharp)
resized = resize(arr, (128, 128), order=3, mode='reflect', 
                anti_aliasing=True, preserve_range=True)
```

---

### Step 3: Upload BraTS ZIP to Drive (~5-10 min)

1. Find ZIP file:
   ```
   c:\Users\rifad\symAD-ECNN\data\brats2021_processed\brats2021_filtered_RAS_4slices_*.zip
   ```

2. Upload to Google Drive:
   ```
   MyDrive/symAD-ECNN/data/
   ```

---

### Step 4: IXI Colab Preprocessing (~3-5 hours)

**Can run in parallel with BraTS!**

1. **Open in Colab**:
   - File: `notebooks/ixi_t1_preprocessing.ipynb`
   - Click "Open in Colab" button

2. **Enable GPU**:
   ```
   Runtime → Change runtime type → Hardware accelerator → T4 GPU → Save
   ```

3. **Run Colab cleanup commands** (from cleanup script output):
   ```python
   # Backup old BraTS test data
   import os
   from datetime import datetime
   
   drive_base = '/content/drive/MyDrive/symAD-ECNN/data'
   timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
   
   old_brats = f'{drive_base}/brats2021_test_filtered'
   if os.path.exists(old_brats):
       backup_brats = f'{drive_base}/brats2021_test_filtered_OLD_{timestamp}'
       os.rename(old_brats, backup_brats)
       print(f'✅ Backed up: {backup_brats}')
   ```

4. **Run all cells** in IXI notebook:
   - Mount Drive
   - Install HD-BET
   - Skull stripping (~3-5 hours)
   - Slice extraction with RAS orientation
   - Resize with bicubic interpolation ✅
   - Train/val split (90/10)

**Output**:
- `MyDrive/symAD-ECNN/data/processed_ixi/train/` - ~22,500 slices
- `MyDrive/symAD-ECNN/data/processed_ixi/val/` - ~2,500 slices

**Critical Settings (Already Correct)**:
```python
# RAS orientation
img_obj = nib.as_closest_canonical(img_obj)

# Bicubic resize (sharp)
slice_resized = resize(slice_img, IMG_SIZE, order=3, 
                      mode='reflect', anti_aliasing=True, 
                      preserve_range=True)
```

---

### Step 5: Extract BraTS in Colab (~2 min)

**Run in Colab after upload completes**:

```python
import os
import zipfile
from glob import glob

BASE_PATH = "/content/drive/MyDrive/symAD-ECNN"
zip_path = f"{BASE_PATH}/data/brats2021_filtered_RAS_4slices_*.zip"

# Find the ZIP file
import glob as g
zip_files = g.glob(zip_path)
if zip_files:
    zip_path = zip_files[0]
    output_folder = f"{BASE_PATH}/data/brats2021_test_filtered"
    
    os.makedirs(output_folder, exist_ok=True)
    
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(output_folder)
    
    print(f"✅ Extracted to: {output_folder}")
    print(f"   Files: {len(glob.glob(f'{output_folder}/*.npy'))}")
else:
    print("❌ ZIP file not found!")
```

---

### Step 6: Retrain CNN-AE

1. **Update paths** in `notebooks/models/02_cnn_autoencoder.ipynb`:
   ```python
   IXI_TRAIN = "/content/drive/MyDrive/symAD-ECNN/data/processed_ixi/train"
   IXI_VAL = "/content/drive/MyDrive/symAD-ECNN/data/processed_ixi/val"
   BRATS_TEST = "/content/drive/MyDrive/symAD-ECNN/data/brats2021_test_filtered"
   ```

2. **Train for 20 epochs**

3. **Verify results**:
   - ✅ Val loss: ~0.02 (similar to before)
   - ✅ **AUROC: 0.75-0.90** (realistic, NOT 1.0!)
   - ✅ Error maps highlight tumors, not format differences

---

## ✅ Success Criteria

### After Preprocessing:

- [x] BraTS: ~5,000 slices (4 per patient)
- [x] IXI: ~25,000 slices (22.5k train, 2.5k val)
- [x] Both: 128×128, [0,1] normalized
- [x] Both: RAS orientation
- [x] Both: Bicubic interpolation (sharp!)
- [x] Both: Uploaded to Google Drive

### After Retraining:

- [ ] AUROC: 0.75-0.90 (not 1.0)
- [ ] Error maps show tumors clearly
- [ ] Loss curves look normal
- [ ] Val loss: ~0.02

---

## 🔍 Troubleshooting

### Q: How do I know if preprocessing is correct?

**Visual Check**:
```python
import numpy as np
import matplotlib.pyplot as plt

# Load sample from each dataset
brats_sample = np.load('path/to/brats/slice_000000.npy')
ixi_sample = np.load('path/to/ixi/slice_000000.npy')

# Plot side-by-side
fig, axes = plt.subplots(1, 2, figsize=(10, 5))
axes[0].imshow(brats_sample, cmap='gray')
axes[0].set_title('BraTS')
axes[1].imshow(ixi_sample, cmap='gray')
axes[1].set_title('IXI')
plt.show()

# Check stats
print(f"BraTS: shape={brats_sample.shape}, range=[{brats_sample.min():.4f}, {brats_sample.max():.4f}]")
print(f"IXI:   shape={ixi_sample.shape}, range=[{ixi_sample.min():.4f}, {ixi_sample.max():.4f}]")
```

**Expected**:
- Both should be sharp (not blurry)
- Both should be 128×128
- Both should be [0, 1] range

### Q: BraTS preprocessing is slow

- Normal: ~60-70 minutes for 1,251 patients
- Use the script (`brats_preprocessing_complete.py`) instead of notebook
- Batch processing helps with memory

### Q: IXI skull stripping fails

- Ensure GPU is enabled in Colab
- Check HD-BET installation
- May need to restart runtime and reinstall

### Q: AUROC still 1.0 after retraining

**Check**:
1. Did you use the NEW processed data? (not old folders)
2. Are both datasets using bicubic interpolation?
3. Are both datasets using RAS orientation?
4. Did you retrain from scratch (not resume old model)?

---

## 📊 Expected Timelines

| Task | Time | Can Parallel? |
|------|------|---------------|
| Cleanup | 2 min | - |
| BraTS local preprocessing | 60-70 min | Yes* |
| Upload BraTS ZIP | 5-10 min | - |
| IXI Colab preprocessing | 3-5 hours | Yes* |
| Extract BraTS in Colab | 2 min | - |
| Retrain CNN-AE | 20-30 min | - |
| **Total Sequential** | **4-6 hours** | |
| **Total Parallel** | **3-5 hours** | *If running BraTS & IXI together |

---

## 🎯 Key Differences from Old Preprocessing

| Aspect | Old (Wrong) | New (Correct) |
|--------|-------------|---------------|
| **BraTS Orientation** | Native (varies) | RAS (standard) ✅ |
| **IXI Orientation** | Native (varies) | RAS (standard) ✅ |
| **BraTS Interpolation** | Linear (order=1) | Bicubic (order=3) ✅ |
| **IXI Interpolation** | Bicubic (order=3) ✅ | Bicubic (order=3) ✅ |
| **Image Quality** | BraTS blurry | Both sharp ✅ |
| **AUROC** | 1.0 (wrong) | 0.75-0.90 (correct) ✅ |

---

## 📁 File Locations Reference

### Local (Windows):
```
Source:
  c:\Users\rifad\symAD-ECNN\data\brats2021\

Scripts:
  c:\Users\rifad\symAD-ECNN\cleanup_and_prep_reprocessing.ps1
  c:\Users\rifad\symAD-ECNN\notebooks\brats_preprocessing_complete.py

Output:
  c:\Users\rifad\symAD-ECNN\data\brats2021_processed\
  c:\Users\rifad\symAD-ECNN\data\brats2021_processed_OLD_*\  (backup)
```

### Google Drive (Colab):
```
IXI:
  MyDrive/symAD-ECNN/data/ixi_t1/raw/                    (source)
  MyDrive/symAD-ECNN/data/processed_ixi/train/           (output)
  MyDrive/symAD-ECNN/data/processed_ixi/val/             (output)

BraTS:
  MyDrive/symAD-ECNN/data/brats2021_filtered_RAS_*.zip   (uploaded)
  MyDrive/symAD-ECNN/data/brats2021_test_filtered/       (extracted)
  MyDrive/symAD-ECNN/data/brats2021_test_filtered_OLD_*/ (backup)
```

---

**Ready to start? Run Step 1 (cleanup script)!**

```powershell
powershell -ExecutionPolicy Bypass -File cleanup_and_prep_reprocessing.ps1
```
