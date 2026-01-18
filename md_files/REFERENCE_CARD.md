# BraTS Preprocessing - Command Reference Card

## 📋 Quick Commands & Paths

### File Paths
```
Source Data:        c:\Users\rifad\symAD-ECNN\data\brats2021\
Notebook:           c:\Users\rifad\symAD-ECNN\notebooks\brats2021_t1_preprocessing.ipynb
Output (Raw):       c:\Users\rifad\symAD-ECNN\data\brats2021_processed\raw_slices\
Output (Filtered):  c:\Users\rifad\symAD-ECNN\data\brats2021_processed\filtered\
Output (Final):     c:\Users\rifad\symAD-ECNN\data\brats2021_processed\resized\
ZIP File:           c:\Users\rifad\symAD-ECNN\data\brats2021_processed_slices_*.zip
```

### Google Drive Paths
```
Upload To:          MyDrive/symAD-ECNN/data/brats2021_test/
IXI Data:           MyDrive/symAD-ECNN/data/processed_ixi/resized_ixi/
```

---

## 🎯 Cell Execution Order

| Cell | What It Does | Time |
|------|--------------|------|
| 1 | Import libraries | <1s |
| 2 | Define paths | <1s |
| 3 | Explore dataset | ~5s |
| 4 | Find T1 files | ~5s |
| 5 | Create folders | <1s |
| 6 | Skip (old copy function) | - |
| 7 | Load sample | ~5s |
| 8 | Define preprocessing functions | <1s |
| 9 | **STEP 1: Extract & normalize slices** | **20-30 min** |
| 10 | Visualize raw slices | ~5s |
| 11 | **STEP 2: Filter slices** | **5-10 min** |
| 12 | Verify filtered | ~5s |
| 13 | Resize configuration | <1s |
| 14 | **STEP 3: Resize to 128×128** | **10-20 min** |
| 15 | Verify resized | ~5s |
| 16 | Visualize final slices | ~5s |
| 17 | Pipeline summary | ~5s |
| 18 | **STEP 4: Create ZIP** | **5-10 min** |
| 19 | Colab extraction code | - |

**Total Time**: ~45-75 minutes

---

## 🔢 Expected Numbers

```
Patient Folders:        ~100-200
T1 Files Found:         ~100-200
Raw Slices Extracted:   ~3000-5000
After Filtering:        ~1000-2000
Final Resized:          ~1000-2000
ZIP File Size:          ~100-300 MB
```

---

## ✅ Validation Checklist

After each step, verify:

### After Cell 9 (Extract):
- [ ] "Total valid slices extracted: XXXX" appears
- [ ] Visualizations show brain scans
- [ ] raw_slices folder has .npy files

### After Cell 11 (Filter):
- [ ] "Filtered slices kept: XXXX" appears
- [ ] filtered folder has .npy files
- [ ] Count is less than raw count

### After Cell 14 (Resize):
- [ ] "Successfully resized: XXXX" appears
- [ ] Sample shape is (128, 128)
- [ ] resized folder has .npy files

### After Cell 18 (ZIP):
- [ ] "ZIP FILE CREATED SUCCESSFULLY!" appears
- [ ] ZIP file exists in data folder
- [ ] File size is reasonable (100-300 MB)

---

## 🛠️ Common Parameters

### Normalization
```python
normalize(x)
# Output: [0.0, 1.0] range
```

### Filtering Thresholds
```python
min_nonzero_ratio = 0.12  # 12% of pixels must be non-zero
min_mean = 0.1            # Mean pixel value > 0.1
```

### Resize Parameters
```python
target_shape = (128, 128)
mode = 'reflect'
anti_aliasing = True
```

---

## 🆘 Emergency Fixes

### Problem: Too Few Slices
**Cell 8**, change:
```python
min_nonzero_ratio = 0.12  # Change to 0.08
min_mean = 0.1            # Change to 0.05
```

### Problem: Out of Memory
**Cell 13**, change:
```python
BATCH_SIZE = 500  # Change to 250 or 100
```

### Problem: Processing Too Slow
Skip intermediate visualizations (cells 10, 16) on first run.

---

## 📊 Sample Statistics

### Good Slice
```python
Shape: (128, 128)
Min: 0.0000
Max: 1.0000
Mean: 0.2500-0.3500
Std: 0.1500-0.2500
```

### Warning Signs
```python
Mean < 0.1        # Too dark (will be filtered)
Max == Min        # Empty slice
Shape != (128,128) # Not resized correctly
Any NaN/Inf       # Corrupted data
```

---

## 💾 Disk Space Requirements

```
Raw Slices:      ~500 MB - 1 GB
Filtered:        ~300 MB - 800 MB
Resized:         ~300 MB - 800 MB
ZIP:             ~100 MB - 300 MB
Total Peak:      ~1.5 GB - 3 GB
```

---

## 🔍 Debug Commands

### Check file counts:
```python
import os
print(len(os.listdir(t1_raw_folder)))
print(len(os.listdir(filtered_folder)))
print(len(os.listdir(resized_folder)))
```

### Check sample file:
```python
import numpy as np
sample = np.load('path/to/slice_000000.npy')
print(f"Shape: {sample.shape}")
print(f"Range: [{sample.min()}, {sample.max()}]")
print(f"Mean: {sample.mean()}")
```

### Check disk space:
```python
import shutil
total, used, free = shutil.disk_usage("C:\\")
print(f"Free: {free // (2**30)} GB")
```

---

## 🎯 Success Criteria

Your preprocessing is successful if:

✅ No errors during execution  
✅ Got 1000-2000 final slices  
✅ All slices are (128, 128) shape  
✅ All values in [0, 1] range  
✅ Visualizations show clear brain images  
✅ ZIP file created successfully  
✅ File size is reasonable (100-300 MB)  

---

## 📤 Upload Process

### Step 1: Local → ZIP
Run cell 18 in notebook

### Step 2: ZIP → Google Drive
1. Open drive.google.com
2. Go to MyDrive/symAD-ECNN/data/
3. Create folder: brats2021_test
4. Upload ZIP file
5. Wait for completion

### Step 3: ZIP → Colab Extract
Copy cell 19 code to Colab and run

---

## 🔗 Quick Links

- **Main Notebook**: `notebooks/data_preprocessing/brats2021_t1_preprocessing.ipynb`
- **Detailed Guide**: `BRATS_PREPROCESSING_GUIDE.md`
- **Quick Start**: `QUICKSTART.md`
- **Comparison**: `IXI_vs_BRATS_COMPARISON.md`
- **Summary**: `PROJECT_SUMMARY.md`

---

## 📞 Troubleshooting Quick Reference

| Error Message | Cell | Fix |
|---------------|------|-----|
| "File not found" | 4 | Check brats_folder path |
| "No T1 files found" | 4 | Verify BraTS data exists |
| "Out of memory" | 14 | Reduce BATCH_SIZE |
| "No space left" | 9/18 | Free up disk space |
| "Corrupted file" | 9 | Noted in output, continues |
| "Shape mismatch" | 15 | Re-run cell 14 |

---

**Keep this reference handy while running the notebook!** 📌
