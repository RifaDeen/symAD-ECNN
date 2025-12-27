# BraTS 2021 Preprocessing Guide
## Local Processing → Google Drive Upload Workflow

---

## 📋 Overview

This guide explains how to preprocess BraTS 2021 T1 MRI data locally and upload it to Google Drive for use in your anomaly detection model.

### **Why This Workflow?**
- **IXI Dataset**: Used for TRAINING (normal brain scans)
- **BraTS Dataset**: Used for TESTING (abnormal brain scans with tumors for anomaly detection)
- **Problem**: BraTS is too large to download in Colab, so we process it locally first

---

## 🔄 IXI Preprocessing Pipeline (Reference)

Your IXI preprocessing in Colab followed these steps:

1. **Download** from Kaggle (3D NIfTI volumes with GM maps)
2. **Extract 2D slices** from 3D volumes
3. **Normalize** each slice to [0, 1] range
4. **Filter** empty slices (mean > 0.1, non-zero ratio > 0.12)
5. **Resize** to 128×128 pixels
6. **Save** as `.npy` files

**Result**: ~2000-3000 normalized 128×128 slices ready for training

---

## 🎯 BraTS Preprocessing Pipeline (Local)

The `brats2021_t1_preprocessing.ipynb` notebook follows the **EXACT SAME PIPELINE** as IXI:

### **Step 1: Extract & Normalize 2D Slices**
- Scans all BraTS T1 files (`*_t1.nii.gz`)
- Extracts each 2D slice from 3D volumes
- Normalizes to [0, 1] using min-max normalization
- Filters during extraction (skips empty slices)
- Saves as `.npy` + `.png` preview

**Output**: `data/brats2021_processed/raw_slices/`

### **Step 2: Additional Filtering**
- Applies same filter as IXI: mean pixel value > 0.1
- Ensures only informative slices are kept
- Matches IXI quality standards

**Output**: `data/brats2021_processed/filtered/`

### **Step 3: Resize to 128×128**
- Resizes all slices to match IXI dimensions
- Uses same parameters: `mode='reflect'`, `anti_aliasing=True`
- Processes in batches (500 at a time) for memory efficiency

**Output**: `data/brats2021_processed/resized/` ← **FINAL DATA**

### **Step 4: Create ZIP for Upload**
- Compresses all `.npy` files into a single zip
- Makes upload to Google Drive easier and faster
- Typical size: 100-300 MB (compressed)

**Output**: `data/brats2021_processed_slices_YYYYMMDD_HHMMSS.zip`

---

## 📂 Folder Structure

```
c:\Users\rifad\symAD-ECNN\
│
├── data\
│   ├── brats2021\                          # Original BraTS dataset
│   │   ├── BraTS2021_00000\
│   │   │   ├── BraTS2021_00000_t1.nii.gz  # T1 MRI (we use this)
│   │   │   ├── BraTS2021_00000_t1ce.nii.gz
│   │   │   ├── BraTS2021_00000_t2.nii.gz
│   │   │   └── ...
│   │   └── ...
│   │
│   ├── brats2021_processed\                # Processed output
│   │   ├── raw_slices\                     # Step 1 output
│   │   ├── filtered\                       # Step 2 output
│   │   └── resized\                        # Step 3 output (FINAL)
│   │
│   └── brats2021_processed_slices_*.zip    # Step 4 output (for upload)
│
└── notebooks\
    └── brats2021_t1_preprocessing.ipynb    # Main preprocessing notebook
```

---

## 🚀 Usage Instructions

### **Part A: Local Processing**

1. **Open the notebook**: `notebooks/brats2021_t1_preprocessing.ipynb`

2. **Run cells 1-4**: Setup and explore dataset
   ```
   - Import libraries
   - Define paths
   - Explore dataset structure
   - Find T1 files
   ```

3. **Run cells 5-10**: Extract and process slices
   ```
   - Step 1: Extract 2D slices (cell 9)
   - Visualize samples (cell 10)
   ```

4. **Run cells 11-12**: Filter slices
   ```
   - Step 2: Filter by mean value (cell 11)
   ```

5. **Run cells 13-15**: Resize slices
   ```
   - Step 3: Resize to 128x128 (cells 13-14)
   ```

6. **Run cells 16-18**: Verify and visualize
   ```
   - Verify final output
   - Visualize processed slices
   - View summary statistics
   ```

7. **Run cell 19**: Create ZIP file for upload
   ```
   - Creates compressed zip file
   - Shows upload instructions
   ```

---

### **Part B: Google Drive Upload**

1. **Open Google Drive** in your browser

2. **Navigate to**: `MyDrive/symAD-ECNN/data/`

3. **Create folder**: `brats2021_test`

4. **Upload** the zip file created in Part A

5. **Wait for upload** to complete (may take 10-30 minutes depending on internet speed)

---

### **Part C: Extract in Google Colab**

1. **Open your Colab notebook**

2. **Copy and run** the extraction code from cell 20 in the preprocessing notebook:

```python
from google.colab import drive
drive.mount('/content/drive')

import os
import zipfile

# Define paths
BASE = "/content/drive/MyDrive/symAD-ECNN"
zip_file = f"{BASE}/data/brats2021_processed_slices_XXXXXXXX_XXXXXX.zip"  # Update filename
output_folder = f"{BASE}/data/brats2021_test"

# Create output folder
os.makedirs(output_folder, exist_ok=True)

# Extract
with zipfile.ZipFile(zip_file, 'r') as zip_ref:
    zip_ref.extractall(output_folder)

# Verify
import numpy as np
files = sorted([f for f in os.listdir(output_folder) if f.endswith('.npy')])
print(f"Total files: {len(files)}")

# Test load
sample = np.load(os.path.join(output_folder, files[0]))
print(f"Shape: {sample.shape}")  # Should be (128, 128)
```

3. **Update the zip filename** in the code above

4. **Run the cell** to extract

---

## ✅ Data Consistency Checks

### **IXI vs BraTS Comparison**

Both datasets should have:
- ✅ Shape: `(128, 128)`
- ✅ Range: `[0.0, 1.0]`
- ✅ Data type: `float32` or `float64`
- ✅ Format: `.npy` files
- ✅ Filtering: mean > 0.1

### **Expected Counts**
- **IXI**: ~2000-3000 slices (training)
- **BraTS**: ~1000-2000 slices (testing)

---

## 🔍 Troubleshooting

### **Problem**: Not enough slices extracted
- **Solution**: Lower the filtering threshold in cell 9 (change `min_mean=0.1` to `0.05`)

### **Problem**: Out of memory during resizing
- **Solution**: Reduce `BATCH_SIZE` in cell 13 (try 250 instead of 500)

### **Problem**: Upload to Drive fails
- **Solution**: Split zip into smaller parts using 7-Zip or WinRAR

### **Problem**: Dimensions don't match IXI
- **Solution**: Verify `TARGET_SIZE = (128, 128)` in cell 13

---

## 📊 Expected Results

After completing all steps, you should have:

```
✅ 1000-2000 BraTS T1 slices
✅ Each slice: 128×128 pixels, normalized [0, 1]
✅ Ready for anomaly detection testing
✅ Uploaded to Google Drive
✅ Accessible in Colab
```

---

## 🎯 Next Steps

After preprocessing is complete:

1. **Train model** on IXI data (normal brains)
2. **Test model** on BraTS data (abnormal brains)
3. **Generate anomaly maps** showing tumor locations
4. **Evaluate performance** using reconstruction error

---

## 📝 Key Differences: IXI vs BraTS

| Aspect | IXI | BraTS |
|--------|-----|-------|
| **Source** | Kaggle (preprocessed GM maps) | Local download (raw T1) |
| **Processing** | In Colab | Local (Windows) |
| **Use Case** | Training (normal) | Testing (abnormal) |
| **Volume Type** | GM maps (`mwp1*`) | T1 MRI (`*_t1.nii.gz`) |
| **Expected Slices** | ~2000-3000 | ~1000-2000 |

---

## 💡 Tips

1. **Save your work**: Run cells in order and save notebook after each major step
2. **Check visualizations**: Always look at sample slices to verify quality
3. **Monitor disk space**: Processing creates ~1-2 GB of intermediate files
4. **Be patient**: Full pipeline takes 30-60 minutes depending on dataset size
5. **Keep originals**: Don't delete the original BraTS data until you verify everything works

---

## 📞 Common Questions

**Q: Why not just use BraTS directly in Colab?**  
A: BraTS dataset is too large (~5GB+) and Kaggle download quota limits apply.

**Q: Can I delete intermediate folders?**  
A: Yes, but keep `resized/` folder and the zip file. You can delete `raw_slices/` and `filtered/`.

**Q: What if I need more or fewer slices?**  
A: Adjust the filtering thresholds in cell 9.

**Q: How do I know if preprocessing worked correctly?**  
A: Check the visualizations in cells 10, 15, and 18. Slices should look like brain scans with good contrast.

---

**Created**: December 26, 2025  
**Version**: 1.0  
**Project**: symAD-ECNN (Symmetric Anomaly Detection with E(n)-Equivariant CNN)
