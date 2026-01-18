# BraTS Preprocessing - Quick Start Guide

## 🚀 5-Minute Setup

### What You Have:
- ✅ BraTS 2021 dataset downloaded at: `c:\Users\rifad\symAD-ECNN\data\brats2021`
- ✅ Preprocessing notebook: `notebooks/data_preprocessing/brats2021_t1_preprocessing.ipynb`

### What You'll Get:
- 📦 Processed BraTS slices (128×128, normalized, .npy format)
- 📦 ZIP file ready for Google Drive upload
- 📦 Data ready for anomaly detection testing

---

## ⚡ Quick Steps

### 1️⃣ Open Notebook
```
Open: notebooks/data_preprocessing/brats2021_t1_preprocessing.ipynb
```

### 2️⃣ Run All Cells in Order
Just click "Run All" or execute cells 1-20 sequentially. The notebook will:
- ✓ Find all T1 files
- ✓ Extract 2D slices
- ✓ Normalize to [0, 1]
- ✓ Filter empty slices
- ✓ Resize to 128×128
- ✓ Create ZIP file

**Time**: ~30-60 minutes (depends on dataset size)

### 3️⃣ Upload to Google Drive
After cell 19 completes:
1. Find the ZIP file at: `c:\Users\rifad\symAD-ECNN\data\brats2021_processed_slices_*.zip`
2. Open Google Drive
3. Navigate to: `MyDrive/symAD-ECNN/data/`
4. Create folder: `brats2021_test`
5. Upload the ZIP file

### 4️⃣ Extract in Colab
Copy code from cell 20 into your Colab notebook and run it.

---

## 📊 Expected Output

```
Step 1: Extract 2D slices → ~3000-5000 raw slices
Step 2: Filter (mean > 0.1) → ~1000-2000 filtered slices  
Step 3: Resize to 128x128 → ~1000-2000 final slices
Step 4: Create ZIP → Single file ~100-300 MB
```

---

## ✅ Success Indicators

Look for these in the notebook output:

✓ "Total T1 files found: XX"  
✓ "Total valid slices extracted: XXXX"  
✓ "Filtered slices kept: XXXX"  
✓ "Successfully resized: XXXX"  
✓ "ZIP FILE CREATED SUCCESSFULLY!"

---

## 🔍 Verify Quality

After processing, check:

1. **Visualizations** (cells 10, 15, 18) - Should show clear brain slices
2. **Sample statistics** (cell 12) - Shape should be (128, 128), range [0, 1]
3. **Final count** (cell 14) - Should have 1000-2000 slices
4. **ZIP file** exists in data folder

---

## 🆘 Quick Troubleshooting

| Problem | Solution |
|---------|----------|
| "No T1 files found" | Check path in cell 4 |
| "Out of memory" | Reduce BATCH_SIZE in cell 13 to 250 |
| Too few slices | Lower thresholds in cell 8 |
| ZIP creation fails | Check disk space (~1-2 GB needed) |

---

## 📝 Important Notes

1. **Don't interrupt** during extraction (cell 9) - it may take 20-30 minutes
2. **Keep notebook open** - some cells take time to complete
3. **Check disk space** - need ~1-2 GB free
4. **Save progress** - notebook auto-saves but manually save after major steps

---

## 🎯 Next Steps After Upload

1. Open your Colab notebook
2. Run the extraction code from cell 20
3. Verify BraTS slices are loaded
4. Compare with IXI dataset format
5. Proceed with model training/testing

---

## 📞 Need Help?

Check these files:
- `BRATS_PREPROCESSING_GUIDE.md` - Detailed explanation
- `IXI_vs_BRATS_COMPARISON.md` - Pipeline comparison
- Notebook cells have detailed comments

---

**Ready to start?** Open the notebook and run cell 1! 🚀
