# IXI Data Splitting - Summary

## What Changed

I've added a **complete train/val split section** (Step 8) to your existing IXI preprocessing notebook:

**File**: `notebooks/preprocessing_ixi.ipynb`

## New Cells Added (14 cells total)

### Section: Step 8 - Split Data into Train/Val (90/10)

1. **Header** - Explains the purpose
2. **Configuration** - Set paths and split ratio
3. **Create Directories** - Make train/val folders
4. **Get Files** - List all .npy files and calculate split
5. **Copy Train Files** - Copy 90% to train folder (with progress)
6. **Copy Val Files** - Copy 10% to val folder (with progress)
7. **Verification** - Check file counts match
8. **Sample Check** - Load samples to verify data integrity
9. **Final Summary** - Show results and next steps
10. **Optional Cleanup** - Delete original folder to save space

## What It Does

```
Input:  ixi_resized/ (all ~16,771 files in one folder)
        ↓
Output: processed_ixi/train/ (~15,094 files - 90%)
        processed_ixi/val/   (~1,677 files - 10%)
```

## How to Use

### In Google Colab:

1. **Upload notebook to Drive** or open from GitHub
2. **Run all cells** up to Step 7 (preprocessing) - if not already done
3. **Run Step 8 cells** (the new split section)
4. **Wait 5-10 minutes** for files to copy
5. **Verify** the final summary shows correct counts
6. **Done!** Data is ready for training

### Configuration (Update if needed)

In the first cell of Step 8, update these paths:
```python
RESIZED_FOLDER = "/content/drive/MyDrive/symAD-ECNN/data/processed_ixi/resized_ixi"
TRAIN_FOLDER = "/content/drive/MyDrive/symAD-ECNN/data/processed_ixi/train"
VAL_FOLDER = "/content/drive/MyDrive/symAD-ECNN/data/processed_ixi/val"
TRAIN_RATIO = 0.9  # 90% train, 10% validation
```

## Features

✅ **Random split** with fixed seed (42) for reproducibility  
✅ **Progress tracking** - Updates every 1000/100 files  
✅ **Verification** - Checks all files copied correctly  
✅ **Sample check** - Loads samples to verify data integrity  
✅ **Copies files** - Keeps original (safer than moving)  
✅ **Optional cleanup** - Can delete original to save space  

## Expected Results

```
Original: ~16,771 files in resized_ixi/

After split:
  Train: ~15,094 files (90.0%)
  Val:   ~1,677 files  (10.0%)
  Total: ~16,771 files ✓
```

## Storage Impact

- **Before split**: ~2 GB (resized_ixi folder only)
- **After split**: ~4 GB (resized + train + val)
- **After cleanup**: ~2 GB (train + val only)

If you run the optional cleanup cell, it deletes the original `resized_ixi` folder to save ~2GB.

## Next Steps After Splitting

1. **Upload BraTS data** to Drive (if not done)
2. **Update training notebooks** with these paths:
   ```python
   IXI_TRAIN_PATH = '/content/drive/MyDrive/symAD-ECNN/data/processed_ixi/train'
   IXI_VAL_PATH = '/content/drive/MyDrive/symAD-ECNN/data/processed_ixi/val'
   ```
3. **Start training** your three models!

## Why This Approach?

✅ **All-in-one**: Preprocessing + splitting in same notebook  
✅ **Organized**: Clear workflow from raw data → train/val  
✅ **Reproducible**: Same split every time (seed=42)  
✅ **Safe**: Copies files (original kept unless you delete)  
✅ **Efficient**: Only split once, use for all three models  

## Troubleshooting

### Issue: "Files not found"
**Solution**: Update `RESIZED_FOLDER` path to match your Drive structure

### Issue: "Out of space"
**Solution**: Run the optional cleanup cell to delete original folder

### Issue: "Copy taking too long"
**Solution**: Normal for ~16K files. Takes 5-10 minutes. Check progress updates.

### Issue: "File count mismatch"
**Solution**: Re-run the split cells. Make sure Drive sync is complete.

---

## Summary

You can now run your preprocessing notebook end-to-end and get train/val split automatically! No need for a separate split notebook.

**Workflow**:
```
preprocessing_ixi.ipynb (Steps 1-7) → Resized data
                       (Step 8)     → Train/val split
                                    → Ready for training!
```
