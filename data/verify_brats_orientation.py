"""
Quick verification script to check if processed BraTS data has correct orientation.
Compares processed .npy files with raw BraTS files processed with as_closest_canonical().
"""

import os
import numpy as np
import nibabel as nib
from skimage.transform import resize
from glob import glob

# Paths
BRATS_RAW = r"c:\Users\rifad\symAD-ECNN\data\brats2021"
BRATS_PROCESSED = r"c:\Users\rifad\symAD-ECNN\data\brats2021_processed\resized"  # Local processed
# OR if checking the filtered version you uploaded to Drive:
# BRATS_PROCESSED = r"path\to\downloaded\brats2021_test_filtered"

print("="*70)
print("BRATS ORIENTATION VERIFICATION")
print("="*70)

# Step 1: Find a raw BraTS T1 file
patient_folders = [f for f in os.listdir(BRATS_RAW) if os.path.isdir(os.path.join(BRATS_RAW, f))]
if not patient_folders:
    print("❌ No patient folders found!")
    exit(1)

# Get first patient's T1 file
first_patient = os.path.join(BRATS_RAW, patient_folders[0])
t1_file = None
for file in os.listdir(first_patient):
    if file.endswith('_t1.nii.gz') and not file.endswith('_t1ce.nii.gz'):
        t1_file = os.path.join(first_patient, file)
        break

if not t1_file:
    print("❌ No T1 file found!")
    exit(1)

print(f"\n📁 Testing with: {os.path.basename(t1_file)}")

# Step 2: Process raw file TWO ways
print("\n🔍 Processing raw file in two ways...")

# Method 1: WITHOUT as_closest_canonical (old way)
print("\n1️⃣ Method 1: WITHOUT orientation correction")
img_old = nib.load(t1_file)
vol_old = img_old.get_fdata()
vol_old_norm = (vol_old - vol_old.min()) / (vol_old.max() - vol_old.min())
middle_slice_old = vol_old_norm[:, :, vol_old.shape[2] // 2]
slice_old_resized = resize(middle_slice_old, (128, 128), mode='reflect', preserve_range=True, anti_aliasing=True)
print(f"   Shape: {slice_old_resized.shape}")
print(f"   Range: [{slice_old_resized.min():.4f}, {slice_old_resized.max():.4f}]")
print(f"   Mean: {slice_old_resized.mean():.4f}")

# Method 2: WITH as_closest_canonical (new way)
print("\n2️⃣ Method 2: WITH orientation correction (as_closest_canonical)")
img_new = nib.load(t1_file)
img_new = nib.as_closest_canonical(img_new)  # Apply orientation fix
vol_new = img_new.get_fdata()
vol_new_norm = (vol_new - vol_new.min()) / (vol_new.max() - vol_new.min())
middle_slice_new = vol_new_norm[:, :, vol_new.shape[2] // 2]
slice_new_resized = resize(middle_slice_new, (128, 128), mode='reflect', preserve_range=True, anti_aliasing=True)
print(f"   Shape: {slice_new_resized.shape}")
print(f"   Range: [{slice_new_resized.min():.4f}, {slice_new_resized.max():.4f}]")
print(f"   Mean: {slice_new_resized.mean():.4f}")

# Step 3: Load a processed file (if available)
if os.path.exists(BRATS_PROCESSED):
    processed_files = sorted([f for f in os.listdir(BRATS_PROCESSED) if f.endswith('.npy')])
    if processed_files:
        print(f"\n3️⃣ Loading existing processed file...")
        # Load first processed file
        sample_processed = np.load(os.path.join(BRATS_PROCESSED, processed_files[0]))
        print(f"   File: {processed_files[0]}")
        print(f"   Shape: {sample_processed.shape}")
        print(f"   Range: [{sample_processed.min():.4f}, {sample_processed.max():.4f}]")
        print(f"   Mean: {sample_processed.mean():.4f}")
        
        # Step 4: Compare
        print("\n" + "="*70)
        print("COMPARISON RESULTS")
        print("="*70)
        
        # Calculate differences
        diff_old = np.abs(slice_old_resized - sample_processed).mean()
        diff_new = np.abs(slice_new_resized - sample_processed).mean()
        
        print(f"\nMean absolute difference:")
        print(f"  Method 1 (no correction) vs Processed: {diff_old:.6f}")
        print(f"  Method 2 (with correction) vs Processed: {diff_new:.6f}")
        
        # Determine which is closer
        if diff_old < diff_new:
            print("\n❌ RESULT: Your processed data was created WITHOUT orientation correction")
            print("   Recommendation: You need to REPROCESS the BraTS data")
            print("   The processed data does not have as_closest_canonical() applied")
        else:
            print("\n✅ RESULT: Your processed data appears to have orientation correction")
            print("   Good news: The data looks like it was processed correctly")
            
        # Also check if slices are identical or just similar
        if diff_new < 0.001:
            print("   ✅ Slices are nearly identical - orientation correction was applied!")
        elif diff_old < 0.001:
            print("   ❌ Slices match the uncorrected version - need to reprocess!")
        else:
            print("   ⚠️  Neither match exactly (different slice selection?)")
            print("   This script compares middle slices - processed data might use different slicing")
            print("   Manual visual inspection recommended")
            
    else:
        print(f"\n⚠️ No .npy files found in {BRATS_PROCESSED}")
        print("   Cannot verify - no processed data to check")
else:
    print(f"\n⚠️ Processed folder not found: {BRATS_PROCESSED}")
    print("   Cannot verify - processed data not available locally")
    print("\n💡 If you have the filtered data from Google Drive:")
    print("   1. Download some sample .npy files")
    print("   2. Update BRATS_PROCESSED path in this script")
    print("   3. Run again")

print("\n" + "="*70)
print("RECOMMENDATION")
print("="*70)
print("\nTo fix the preprocessing pipeline:")
print("1. The BraTS preprocessing notebook needs as_closest_canonical()")
print("2. This should be added to the ACTUAL processing loop (Section 9)")
print("3. Not just the test cell at the end")
print("\nWould you like me to update the notebook with the fix?")
print("="*70)
