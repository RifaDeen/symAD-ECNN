"""
Test different IXI orientations to see which matches processed BraTS data.
This will tell us if we can match IXI to BraTS without reprocessing BraTS.
"""

import os
import numpy as np
import nibabel as nib
from skimage.transform import resize
from glob import glob

# Paths
IXI_RAW_LOCAL = r"c:\Users\rifad\symAD-ECNN\data\ixi_t1\raw"  # If you have IXI raw locally
BRATS_PROCESSED = r"c:\Users\rifad\symAD-ECNN\data\brats2021_processed\resized"

print("="*70)
print("IXI ORIENTATION MATCHING TEST")
print("="*70)

# Check if IXI raw data exists locally
if not os.path.exists(IXI_RAW_LOCAL):
    print(f"\n❌ IXI raw data not found at: {IXI_RAW_LOCAL}")
    print("\n💡 This test needs raw IXI .nii files.")
    print("   Options:")
    print("   1. Download a few sample IXI files from Kaggle to test")
    print("   2. Run this test in Google Colab where IXI data is located")
    print("\n📋 To run in Colab, use this code:")
    print("-" * 70)
    print("""
# Colab version:
IXI_RAW = "/content/drive/MyDrive/symAD-ECNN/data/ixi_t1/raw"
BRATS_PROCESSED_SAMPLE = "/content/drive/MyDrive/symAD-ECNN/data/brats2021_test_filtered"

# Download and test a few files to compare orientations
""")
    print("-" * 70)
    exit(1)

# Load sample IXI file
ixi_files = [f for f in os.listdir(IXI_RAW_LOCAL) if f.endswith('.nii') or f.endswith('.nii.gz')]
if not ixi_files:
    print(f"❌ No .nii files found in {IXI_RAW_LOCAL}")
    exit(1)

ixi_file = os.path.join(IXI_RAW_LOCAL, ixi_files[0])
print(f"\n📁 Testing IXI file: {os.path.basename(ixi_file)}")

# Load processed BraTS sample
if not os.path.exists(BRATS_PROCESSED):
    print(f"❌ BraTS processed folder not found: {BRATS_PROCESSED}")
    exit(1)

brats_files = sorted([f for f in os.listdir(BRATS_PROCESSED) if f.endswith('.npy')])
if not brats_files:
    print(f"❌ No processed BraTS files found")
    exit(1)

brats_sample = np.load(os.path.join(BRATS_PROCESSED, brats_files[0]))
print(f"📁 BraTS reference: {brats_files[0]}")
print(f"   Shape: {brats_sample.shape}, Mean: {brats_sample.mean():.4f}")

# Test different orientation codes
orientation_codes = [
    ('Native', None),           # No conversion
    ('RAS', 'RAS'),             # Standard radiological (as_closest_canonical)
    ('LPS', 'LPS'),             # DICOM standard
    ('LAS', 'LAS'),
    ('RPS', 'RPS'),
    ('LPI', 'LPI'),
    ('RPI', 'RPI'),
    ('LAI', 'LAI'),
    ('RAI', 'RAI'),
]

print("\n" + "="*70)
print("TESTING ORIENTATIONS")
print("="*70)

results = []

for name, code in orientation_codes:
    try:
        # Load IXI file
        img = nib.load(ixi_file)
        
        # Apply orientation if specified
        if code is not None:
            # Convert to specified orientation using nibabel's orientation tools
            target_ornt = nib.orientations.axcodes2ornt(code)
            current_ornt = nib.orientations.io_orientation(img.affine)
            ornt_transform = nib.orientations.ornt_transform(current_ornt, target_ornt)
            img_data = nib.orientations.apply_orientation(img.get_fdata(), ornt_transform)
            vol = img_data
        else:
            # Native - no transformation
            vol = img.get_fdata()
        
        # Normalize
        vol_norm = (vol - vol.min()) / (vol.max() - vol.min())
        
        # Extract middle slice (axis 2)
        middle_slice = vol_norm[:, :, vol.shape[2] // 2]
        
        # Resize to 128x128
        slice_resized = resize(middle_slice, (128, 128), mode='reflect', 
                              preserve_range=True, anti_aliasing=True)
        
        # Compare to BraTS
        diff = np.abs(slice_resized - brats_sample).mean()
        
        print(f"\n{name} orientation:")
        print(f"   Volume shape: {vol.shape}")
        print(f"   Slice mean: {slice_resized.mean():.4f}")
        print(f"   Diff from BraTS: {diff:.6f}")
        
        results.append((name, code, diff))
        
    except Exception as e:
        print(f"\n{name} orientation: ❌ Error - {e}")

# Find best match
if results:
    print("\n" + "="*70)
    print("RESULTS")
    print("="*70)
    
    results.sort(key=lambda x: x[2])  # Sort by difference
    
    print("\nOrientations ranked by similarity to BraTS:")
    for i, (name, code, diff) in enumerate(results, 1):
        marker = "🎯" if i == 1 else "  "
        print(f"{marker} {i}. {name:15s} - Difference: {diff:.6f}")
    
    best_name, best_code, best_diff = results[0]
    
    print(f"\n{'='*70}")
    print(f"🎯 BEST MATCH: {best_name} orientation")
    print(f"{'='*70}")
    
    if best_name == 'Native':
        print("\n✅ GOOD NEWS!")
        print("   IXI native orientation matches BraTS processed data")
        print("   → Skip as_closest_canonical() in IXI preprocessing")
        print("   → Just process IXI without orientation correction")
    elif best_name == 'RAS':
        print("\n⚠️ BraTS needs reprocessing")
        print("   IXI with RAS (as_closest_canonical) would match")
        print("   → But your BraTS data was NOT processed with RAS")
        print("   → You need to reprocess BraTS with as_closest_canonical()")
    
    if best_diff < 0.01:
        print(f"\n✅ Very close match (diff < 0.01)")
        print("   → High confidence this is the right orientation")
    elif best_diff < 0.05:
        print(f"\n⚠️ Moderate match (diff < 0.05)")
        print("   → Likely correct but verify visually")
    else:
        print(f"\n❌ Poor match (diff > 0.05)")
        print("   → Different slices selected, or other preprocessing differences")
        print("   → Manual visual inspection needed")

print("\n" + "="*70)
