"""
IXI Orientation Matching Test - Google Colab Version
Test if IXI native orientation matches processed BraTS data.

INSTRUCTIONS:
1. Upload this script to Colab or copy the code into a cell
2. Make sure Drive is mounted
3. Run to see which orientation matches BraTS
"""

import os
import numpy as np
import nibabel as nib
from skimage.transform import resize

# === COLAB PATHS ===
IXI_RAW = "/content/drive/MyDrive/symAD-ECNN/data/ixi_t1/raw"
BRATS_PROCESSED = "/content/drive/MyDrive/symAD-ECNN/data/brats2021_test_filtered"

print("="*70)
print("IXI ORIENTATION MATCHING TEST")
print("="*70)

# Check paths
if not os.path.exists(IXI_RAW):
    print(f"❌ IXI raw path not found: {IXI_RAW}")
    print("   Make sure Google Drive is mounted!")
    exit(1)

if not os.path.exists(BRATS_PROCESSED):
    print(f"❌ BraTS processed path not found: {BRATS_PROCESSED}")
    exit(1)

# Load sample IXI file
ixi_files = sorted([f for f in os.listdir(IXI_RAW) if f.endswith('.nii') or f.endswith('.nii.gz')])
if not ixi_files:
    print(f"❌ No .nii files in {IXI_RAW}")
    exit(1)

ixi_file = os.path.join(IXI_RAW, ixi_files[0])
print(f"\n📁 Testing IXI: {os.path.basename(ixi_file)}")

# Load processed BraTS sample (use multiple for better comparison)
brats_files = sorted([f for f in os.listdir(BRATS_PROCESSED) if f.endswith('.npy')])
if not brats_files:
    print(f"❌ No .npy files in {BRATS_PROCESSED}")
    exit(1)

# Load 5 random BraTS samples for robust comparison
import random
test_brats = random.sample(brats_files, min(5, len(brats_files)))
brats_samples = [np.load(os.path.join(BRATS_PROCESSED, f)) for f in test_brats]
print(f"📁 Loaded {len(brats_samples)} BraTS samples for comparison")

print("\n" + "="*70)
print("TESTING ORIENTATIONS")
print("="*70)

def process_ixi_slice(img_obj, apply_canonical=False):
    """Process IXI file and extract middle slice."""
    if apply_canonical:
        img_obj = nib.as_closest_canonical(img_obj)
    
    vol = img_obj.get_fdata()
    vol_norm = (vol - vol.min()) / (vol.max() - vol.min())
    
    # Extract middle slice (axis 2 - like BraTS)
    middle_slice = vol_norm[:, :, vol.shape[2] // 2]
    
    # Resize to 128x128 with same parameters
    slice_resized = resize(middle_slice, (128, 128), mode='reflect', 
                          preserve_range=True, anti_aliasing=True)
    return slice_resized

# Test 1: Native orientation (no correction)
print("\n1️⃣ Testing: Native orientation (no as_closest_canonical)")
img_native = nib.load(ixi_file)
ixi_native = process_ixi_slice(img_native, apply_canonical=False)
print(f"   IXI shape: {img_native.shape}")
print(f"   IXI slice mean: {ixi_native.mean():.4f}")

# Compare to all BraTS samples
diffs_native = [np.abs(ixi_native - brats).mean() for brats in brats_samples]
avg_diff_native = np.mean(diffs_native)
print(f"   Avg diff from BraTS: {avg_diff_native:.6f}")

# Test 2: RAS orientation (with as_closest_canonical)
print("\n2️⃣ Testing: RAS orientation (with as_closest_canonical)")
img_ras = nib.load(ixi_file)
ixi_ras = process_ixi_slice(img_ras, apply_canonical=True)
print(f"   IXI shape: {img_ras.shape}")
print(f"   IXI slice mean: {ixi_ras.mean():.4f}")

# Compare to all BraTS samples
diffs_ras = [np.abs(ixi_ras - brats).mean() for brats in brats_samples]
avg_diff_ras = np.mean(diffs_ras)
print(f"   Avg diff from BraTS: {avg_diff_ras:.6f}")

# Results
print("\n" + "="*70)
print("RESULTS")
print("="*70)

print(f"\nAverage differences:")
print(f"  Native (no correction):  {avg_diff_native:.6f}")
print(f"  RAS (with correction):   {avg_diff_ras:.6f}")

difference_ratio = abs(avg_diff_native - avg_diff_ras) / min(avg_diff_native, avg_diff_ras)

if avg_diff_native < avg_diff_ras:
    winner = "Native"
    print(f"\n🎯 WINNER: Native orientation (no as_closest_canonical)")
    print(f"   Difference is {difference_ratio*100:.1f}% better")
    
    if avg_diff_native < 0.02:
        print("\n✅ RECOMMENDATION: Skip as_closest_canonical() in IXI preprocessing")
        print("   Your BraTS data appears to be in native orientation")
        print("   → Remove nib.as_closest_canonical() from IXI notebook")
        print("   → This saves you from reprocessing BraTS!")
    else:
        print("\n⚠️ WARNING: Difference is still large (>0.02)")
        print("   → Might be due to different slice selection")
        print("   → Visual inspection recommended")
else:
    winner = "RAS"
    print(f"\n🎯 WINNER: RAS orientation (with as_closest_canonical)")
    print(f"   Difference is {difference_ratio*100:.1f}% better")
    
    if avg_diff_ras < 0.02:
        print("\n❌ RECOMMENDATION: Reprocess BraTS with as_closest_canonical()")
        print("   Your BraTS data needs orientation correction")
        print("   → Keep nib.as_closest_canonical() in IXI notebook")
        print("   → Add it to BraTS preprocessing and reprocess")
    else:
        print("\n⚠️ WARNING: Difference is still large (>0.02)")
        print("   → Might be due to different slice selection")
        print("   → Visual inspection recommended")

# Visual comparison
print("\n" + "="*70)
print("VISUAL COMPARISON")
print("="*70)

import matplotlib.pyplot as plt

fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# Row 1: IXI Native
axes[0, 0].imshow(ixi_native, cmap='gray')
axes[0, 0].set_title('IXI Native', fontweight='bold')
axes[0, 0].axis('off')

axes[0, 1].imshow(brats_samples[0], cmap='gray')
axes[0, 1].set_title('BraTS Sample 1', fontweight='bold')
axes[0, 1].axis('off')

axes[0, 2].imshow(np.abs(ixi_native - brats_samples[0]), cmap='hot')
axes[0, 2].set_title(f'Difference\nMean: {diffs_native[0]:.4f}', fontweight='bold')
axes[0, 2].axis('off')

# Row 2: IXI RAS
axes[1, 0].imshow(ixi_ras, cmap='gray')
axes[1, 0].set_title('IXI RAS (canonical)', fontweight='bold')
axes[1, 0].axis('off')

axes[1, 1].imshow(brats_samples[0], cmap='gray')
axes[1, 1].set_title('BraTS Sample 1', fontweight='bold')
axes[1, 1].axis('off')

axes[1, 2].imshow(np.abs(ixi_ras - brats_samples[0]), cmap='hot')
axes[1, 2].set_title(f'Difference\nMean: {diffs_ras[0]:.4f}', fontweight='bold')
axes[1, 2].axis('off')

plt.suptitle(f'Winner: {winner} orientation', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.show()

print("\n📊 Look at the difference maps (red/yellow):")
print("   - Brighter = more difference")
print("   - The row with LESS brightness wins")
print("="*70)
