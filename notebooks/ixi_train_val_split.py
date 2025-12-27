# IXI Data Train/Val Split Script
# Run this in Google Colab to split your preprocessed IXI data into 90/10 train/val

"""
This script splits your preprocessed IXI data (all in one folder) into:
- 90% training set
- 10% validation set

Run this ONCE before training your models.
"""

from google.colab import drive
import os
import shutil
import random
from pathlib import Path

# Mount Google Drive
print("Mounting Google Drive...")
drive.mount('/content/drive')

# ============================================
# CONFIGURATION - Update these paths!
# ============================================

# Path to your current resized IXI data (all files in one folder)
SOURCE_FOLDER = '/content/drive/MyDrive/[YOUR_FOLDER]/ixi_resized'  # UPDATE THIS!

# Destination folder for split data
DEST_BASE = '/content/drive/MyDrive/symAD-ECNN/data/processed_ixi'

# Split ratio
TRAIN_RATIO = 0.9  # 90% train, 10% validation

# ============================================
# CREATE DIRECTORY STRUCTURE
# ============================================

print("\n" + "="*50)
print("IXI Data Train/Val Split")
print("="*50 + "\n")

# Create destination folders
train_dir = os.path.join(DEST_BASE, 'train')
val_dir = os.path.join(DEST_BASE, 'val')

os.makedirs(train_dir, exist_ok=True)
os.makedirs(val_dir, exist_ok=True)

print(f"Created folders:")
print(f"  Train: {train_dir}")
print(f"  Val: {val_dir}")

# ============================================
# CHECK SOURCE DATA
# ============================================

print(f"\nChecking source data: {SOURCE_FOLDER}")

if not os.path.exists(SOURCE_FOLDER):
    print(f"\n❌ ERROR: Source folder not found!")
    print(f"Please update SOURCE_FOLDER path at top of script.")
    print(f"\nCurrent path: {SOURCE_FOLDER}")
    exit()

# Get all .npy files
all_files = [f for f in os.listdir(SOURCE_FOLDER) if f.endswith('.npy')]
total_files = len(all_files)

if total_files == 0:
    print(f"\n❌ ERROR: No .npy files found in source folder!")
    exit()

print(f"✓ Found {total_files} .npy files")

# ============================================
# SPLIT DATA
# ============================================

print("\nSplitting data...")

# Shuffle files for random split
random.seed(42)  # For reproducibility
random.shuffle(all_files)

# Calculate split point
split_idx = int(total_files * TRAIN_RATIO)

train_files = all_files[:split_idx]
val_files = all_files[split_idx:]

print(f"  Train: {len(train_files)} files ({len(train_files)/total_files*100:.1f}%)")
print(f"  Val: {len(val_files)} files ({len(val_files)/total_files*100:.1f}%)")

# ============================================
# COPY OR MOVE FILES
# ============================================

print("\nChoose operation:")
print("  1. COPY files (keeps original, safer)")
print("  2. MOVE files (deletes from original, saves space)")

operation = input("Enter choice (1/2): ").strip()

if operation == "1":
    print("\nCopying files to train/val folders...")
    copy_func = shutil.copy2
    action = "Copying"
elif operation == "2":
    print("\nMoving files to train/val folders...")
    copy_func = shutil.move
    action = "Moving"
else:
    print("Invalid choice. Defaulting to COPY.")
    copy_func = shutil.copy2
    action = "Copying"

# Copy/Move training files
print(f"\n{action} training files...")
for i, filename in enumerate(train_files):
    src = os.path.join(SOURCE_FOLDER, filename)
    dst = os.path.join(train_dir, filename)
    copy_func(src, dst)
    
    if (i + 1) % 1000 == 0:
        print(f"  Progress: {i + 1}/{len(train_files)} files...")

print(f"✓ {len(train_files)} files in train folder")

# Copy/Move validation files
print(f"\n{action} validation files...")
for i, filename in enumerate(val_files):
    src = os.path.join(SOURCE_FOLDER, filename)
    dst = os.path.join(val_dir, filename)
    copy_func(src, dst)
    
    if (i + 1) % 100 == 0:
        print(f"  Progress: {i + 1}/{len(val_files)} files...")

print(f"✓ {len(val_files)} files in val folder")

# ============================================
# VERIFY SPLIT
# ============================================

print("\n" + "="*50)
print("Verification")
print("="*50 + "\n")

# Count files in each folder
train_count = len([f for f in os.listdir(train_dir) if f.endswith('.npy')])
val_count = len([f for f in os.listdir(val_dir) if f.endswith('.npy')])
total_split = train_count + val_count

print(f"Files in train folder: {train_count}")
print(f"Files in val folder: {val_count}")
print(f"Total after split: {total_split}")
print(f"Original total: {total_files}")

if operation == "2":
    # Check if source folder is now empty (for MOVE operation)
    remaining = len([f for f in os.listdir(SOURCE_FOLDER) if f.endswith('.npy')])
    print(f"Files remaining in source: {remaining}")

# Check if numbers match
if total_split == total_files:
    print("\n✅ SUCCESS! Data split completed successfully.")
else:
    print(f"\n⚠️ WARNING: File count mismatch!")
    print(f"  Expected: {total_files}")
    print(f"  Got: {total_split}")

# ============================================
# SAMPLE CHECK
# ============================================

print("\n" + "="*50)
print("Sample Check")
print("="*50 + "\n")

# Load one sample from each set to verify
import numpy as np

try:
    # Check train sample
    train_sample = os.path.join(train_dir, os.listdir(train_dir)[0])
    train_data = np.load(train_sample)
    print(f"✓ Train sample shape: {train_data.shape}")
    
    # Check val sample
    val_sample = os.path.join(val_dir, os.listdir(val_dir)[0])
    val_data = np.load(val_sample)
    print(f"✓ Val sample shape: {val_data.shape}")
    
    print("\n✅ All samples loaded successfully!")
    
except Exception as e:
    print(f"\n❌ ERROR loading samples: {e}")

# ============================================
# SUMMARY
# ============================================

print("\n" + "="*50)
print("SUMMARY")
print("="*50 + "\n")

print(f"✓ Data split complete!")
print(f"✓ Training set: {train_count} samples ({train_count/total_files*100:.1f}%)")
print(f"✓ Validation set: {val_count} samples ({val_count/total_files*100:.1f}%)")
print(f"\nData ready for training at:")
print(f"  {DEST_BASE}/train/")
print(f"  {DEST_BASE}/val/")

# ============================================
# NEXT STEPS
# ============================================

print("\n" + "="*50)
print("NEXT STEPS")
print("="*50 + "\n")

print("1. Update your training notebooks with these paths:")
print(f"   IXI_TRAIN_PATH = '{train_dir}'")
print(f"   IXI_VAL_PATH = '{val_dir}'")
print("\n2. Start training your models!")
print("\n3. (Optional) If you used MOVE, you can delete the empty source folder")

print("\n✅ Ready to train!")
