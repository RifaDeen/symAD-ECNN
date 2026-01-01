"""
Count exact number of filtered slices per BraTS patient.
Run locally to generate patient_slice_counts.txt for Colab filtering.
"""

import os
import numpy as np
import nibabel as nib
from tqdm import tqdm
import json

# Paths
BRATS_FOLDER = r"c:\Users\rifad\symAD-ECNN\data\brats2021"
OUTPUT_FILE = r"c:\Users\rifad\symAD-ECNN\data\patient_slice_counts.json"

def is_valid_slice(slice_array, min_nonzero_ratio=0.12, min_mean=0.1):
    """Check if slice is valid (same as preprocessing)"""
    nonzero_ratio = np.count_nonzero(slice_array) / slice_array.size
    if nonzero_ratio < min_nonzero_ratio:
        return False
    
    if slice_array.max() - slice_array.min() < 1e-6:
        return False
    
    normalized = (slice_array - slice_array.min()) / (slice_array.max() - slice_array.min())
    if normalized.mean() < min_mean:
        return False
    
    return True

print("🔍 Counting valid slices per BraTS patient...")
print(f"Source: {BRATS_FOLDER}\n")

# Get patient folders
patient_folders = sorted([d for d in os.listdir(BRATS_FOLDER) 
                         if os.path.isdir(os.path.join(BRATS_FOLDER, d)) 
                         and d.startswith('BraTS2021_')])

print(f"Found {len(patient_folders)} patient folders\n")

# Count slices per patient
patient_slice_counts = []
patient_names = []
total_slices = 0

for patient_folder in tqdm(patient_folders, desc="Processing patients"):
    patient_path = os.path.join(BRATS_FOLDER, patient_folder)
    
    # Find T1 file (not T1CE)
    t1_files = [f for f in os.listdir(patient_path) 
                if f.endswith('_t1.nii.gz') and not f.endswith('_t1ce.nii.gz')]
    
    if not t1_files:
        print(f"⚠️  No T1 file found for {patient_folder}")
        continue
    
    t1_file = os.path.join(patient_path, t1_files[0])
    
    try:
        # Load volume
        nii_img = nib.load(t1_file)
        vol = nii_img.get_fdata()
        Z = vol.shape[2]
        
        # Count valid slices (matching preprocessing logic)
        valid_count = 0
        for s in range(Z):
            slice_2d = vol[:, :, s]
            if is_valid_slice(slice_2d, min_nonzero_ratio=0.12, min_mean=0.1):
                valid_count += 1
        
        patient_slice_counts.append(valid_count)
        patient_names.append(patient_folder)
        total_slices += valid_count
        
    except Exception as e:
        print(f"❌ Error processing {patient_folder}: {e}")
        continue

# Statistics
patient_slice_counts_np = np.array(patient_slice_counts)
print(f"\n" + "="*60)
print("PATIENT SLICE COUNT SUMMARY")
print("="*60)
print(f"Patients processed: {len(patient_slice_counts)}")
print(f"Total valid slices: {total_slices:,}")
print(f"Avg slices/patient: {patient_slice_counts_np.mean():.1f} ± {patient_slice_counts_np.std():.1f}")
print(f"Min slices/patient: {patient_slice_counts_np.min()}")
print(f"Max slices/patient: {patient_slice_counts_np.max()}")
print("="*60)

# Save to JSON
output_data = {
    'patient_names': patient_names,
    'slice_counts': patient_slice_counts,
    'total_patients': len(patient_slice_counts),
    'total_slices': total_slices,
    'mean_slices': float(patient_slice_counts_np.mean()),
    'std_slices': float(patient_slice_counts_np.std())
}

with open(OUTPUT_FILE, 'w') as f:
    json.dump(output_data, f, indent=2)

print(f"\n✅ Patient slice counts saved to: {OUTPUT_FILE}")
print(f"\n📤 Upload this file to Google Drive at:")
print(f"   MyDrive/symAD-ECNN/data/patient_slice_counts.json")
print(f"\nThen use it in Colab for accurate patient-based filtering!")
