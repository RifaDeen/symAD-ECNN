"""
Extract BraTS Mask Pairs for Localization Evaluation.

This script extracts paired tumor mask and T1 image slices from
BraTS 2021 dataset for pixel-level anomaly localization evaluation.

Supports dissertation Chapter 8.5 localization analysis.

Author: SymAD-ECNN Project
Purpose: Prepare ground truth masks for localization metrics
"""

import sys
from pathlib import Path
from typing import Dict, Optional, List, Tuple, Union
import json
from datetime import datetime

import numpy as np

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import (
    EVALUATIONS_ROOT, JSON_DIR, FIGURES_DIR,
    ensure_directories_exist
)
from path_utils import (
    get_drive_project_root, find_data_paths
)
from io_utils import (
    save_json, log_message,
    start_experiment_log, end_experiment_log
)

# Try to import nibabel for NIfTI files
try:
    import nibabel as nib
    NIBABEL_AVAILABLE = True
except ImportError:
    NIBABEL_AVAILABLE = False
    print("Warning: nibabel not available. Install with: pip install nibabel")

# Try to import PIL for image saving
try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False


# =============================================================================
# BRATS DATA STRUCTURES
# =============================================================================

# BraTS 2021 segmentation labels
BRATS_LABELS = {
    0: "background",
    1: "necrotic_core",  # NCR - Necrotic tumor core
    2: "edema",          # ED - Peritumoral edema
    4: "enhancing_tumor" # ET - GD-enhancing tumor
}

# Combined tumor regions
BRATS_REGIONS = {
    "whole_tumor": [1, 2, 4],     # WT = NCR + ED + ET
    "tumor_core": [1, 4],          # TC = NCR + ET
    "enhancing_tumor": [4],        # ET only
}


# =============================================================================
# MASK EXTRACTION FUNCTIONS
# =============================================================================

def find_brats_volumes(
    brats_dir: Path,
    required_files: Optional[List[str]] = None
) -> List[Dict[str, Path]]:
    """
    Find all BraTS patient volumes with required modalities.
    
    Args:
        brats_dir: Root directory of BraTS dataset.
        required_files: List of required file suffixes (e.g., ['t1.nii.gz', 'seg.nii.gz']).
        
    Returns:
        List of dictionaries with paths to each modality.
    """
    if required_files is None:
        required_files = ['t1.nii.gz', 'seg.nii.gz']
    
    brats_dir = Path(brats_dir)
    volumes = []
    
    # Find patient directories (BraTS2021_XXXXX)
    for patient_dir in sorted(brats_dir.iterdir()):
        if not patient_dir.is_dir():
            continue
        if not patient_dir.name.startswith('BraTS'):
            continue
        
        patient_id = patient_dir.name
        patient_files = {}
        
        # Look for required files
        all_found = True
        for suffix in required_files:
            # Try common naming patterns
            patterns = [
                f"{patient_id}_{suffix}",
                f"*_{suffix}",
                f"*{suffix}"
            ]
            
            found_file = None
            for pattern in patterns:
                matches = list(patient_dir.glob(pattern))
                if matches:
                    found_file = matches[0]
                    break
            
            if found_file:
                # Map suffix to key
                key = suffix.replace('.nii.gz', '').replace('.nii', '')
                patient_files[key] = found_file
            else:
                all_found = False
                break
        
        if all_found and patient_files:
            patient_files['patient_id'] = patient_id
            patient_files['patient_dir'] = patient_dir
            volumes.append(patient_files)
    
    return volumes


def load_nifti_volume(filepath: Path) -> Tuple[np.ndarray, object]:
    """
    Load a NIfTI volume.
    
    Args:
        filepath: Path to NIfTI file.
        
    Returns:
        Tuple of (data array, affine matrix).
    """
    if not NIBABEL_AVAILABLE:
        raise ImportError("nibabel is required for NIfTI loading")
    
    img = nib.load(str(filepath))
    data = img.get_fdata()
    affine = img.affine
    
    return data, affine


def extract_tumor_mask(
    seg_data: np.ndarray,
    region: str = "whole_tumor"
) -> np.ndarray:
    """
    Extract binary tumor mask from segmentation volume.
    
    Args:
        seg_data: Segmentation volume with BraTS labels.
        region: Which tumor region to extract.
        
    Returns:
        Binary mask array.
    """
    if region not in BRATS_REGIONS:
        print(f"Warning: Unknown region '{region}', using whole_tumor")
        region = "whole_tumor"
    
    labels = BRATS_REGIONS[region]
    mask = np.isin(seg_data, labels).astype(np.uint8)
    
    return mask


def find_tumor_slices(
    mask_volume: np.ndarray,
    axis: int = 2,
    min_tumor_ratio: float = 0.01
) -> List[int]:
    """
    Find slice indices that contain tumor.
    
    Args:
        mask_volume: Binary tumor mask volume.
        axis: Axis along which to find slices (2 = axial).
        min_tumor_ratio: Minimum ratio of tumor pixels to count as tumor slice.
        
    Returns:
        List of slice indices with tumor.
    """
    tumor_slices = []
    
    for i in range(mask_volume.shape[axis]):
        if axis == 0:
            slice_mask = mask_volume[i, :, :]
        elif axis == 1:
            slice_mask = mask_volume[:, i, :]
        else:
            slice_mask = mask_volume[:, :, i]
        
        tumor_ratio = np.sum(slice_mask) / slice_mask.size
        if tumor_ratio >= min_tumor_ratio:
            tumor_slices.append(i)
    
    return tumor_slices


def extract_slice_pairs(
    t1_volume: np.ndarray,
    mask_volume: np.ndarray,
    slice_indices: List[int],
    axis: int = 2
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Extract paired T1 and mask slices.
    
    Args:
        t1_volume: T1-weighted image volume.
        mask_volume: Binary tumor mask volume.
        slice_indices: Indices of slices to extract.
        axis: Axis along which to slice.
        
    Returns:
        List of (t1_slice, mask_slice) tuples.
    """
    pairs = []
    
    for idx in slice_indices:
        if axis == 0:
            t1_slice = t1_volume[idx, :, :]
            mask_slice = mask_volume[idx, :, :]
        elif axis == 1:
            t1_slice = t1_volume[:, idx, :]
            mask_slice = mask_volume[:, idx, :]
        else:
            t1_slice = t1_volume[:, :, idx]
            mask_slice = mask_volume[:, :, idx]
        
        pairs.append((t1_slice.copy(), mask_slice.copy()))
    
    return pairs


def normalize_slice(slice_data: np.ndarray) -> np.ndarray:
    """
    Normalize a slice to 0-255 range.
    
    Args:
        slice_data: Input slice data.
        
    Returns:
        Normalized uint8 array.
    """
    # Remove extreme outliers
    p1, p99 = np.percentile(slice_data, [1, 99])
    clipped = np.clip(slice_data, p1, p99)
    
    # Normalize to 0-255
    if clipped.max() > clipped.min():
        normalized = (clipped - clipped.min()) / (clipped.max() - clipped.min())
    else:
        normalized = np.zeros_like(clipped)
    
    return (normalized * 255).astype(np.uint8)


# =============================================================================
# MAIN EXTRACTION FUNCTIONS
# =============================================================================

def extract_brats_mask_pairs(
    brats_dir: Optional[Path] = None,
    output_dir: Optional[Path] = None,
    region: str = "whole_tumor",
    max_patients: int = 10,
    slices_per_patient: int = 5,
    save_images: bool = True,
    verbose: bool = True
) -> Dict:
    """
    Extract paired T1 and mask slices from BraTS dataset.
    
    Args:
        brats_dir: Path to BraTS data directory.
        output_dir: Where to save extracted pairs.
        region: Tumor region to extract masks for.
        max_patients: Maximum number of patients to process.
        slices_per_patient: Number of slices per patient.
        save_images: Whether to save as PNG images.
        verbose: Whether to print progress.
        
    Returns:
        Dictionary with extraction summary and file paths.
    """
    if not NIBABEL_AVAILABLE:
        print("ERROR: nibabel is required for BraTS mask extraction.")
        print("Install with: pip install nibabel")
        return {"error": "nibabel not available"}
    
    ensure_directories_exist()
    
    # Find BraTS directory
    if brats_dir is None:
        data_paths = find_data_paths()
        if data_paths.get('brats_raw'):
            brats_dir = Path(data_paths['brats_raw'])
        else:
            # Try common locations
            for candidate in [
                Path('/content/drive/MyDrive/symAD-ECNN/data/brats2021'),
                Path('/content/drive/MyDrive/data/brats2021'),
                get_drive_project_root() / 'data' / 'brats2021'
            ]:
                if candidate.exists():
                    brats_dir = candidate
                    break
    
    if brats_dir is None or not Path(brats_dir).exists():
        print(f"ERROR: BraTS directory not found: {brats_dir}")
        return {"error": "BraTS directory not found"}
    
    brats_dir = Path(brats_dir)
    
    # Set output directory
    if output_dir is None:
        output_dir = EVALUATIONS_ROOT / 'localization' / 'mask_pairs'
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Start logging
    log_name = start_experiment_log(
        "extract_brats_masks",
        params={
            "brats_dir": str(brats_dir),
            "region": region,
            "max_patients": max_patients
        }
    )
    
    if verbose:
        print("=" * 60)
        print("EXTRACTING BRATS MASK PAIRS")
        print("=" * 60)
        print(f"BraTS directory: {brats_dir}")
        print(f"Output directory: {output_dir}")
        print(f"Region: {region}")
        print("-" * 60)
    
    # Find volumes
    volumes = find_brats_volumes(brats_dir)
    
    if not volumes:
        print("ERROR: No BraTS volumes found.")
        return {"error": "No volumes found"}
    
    if verbose:
        print(f"Found {len(volumes)} patient volumes")
    
    # Select subset
    selected_volumes = volumes[:max_patients]
    
    # Extract pairs
    all_pairs = []
    extraction_summary = []
    
    for vol_info in selected_volumes:
        patient_id = vol_info['patient_id']
        
        if verbose:
            print(f"\nProcessing: {patient_id}")
        
        try:
            # Load volumes
            t1_data, _ = load_nifti_volume(vol_info['t1'])
            seg_data, _ = load_nifti_volume(vol_info['seg'])
            
            # Extract tumor mask
            mask_data = extract_tumor_mask(seg_data, region)
            
            # Find tumor slices
            tumor_slices = find_tumor_slices(mask_data)
            
            if not tumor_slices:
                if verbose:
                    print(f"  No tumor slices found for {patient_id}")
                continue
            
            # Select evenly spaced slices
            if len(tumor_slices) > slices_per_patient:
                indices = np.linspace(0, len(tumor_slices)-1, slices_per_patient, dtype=int)
                selected_slices = [tumor_slices[i] for i in indices]
            else:
                selected_slices = tumor_slices
            
            # Extract pairs
            pairs = extract_slice_pairs(t1_data, mask_data, selected_slices)
            
            patient_files = []
            
            for i, (t1_slice, mask_slice) in enumerate(pairs):
                slice_idx = selected_slices[i]
                
                # Normalize T1
                t1_normalized = normalize_slice(t1_slice)
                mask_uint8 = (mask_slice * 255).astype(np.uint8)
                
                if save_images and PIL_AVAILABLE:
                    # Save T1 image
                    t1_filename = f"{patient_id}_slice{slice_idx:03d}_t1.png"
                    t1_path = output_dir / t1_filename
                    Image.fromarray(t1_normalized).save(t1_path)
                    
                    # Save mask image
                    mask_filename = f"{patient_id}_slice{slice_idx:03d}_mask.png"
                    mask_path = output_dir / mask_filename
                    Image.fromarray(mask_uint8).save(mask_path)
                    
                    patient_files.append({
                        "t1": str(t1_path),
                        "mask": str(mask_path),
                        "slice_idx": slice_idx
                    })
                
                all_pairs.append({
                    "patient_id": patient_id,
                    "slice_idx": slice_idx,
                    "tumor_ratio": np.mean(mask_slice)
                })
            
            extraction_summary.append({
                "patient_id": patient_id,
                "total_tumor_slices": len(tumor_slices),
                "extracted_slices": len(pairs),
                "files": patient_files
            })
            
            if verbose:
                print(f"  Extracted {len(pairs)} slices")
            
        except Exception as e:
            if verbose:
                print(f"  Error processing {patient_id}: {e}")
            log_message(f"Error with {patient_id}: {e}", log_name)
    
    # Summary
    summary = {
        "total_patients": len(extraction_summary),
        "total_pairs": len(all_pairs),
        "region": region,
        "output_dir": str(output_dir),
        "brats_dir": str(brats_dir),
        "timestamp": datetime.now().isoformat(),
        "patients": extraction_summary
    }
    
    # Save summary
    save_json(summary, "brats_mask_extraction_summary.json")
    
    # Save pair index
    pair_index = {
        "pairs": all_pairs,
        "region": region,
        "output_dir": str(output_dir)
    }
    pair_index_path = output_dir / "pair_index.json"
    with open(pair_index_path, 'w') as f:
        json.dump(pair_index, f, indent=2)
    
    if verbose:
        print("\n" + "=" * 60)
        print("EXTRACTION COMPLETE")
        print("=" * 60)
        print(f"Patients processed: {summary['total_patients']}")
        print(f"Total pairs extracted: {summary['total_pairs']}")
        print(f"Output directory: {output_dir}")
        print("=" * 60)
    
    end_experiment_log(log_name, summary=summary)
    
    return summary


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def load_mask_pair(
    pair_dir: Path,
    patient_id: str,
    slice_idx: int
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Load a saved T1/mask pair.
    
    Args:
        pair_dir: Directory containing pair images.
        patient_id: Patient identifier.
        slice_idx: Slice index.
        
    Returns:
        Tuple of (t1_array, mask_array) or (None, None) if not found.
    """
    pair_dir = Path(pair_dir)
    
    t1_path = pair_dir / f"{patient_id}_slice{slice_idx:03d}_t1.png"
    mask_path = pair_dir / f"{patient_id}_slice{slice_idx:03d}_mask.png"
    
    if not t1_path.exists() or not mask_path.exists():
        return None, None
    
    try:
        t1_img = Image.open(t1_path)
        mask_img = Image.open(mask_path)
        
        return np.array(t1_img), np.array(mask_img)
    except Exception as e:
        print(f"Error loading pair: {e}")
        return None, None


def get_all_mask_pairs(pair_dir: Path) -> List[Dict]:
    """
    Get list of all available mask pairs.
    
    Args:
        pair_dir: Directory containing pair images.
        
    Returns:
        List of pair info dictionaries.
    """
    pair_dir = Path(pair_dir)
    index_path = pair_dir / "pair_index.json"
    
    if index_path.exists():
        with open(index_path, 'r') as f:
            index = json.load(f)
        return index.get('pairs', [])
    
    # Fallback: scan directory
    pairs = []
    for t1_path in pair_dir.glob("*_t1.png"):
        parts = t1_path.stem.rsplit('_', 2)
        if len(parts) >= 2:
            patient_id = parts[0]
            slice_str = parts[1].replace('slice', '')
            try:
                slice_idx = int(slice_str)
                pairs.append({
                    "patient_id": patient_id,
                    "slice_idx": slice_idx,
                    "t1_path": str(t1_path),
                    "mask_path": str(t1_path).replace('_t1.png', '_mask.png')
                })
            except ValueError:
                continue
    
    return pairs


if __name__ == "__main__":
    print("Extracting BraTS mask pairs...")
    
    summary = extract_brats_mask_pairs(
        max_patients=5,
        slices_per_patient=3
    )
    
    print(f"\nExtraction summary: {summary}")
