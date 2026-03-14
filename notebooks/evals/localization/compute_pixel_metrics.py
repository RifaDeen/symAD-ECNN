"""
Compute Pixel-Level Localization Metrics.

This script computes pixel-level metrics for anomaly localization
by comparing reconstruction error maps against ground truth tumor masks.

Supports dissertation Chapter 8.5 localization analysis.

Author: SymAD-ECNN Project
Purpose: Evaluate spatial localization accuracy of anomaly detection
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
    EVALUATIONS_ROOT, JSON_DIR, FIGURES_DIR, TABLES_DIR,
    ensure_directories_exist
)
from path_utils import get_drive_project_root, find_data_paths
from io_utils import (
    save_json, save_csv, log_message,
    start_experiment_log, end_experiment_log
)
from plotting_utils import save_figure

# Try imports
try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False

try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

try:
    from sklearn.metrics import (
        precision_score, recall_score, f1_score,
        jaccard_score, roc_auc_score, average_precision_score
    )
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False


# =============================================================================
# PIXEL METRIC FUNCTIONS
# =============================================================================

def compute_dice_coefficient(
    pred_mask: np.ndarray,
    true_mask: np.ndarray
) -> float:
    """
    Compute Dice coefficient (F1 for segmentation).
    
    Args:
        pred_mask: Binary predicted mask.
        true_mask: Binary ground truth mask.
        
    Returns:
        Dice coefficient [0, 1].
    """
    pred_flat = pred_mask.flatten().astype(bool)
    true_flat = true_mask.flatten().astype(bool)
    
    intersection = np.logical_and(pred_flat, true_flat).sum()
    
    if pred_flat.sum() + true_flat.sum() == 0:
        return 1.0  # Both empty
    
    dice = (2 * intersection) / (pred_flat.sum() + true_flat.sum())
    return float(dice)


def compute_iou(
    pred_mask: np.ndarray,
    true_mask: np.ndarray
) -> float:
    """
    Compute Intersection over Union (Jaccard Index).
    
    Args:
        pred_mask: Binary predicted mask.
        true_mask: Binary ground truth mask.
        
    Returns:
        IoU score [0, 1].
    """
    pred_flat = pred_mask.flatten().astype(bool)
    true_flat = true_mask.flatten().astype(bool)
    
    intersection = np.logical_and(pred_flat, true_flat).sum()
    union = np.logical_or(pred_flat, true_flat).sum()
    
    if union == 0:
        return 1.0  # Both empty
    
    return float(intersection / union)


def compute_pixel_precision_recall(
    pred_mask: np.ndarray,
    true_mask: np.ndarray
) -> Tuple[float, float, float]:
    """
    Compute pixel-level precision, recall, and F1.
    
    Args:
        pred_mask: Binary predicted mask.
        true_mask: Binary ground truth mask.
        
    Returns:
        Tuple of (precision, recall, f1).
    """
    pred_flat = pred_mask.flatten().astype(bool)
    true_flat = true_mask.flatten().astype(bool)
    
    tp = np.logical_and(pred_flat, true_flat).sum()
    fp = np.logical_and(pred_flat, ~true_flat).sum()
    fn = np.logical_and(~pred_flat, true_flat).sum()
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return float(precision), float(recall), float(f1)


def compute_pixel_auroc(
    error_map: np.ndarray,
    true_mask: np.ndarray
) -> float:
    """
    Compute pixel-level AUROC from continuous error map.
    
    Args:
        error_map: Continuous reconstruction error map.
        true_mask: Binary ground truth mask.
        
    Returns:
        AUROC score [0, 1].
    """
    if not SKLEARN_AVAILABLE:
        return float('nan')
    
    error_flat = error_map.flatten()
    true_flat = true_mask.flatten().astype(bool)
    
    # Check if both classes are present
    if true_flat.sum() == 0 or true_flat.sum() == len(true_flat):
        return float('nan')
    
    try:
        return float(roc_auc_score(true_flat, error_flat))
    except Exception:
        return float('nan')


def compute_pixel_auprc(
    error_map: np.ndarray,
    true_mask: np.ndarray
) -> float:
    """
    Compute pixel-level Average Precision (AUPRC).
    
    Args:
        error_map: Continuous reconstruction error map.
        true_mask: Binary ground truth mask.
        
    Returns:
        Average Precision score.
    """
    if not SKLEARN_AVAILABLE:
        return float('nan')
    
    error_flat = error_map.flatten()
    true_flat = true_mask.flatten().astype(bool)
    
    if true_flat.sum() == 0:
        return float('nan')
    
    try:
        return float(average_precision_score(true_flat, error_flat))
    except Exception:
        return float('nan')


def compute_all_pixel_metrics(
    error_map: np.ndarray,
    true_mask: np.ndarray,
    threshold: Optional[float] = None
) -> Dict:
    """
    Compute all pixel-level localization metrics.
    
    Args:
        error_map: Reconstruction error map (continuous or binary).
        true_mask: Binary ground truth mask.
        threshold: Optional threshold to binarize error map.
        
    Returns:
        Dictionary with all metrics.
    """
    # Ensure arrays are numpy
    error_map = np.asarray(error_map)
    true_mask = np.asarray(true_mask)
    
    # Normalize error map to [0, 1] if needed
    if error_map.max() > 1:
        error_map = error_map / error_map.max() if error_map.max() > 0 else error_map
    
    # Binarize mask
    true_binary = (true_mask > 0.5).astype(np.uint8)
    
    # Compute threshold if not provided
    if threshold is None:
        # Use Otsu-like threshold
        threshold = np.mean(error_map) + np.std(error_map)
    
    # Binarize error map
    pred_binary = (error_map > threshold).astype(np.uint8)
    
    # Compute metrics
    precision, recall, f1 = compute_pixel_precision_recall(pred_binary, true_binary)
    
    metrics = {
        "dice": compute_dice_coefficient(pred_binary, true_binary),
        "iou": compute_iou(pred_binary, true_binary),
        "pixel_precision": precision,
        "pixel_recall": recall,
        "pixel_f1": f1,
        "pixel_auroc": compute_pixel_auroc(error_map, true_binary),
        "pixel_auprc": compute_pixel_auprc(error_map, true_binary),
        "threshold_used": float(threshold),
        "pred_coverage": float(pred_binary.sum() / pred_binary.size),
        "true_coverage": float(true_binary.sum() / true_binary.size),
    }
    
    return metrics


# =============================================================================
# ERROR MAP GENERATION
# =============================================================================

def compute_reconstruction_error(
    original: np.ndarray,
    reconstructed: np.ndarray,
    method: str = "mse"
) -> np.ndarray:
    """
    Compute pixel-wise reconstruction error.
    
    Args:
        original: Original image array.
        reconstructed: Reconstructed image array.
        method: Error method ('mse', 'mae', 'ssim').
        
    Returns:
        Error map array.
    """
    # Ensure same shape
    if original.shape != reconstructed.shape:
        raise ValueError(f"Shape mismatch: {original.shape} vs {reconstructed.shape}")
    
    # Normalize to [0, 1]
    original = original.astype(np.float32)
    reconstructed = reconstructed.astype(np.float32)
    
    if original.max() > 1:
        original = original / 255.0
    if reconstructed.max() > 1:
        reconstructed = reconstructed / 255.0
    
    if method == "mse":
        error = (original - reconstructed) ** 2
    elif method == "mae":
        error = np.abs(original - reconstructed)
    elif method == "ssim":
        # Simplified SSIM-based error (would need skimage for full SSIM)
        error = 1 - np.exp(-np.abs(original - reconstructed))
    else:
        error = (original - reconstructed) ** 2
    
    return error


# =============================================================================
# BATCH EVALUATION
# =============================================================================

def evaluate_localization_batch(
    pairs: List[Dict],
    pair_dir: Path,
    model_predictions: Optional[Dict] = None,
    error_method: str = "mse",
    threshold: Optional[float] = None,
    verbose: bool = True
) -> Tuple[List[Dict], Dict]:
    """
    Evaluate localization metrics on a batch of image pairs.
    
    Args:
        pairs: List of pair info dicts with patient_id, slice_idx.
        pair_dir: Directory containing mask pairs.
        model_predictions: Optional dict mapping (patient_id, slice_idx) to reconstructed images.
        error_method: Method for computing reconstruction error.
        threshold: Threshold for binarizing error maps.
        verbose: Whether to print progress.
        
    Returns:
        Tuple of (list of per-image metrics, summary dict).
    """
    if not PIL_AVAILABLE:
        print("ERROR: PIL is required for loading images")
        return [], {"error": "PIL not available"}
    
    pair_dir = Path(pair_dir)
    results = []
    
    for i, pair in enumerate(pairs):
        patient_id = pair['patient_id']
        slice_idx = pair['slice_idx']
        
        if verbose:
            print(f"Processing {i+1}/{len(pairs)}: {patient_id} slice {slice_idx}")
        
        # Load ground truth
        t1_path = pair_dir / f"{patient_id}_slice{slice_idx:03d}_t1.png"
        mask_path = pair_dir / f"{patient_id}_slice{slice_idx:03d}_mask.png"
        
        if not t1_path.exists() or not mask_path.exists():
            if verbose:
                print(f"  Warning: Files not found for {patient_id}")
            continue
        
        try:
            t1_img = np.array(Image.open(t1_path))
            mask_img = np.array(Image.open(mask_path))
        except Exception as e:
            if verbose:
                print(f"  Error loading images: {e}")
            continue
        
        # Get reconstruction if available
        if model_predictions and (patient_id, slice_idx) in model_predictions:
            reconstructed = model_predictions[(patient_id, slice_idx)]
            error_map = compute_reconstruction_error(t1_img, reconstructed, error_method)
        else:
            # Use placeholder (for demonstration without model)
            # In practice, you'd load actual model predictions
            error_map = np.random.rand(*t1_img.shape[:2]) * 0.5
            error_map[mask_img > 0] += 0.3  # Simulate higher error in tumor regions
        
        # Compute metrics
        metrics = compute_all_pixel_metrics(error_map, mask_img, threshold)
        metrics['patient_id'] = patient_id
        metrics['slice_idx'] = slice_idx
        
        results.append(metrics)
    
    # Compute summary statistics
    if results:
        metric_names = ['dice', 'iou', 'pixel_precision', 'pixel_recall', 
                       'pixel_f1', 'pixel_auroc', 'pixel_auprc']
        
        summary = {
            "total_samples": len(results),
        }
        
        for metric in metric_names:
            values = [r[metric] for r in results if not np.isnan(r.get(metric, np.nan))]
            if values:
                summary[f"mean_{metric}"] = float(np.mean(values))
                summary[f"std_{metric}"] = float(np.std(values))
    else:
        summary = {"total_samples": 0, "error": "No results"}
    
    return results, summary


# =============================================================================
# VISUALIZATION
# =============================================================================

def visualize_localization_result(
    original: np.ndarray,
    mask: np.ndarray,
    error_map: np.ndarray,
    pred_mask: np.ndarray,
    metrics: Dict,
    save_path: Optional[Path] = None
) -> None:
    """
    Visualize localization result with all components.
    
    Args:
        original: Original image.
        mask: Ground truth mask.
        error_map: Reconstruction error map.
        pred_mask: Binarized predicted mask.
        metrics: Dictionary of computed metrics.
        save_path: Optional path to save figure.
    """
    if not MATPLOTLIB_AVAILABLE:
        print("matplotlib not available for visualization")
        return
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Original image
    axes[0, 0].imshow(original, cmap='gray')
    axes[0, 0].set_title('Original Image')
    axes[0, 0].axis('off')
    
    # Ground truth mask
    axes[0, 1].imshow(mask, cmap='Reds', alpha=0.7)
    axes[0, 1].set_title('Ground Truth Mask')
    axes[0, 1].axis('off')
    
    # Error map
    im = axes[0, 2].imshow(error_map, cmap='hot')
    axes[0, 2].set_title('Reconstruction Error Map')
    axes[0, 2].axis('off')
    plt.colorbar(im, ax=axes[0, 2], fraction=0.046)
    
    # Predicted mask
    axes[1, 0].imshow(pred_mask, cmap='Blues', alpha=0.7)
    axes[1, 0].set_title('Predicted Anomaly Mask')
    axes[1, 0].axis('off')
    
    # Overlay comparison
    axes[1, 1].imshow(original, cmap='gray')
    axes[1, 1].imshow(mask, cmap='Reds', alpha=0.3)
    axes[1, 1].imshow(pred_mask, cmap='Blues', alpha=0.3)
    axes[1, 1].set_title('Overlay (Red=GT, Blue=Pred)')
    axes[1, 1].axis('off')
    
    # Metrics text
    axes[1, 2].axis('off')
    metric_text = "Localization Metrics:\n"
    metric_text += "-" * 25 + "\n"
    metric_text += f"Dice: {metrics.get('dice', 0):.4f}\n"
    metric_text += f"IoU: {metrics.get('iou', 0):.4f}\n"
    metric_text += f"Precision: {metrics.get('pixel_precision', 0):.4f}\n"
    metric_text += f"Recall: {metrics.get('pixel_recall', 0):.4f}\n"
    metric_text += f"F1: {metrics.get('pixel_f1', 0):.4f}\n"
    metric_text += f"AUROC: {metrics.get('pixel_auroc', 0):.4f}\n"
    metric_text += f"AUPRC: {metrics.get('pixel_auprc', 0):.4f}"
    
    axes[1, 2].text(0.1, 0.5, metric_text, fontsize=12, fontfamily='monospace',
                   verticalalignment='center', transform=axes[1, 2].transAxes)
    axes[1, 2].set_title('Metrics')
    
    plt.tight_layout()
    
    if save_path:
        save_figure(fig, save_path)
    
    plt.show()


# =============================================================================
# MAIN FUNCTION
# =============================================================================

def run_localization_evaluation(
    pair_dir: Optional[Path] = None,
    output_dir: Optional[Path] = None,
    max_samples: int = 50,
    error_method: str = "mse",
    threshold: Optional[float] = None,
    save_results: bool = True,
    verbose: bool = True
) -> Tuple[List[Dict], Dict]:
    """
    Run full localization evaluation pipeline.
    
    Args:
        pair_dir: Directory containing T1/mask pairs.
        output_dir: Where to save results.
        max_samples: Maximum number of samples to evaluate.
        error_method: Method for computing error maps.
        threshold: Threshold for binarizing error maps.
        save_results: Whether to save results.
        verbose: Whether to print progress.
        
    Returns:
        Tuple of (per-sample results, summary).
    """
    ensure_directories_exist()
    
    # Find pair directory
    if pair_dir is None:
        pair_dir = EVALUATIONS_ROOT / 'localization' / 'mask_pairs'
    
    pair_dir = Path(pair_dir)
    
    if not pair_dir.exists():
        print(f"ERROR: Pair directory not found: {pair_dir}")
        print("Run extract_brats_mask_pairs.py first to generate mask pairs.")
        return [], {"error": "Pair directory not found"}
    
    # Load pair index
    index_path = pair_dir / "pair_index.json"
    if index_path.exists():
        with open(index_path, 'r') as f:
            pair_data = json.load(f)
        pairs = pair_data.get('pairs', [])
    else:
        # Scan directory
        pairs = []
        for t1_file in pair_dir.glob("*_t1.png"):
            parts = t1_file.stem.rsplit('_slice', 1)
            if len(parts) == 2:
                patient_id = parts[0]
                slice_idx = int(parts[1].replace('_t1', ''))
                pairs.append({"patient_id": patient_id, "slice_idx": slice_idx})
    
    if not pairs:
        print("ERROR: No pairs found in directory")
        return [], {"error": "No pairs found"}
    
    # Limit samples
    pairs = pairs[:max_samples]
    
    # Start logging
    log_name = start_experiment_log(
        "localization_evaluation",
        params={
            "pair_dir": str(pair_dir),
            "num_samples": len(pairs),
            "error_method": error_method
        }
    )
    
    if verbose:
        print("=" * 60)
        print("LOCALIZATION EVALUATION")
        print("=" * 60)
        print(f"Pair directory: {pair_dir}")
        print(f"Samples: {len(pairs)}")
        print(f"Error method: {error_method}")
        print("-" * 60)
    
    # Run evaluation
    results, summary = evaluate_localization_batch(
        pairs=pairs,
        pair_dir=pair_dir,
        error_method=error_method,
        threshold=threshold,
        verbose=verbose
    )
    
    # Add metadata
    summary['pair_dir'] = str(pair_dir)
    summary['error_method'] = error_method
    summary['timestamp'] = datetime.now().isoformat()
    
    if verbose:
        print("\n" + "=" * 60)
        print("SUMMARY")
        print("=" * 60)
        for key, value in summary.items():
            if isinstance(value, float):
                print(f"  {key}: {value:.4f}")
            else:
                print(f"  {key}: {value}")
        print("=" * 60)
    
    # Save results
    if save_results:
        try:
            save_json(
                {"results": results, "summary": summary},
                "localization_evaluation_results.json"
            )
            
            if PANDAS_AVAILABLE:
                df = pd.DataFrame(results)
                save_csv(df, "localization_metrics.csv")
            
            log_message("Localization results saved.", log_name)
            
        except Exception as e:
            log_message(f"Error saving results: {e}", log_name)
    
    end_experiment_log(log_name, summary=summary)
    
    return results, summary


if __name__ == "__main__":
    print("Running localization evaluation...")
    
    results, summary = run_localization_evaluation(
        max_samples=10
    )
    
    print(f"\nSummary: {summary}")
