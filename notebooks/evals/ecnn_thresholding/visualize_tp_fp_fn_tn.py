"""
Visualize TP/FP/FN/TN Samples for ECNN Evaluation.

This script creates visualization panels showing examples of:
- True Positives (correctly detected anomalies)
- False Positives (normal samples incorrectly flagged as anomalies)
- False Negatives (missed anomalies)
- True Negatives (correctly identified normal samples)

Supports dissertation section 8.5 (Further Evaluations) for qualitative analysis.

Author: SymAD-ECNN Project
Purpose: Generate TP/FP/FN/TN visualization panels for error analysis
"""

import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy import ndimage

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import (
    FIGURES_DIR, ECNN_DEFAULT_ERROR_MODE, ECNN_DEFAULT_MIN_BRAIN_PIXELS,
    DEFAULT_FIGURE_DPI, ensure_directories_exist
)
from path_utils import find_data_paths
from metrics_utils import compute_score
from plotting_utils import plot_tp_fp_fn_tn_panels, save_figure
from io_utils import save_json, log_message
from ecnn_model_loader import get_model_for_inference, compute_error_maps


# =============================================================================
# SAMPLE COLLECTION
# =============================================================================

def collect_classified_samples(
    model: torch.nn.Module,
    normal_images: List[np.ndarray],
    anomaly_images: List[np.ndarray],
    threshold: float,
    score_method: str = "mean",
    error_mode: str = "abs",
    device: str = "cuda",
    max_samples: int = 20
) -> Dict[str, List[Dict]]:
    """
    Classify samples and collect examples of TP, FP, FN, TN.
    
    Args:
        model: The ECNN model.
        normal_images: List of normal sample images (numpy arrays).
        anomaly_images: List of anomaly sample images (numpy arrays).
        threshold: Classification threshold.
        score_method: Scoring method to use.
        error_mode: Error computation mode.
        device: Device for inference.
        max_samples: Maximum samples to collect per category.
        
    Returns:
        Dictionary with keys 'tp', 'fp', 'fn', 'tn', each containing sample dicts.
    """
    model.eval()
    
    samples = {
        "tp": [],  # True Positive: anomaly correctly detected
        "fp": [],  # False Positive: normal incorrectly flagged
        "fn": [],  # False Negative: anomaly missed
        "tn": [],  # True Negative: normal correctly identified
    }
    
    def process_batch(images, is_anomaly):
        category_pos = "tp" if is_anomaly else "fp"  # When predicted positive
        category_neg = "fn" if is_anomaly else "tn"  # When predicted negative
        
        for img in images:
            # Prepare input
            if img.ndim == 2:
                x = torch.from_numpy(img[np.newaxis, np.newaxis, :, :]).float().to(device)
            else:
                x = torch.from_numpy(img[np.newaxis, :, :, :]).float().to(device)
            
            # Get reconstruction and error
            with torch.no_grad():
                recon, error_map = compute_error_maps(model, x, error_mode)
            
            # Convert to numpy
            input_np = x.squeeze().cpu().numpy()
            recon_np = recon.squeeze().cpu().numpy()
            error_np = error_map.squeeze().cpu().numpy()
            
            # Create brain mask
            brain_mask = input_np > 0.01
            
            if brain_mask.sum() < ECNN_DEFAULT_MIN_BRAIN_PIXELS:
                continue
            
            # Compute score
            score = compute_score(error_np, brain_mask, method=score_method)
            
            # Classify
            predicted_anomaly = score >= threshold
            
            sample_data = {
                "input": input_np,
                "reconstruction": recon_np,
                "error_map": error_np,
                "smoothed_error_map": ndimage.gaussian_filter(error_np, sigma=2.0).astype(np.float32),
                "score": score,
                "threshold": threshold,
                "is_anomaly": is_anomaly,
                "predicted_anomaly": predicted_anomaly,
            }
            
            if predicted_anomaly:
                if len(samples[category_pos]) < max_samples:
                    samples[category_pos].append(sample_data)
            else:
                if len(samples[category_neg]) < max_samples:
                    samples[category_neg].append(sample_data)
            
            # Early stop if we have enough samples
            all_full = all(len(samples[k]) >= max_samples for k in samples)
            if all_full:
                return True
        
        return False
    
    # Process normal samples
    print("Processing normal samples...")
    process_batch(normal_images, is_anomaly=False)
    
    # Process anomaly samples
    print("Processing anomaly samples...")
    process_batch(anomaly_images, is_anomaly=True)
    
    # Report counts
    for category, sample_list in samples.items():
        print(f"  {category.upper()}: {len(sample_list)} samples")
    
    return samples


def load_images_from_folder(
    folder_path: Path,
    max_images: int = 100,
    extensions: Tuple[str, ...] = ('.png', '.jpg', '.jpeg', '.npy')
) -> List[np.ndarray]:
    """
    Load images from a folder.
    
    Args:
        folder_path: Path to image folder.
        max_images: Maximum images to load.
        extensions: Valid extensions.
        
    Returns:
        List of image arrays.
    """
    images = []
    folder_path = Path(folder_path)
    
    if not folder_path.exists():
        print(f"Warning: Folder not found: {folder_path}")
        return images
    
    # Collect image paths
    image_paths = []
    for ext in extensions:
        image_paths.extend(list(folder_path.glob(f"*{ext}")))
        image_paths.extend(list(folder_path.glob(f"**/*{ext}")))
    
    image_paths = sorted(list(set(image_paths)))[:max_images]
    
    for path in tqdm(image_paths, desc="Loading images"):
        try:
            if path.suffix == '.npy':
                img = np.load(path)
            else:
                img = np.array(Image.open(path).convert('L'))
            
            # Normalize
            if img.max() > 1:
                img = img.astype(np.float32) / 255.0
            
            images.append(img)
        except Exception as e:
            print(f"Warning: Could not load {path}: {e}")
    
    return images


# =============================================================================
# VISUALIZATION
# =============================================================================

def create_sample_panel(
    samples: List[Dict],
    category: str,
    n_rows: int = 4,
    save_path: Optional[Path] = None,
    smooth_sigma: float = 2.0,
) -> plt.Figure:
    """
    Create a panel figure for one category of samples.
    
    Args:
        samples: List of sample dictionaries.
        category: Category name (tp, fp, fn, tn).
        n_rows: Number of rows to display.
        save_path: Optional path to save figure.
        
    Returns:
        Matplotlib figure.
    """
    category_titles = {
        "tp": "True Positives - Correctly Detected Anomalies",
        "fp": "False Positives - Normal Samples Incorrectly Flagged",
        "fn": "False Negatives - Missed Anomalies",
        "tn": "True Negatives - Correctly Identified Normal Samples",
    }
    
    category_colors = {
        "tp": "green",
        "fp": "orange",
        "fn": "red",
        "tn": "blue",
    }
    
    title = category_titles.get(category, f"Category: {category}")
    color = category_colors.get(category, "black")
    
    n_samples = min(len(samples), n_rows)
    
    if n_samples == 0:
        fig, ax = plt.subplots(figsize=(12, 2))
        ax.text(0.5, 0.5, f"No {category.upper()} samples available",
                ha='center', va='center', fontsize=14)
        ax.axis('off')
        fig.suptitle(title, fontsize=14, color=color)
        
        if save_path:
            fig.savefig(save_path, dpi=DEFAULT_FIGURE_DPI, bbox_inches='tight')
            print(f"Figure saved: {save_path}")
        
        return fig
    
    fig, axes = plt.subplots(n_samples, 4, figsize=(16, 4 * n_samples))
    
    if n_samples == 1:
        axes = axes.reshape(1, -1)
    
    fig.suptitle(title, fontsize=14, color=color, y=1.02)
    
    for i, sample in enumerate(samples[:n_samples]):
        input_img = sample["input"]
        recon = sample["reconstruction"]
        error = sample["error_map"]
        smooth_error = sample.get("smoothed_error_map")
        if smooth_error is None:
            smooth_error = ndimage.gaussian_filter(error, sigma=smooth_sigma).astype(np.float32)
        score = sample["score"]
        
        # Input image
        axes[i, 0].imshow(input_img, cmap='gray')
        axes[i, 0].set_title(f"Input (Score: {score:.4f})" if i == 0 else f"Score: {score:.4f}",
                            fontsize=10)
        axes[i, 0].axis('off')
        
        # Reconstruction
        axes[i, 1].imshow(recon, cmap='gray')
        if i == 0:
            axes[i, 1].set_title("Reconstruction", fontsize=10)
        axes[i, 1].axis('off')
        
        # Error map
        im = axes[i, 2].imshow(error, cmap='hot')
        if i == 0:
            axes[i, 2].set_title("Error Map", fontsize=10)
        axes[i, 2].axis('off')
        
        # Overlay
        axes[i, 3].imshow(input_img, cmap='gray')
        if smooth_error.max() > 0:
            smooth_normalized = smooth_error / smooth_error.max()
            axes[i, 3].imshow(smooth_normalized, cmap='jet', alpha=0.5)
        if i == 0:
            axes[i, 3].set_title(f"Smoothed Heatmap Overlay (σ={smooth_sigma:g})", fontsize=10)
        axes[i, 3].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=DEFAULT_FIGURE_DPI, bbox_inches='tight', facecolor='white')
        print(f"Figure saved: {save_path}")
    
    return fig


def create_all_panels(
    samples: Dict[str, List[Dict]],
    output_dir: Optional[Path] = None,
    n_rows: int = 4,
    smooth_sigma: float = 2.0,
) -> Dict[str, plt.Figure]:
    """
    Create panels for all categories.
    
    Args:
        samples: Dictionary of samples by category.
        output_dir: Directory to save figures.
        n_rows: Rows per panel.
        
    Returns:
        Dictionary of figures by category.
    """
    if output_dir is None:
        output_dir = FIGURES_DIR / "tp_fp_fn_tn"
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    figures = {}
    
    for category in ["tp", "fp", "fn", "tn"]:
        save_path = output_dir / f"{category}_samples.png"
        fig = create_sample_panel(
            samples.get(category, []),
            category,
            n_rows=n_rows,
            save_path=save_path,
            smooth_sigma=smooth_sigma,
        )
        figures[category] = fig
    
    return figures


def create_summary_grid(
    samples: Dict[str, List[Dict]],
    save_path: Optional[Path] = None,
    smooth_sigma: float = 2.0,
) -> plt.Figure:
    """
    Create a summary grid showing one example from each category.
    
    Args:
        samples: Dictionary of samples by category.
        save_path: Optional path to save figure.
        
    Returns:
        Summary figure.
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    
    categories = [
        ("tp", "True Positive", "green", (0, 0)),
        ("fp", "False Positive", "orange", (0, 1)),
        ("fn", "False Negative", "red", (1, 0)),
        ("tn", "True Negative", "blue", (1, 1)),
    ]
    
    for cat, title, color, (row, col) in categories:
        ax = axes[row, col]
        
        if samples.get(cat):
            sample = samples[cat][0]
            
            # Show input with error overlay
            ax.imshow(sample["input"], cmap='gray')
            smooth_error = sample.get("smoothed_error_map")
            if smooth_error is None:
                smooth_error = ndimage.gaussian_filter(sample["error_map"], sigma=smooth_sigma).astype(np.float32)
            if smooth_error.max() > 0:
                smooth_norm = smooth_error / smooth_error.max()
                ax.imshow(smooth_norm, cmap='jet', alpha=0.4)
            
            ax.set_title(f"{title}\nScore: {sample['score']:.4f}", fontsize=12, color=color)
        else:
            ax.text(0.5, 0.5, f"No {cat.upper()} sample", ha='center', va='center')
            ax.set_title(title, fontsize=12, color=color)
        
        ax.axis('off')
    
    fig.suptitle("Classification Summary - One Sample Per Category", fontsize=14, y=1.02)
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=DEFAULT_FIGURE_DPI, bbox_inches='tight', facecolor='white')
        print(f"Figure saved: {save_path}")
    
    return fig


# =============================================================================
# MAIN FUNCTION
# =============================================================================

def visualize_tp_fp_fn_tn(
    checkpoint_path: Optional[Path] = None,
    normal_data_path: Optional[Path] = None,
    anomaly_data_path: Optional[Path] = None,
    threshold: float = None,
    score_method: str = "mean",
    error_mode: Optional[str] = None,
    smooth_sigma: float = 2.0,
    output_dir: Optional[Path] = None,
    max_samples_per_category: int = 8,
    n_rows_per_panel: int = 4,
    device: str = None
) -> Dict[str, List[Dict]]:
    """
    Main function to create TP/FP/FN/TN visualizations.
    
    Args:
        checkpoint_path: Path to ECNN checkpoint.
        normal_data_path: Path to normal data.
        anomaly_data_path: Path to anomaly data.
        threshold: Classification threshold (auto-computed if None).
        score_method: Score method to use.
        error_mode: Error-map mode ("abs" or "squared"). If None, uses
            ``ECNN_DEFAULT_ERROR_MODE``.
        smooth_sigma: Gaussian smoothing sigma for heatmap overlays.
        output_dir: Output directory for figures.
        max_samples_per_category: Max samples per category.
        n_rows_per_panel: Rows per visualization panel.
        device: Device for inference.
        
    Returns:
        Dictionary of classified samples.
    """
    # Setup
    ensure_directories_exist()
    
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    if output_dir is None:
        output_dir = FIGURES_DIR / "tp_fp_fn_tn"

    selected_error_mode = error_mode or ECNN_DEFAULT_ERROR_MODE
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("TP/FP/FN/TN Visualization")
    print("=" * 60)
    
    # Load model
    print("\nLoading ECNN model...")
    model, model_info = get_model_for_inference(checkpoint_path, device)
    print(f"Model loaded from: {model_info.get('path', 'unknown')}")
    
    # Find data paths if not provided
    if normal_data_path is None or anomaly_data_path is None:
        data_paths = find_data_paths()
        
        if normal_data_path is None:
            normal_data_path = data_paths.get("ixi_val") or data_paths.get("ixi_test")
        
        if anomaly_data_path is None:
            anomaly_data_path = data_paths.get("brats_test")
    
    print(f"\nNormal data: {normal_data_path}")
    print(f"Anomaly data: {anomaly_data_path}")
    
    # Load images
    print("\nLoading images...")
    normal_images = load_images_from_folder(normal_data_path, max_images=100)
    anomaly_images = load_images_from_folder(anomaly_data_path, max_images=100)
    
    print(f"Normal images loaded: {len(normal_images)}")
    print(f"Anomaly images loaded: {len(anomaly_images)}")
    
    if len(normal_images) == 0 or len(anomaly_images) == 0:
        raise ValueError("Insufficient images loaded for visualization.")
    
    # Auto-compute threshold if not provided
    if threshold is None:
        print("\nComputing threshold from normal samples...")
        
        # Compute scores for normal samples to set threshold
        normal_scores = []
        model.eval()
        
        with torch.no_grad():
            for img in tqdm(normal_images[:50], desc="Computing normal scores"):
                if img.ndim == 2:
                    x = torch.from_numpy(img[np.newaxis, np.newaxis, :, :]).float().to(device)
                else:
                    x = torch.from_numpy(img[np.newaxis, :, :, :]).float().to(device)
                
                _, error_map = compute_error_maps(model, x, selected_error_mode)
                error_np = error_map.squeeze().cpu().numpy()
                brain_mask = x.squeeze().cpu().numpy() > 0.01
                
                if brain_mask.sum() >= ECNN_DEFAULT_MIN_BRAIN_PIXELS:
                    score = compute_score(error_np, brain_mask, method=score_method)
                    normal_scores.append(score)
        
        # Set threshold at 90th percentile of normal scores (FPR ~10%)
        threshold = np.percentile(normal_scores, 90)
        print(f"Auto-computed threshold (P90 of normal): {threshold:.6f}")
    
    # Collect classified samples
    print("\nClassifying samples...")
    samples = collect_classified_samples(
        model=model,
        normal_images=normal_images,
        anomaly_images=anomaly_images,
        threshold=threshold,
        score_method=score_method,
        error_mode=selected_error_mode,
        device=device,
        max_samples=max_samples_per_category
    )
    
    # Create visualizations
    print("\nCreating visualization panels...")
    figures = create_all_panels(samples, output_dir, n_rows=n_rows_per_panel, smooth_sigma=smooth_sigma)
    
    # Create summary grid
    summary_path = output_dir / "classification_summary.png"
    create_summary_grid(samples, save_path=summary_path, smooth_sigma=smooth_sigma)
    
    # Save sample metadata
    sample_counts = {k: len(v) for k, v in samples.items()}
    metadata = {
        "threshold": float(threshold),
        "score_method": score_method,
        "error_mode": selected_error_mode,
        "smooth_sigma": float(smooth_sigma),
        "sample_counts": sample_counts,
        "normal_data_path": str(normal_data_path),
        "anomaly_data_path": str(anomaly_data_path),
    }
    save_json(metadata, "tp_fp_fn_tn_metadata.json", subdir="ecnn_threshold_experiments")
    
    print("\n" + "=" * 60)
    print("VISUALIZATION COMPLETE")
    print("=" * 60)
    print(f"Output directory: {output_dir}")
    print(f"Sample counts: {sample_counts}")
    
    return samples


if __name__ == "__main__":
    try:
        samples = visualize_tp_fp_fn_tn()
    except Exception as e:
        print(f"Error: {e}")
        raise
