"""
Run ECNN Threshold Experiments.

This script evaluates the trained ECNN model with different scoring methods
and threshold strategies to find optimal configurations for anomaly detection.

Supports dissertation section 8.5 (Further Evaluations).

Author: SymAD-ECNN Project
Purpose: Systematic threshold and scoring method ablation for ECNN
"""

import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from tqdm import tqdm
import json

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import (
    DRIVE_PROJECT_ROOT, DATA_DIR, JSON_DIR, TABLES_DIR, LOGS_DIR,
    ECNN_DEFAULT_SCORE_METHODS, ECNN_DEFAULT_FPRS, ECNN_DEFAULT_ERROR_MODE,
    ECNN_DEFAULT_MIN_BRAIN_PIXELS, ECNN_DEFAULT_EXPERIMENTS,
    ensure_directories_exist
)
from path_utils import (
    get_drive_project_root, find_ecnn_checkpoint, find_data_paths,
    require_file
)
from metrics_utils import (
    compute_score, threshold_from_normal_scores, threshold_youden_j,
    compute_full_metrics, create_experiment_result
)
from io_utils import (
    save_json, save_csv, save_markdown_table,
    start_experiment_log, end_experiment_log, log_message
)
from ecnn_model_loader import get_model_for_inference, compute_error_maps


# =============================================================================
# DATASET CLASSES
# =============================================================================

class ImageFolderDataset(Dataset):
    """
    Simple dataset for loading preprocessed MRI slices from a folder.
    """
    
    def __init__(
        self,
        folder_path: Path,
        extensions: Tuple[str, ...] = ('.png', '.jpg', '.jpeg', '.npy'),
        transform=None
    ):
        """
        Initialize dataset.
        
        Args:
            folder_path: Path to folder containing images.
            extensions: Valid file extensions.
            transform: Optional transform to apply.
        """
        self.folder_path = Path(folder_path)
        self.transform = transform
        
        # Find all image files
        self.image_paths = []
        if self.folder_path.exists():
            for ext in extensions:
                self.image_paths.extend(list(self.folder_path.glob(f"*{ext}")))
                self.image_paths.extend(list(self.folder_path.glob(f"**/*{ext}")))
        
        # Remove duplicates and sort
        self.image_paths = sorted(list(set(self.image_paths)))
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        
        # Load image based on extension
        if img_path.suffix == '.npy':
            image = np.load(img_path)
        else:
            image = np.array(Image.open(img_path).convert('L'))
        
        # Normalize to [0, 1]
        if image.max() > 1:
            image = image.astype(np.float32) / 255.0
        
        # Add channel dimension if needed
        if image.ndim == 2:
            image = image[np.newaxis, :, :]
        
        if self.transform:
            image = self.transform(image)
        
        return torch.from_numpy(image).float(), str(img_path)


# =============================================================================
# SCORE COMPUTATION
# =============================================================================

def compute_scores_for_dataset(
    model: torch.nn.Module,
    dataloader: DataLoader,
    score_method: str = "mean",
    error_mode: str = "abs",
    device: str = "cuda",
    min_brain_pixels: int = 50
) -> Tuple[np.ndarray, List[str]]:
    """
    Compute anomaly scores for all images in a dataset.
    
    Args:
        model: The ECNN model.
        dataloader: DataLoader for the dataset.
        score_method: Scoring method ("mean", "p95", "p90").
        error_mode: Error computation mode.
        device: Device for inference.
        min_brain_pixels: Minimum brain pixels for valid score.
        
    Returns:
        Tuple of (scores array, file paths list).
    """
    model.eval()
    scores = []
    paths = []
    
    with torch.no_grad():
        for images, image_paths in tqdm(dataloader, desc=f"Computing {score_method} scores"):
            images = images.to(device)
            
            # Compute reconstructions and error maps
            recons, error_maps = compute_error_maps(model, images, error_mode)
            
            # Compute scores for each image in batch
            for i in range(images.shape[0]):
                error_map = error_maps[i, 0].cpu().numpy()
                
                # Create simple brain mask (non-zero pixels)
                brain_mask = images[i, 0].cpu().numpy() > 0.01
                
                if brain_mask.sum() < min_brain_pixels:
                    # Skip slices with insufficient brain content
                    continue
                
                # Compute score
                score = compute_score(error_map, brain_mask, method=score_method)
                scores.append(score)
                paths.append(image_paths[i])
    
    return np.array(scores), paths


# =============================================================================
# EXPERIMENT EXECUTION
# =============================================================================

def run_single_experiment(
    experiment_name: str,
    score_method: str,
    threshold_method: str,
    threshold_param: Optional[float],
    normal_scores: np.ndarray,
    anomaly_scores: np.ndarray,
    reference_threshold: Optional[float] = None
) -> Dict:
    """
    Run a single threshold experiment.
    
    Args:
        experiment_name: Name/ID for this experiment.
        score_method: Score computation method used.
        threshold_method: Threshold determination method ("fpr", "youden", "reference").
        threshold_param: Parameter for threshold method (e.g., target FPR).
        normal_scores: Scores from normal validation samples.
        anomaly_scores: Scores from anomaly test samples.
        reference_threshold: Reference threshold from training (for "reference" method).
        
    Returns:
        Experiment result dictionary.
    """
    # Determine threshold
    if threshold_method == "fpr":
        threshold = threshold_from_normal_scores(normal_scores, target_fpr=threshold_param)
        threshold_desc = f"FPR={threshold_param:.0%}"
    elif threshold_method == "youden":
        # Need all labels for Youden
        all_scores = np.concatenate([normal_scores, anomaly_scores])
        all_labels = np.concatenate([np.zeros(len(normal_scores)), np.ones(len(anomaly_scores))])
        threshold, _ = threshold_youden_j(all_labels, all_scores)
        threshold_desc = "Youden J"
    elif threshold_method == "reference":
        if reference_threshold is None:
            # Fall back to median of normal scores if no reference
            threshold = np.percentile(normal_scores, 90)
            threshold_desc = "Reference (P90 fallback)"
        else:
            threshold = reference_threshold
            threshold_desc = "Reference (original)"
    else:
        raise ValueError(f"Unknown threshold method: {threshold_method}")
    
    # Combine scores and labels for evaluation
    all_scores = np.concatenate([normal_scores, anomaly_scores])
    all_labels = np.concatenate([np.zeros(len(normal_scores)), np.ones(len(anomaly_scores))])
    
    # Compute metrics
    metrics = compute_full_metrics(all_labels, all_scores, threshold)
    
    # Create result
    result = create_experiment_result(
        experiment_name=experiment_name,
        score_method=score_method,
        threshold_method=f"{threshold_method}_{threshold_desc}",
        threshold_value=threshold,
        metrics=metrics,
        normal_scores=normal_scores,
        anomaly_scores=anomaly_scores,
        notes=f"Score: {score_method}, Threshold: {threshold_desc}"
    )
    
    return result


def run_all_experiments(
    model: torch.nn.Module,
    normal_dataloader: DataLoader,
    anomaly_dataloader: DataLoader,
    experiments: List[Tuple] = None,
    device: str = "cuda",
    error_mode: str = "abs",
    reference_threshold: Optional[float] = None,
    log_name: str = None
) -> Tuple[List[Dict], Dict]:
    """
    Run all threshold experiments.
    
    Args:
        model: The ECNN model.
        normal_dataloader: DataLoader for normal validation data.
        anomaly_dataloader: DataLoader for anomaly test data.
        experiments: List of (score_method, threshold_method, threshold_param) tuples.
        device: Device for inference.
        error_mode: Error computation mode.
        reference_threshold: Optional reference threshold from training.
        log_name: Name for logging.
        
    Returns:
        Tuple of (list of results, summary dict).
    """
    if experiments is None:
        experiments = ECNN_DEFAULT_EXPERIMENTS
    
    if log_name:
        log_message(f"Running {len(experiments)} experiments", log_name)
    
    # Pre-compute scores for each scoring method
    score_cache = {}
    
    for score_method, _, _ in experiments:
        if score_method not in score_cache:
            if log_name:
                log_message(f"Computing {score_method} scores...", log_name)
            
            # Normal scores
            normal_scores, _ = compute_scores_for_dataset(
                model, normal_dataloader, score_method, error_mode, device
            )
            
            # Anomaly scores
            anomaly_scores, _ = compute_scores_for_dataset(
                model, anomaly_dataloader, score_method, error_mode, device
            )
            
            score_cache[score_method] = {
                "normal": normal_scores,
                "anomaly": anomaly_scores
            }
            
            if log_name:
                log_message(
                    f"  Normal samples: {len(normal_scores)}, Anomaly samples: {len(anomaly_scores)}",
                    log_name
                )
    
    # Run experiments
    results = []
    
    for i, (score_method, threshold_method, threshold_param) in enumerate(experiments):
        exp_name = f"exp_{i+1:02d}_{score_method}_{threshold_method}"
        if threshold_param is not None:
            exp_name += f"_{threshold_param}"
        
        if log_name:
            log_message(f"Running experiment: {exp_name}", log_name)
        
        normal_scores = score_cache[score_method]["normal"]
        anomaly_scores = score_cache[score_method]["anomaly"]
        
        result = run_single_experiment(
            experiment_name=exp_name,
            score_method=score_method,
            threshold_method=threshold_method,
            threshold_param=threshold_param,
            normal_scores=normal_scores,
            anomaly_scores=anomaly_scores,
            reference_threshold=reference_threshold
        )
        
        results.append(result)
        
        if log_name:
            metrics = result["metrics"]
            log_message(
                f"  AUROC: {metrics['auroc']:.4f}, Acc: {metrics['accuracy']:.4f}, "
                f"Recall: {metrics['recall']:.4f}, Spec: {metrics['specificity']:.4f}",
                log_name
            )
    
    # Create summary
    summary = {
        "n_experiments": len(results),
        "score_methods_used": list(set(r["score_method"] for r in results)),
        "best_auroc": max(r["metrics"]["auroc"] for r in results),
        "best_recall": max(r["metrics"]["recall"] for r in results),
        "best_specificity": max(r["metrics"]["specificity"] for r in results),
        "best_f1": max(r["metrics"]["f1_score"] for r in results),
    }
    
    # Find best experiments
    best_auroc_exp = max(results, key=lambda r: r["metrics"]["auroc"])
    best_recall_exp = max(results, key=lambda r: r["metrics"]["recall"])
    best_spec_exp = max(results, key=lambda r: r["metrics"]["specificity"])
    best_f1_exp = max(results, key=lambda r: r["metrics"]["f1_score"])
    
    summary["best_auroc_experiment"] = best_auroc_exp["experiment_name"]
    summary["best_recall_experiment"] = best_recall_exp["experiment_name"]
    summary["best_specificity_experiment"] = best_spec_exp["experiment_name"]
    summary["best_f1_experiment"] = best_f1_exp["experiment_name"]
    
    return results, summary


# =============================================================================
# OUTPUT GENERATION
# =============================================================================

def results_to_dataframe(results: List[Dict]) -> "pd.DataFrame":
    """Convert experiment results to DataFrame."""
    import pandas as pd
    
    rows = []
    for r in results:
        row = {
            "experiment": r["experiment_name"],
            "score_method": r["score_method"],
            "threshold_method": r["threshold_method"],
            "threshold": r["threshold_value"],
            **{k: v for k, v in r["metrics"].items() if isinstance(v, (int, float))}
        }
        rows.append(row)
    
    df = pd.DataFrame(rows)
    return df.sort_values("auroc", ascending=False).reset_index(drop=True)


def save_experiment_outputs(
    results: List[Dict],
    summary: Dict,
    output_subdir: str = "ecnn_threshold_experiments"
) -> Dict[str, Path]:
    """
    Save all experiment outputs to Drive.
    
    Args:
        results: List of experiment results.
        summary: Summary dictionary.
        output_subdir: Subdirectory for JSON outputs.
        
    Returns:
        Dictionary of output file paths.
    """
    import pandas as pd
    
    output_paths = {}
    
    # Save individual experiment JSONs
    for result in results:
        json_path = save_json(
            result,
            f"{result['experiment_name']}.json",
            subdir=output_subdir
        )
        output_paths[result["experiment_name"]] = json_path
    
    # Save summary JSON
    summary_path = save_json(
        {
            "summary": summary,
            "experiments": [r["experiment_name"] for r in results]
        },
        "experiments_summary.json",
        subdir=output_subdir
    )
    output_paths["summary_json"] = summary_path
    
    # Create and save DataFrame
    df = results_to_dataframe(results)
    
    csv_path = save_csv(df, "ecnn_threshold_experiments.csv")
    output_paths["csv"] = csv_path
    
    md_path = save_markdown_table(
        df,
        "ecnn_threshold_experiments.md",
        title="ECNN Threshold Experiment Results"
    )
    output_paths["markdown"] = md_path
    
    return output_paths


# =============================================================================
# MAIN FUNCTION
# =============================================================================

def run_ecnn_threshold_experiments(
    checkpoint_path: Optional[Path] = None,
    normal_data_path: Optional[Path] = None,
    anomaly_data_path: Optional[Path] = None,
    experiments: List[Tuple] = None,
    reference_threshold: Optional[float] = None,
    batch_size: int = 16,
    device: str = None
) -> Tuple[List[Dict], Dict]:
    """
    Main function to run all ECNN threshold experiments.
    
    Args:
        checkpoint_path: Path to ECNN checkpoint (auto-detected if None).
        normal_data_path: Path to normal validation data (auto-detected if None).
        anomaly_data_path: Path to anomaly test data (auto-detected if None).
        experiments: Custom experiments list (uses defaults if None).
        reference_threshold: Reference threshold from training.
        batch_size: Batch size for inference.
        device: Device for inference.
        
    Returns:
        Tuple of (results list, summary dict).
    """
    # Setup
    ensure_directories_exist()
    
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    log_name = start_experiment_log(
        "ecnn_threshold_experiments",
        params={
            "device": device,
            "batch_size": batch_size,
            "n_experiments": len(experiments) if experiments else len(ECNN_DEFAULT_EXPERIMENTS)
        }
    )
    
    try:
        # Load model
        log_message("Loading ECNN model...", log_name)
        model, model_info = get_model_for_inference(checkpoint_path, device)
        log_message(f"Model loaded from: {model_info.get('path', 'unknown')}", log_name)
        
        # Find data paths if not provided
        if normal_data_path is None or anomaly_data_path is None:
            data_paths = find_data_paths()
            
            if normal_data_path is None:
                normal_data_path = data_paths.get("ixi_val") or data_paths.get("ixi_test")
                if normal_data_path is None:
                    raise FileNotFoundError("Could not find normal validation data in Drive.")
            
            if anomaly_data_path is None:
                anomaly_data_path = data_paths.get("brats_test")
                if anomaly_data_path is None:
                    raise FileNotFoundError("Could not find anomaly test data in Drive.")
        
        log_message(f"Normal data: {normal_data_path}", log_name)
        log_message(f"Anomaly data: {anomaly_data_path}", log_name)
        
        # Create datasets and dataloaders
        normal_dataset = ImageFolderDataset(normal_data_path)
        anomaly_dataset = ImageFolderDataset(anomaly_data_path)
        
        log_message(f"Normal samples: {len(normal_dataset)}", log_name)
        log_message(f"Anomaly samples: {len(anomaly_dataset)}", log_name)
        
        if len(normal_dataset) == 0:
            raise ValueError(f"No images found in normal data path: {normal_data_path}")
        if len(anomaly_dataset) == 0:
            raise ValueError(f"No images found in anomaly data path: {anomaly_data_path}")
        
        normal_loader = DataLoader(normal_dataset, batch_size=batch_size, shuffle=False)
        anomaly_loader = DataLoader(anomaly_dataset, batch_size=batch_size, shuffle=False)
        
        # Run experiments
        results, summary = run_all_experiments(
            model=model,
            normal_dataloader=normal_loader,
            anomaly_dataloader=anomaly_loader,
            experiments=experiments,
            device=device,
            error_mode=ECNN_DEFAULT_ERROR_MODE,
            reference_threshold=reference_threshold,
            log_name=log_name
        )
        
        # Save outputs
        log_message("Saving outputs...", log_name)
        output_paths = save_experiment_outputs(results, summary)
        
        log_message(f"Outputs saved to: {list(output_paths.values())[:3]}...", log_name)
        
        end_experiment_log(log_name, summary=summary)
        
        return results, summary
        
    except Exception as e:
        log_message(f"ERROR: {e}", log_name)
        end_experiment_log(log_name, summary={"error": str(e)})
        raise


if __name__ == "__main__":
    print("=" * 60)
    print("ECNN Threshold Experiments")
    print("=" * 60)
    
    try:
        results, summary = run_ecnn_threshold_experiments()
        
        print("\n" + "=" * 60)
        print("SUMMARY")
        print("=" * 60)
        print(f"Experiments run: {summary['n_experiments']}")
        print(f"Best AUROC: {summary['best_auroc']:.4f} ({summary['best_auroc_experiment']})")
        print(f"Best Recall: {summary['best_recall']:.4f} ({summary['best_recall_experiment']})")
        print(f"Best Specificity: {summary['best_specificity']:.4f} ({summary['best_specificity_experiment']})")
        print(f"Best F1: {summary['best_f1']:.4f} ({summary['best_f1_experiment']})")
        
    except Exception as e:
        print(f"Error: {e}")
        raise
