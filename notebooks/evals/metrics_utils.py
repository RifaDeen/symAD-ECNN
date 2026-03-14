"""
Metrics utilities for SymAD-ECNN evaluation package.

This module provides functions for computing evaluation metrics,
score calculations, threshold determination, and result handling.

Author: SymAD-ECNN Project
Purpose: Reusable metric functions for dissertation Chapter 8 evaluations
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Union
import json
from sklearn.metrics import (
    roc_auc_score,
    precision_recall_curve,
    auc,
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_curve,
)


# =============================================================================
# SCORE COMPUTATION FUNCTIONS
# =============================================================================

def score_mean(error_map: np.ndarray, brain_mask: Optional[np.ndarray] = None) -> float:
    """
    Compute mean anomaly score from error map.
    
    The mean score is the average of error values within the brain region.
    Lower values indicate normal tissue, higher values indicate anomalies.
    
    Args:
        error_map: 2D array of reconstruction errors (H, W).
        brain_mask: Optional binary mask of brain region. If None, uses entire image.
        
    Returns:
        Mean error value as anomaly score.
    """
    if brain_mask is not None:
        if brain_mask.sum() == 0:
            return 0.0
        values = error_map[brain_mask > 0]
    else:
        values = error_map.flatten()
    
    if len(values) == 0:
        return 0.0
    
    return float(np.mean(values))


def score_percentile(
    error_map: np.ndarray,
    brain_mask: Optional[np.ndarray] = None,
    percentile: float = 95
) -> float:
    """
    Compute percentile-based anomaly score from error map.
    
    The percentile score focuses on the high-error regions, which are more
    likely to represent anomalies. P95 captures the top 5% of errors.
    
    Args:
        error_map: 2D array of reconstruction errors (H, W).
        brain_mask: Optional binary mask of brain region. If None, uses entire image.
        percentile: Percentile to compute (default 95 for P95).
        
    Returns:
        Percentile error value as anomaly score.
    """
    if brain_mask is not None:
        if brain_mask.sum() == 0:
            return 0.0
        values = error_map[brain_mask > 0]
    else:
        values = error_map.flatten()
    
    if len(values) == 0:
        return 0.0
    
    return float(np.percentile(values, percentile))


def compute_score(
    error_map: np.ndarray,
    brain_mask: Optional[np.ndarray] = None,
    method: str = "mean"
) -> float:
    """
    Compute anomaly score using specified method.
    
    Args:
        error_map: 2D array of reconstruction errors.
        brain_mask: Optional binary mask of brain region.
        method: Scoring method - "mean", "p95", "p90", "max".
        
    Returns:
        Computed anomaly score.
        
    Raises:
        ValueError: If method is not recognized.
    """
    method = method.lower()
    
    if method == "mean":
        return score_mean(error_map, brain_mask)
    elif method == "p95":
        return score_percentile(error_map, brain_mask, 95)
    elif method == "p90":
        return score_percentile(error_map, brain_mask, 90)
    elif method == "p99":
        return score_percentile(error_map, brain_mask, 99)
    elif method == "max":
        if brain_mask is not None and brain_mask.sum() > 0:
            return float(np.max(error_map[brain_mask > 0]))
        return float(np.max(error_map))
    else:
        raise ValueError(f"Unknown score method: {method}. Use 'mean', 'p95', 'p90', 'p99', or 'max'.")


# =============================================================================
# THRESHOLD COMPUTATION FUNCTIONS
# =============================================================================

def threshold_from_normal_scores(
    normal_scores: np.ndarray,
    target_fpr: float = 0.10
) -> float:
    """
    Compute threshold from normal validation scores to achieve target FPR.
    
    This method sets the threshold such that a specified percentage of normal
    samples will be incorrectly classified as anomalies (false positives).
    
    Args:
        normal_scores: Array of anomaly scores from normal samples.
        target_fpr: Target false positive rate (default 0.10 = 10%).
        
    Returns:
        Threshold value that achieves approximately the target FPR.
        
    Example:
        If target_fpr=0.10, the threshold will be set at the 90th percentile
        of normal scores, ensuring ~10% of normal samples exceed it.
    """
    normal_scores = np.array(normal_scores)
    if len(normal_scores) == 0:
        raise ValueError("Cannot compute threshold from empty normal scores array.")
    
    # The threshold is set at (1 - target_fpr) percentile
    # E.g., for FPR=0.10, we use the 90th percentile
    percentile = (1 - target_fpr) * 100
    threshold = float(np.percentile(normal_scores, percentile))
    
    return threshold


def threshold_youden_j(
    y_true: np.ndarray,
    y_scores: np.ndarray
) -> Tuple[float, float]:
    """
    Compute optimal threshold using Youden's J statistic.
    
    Youden's J = Sensitivity + Specificity - 1
    The optimal threshold maximizes this value.
    
    Args:
        y_true: Binary ground truth labels (0=normal, 1=anomaly).
        y_scores: Continuous anomaly scores.
        
    Returns:
        Tuple of (optimal_threshold, max_youden_j).
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    
    # Youden's J statistic
    j_scores = tpr - fpr
    optimal_idx = np.argmax(j_scores)
    
    optimal_threshold = thresholds[optimal_idx]
    max_j = j_scores[optimal_idx]
    
    return float(optimal_threshold), float(max_j)


# =============================================================================
# BINARY CLASSIFICATION METRICS
# =============================================================================

def compute_binary_metrics(
    y_true: np.ndarray,
    y_score: np.ndarray,
    threshold: float
) -> Dict[str, float]:
    """
    Compute binary classification metrics at a given threshold.
    
    Args:
        y_true: Binary ground truth labels (0=normal, 1=anomaly).
        y_score: Continuous anomaly scores.
        threshold: Classification threshold (scores >= threshold -> anomaly).
        
    Returns:
        Dictionary containing all computed metrics.
    """
    y_true = np.array(y_true)
    y_score = np.array(y_score)
    
    # Binary predictions
    y_pred = (y_score >= threshold).astype(int)
    
    # Confusion matrix
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    
    # Core metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)  # Also called sensitivity
    f1 = f1_score(y_true, y_pred, zero_division=0)
    
    # Specificity = TN / (TN + FP)
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    
    # False positive rate = FP / (FP + TN)
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    
    # False negative rate = FN / (FN + TP)
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0
    
    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "sensitivity": float(recall),  # Alias for recall
        "specificity": float(specificity),
        "f1_score": float(f1),
        "fpr": float(fpr),
        "fnr": float(fnr),
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn),
        "total_samples": int(len(y_true)),
        "total_positive": int(np.sum(y_true == 1)),
        "total_negative": int(np.sum(y_true == 0)),
    }


def compute_auroc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """
    Compute Area Under the ROC Curve (AUROC).
    
    AUROC measures the model's ability to discriminate between normal and
    anomalous samples across all possible thresholds.
    
    Args:
        y_true: Binary ground truth labels.
        y_score: Continuous anomaly scores.
        
    Returns:
        AUROC value (0.5 = random, 1.0 = perfect).
    """
    try:
        return float(roc_auc_score(y_true, y_score))
    except ValueError:
        # Handle cases with only one class
        return 0.5


def compute_auprc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """
    Compute Area Under the Precision-Recall Curve (AUPRC).
    
    AUPRC is especially useful for imbalanced datasets where the positive
    class (anomalies) is rare.
    
    Args:
        y_true: Binary ground truth labels.
        y_score: Continuous anomaly scores.
        
    Returns:
        AUPRC value.
    """
    try:
        precision, recall, _ = precision_recall_curve(y_true, y_score)
        return float(auc(recall, precision))
    except ValueError:
        return 0.0


def compute_full_metrics(
    y_true: np.ndarray,
    y_score: np.ndarray,
    threshold: float
) -> Dict[str, float]:
    """
    Compute all metrics including AUROC and AUPRC.
    
    Args:
        y_true: Binary ground truth labels.
        y_score: Continuous anomaly scores.
        threshold: Classification threshold.
        
    Returns:
        Dictionary with all metrics.
    """
    metrics = compute_binary_metrics(y_true, y_score, threshold)
    metrics["auroc"] = compute_auroc(y_true, y_score)
    metrics["auprc"] = compute_auprc(y_true, y_score)
    return metrics


# =============================================================================
# RESULT JSON HANDLING
# =============================================================================

def load_results_json(path: Union[str, Path]) -> Optional[Dict]:
    """
    Load a results JSON file with error handling.
    
    Args:
        path: Path to the JSON file.
        
    Returns:
        Parsed JSON as dictionary, or None if loading fails.
    """
    path = Path(path)
    if not path.exists():
        print(f"Warning: Results file not found: {path}")
        return None
    
    try:
        with open(path, 'r') as f:
            return json.load(f)
    except json.JSONDecodeError as e:
        print(f"Warning: Failed to parse JSON {path}: {e}")
        return None
    except Exception as e:
        print(f"Warning: Error loading {path}: {e}")
        return None


def load_all_results_jsons(paths: List[Union[str, Path]]) -> List[Dict]:
    """
    Load multiple results JSON files.
    
    Args:
        paths: List of paths to JSON files.
        
    Returns:
        List of successfully loaded JSON dictionaries.
    """
    results = []
    for path in paths:
        data = load_results_json(path)
        if data is not None:
            # Add source file info
            data["_source_file"] = str(path)
            results.append(data)
    return results


def normalize_result_schema(result: Dict) -> Dict:
    """
    Normalize a result JSON to a standard schema.
    
    Different training notebooks may save results in slightly different formats.
    This function normalizes them to a consistent schema.
    
    Args:
        result: Raw result dictionary from JSON.
        
    Returns:
        Normalized result dictionary.
    """
    normalized = {
        "model_name": None,
        "auroc": None,
        "accuracy": None,
        "precision": None,
        "recall": None,
        "specificity": None,
        "f1_score": None,
        "threshold": None,
        "score_method": None,
        "training_epochs": None,
        "notes": None,
        "_source_file": result.get("_source_file"),
    }
    
    # Try to extract model name
    for key in ["model_name", "model", "name", "architecture"]:
        if key in result:
            normalized["model_name"] = result[key]
            break
    
    # Try to extract from filename if not found
    if normalized["model_name"] is None and "_source_file" in result:
        filename = Path(result["_source_file"]).stem
        normalized["model_name"] = filename
    
    # Extract metrics with various possible key names
    metric_aliases = {
        "auroc": ["auroc", "auc_roc", "roc_auc", "auc"],
        "accuracy": ["accuracy", "acc"],
        "precision": ["precision", "prec"],
        "recall": ["recall", "sensitivity", "sens", "tpr"],
        "specificity": ["specificity", "spec", "tnr"],
        "f1_score": ["f1_score", "f1", "f_score"],
        "threshold": ["threshold", "thresh", "optimal_threshold"],
        "score_method": ["score_method", "scoring", "score_type"],
        "training_epochs": ["epochs", "training_epochs", "num_epochs"],
    }
    
    for standard_key, aliases in metric_aliases.items():
        for alias in aliases:
            # Check direct keys
            if alias in result:
                normalized[standard_key] = result[alias]
                break
            # Check nested under 'metrics' or 'results'
            for nested_key in ["metrics", "results", "evaluation"]:
                if nested_key in result and isinstance(result[nested_key], dict):
                    if alias in result[nested_key]:
                        normalized[standard_key] = result[nested_key][alias]
                        break
    
    # Copy any extra fields
    normalized["_raw"] = result
    
    return normalized


def results_to_dataframe(results_list: List[Dict]) -> pd.DataFrame:
    """
    Convert a list of result dictionaries to a pandas DataFrame.
    
    Args:
        results_list: List of result dictionaries.
        
    Returns:
        DataFrame with normalized columns.
    """
    if not results_list:
        return pd.DataFrame()
    
    # Normalize all results
    normalized = [normalize_result_schema(r) for r in results_list]
    
    # Create DataFrame
    df = pd.DataFrame(normalized)
    
    # Select and order important columns
    display_cols = [
        "model_name", "auroc", "accuracy", "precision", 
        "recall", "specificity", "f1_score", "threshold", 
        "score_method", "training_epochs", "notes"
    ]
    
    # Only include columns that exist
    cols = [c for c in display_cols if c in df.columns]
    df = df[cols]
    
    # Sort by AUROC descending if available
    if "auroc" in df.columns:
        df = df.sort_values("auroc", ascending=False, na_position="last")
    
    return df.reset_index(drop=True)


# =============================================================================
# EXPERIMENT RESULTS STRUCTURE
# =============================================================================

def create_experiment_result(
    experiment_name: str,
    score_method: str,
    threshold_method: str,
    threshold_value: float,
    metrics: Dict[str, float],
    normal_scores: Optional[np.ndarray] = None,
    anomaly_scores: Optional[np.ndarray] = None,
    notes: str = None
) -> Dict:
    """
    Create a standardized experiment result dictionary.
    
    Args:
        experiment_name: Name/identifier for this experiment.
        score_method: Score computation method used.
        threshold_method: Threshold determination method used.
        threshold_value: The actual threshold value.
        metrics: Dictionary of computed metrics.
        normal_scores: Optional array of normal sample scores.
        anomaly_scores: Optional array of anomaly sample scores.
        notes: Optional notes about this experiment.
        
    Returns:
        Standardized experiment result dictionary.
    """
    result = {
        "experiment_name": experiment_name,
        "score_method": score_method,
        "threshold_method": threshold_method,
        "threshold_value": float(threshold_value),
        "metrics": metrics,
        "notes": notes,
    }
    
    # Add score statistics if provided
    if normal_scores is not None:
        normal_scores = np.array(normal_scores)
        result["normal_score_stats"] = {
            "count": int(len(normal_scores)),
            "mean": float(np.mean(normal_scores)),
            "std": float(np.std(normal_scores)),
            "min": float(np.min(normal_scores)),
            "max": float(np.max(normal_scores)),
            "p50": float(np.percentile(normal_scores, 50)),
            "p90": float(np.percentile(normal_scores, 90)),
            "p95": float(np.percentile(normal_scores, 95)),
        }
    
    if anomaly_scores is not None:
        anomaly_scores = np.array(anomaly_scores)
        result["anomaly_score_stats"] = {
            "count": int(len(anomaly_scores)),
            "mean": float(np.mean(anomaly_scores)),
            "std": float(np.std(anomaly_scores)),
            "min": float(np.min(anomaly_scores)),
            "max": float(np.max(anomaly_scores)),
            "p50": float(np.percentile(anomaly_scores, 50)),
            "p90": float(np.percentile(anomaly_scores, 90)),
            "p95": float(np.percentile(anomaly_scores, 95)),
        }
    
    return result


# =============================================================================
# COMPARISON UTILITIES
# =============================================================================

def rank_models(
    df: pd.DataFrame,
    metric: str = "auroc",
    ascending: bool = False
) -> pd.DataFrame:
    """
    Rank models by a specified metric.
    
    Args:
        df: DataFrame with model results.
        metric: Column name to rank by.
        ascending: If True, lower is better.
        
    Returns:
        DataFrame sorted with rank column added.
    """
    if metric not in df.columns:
        print(f"Warning: Metric '{metric}' not in DataFrame columns.")
        return df
    
    ranked = df.sort_values(metric, ascending=ascending, na_position="last").reset_index(drop=True)
    ranked.insert(0, "rank", range(1, len(ranked) + 1))
    return ranked


def format_metrics_table(
    df: pd.DataFrame,
    float_format: str = ".4f"
) -> pd.DataFrame:
    """
    Format a metrics DataFrame for display.
    
    Args:
        df: DataFrame with metrics.
        float_format: Format string for float values.
        
    Returns:
        DataFrame with formatted values.
    """
    formatted = df.copy()
    
    # Format float columns
    float_cols = formatted.select_dtypes(include=[np.floating]).columns
    for col in float_cols:
        formatted[col] = formatted[col].apply(
            lambda x: f"{x:{float_format}}" if pd.notna(x) else "-"
        )
    
    return formatted


if __name__ == "__main__":
    # Simple test
    print("Metrics utilities loaded successfully.")
    
    # Example usage
    y_true = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    y_scores = np.array([0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9])
    
    print("\nExample metrics computation:")
    print(f"AUROC: {compute_auroc(y_true, y_scores):.4f}")
    print(f"AUPRC: {compute_auprc(y_true, y_scores):.4f}")
    
    threshold = 0.5
    metrics = compute_binary_metrics(y_true, y_scores, threshold)
    print(f"\nAt threshold {threshold}:")
    print(f"  Accuracy: {metrics['accuracy']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall: {metrics['recall']:.4f}")
    print(f"  Specificity: {metrics['specificity']:.4f}")
    print(f"  F1: {metrics['f1_score']:.4f}")
