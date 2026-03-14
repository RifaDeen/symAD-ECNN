"""
Plotting utilities for SymAD-ECNN evaluation package.

This module provides functions for creating publication-quality figures
for the dissertation, including ROC curves, PR curves, confusion matrices,
and anomaly visualization panels.

Author: SymAD-ECNN Project
Purpose: Visualization functions for dissertation Chapter 8
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
import pandas as pd
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Union
from sklearn.metrics import roc_curve, precision_recall_curve, confusion_matrix

# Import configuration
try:
    from config import (
        FIGURES_DIR, DEFAULT_FIGURE_DPI, DEFAULT_FIGURE_FORMAT,
        HEATMAP_CMAP, ERROR_CMAP, BRAIN_CMAP
    )
except ImportError:
    # Fallback defaults if config not available
    FIGURES_DIR = Path("/content/drive/MyDrive/symAD-ECNN/evaluations/figures")
    DEFAULT_FIGURE_DPI = 150
    DEFAULT_FIGURE_FORMAT = "png"
    HEATMAP_CMAP = "hot"
    ERROR_CMAP = "jet"
    BRAIN_CMAP = "gray"


# =============================================================================
# FIGURE SAVING UTILITY
# =============================================================================

def save_figure(
    fig: plt.Figure,
    filename: str,
    subdir: Optional[str] = None,
    dpi: int = DEFAULT_FIGURE_DPI,
    fmt: str = DEFAULT_FIGURE_FORMAT,
    close: bool = True
) -> Path:
    """
    Save a matplotlib figure to the evaluation figures directory.
    
    Args:
        fig: Matplotlib figure object.
        filename: Base filename (without extension).
        subdir: Optional subdirectory within figures folder.
        dpi: Dots per inch for saved figure.
        fmt: File format (png, pdf, svg).
        close: Whether to close the figure after saving.
        
    Returns:
        Path to the saved figure.
    """
    # Determine save directory
    save_dir = FIGURES_DIR
    if subdir:
        save_dir = FIGURES_DIR / subdir
    
    # Create directory if needed
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Build full path
    filepath = save_dir / f"{filename}.{fmt}"
    
    # Save figure
    fig.savefig(filepath, dpi=dpi, bbox_inches='tight', facecolor='white')
    
    if close:
        plt.close(fig)
    
    print(f"Figure saved: {filepath}")
    return filepath


# =============================================================================
# ROC AND PR CURVES
# =============================================================================

def plot_roc_curve(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    auroc: Optional[float] = None,
    title: str = "ROC Curve",
    label: str = None,
    ax: Optional[plt.Axes] = None,
    color: str = "blue",
    show_diagonal: bool = True
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plot ROC curve for binary classification.
    
    Args:
        y_true: Binary ground truth labels.
        y_scores: Continuous anomaly scores.
        auroc: Pre-computed AUROC (computed if None).
        title: Plot title.
        label: Legend label for the curve.
        ax: Existing axes to plot on (creates new if None).
        color: Line color.
        show_diagonal: Whether to show diagonal reference line.
        
    Returns:
        Tuple of (figure, axes).
    """
    # Compute ROC curve
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    
    # Compute AUROC if not provided
    if auroc is None:
        from sklearn.metrics import roc_auc_score
        try:
            auroc = roc_auc_score(y_true, y_scores)
        except:
            auroc = 0.5
    
    # Create figure if needed
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    else:
        fig = ax.figure
    
    # Plot ROC curve
    curve_label = label if label else f"Model (AUROC = {auroc:.4f})"
    ax.plot(fpr, tpr, color=color, lw=2, label=curve_label)
    
    # Plot diagonal reference line
    if show_diagonal:
        ax.plot([0, 1], [0, 1], 'k--', lw=1, label='Random (AUROC = 0.5)')
    
    # Configure plot
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate (Recall)', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(loc='lower right', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    return fig, ax


def plot_multiple_roc_curves(
    results: List[Dict],
    title: str = "ROC Curve Comparison",
    save_path: Optional[str] = None
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plot multiple ROC curves for model comparison.
    
    Args:
        results: List of dicts with 'y_true', 'y_scores', 'label', optional 'auroc'.
        title: Plot title.
        save_path: Optional path to save figure.
        
    Returns:
        Tuple of (figure, axes).
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Color palette for multiple curves
    colors = plt.cm.tab10(np.linspace(0, 1, len(results)))
    
    for i, result in enumerate(results):
        y_true = result['y_true']
        y_scores = result['y_scores']
        label = result.get('label', f'Model {i+1}')
        auroc = result.get('auroc')
        
        plot_roc_curve(
            y_true, y_scores, auroc=auroc,
            label=label, ax=ax, color=colors[i],
            show_diagonal=(i == 0)  # Only show diagonal once
        )
    
    ax.set_title(title, fontsize=14)
    ax.legend(loc='lower right', fontsize=9)
    
    if save_path:
        fig.savefig(save_path, dpi=DEFAULT_FIGURE_DPI, bbox_inches='tight')
        print(f"Figure saved: {save_path}")
    
    return fig, ax


def plot_pr_curve(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    auprc: Optional[float] = None,
    title: str = "Precision-Recall Curve",
    label: str = None,
    ax: Optional[plt.Axes] = None,
    color: str = "blue"
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plot Precision-Recall curve.
    
    Args:
        y_true: Binary ground truth labels.
        y_scores: Continuous anomaly scores.
        auprc: Pre-computed AUPRC (computed if None).
        title: Plot title.
        label: Legend label.
        ax: Existing axes.
        color: Line color.
        
    Returns:
        Tuple of (figure, axes).
    """
    # Compute PR curve
    precision, recall, thresholds = precision_recall_curve(y_true, y_scores)
    
    # Compute AUPRC if not provided
    if auprc is None:
        from sklearn.metrics import auc
        auprc = auc(recall, precision)
    
    # Create figure if needed
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    else:
        fig = ax.figure
    
    # Plot PR curve
    curve_label = label if label else f"Model (AUPRC = {auprc:.4f})"
    ax.plot(recall, precision, color=color, lw=2, label=curve_label)
    
    # Baseline (random classifier)
    baseline = np.sum(y_true) / len(y_true)
    ax.axhline(y=baseline, color='gray', linestyle='--', lw=1, 
               label=f'Baseline (Prevalence = {baseline:.3f})')
    
    # Configure plot
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('Recall', fontsize=12)
    ax.set_ylabel('Precision', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    return fig, ax


# =============================================================================
# CONFUSION MATRIX
# =============================================================================

def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: List[str] = None,
    title: str = "Confusion Matrix",
    normalize: bool = False,
    cmap: str = "Blues",
    ax: Optional[plt.Axes] = None,
    save_path: Optional[str] = None
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plot confusion matrix as a heatmap.
    
    Args:
        y_true: Ground truth labels.
        y_pred: Predicted labels.
        class_names: Names for classes (default: Normal, Anomaly).
        title: Plot title.
        normalize: Whether to show percentages instead of counts.
        cmap: Colormap name.
        ax: Existing axes.
        save_path: Optional path to save figure.
        
    Returns:
        Tuple of (figure, axes).
    """
    if class_names is None:
        class_names = ['Normal', 'Anomaly']
    
    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        fmt = '.2%'
    else:
        fmt = 'd'
    
    # Create figure if needed
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    else:
        fig = ax.figure
    
    # Plot heatmap
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    
    # Configure ticks and labels
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=class_names,
           yticklabels=class_names,
           ylabel='True Label',
           xlabel='Predicted Label',
           title=title)
    
    # Rotate x labels
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right', rotation_mode='anchor')
    
    # Add text annotations
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            value = cm[i, j]
            if normalize:
                text = f'{value:.1%}'
            else:
                text = f'{value:d}'
            ax.text(j, i, text, ha='center', va='center',
                    color='white' if value > thresh else 'black', fontsize=14)
    
    if save_path:
        fig.savefig(save_path, dpi=DEFAULT_FIGURE_DPI, bbox_inches='tight')
        print(f"Figure saved: {save_path}")
    
    return fig, ax


# =============================================================================
# METRIC COMPARISON PLOTS
# =============================================================================

def plot_metric_comparison(
    df: pd.DataFrame,
    metrics: List[str] = None,
    model_col: str = 'model_name',
    title: str = 'Model Comparison',
    figsize: Tuple[int, int] = (12, 6),
    save_path: Optional[str] = None
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plot bar chart comparing metrics across models.
    
    Args:
        df: DataFrame with model results.
        metrics: List of metric columns to plot.
        model_col: Column containing model names.
        title: Plot title.
        figsize: Figure size.
        save_path: Optional path to save figure.
        
    Returns:
        Tuple of (figure, axes).
    """
    if metrics is None:
        metrics = ['auroc', 'accuracy', 'precision', 'recall', 'f1_score']
    
    # Filter to available metrics
    available_metrics = [m for m in metrics if m in df.columns]
    if not available_metrics:
        print("Warning: No matching metrics found in DataFrame.")
        return None, None
    
    # Prepare data
    models = df[model_col].tolist()
    x = np.arange(len(models))
    width = 0.8 / len(available_metrics)
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot bars for each metric
    colors = plt.cm.Set2(np.linspace(0, 1, len(available_metrics)))
    for i, metric in enumerate(available_metrics):
        values = df[metric].fillna(0).astype(float).tolist()
        offset = (i - len(available_metrics)/2 + 0.5) * width
        bars = ax.bar(x + offset, values, width, label=metric.upper(), color=colors[i])
        
        # Add value labels on bars
        for bar, val in zip(bars, values):
            if val > 0:
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                        f'{val:.3f}', ha='center', va='bottom', fontsize=8, rotation=90)
    
    # Configure plot
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=45, ha='right')
    ax.legend(loc='upper right')
    ax.set_ylim([0, 1.15])
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=DEFAULT_FIGURE_DPI, bbox_inches='tight')
        print(f"Figure saved: {save_path}")
    
    return fig, ax


def plot_score_histograms(
    normal_scores: np.ndarray,
    anomaly_scores: np.ndarray,
    threshold: Optional[float] = None,
    title: str = 'Anomaly Score Distribution',
    bins: int = 50,
    ax: Optional[plt.Axes] = None,
    save_path: Optional[str] = None
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plot histograms of anomaly scores for normal and anomalous samples.
    
    Args:
        normal_scores: Scores from normal samples.
        anomaly_scores: Scores from anomalous samples.
        threshold: Optional threshold line to draw.
        title: Plot title.
        bins: Number of histogram bins.
        ax: Existing axes.
        save_path: Optional path to save figure.
        
    Returns:
        Tuple of (figure, axes).
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    else:
        fig = ax.figure
    
    # Determine common range
    all_scores = np.concatenate([normal_scores, anomaly_scores])
    score_range = (np.min(all_scores), np.max(all_scores))
    
    # Plot histograms
    ax.hist(normal_scores, bins=bins, range=score_range, alpha=0.6, 
            color='blue', label=f'Normal (n={len(normal_scores)})', density=True)
    ax.hist(anomaly_scores, bins=bins, range=score_range, alpha=0.6, 
            color='red', label=f'Anomaly (n={len(anomaly_scores)})', density=True)
    
    # Draw threshold line if provided
    if threshold is not None:
        ax.axvline(x=threshold, color='green', linestyle='--', lw=2, 
                   label=f'Threshold = {threshold:.4f}')
    
    # Configure plot
    ax.set_xlabel('Anomaly Score', fontsize=12)
    ax.set_ylabel('Density', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    if save_path:
        fig.savefig(save_path, dpi=DEFAULT_FIGURE_DPI, bbox_inches='tight')
        print(f"Figure saved: {save_path}")
    
    return fig, ax


# =============================================================================
# ANOMALY VISUALIZATION PANELS
# =============================================================================

def plot_reconstruction_panel(
    input_img: np.ndarray,
    reconstruction: np.ndarray,
    error_map: Optional[np.ndarray] = None,
    brain_mask: Optional[np.ndarray] = None,
    title: str = None,
    save_path: Optional[str] = None
) -> Tuple[plt.Figure, np.ndarray]:
    """
    Plot 4-column panel: input, reconstruction, raw error, heatmap.
    
    Args:
        input_img: Original input image.
        reconstruction: Reconstructed image.
        error_map: Error map (computed if None).
        brain_mask: Optional brain mask for overlay.
        title: Optional super title.
        save_path: Optional path to save figure.
        
    Returns:
        Tuple of (figure, axes array).
    """
    # Compute error map if not provided
    if error_map is None:
        error_map = np.abs(input_img - reconstruction)
    
    # Create figure
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    
    # Panel 1: Input
    axes[0].imshow(input_img, cmap='gray')
    axes[0].set_title('Input', fontsize=12)
    axes[0].axis('off')
    
    # Panel 2: Reconstruction
    axes[1].imshow(reconstruction, cmap='gray')
    axes[1].set_title('Reconstruction', fontsize=12)
    axes[1].axis('off')
    
    # Panel 3: Raw Error
    im = axes[2].imshow(error_map, cmap='hot')
    axes[2].set_title('Error Map', fontsize=12)
    axes[2].axis('off')
    plt.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)
    
    # Panel 4: Heatmap overlay
    axes[3].imshow(input_img, cmap='gray')
    
    # Create masked error for overlay
    masked_error = error_map.copy()
    if brain_mask is not None:
        masked_error[brain_mask == 0] = 0
    
    # Normalize for overlay
    if masked_error.max() > 0:
        masked_error = masked_error / masked_error.max()
    
    # Apply heatmap with transparency
    overlay = axes[3].imshow(masked_error, cmap='jet', alpha=0.5)
    axes[3].set_title('Heatmap Overlay', fontsize=12)
    axes[3].axis('off')
    plt.colorbar(overlay, ax=axes[3], fraction=0.046, pad=0.04)
    
    if title:
        fig.suptitle(title, fontsize=14, y=1.02)
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=DEFAULT_FIGURE_DPI, bbox_inches='tight')
        print(f"Figure saved: {save_path}")
    
    return fig, axes


def plot_tp_fp_fn_tn_panels(
    samples: Dict[str, List[Dict]],
    n_samples: int = 4,
    save_dir: Optional[Path] = None
) -> Dict[str, plt.Figure]:
    """
    Create separate panel figures for TP, FP, FN, TN samples.
    
    Args:
        samples: Dictionary with keys 'tp', 'fp', 'fn', 'tn', each containing
                 list of dicts with 'input', 'reconstruction', 'error_map', 'score'.
        n_samples: Number of samples to show per category.
        save_dir: Directory to save figures.
        
    Returns:
        Dictionary of figures by category.
    """
    figures = {}
    
    category_info = {
        'tp': ('True Positives (Correctly Detected Anomalies)', 'green'),
        'fp': ('False Positives (Normal Misclassified as Anomaly)', 'orange'),
        'fn': ('False Negatives (Missed Anomalies)', 'red'),
        'tn': ('True Negatives (Correctly Identified Normal)', 'blue'),
    }
    
    for category, (title, color) in category_info.items():
        if category not in samples or not samples[category]:
            print(f"No samples available for {category.upper()}")
            continue
        
        # Get samples
        cat_samples = samples[category][:n_samples]
        n = len(cat_samples)
        
        if n == 0:
            continue
        
        # Create figure
        fig, axes = plt.subplots(n, 4, figsize=(16, 4*n))
        if n == 1:
            axes = axes.reshape(1, -1)
        
        fig.suptitle(title, fontsize=14, color=color, y=1.02)
        
        for i, sample in enumerate(cat_samples):
            input_img = sample.get('input', np.zeros((128, 128)))
            recon = sample.get('reconstruction', np.zeros((128, 128)))
            error = sample.get('error_map', np.abs(input_img - recon))
            score = sample.get('score', 0.0)
            
            # Input
            axes[i, 0].imshow(input_img, cmap='gray')
            axes[i, 0].set_title(f'Input (Score: {score:.4f})' if i == 0 else f'Score: {score:.4f}', fontsize=10)
            axes[i, 0].axis('off')
            
            # Reconstruction
            axes[i, 1].imshow(recon, cmap='gray')
            if i == 0:
                axes[i, 1].set_title('Reconstruction', fontsize=10)
            axes[i, 1].axis('off')
            
            # Error
            axes[i, 2].imshow(error, cmap='hot')
            if i == 0:
                axes[i, 2].set_title('Error Map', fontsize=10)
            axes[i, 2].axis('off')
            
            # Overlay
            axes[i, 3].imshow(input_img, cmap='gray')
            if error.max() > 0:
                axes[i, 3].imshow(error / error.max(), cmap='jet', alpha=0.5)
            if i == 0:
                axes[i, 3].set_title('Overlay', fontsize=10)
            axes[i, 3].axis('off')
        
        plt.tight_layout()
        
        figures[category] = fig
        
        # Save if directory provided
        if save_dir:
            save_path = Path(save_dir) / f"{category}_samples.{DEFAULT_FIGURE_FORMAT}"
            fig.savefig(save_path, dpi=DEFAULT_FIGURE_DPI, bbox_inches='tight')
            print(f"Figure saved: {save_path}")
    
    return figures


# =============================================================================
# RADAR CHART FOR MODEL COMPARISON
# =============================================================================

def plot_radar_comparison(
    models: List[str],
    metrics_data: List[List[float]],
    metric_names: List[str] = None,
    title: str = "Model Performance Comparison",
    save_path: Optional[str] = None
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Create radar chart comparing multiple models.
    
    Args:
        models: List of model names.
        metrics_data: List of metric lists, one per model.
        metric_names: Names of metrics.
        title: Plot title.
        save_path: Optional save path.
        
    Returns:
        Tuple of (figure, axes).
    """
    if metric_names is None:
        metric_names = ['AUROC', 'Accuracy', 'Precision', 'Recall', 'F1']
    
    num_vars = len(metric_names)
    
    # Compute angle for each metric
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]  # Complete the loop
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
    
    # Colors for different models
    colors = plt.cm.Set1(np.linspace(0, 1, len(models)))
    
    for i, (model, metrics) in enumerate(zip(models, metrics_data)):
        values = metrics + metrics[:1]  # Complete the loop
        ax.plot(angles, values, 'o-', linewidth=2, label=model, color=colors[i])
        ax.fill(angles, values, alpha=0.1, color=colors[i])
    
    # Configure plot
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metric_names, fontsize=11)
    ax.set_ylim([0, 1])
    ax.set_title(title, fontsize=14, pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
    ax.grid(True)
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=DEFAULT_FIGURE_DPI, bbox_inches='tight')
        print(f"Figure saved: {save_path}")
    
    return fig, ax


# =============================================================================
# THRESHOLD ANALYSIS PLOT
# =============================================================================

def plot_threshold_analysis(
    thresholds: np.ndarray,
    metrics_at_thresholds: Dict[str, np.ndarray],
    optimal_threshold: Optional[float] = None,
    title: str = "Metrics vs Threshold",
    save_path: Optional[str] = None
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plot how metrics change with different threshold values.
    
    Args:
        thresholds: Array of threshold values.
        metrics_at_thresholds: Dict mapping metric names to arrays of values.
        optimal_threshold: Optional optimal threshold to highlight.
        title: Plot title.
        save_path: Optional save path.
        
    Returns:
        Tuple of (figure, axes).
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    colors = {'precision': 'blue', 'recall': 'red', 'f1_score': 'green', 
              'specificity': 'orange', 'accuracy': 'purple'}
    
    for metric_name, values in metrics_at_thresholds.items():
        color = colors.get(metric_name, 'gray')
        ax.plot(thresholds, values, label=metric_name.replace('_', ' ').title(), 
                color=color, lw=2)
    
    if optimal_threshold is not None:
        ax.axvline(x=optimal_threshold, color='black', linestyle='--', 
                   label=f'Optimal ({optimal_threshold:.4f})')
    
    ax.set_xlabel('Threshold', fontsize=12)
    ax.set_ylabel('Metric Value', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1])
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=DEFAULT_FIGURE_DPI, bbox_inches='tight')
        print(f"Figure saved: {save_path}")
    
    return fig, ax


if __name__ == "__main__":
    print("Plotting utilities loaded successfully.")
    print(f"Default figure directory: {FIGURES_DIR}")
