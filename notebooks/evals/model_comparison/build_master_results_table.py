"""
Build Master Results Table for SymAD-ECNN Model Comparison.

This script gathers all model result JSON files from Google Drive,
normalizes their schemas, and produces a unified comparison table
for dissertation Chapter 8.3 and 8.4.

Author: SymAD-ECNN Project
Purpose: Generate master model comparison tables for benchmarking
"""

import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import json
import pandas as pd
import numpy as np

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import (
    DRIVE_PROJECT_ROOT, RESULTS_DIR, TABLES_DIR, JSON_DIR,
    KNOWN_MODELS, ensure_directories_exist
)
from path_utils import (
    get_drive_project_root, find_results_jsons, validate_paths
)
from metrics_utils import (
    load_results_json, load_all_results_jsons, 
    normalize_result_schema, results_to_dataframe,
    rank_models, format_metrics_table
)
from io_utils import (
    save_json, save_csv, save_markdown_table,
    log_message, start_experiment_log, end_experiment_log
)


# =============================================================================
# RESULT SCHEMA NORMALIZATION
# =============================================================================

def identify_model_type(result: Dict, filepath: Path) -> str:
    """
    Identify the model type from result data or filepath.
    
    Args:
        result: Result dictionary.
        filepath: Path to the result file.
        
    Returns:
        Identified model type string.
    """
    # First check explicit model name in result
    for key in ["model_name", "model", "name", "architecture"]:
        if key in result:
            name = result[key].lower()
            for model_type, aliases in KNOWN_MODELS.items():
                if any(alias in name for alias in aliases):
                    return model_type
    
    # Check filepath
    filename = filepath.stem.lower()
    parent_dir = filepath.parent.name.lower()
    
    for model_type, aliases in KNOWN_MODELS.items():
        if any(alias in filename or alias in parent_dir for alias in aliases):
            return model_type
    
    return "unknown"


def extract_metrics_from_result(result: Dict) -> Dict[str, Optional[float]]:
    """
    Extract metrics from a result dictionary with various schema support.
    
    Args:
        result: Raw result dictionary.
        
    Returns:
        Dictionary of normalized metrics.
    """
    metrics = {
        "auroc": None,
        "accuracy": None,
        "precision": None,
        "recall": None,
        "specificity": None,
        "f1_score": None,
        "threshold": None,
    }
    
    # Define aliases for each metric
    metric_aliases = {
        "auroc": ["auroc", "auc_roc", "roc_auc", "auc", "AUROC"],
        "accuracy": ["accuracy", "acc", "Accuracy"],
        "precision": ["precision", "prec", "Precision", "ppv"],
        "recall": ["recall", "sensitivity", "sens", "tpr", "Recall", "Sensitivity"],
        "specificity": ["specificity", "spec", "tnr", "Specificity"],
        "f1_score": ["f1_score", "f1", "f_score", "F1", "F1_score"],
        "threshold": ["threshold", "thresh", "optimal_threshold", "Threshold"],
    }
    
    # Search locations in the result dict
    search_locations = [
        result,
        result.get("metrics", {}),
        result.get("results", {}),
        result.get("evaluation", {}),
        result.get("test_metrics", {}),
        result.get("best_metrics", {}),
    ]
    
    for metric_key, aliases in metric_aliases.items():
        for location in search_locations:
            if not isinstance(location, dict):
                continue
            for alias in aliases:
                if alias in location:
                    value = location[alias]
                    if isinstance(value, (int, float)) and not np.isnan(value):
                        metrics[metric_key] = float(value)
                        break
            if metrics[metric_key] is not None:
                break
    
    return metrics


def build_master_dataframe(results_dict: Dict[str, List[Path]]) -> pd.DataFrame:
    """
    Build a master DataFrame from all result files.
    
    Args:
        results_dict: Dictionary mapping category to list of file paths.
        
    Returns:
        DataFrame with all model results.
    """
    rows = []
    processed_files = set()
    
    all_files = results_dict.get("all", [])
    
    for filepath in all_files:
        if filepath in processed_files:
            continue
        
        result = load_results_json(filepath)
        if result is None:
            continue
        
        processed_files.add(filepath)
        
        # Identify model type
        model_type = identify_model_type(result, filepath)
        
        # Extract metrics
        metrics = extract_metrics_from_result(result)
        
        # Build row
        row = {
            "model_name": result.get("model_name", filepath.stem),
            "model_type": model_type,
            "source_file": str(filepath),
            **metrics,
        }
        
        # Extract additional info if available
        for key in ["training_epochs", "score_method", "notes"]:
            if key in result:
                row[key] = result[key]
        
        rows.append(row)
    
    if not rows:
        return pd.DataFrame()
    
    df = pd.DataFrame(rows)
    
    # Sort by AUROC descending
    if "auroc" in df.columns:
        df = df.sort_values("auroc", ascending=False, na_position="last")
    
    return df.reset_index(drop=True)


# =============================================================================
# TABLE GENERATION
# =============================================================================

def generate_chapter8_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate a publication-ready table for Chapter 8.
    
    Args:
        df: Master results DataFrame.
        
    Returns:
        Formatted DataFrame for dissertation.
    """
    # Select columns for publication
    columns = [
        "model_name", "auroc", "accuracy", "precision", 
        "recall", "specificity", "f1_score"
    ]
    
    # Filter to available columns
    available_cols = [c for c in columns if c in df.columns]
    
    chapter_df = df[available_cols].copy()
    
    # Add rank column
    chapter_df.insert(0, "Rank", range(1, len(chapter_df) + 1))
    
    # Rename columns for publication
    column_names = {
        "model_name": "Model",
        "auroc": "AUROC",
        "accuracy": "Accuracy",
        "precision": "Precision",
        "recall": "Recall",
        "specificity": "Specificity",
        "f1_score": "F1 Score",
    }
    chapter_df = chapter_df.rename(columns=column_names)
    
    return chapter_df


def generate_ranked_tables(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """
    Generate tables ranked by different metrics.
    
    Args:
        df: Master results DataFrame.
        
    Returns:
        Dictionary of ranked DataFrames by metric.
    """
    ranked_tables = {}
    
    metrics_to_rank = ["auroc", "accuracy", "recall", "specificity", "f1_score"]
    
    for metric in metrics_to_rank:
        if metric in df.columns and df[metric].notna().any():
            ranked_df = df.sort_values(metric, ascending=False, na_position="last")
            ranked_df = ranked_df.reset_index(drop=True)
            ranked_df.insert(0, "rank", range(1, len(ranked_df) + 1))
            ranked_tables[metric] = ranked_df
    
    return ranked_tables


# =============================================================================
# MAIN FUNCTION
# =============================================================================

def build_master_results_table(
    root: Optional[Path] = None,
    save_outputs: bool = True,
    verbose: bool = True
) -> Tuple[pd.DataFrame, Dict]:
    """
    Main function to build the master results table.
    
    Args:
        root: Project root path (uses Drive default if None).
        save_outputs: Whether to save outputs to Drive.
        verbose: Whether to print progress messages.
        
    Returns:
        Tuple of (master DataFrame, metadata dict).
    """
    # Ensure output directories exist
    ensure_directories_exist()
    
    # Start logging
    log_name = start_experiment_log(
        "build_master_results",
        params={"root": str(root) if root else "default"}
    )
    
    # Get project root
    if root is None:
        try:
            root = get_drive_project_root()
        except FileNotFoundError as e:
            log_message(f"Error: {e}", log_name)
            raise
    
    if verbose:
        print(f"Project root: {root}")
    
    # Find all result JSON files
    results_dict = find_results_jsons(root)
    
    n_files = len(results_dict.get("all", []))
    log_message(f"Found {n_files} result JSON files", log_name)
    
    if n_files == 0:
        log_message("No result files found. Check that results are in Drive.", log_name)
        return pd.DataFrame(), {"error": "No result files found"}
    
    # Build master DataFrame
    master_df = build_master_dataframe(results_dict)
    
    if master_df.empty:
        log_message("Failed to build master DataFrame - check result file formats", log_name)
        return pd.DataFrame(), {"error": "Failed to parse result files"}
    
    log_message(f"Built master table with {len(master_df)} models", log_name)
    
    # Generate publication table
    chapter8_df = generate_chapter8_table(master_df)
    
    # Generate ranked tables
    ranked_tables = generate_ranked_tables(master_df)
    
    # Prepare metadata
    metadata = {
        "n_models": len(master_df),
        "n_source_files": n_files,
        "models_found": master_df["model_name"].tolist(),
        "model_types": master_df["model_type"].value_counts().to_dict() if "model_type" in master_df.columns else {},
        "metrics_available": [c for c in master_df.columns if c not in ["model_name", "model_type", "source_file"]],
    }
    
    # Save outputs
    if save_outputs:
        try:
            # Save master CSV
            save_csv(master_df, "master_model_results.csv")
            
            # Save master markdown
            save_markdown_table(
                chapter8_df, 
                "master_model_results.md",
                title="Model Comparison Results"
            )
            
            # Save master JSON
            save_json(
                {
                    "models": master_df.to_dict(orient="records"),
                    "metadata": metadata,
                },
                "master_model_results.json"
            )
            
            # Save ranked tables
            for metric, ranked_df in ranked_tables.items():
                save_csv(ranked_df, f"ranked_by_{metric}.csv")
            
            log_message("All outputs saved successfully", log_name)
            
        except Exception as e:
            log_message(f"Error saving outputs: {e}", log_name)
    
    # End logging
    end_experiment_log(log_name, summary={
        "n_models": len(master_df),
        "best_auroc_model": master_df.iloc[0]["model_name"] if not master_df.empty else "N/A",
        "best_auroc": f"{master_df.iloc[0]['auroc']:.4f}" if not master_df.empty and pd.notna(master_df.iloc[0].get('auroc')) else "N/A",
    })
    
    if verbose:
        print("\n" + "=" * 60)
        print("MASTER RESULTS TABLE")
        print("=" * 60)
        print(chapter8_df.to_string(index=False))
        print("=" * 60)
    
    return master_df, metadata


if __name__ == "__main__":
    print("Building master results table...")
    print("=" * 60)
    
    try:
        df, metadata = build_master_results_table()
        
        print(f"\nSummary:")
        print(f"  Models found: {metadata.get('n_models', 0)}")
        print(f"  Source files: {metadata.get('n_source_files', 0)}")
        
        if metadata.get('model_types'):
            print(f"\nModel types:")
            for model_type, count in metadata['model_types'].items():
                print(f"    {model_type}: {count}")
        
    except Exception as e:
        print(f"Error: {e}")
        raise
