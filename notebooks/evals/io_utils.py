"""
I/O utilities for SymAD-ECNN evaluation package.

This module provides helper functions for file I/O operations including
JSON, CSV, markdown tables, and logging. All outputs are saved to the
Google Drive evaluations directory.

Author: SymAD-ECNN Project
Purpose: Centralized I/O functions for evaluation outputs
"""

import json
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Union, Any
import numpy as np

# Import configuration
try:
    from config import (
        EVALUATIONS_ROOT, TABLES_DIR, FIGURES_DIR, 
        JSON_DIR, LOGS_DIR, ensure_directories_exist
    )
except ImportError:
    # Fallback defaults
    EVALUATIONS_ROOT = Path("/content/drive/MyDrive/symAD-ECNN/evaluations")
    TABLES_DIR = EVALUATIONS_ROOT / "tables"
    JSON_DIR = EVALUATIONS_ROOT / "json"
    LOGS_DIR = EVALUATIONS_ROOT / "logs"


# =============================================================================
# DIRECTORY MANAGEMENT
# =============================================================================

def ensure_dir(path: Union[str, Path]) -> Path:
    """
    Ensure a directory exists, creating it if necessary.
    
    Args:
        path: Directory path.
        
    Returns:
        Path object for the directory.
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_output_path(
    filename: str,
    output_type: str = "json",
    subdir: Optional[str] = None
) -> Path:
    """
    Get the full output path for a file.
    
    Args:
        filename: Base filename.
        output_type: Type of output - "json", "table", "figure", "log".
        subdir: Optional subdirectory within the output type folder.
        
    Returns:
        Full path for the output file.
    """
    type_dirs = {
        "json": JSON_DIR,
        "table": TABLES_DIR,
        "csv": TABLES_DIR,
        "figure": FIGURES_DIR,
        "log": LOGS_DIR,
    }
    
    base_dir = type_dirs.get(output_type.lower(), EVALUATIONS_ROOT)
    
    if subdir:
        output_dir = base_dir / subdir
    else:
        output_dir = base_dir
    
    ensure_dir(output_dir)
    return output_dir / filename


# =============================================================================
# JSON I/O
# =============================================================================

class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder that handles numpy types."""
    
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, Path):
            return str(obj)
        return super().default(obj)


def save_json(
    data: Dict,
    filename: str,
    subdir: Optional[str] = None,
    indent: int = 2
) -> Path:
    """
    Save data to a JSON file in the evaluations directory.
    
    Args:
        data: Dictionary to save.
        filename: Filename (with or without .json extension).
        subdir: Optional subdirectory within json folder.
        indent: JSON indentation level.
        
    Returns:
        Path to the saved file.
    """
    if not filename.endswith('.json'):
        filename = f"{filename}.json"
    
    filepath = get_output_path(filename, "json", subdir)
    
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=indent, cls=NumpyEncoder)
    
    print(f"JSON saved: {filepath}")
    return filepath


def load_json(filepath: Union[str, Path]) -> Optional[Dict]:
    """
    Load a JSON file with error handling.
    
    Args:
        filepath: Path to JSON file.
        
    Returns:
        Parsed JSON data or None if loading fails.
    """
    filepath = Path(filepath)
    
    if not filepath.exists():
        print(f"Warning: JSON file not found: {filepath}")
        return None
    
    try:
        with open(filepath, 'r') as f:
            return json.load(f)
    except json.JSONDecodeError as e:
        print(f"Error parsing JSON {filepath}: {e}")
        return None
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        return None


def save_experiment_results(
    results: Dict,
    experiment_name: str,
    subdir: Optional[str] = None,
    add_timestamp: bool = True
) -> Path:
    """
    Save experiment results with optional timestamp.
    
    Args:
        results: Results dictionary.
        experiment_name: Base name for the experiment.
        subdir: Optional subdirectory.
        add_timestamp: Whether to add timestamp to filename.
        
    Returns:
        Path to saved file.
    """
    # Add metadata
    results["_metadata"] = {
        "experiment_name": experiment_name,
        "saved_at": datetime.now().isoformat(),
    }
    
    if add_timestamp:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{experiment_name}_{timestamp}.json"
    else:
        filename = f"{experiment_name}.json"
    
    return save_json(results, filename, subdir)


# =============================================================================
# CSV I/O
# =============================================================================

def save_csv(
    df: pd.DataFrame,
    filename: str,
    subdir: Optional[str] = None,
    index: bool = False
) -> Path:
    """
    Save DataFrame to CSV in the tables directory.
    
    Args:
        df: DataFrame to save.
        filename: Filename (with or without .csv extension).
        subdir: Optional subdirectory.
        index: Whether to include row index.
        
    Returns:
        Path to saved file.
    """
    if not filename.endswith('.csv'):
        filename = f"{filename}.csv"
    
    filepath = get_output_path(filename, "csv", subdir)
    df.to_csv(filepath, index=index)
    
    print(f"CSV saved: {filepath}")
    return filepath


def load_csv(filepath: Union[str, Path]) -> Optional[pd.DataFrame]:
    """
    Load a CSV file with error handling.
    
    Args:
        filepath: Path to CSV file.
        
    Returns:
        DataFrame or None if loading fails.
    """
    filepath = Path(filepath)
    
    if not filepath.exists():
        print(f"Warning: CSV file not found: {filepath}")
        return None
    
    try:
        return pd.read_csv(filepath)
    except Exception as e:
        print(f"Error loading CSV {filepath}: {e}")
        return None


# =============================================================================
# MARKDOWN TABLE I/O
# =============================================================================

def save_markdown_table(
    df: pd.DataFrame,
    filename: str,
    title: Optional[str] = None,
    subdir: Optional[str] = None,
    float_format: str = ".4f"
) -> Path:
    """
    Save DataFrame as a markdown table.
    
    Args:
        df: DataFrame to save.
        filename: Filename (with or without .md extension).
        title: Optional title for the table.
        subdir: Optional subdirectory.
        float_format: Format string for float values.
        
    Returns:
        Path to saved file.
    """
    if not filename.endswith('.md'):
        filename = f"{filename}.md"
    
    filepath = get_output_path(filename, "table", subdir)
    
    # Format float columns
    formatted_df = df.copy()
    for col in formatted_df.select_dtypes(include=[np.floating]).columns:
        formatted_df[col] = formatted_df[col].apply(
            lambda x: f"{x:{float_format}}" if pd.notna(x) else "-"
        )
    
    # Generate markdown
    md_content = []
    
    if title:
        md_content.append(f"# {title}\n")
    
    md_content.append(formatted_df.to_markdown(index=False))
    md_content.append(f"\n\n*Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*")
    
    with open(filepath, 'w') as f:
        f.write('\n'.join(md_content))
    
    print(f"Markdown table saved: {filepath}")
    return filepath


def df_to_markdown_string(
    df: pd.DataFrame,
    float_format: str = ".4f"
) -> str:
    """
    Convert DataFrame to markdown string for notebook display.
    
    Args:
        df: DataFrame to convert.
        float_format: Format string for float values.
        
    Returns:
        Markdown formatted string.
    """
    formatted_df = df.copy()
    for col in formatted_df.select_dtypes(include=[np.floating]).columns:
        formatted_df[col] = formatted_df[col].apply(
            lambda x: f"{x:{float_format}}" if pd.notna(x) else "-"
        )
    return formatted_df.to_markdown(index=False)


# =============================================================================
# LOGGING
# =============================================================================

def get_log_path(log_name: str) -> Path:
    """
    Get path for a log file.
    
    Args:
        log_name: Base name for the log.
        
    Returns:
        Full path for the log file.
    """
    ensure_dir(LOGS_DIR)
    return LOGS_DIR / f"{log_name}.txt"


def log_message(
    message: str,
    log_name: str = "evaluation_log",
    print_msg: bool = True
) -> None:
    """
    Log a message to file and optionally print.
    
    Args:
        message: Message to log.
        log_name: Name of the log file.
        print_msg: Whether to also print the message.
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    formatted_msg = f"[{timestamp}] {message}"
    
    log_path = get_log_path(log_name)
    
    with open(log_path, 'a') as f:
        f.write(formatted_msg + "\n")
    
    if print_msg:
        print(formatted_msg)


def start_experiment_log(
    experiment_name: str,
    params: Optional[Dict] = None
) -> str:
    """
    Start a new experiment log.
    
    Args:
        experiment_name: Name of the experiment.
        params: Optional parameters to log.
        
    Returns:
        The log name.
    """
    log_name = f"{experiment_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    log_message(f"=" * 60, log_name, print_msg=False)
    log_message(f"Experiment: {experiment_name}", log_name)
    log_message(f"Started at: {datetime.now().isoformat()}", log_name)
    
    if params:
        log_message("Parameters:", log_name)
        for key, value in params.items():
            log_message(f"  {key}: {value}", log_name)
    
    log_message(f"=" * 60, log_name, print_msg=False)
    
    return log_name


def end_experiment_log(
    log_name: str,
    summary: Optional[Dict] = None
) -> None:
    """
    End an experiment log with optional summary.
    
    Args:
        log_name: Name of the log file.
        summary: Optional summary dict to log.
    """
    log_message(f"=" * 60, log_name, print_msg=False)
    log_message(f"Experiment completed at: {datetime.now().isoformat()}", log_name)
    
    if summary:
        log_message("Summary:", log_name)
        for key, value in summary.items():
            log_message(f"  {key}: {value}", log_name)
    
    log_message(f"=" * 60, log_name, print_msg=False)


# =============================================================================
# METADATA UTILITIES
# =============================================================================

def save_experiment_metadata(
    experiment_name: str,
    metadata: Dict,
    subdir: Optional[str] = None
) -> Path:
    """
    Save experiment metadata to JSON.
    
    Args:
        experiment_name: Name of the experiment.
        metadata: Metadata dictionary.
        subdir: Optional subdirectory.
        
    Returns:
        Path to saved metadata file.
    """
    metadata["_experiment_name"] = experiment_name
    metadata["_created_at"] = datetime.now().isoformat()
    
    filename = f"{experiment_name}_metadata.json"
    return save_json(metadata, filename, subdir)


def create_summary_report(
    title: str,
    sections: Dict[str, str],
    filename: str,
    subdir: Optional[str] = None
) -> Path:
    """
    Create a summary report in markdown format.
    
    Args:
        title: Report title.
        sections: Dictionary mapping section titles to content.
        filename: Output filename.
        subdir: Optional subdirectory.
        
    Returns:
        Path to saved report.
    """
    if not filename.endswith('.md'):
        filename = f"{filename}.md"
    
    filepath = get_output_path(filename, "log", subdir)
    
    content = [f"# {title}\n"]
    content.append(f"*Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*\n")
    
    for section_title, section_content in sections.items():
        content.append(f"\n## {section_title}\n")
        content.append(section_content)
    
    with open(filepath, 'w') as f:
        f.write('\n'.join(content))
    
    print(f"Report saved: {filepath}")
    return filepath


# =============================================================================
# INITIALIZATION
# =============================================================================

def initialize_output_directories() -> Dict[str, bool]:
    """
    Initialize all output directories.
    
    Returns:
        Dictionary indicating which directories were created successfully.
    """
    directories = {
        "evaluations_root": EVALUATIONS_ROOT,
        "tables": TABLES_DIR,
        "json": JSON_DIR,
        "logs": LOGS_DIR,
    }
    
    status = {}
    for name, path in directories.items():
        try:
            path.mkdir(parents=True, exist_ok=True)
            status[name] = True
            print(f"Directory ready: {path}")
        except Exception as e:
            status[name] = False
            print(f"Failed to create {path}: {e}")
    
    return status


if __name__ == "__main__":
    print("I/O utilities loaded successfully.")
    print(f"\nDefault output paths:")
    print(f"  JSON:   {JSON_DIR}")
    print(f"  Tables: {TABLES_DIR}")
    print(f"  Logs:   {LOGS_DIR}")
