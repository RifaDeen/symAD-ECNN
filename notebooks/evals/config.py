"""
Configuration module for SymAD-ECNN evaluation package.

This module defines all paths, directories, and default parameters used across
the evaluation notebooks and scripts. All paths assume execution in Google Colab
with Google Drive mounted.
"""

from pathlib import Path
import os

# =============================================================================
# GOOGLE DRIVE PROJECT PATHS
# =============================================================================

# Root directory of the project in Google Drive (Colab execution)
DRIVE_PROJECT_ROOT = Path("/content/drive/MyDrive/symAD-ECNN")

# Evaluation outputs directory in Google Drive
EVALUATIONS_ROOT = DRIVE_PROJECT_ROOT / "evaluations"

# Subdirectories for evaluation outputs
TABLES_DIR = EVALUATIONS_ROOT / "tables"
FIGURES_DIR = EVALUATIONS_ROOT / "figures"
JSON_DIR = EVALUATIONS_ROOT / "json"
LOGS_DIR = EVALUATIONS_ROOT / "logs"

# =============================================================================
# PROJECT DATA AND MODEL PATHS
# =============================================================================

# Models directory
MODELS_DIR = DRIVE_PROJECT_ROOT / "models"
SAVED_MODELS_DIR = MODELS_DIR / "saved_models"

# Results directory (contains training result JSONs)
RESULTS_DIR = DRIVE_PROJECT_ROOT / "results"

# Data directory
DATA_DIR = DRIVE_PROJECT_ROOT / "data"
IXI_DATA_DIR = DATA_DIR / "ixi_t1"
BRATS_DATA_DIR = DATA_DIR / "brats_t1"
BRATS_RAW_DIR = DATA_DIR / "brats2021"

# Demo app paths
DEMO_APP_DIR = DRIVE_PROJECT_ROOT / "demo_app"
BACKEND_DIR = DEMO_APP_DIR / "backend"
FRONTEND_DIR = DEMO_APP_DIR / "frontend"

# =============================================================================
# ECNN EVALUATION DEFAULT PARAMETERS
# =============================================================================

# Score computation methods for anomaly detection
# - "mean": Mean of error values within brain mask
# - "p95": 95th percentile of error values within brain mask
# - "p90": 90th percentile of error values within brain mask
ECNN_DEFAULT_SCORE_METHODS = ["mean", "p95", "p90"]

# Target false positive rates for threshold computation
# Used to set thresholds based on normal validation data
ECNN_DEFAULT_FPRS = [0.05, 0.10, 0.20]

# Error computation mode
# - "abs": Absolute error |input - reconstruction|
# - "squared": Squared error (input - reconstruction)^2
ECNN_DEFAULT_ERROR_MODE = "abs"

# Minimum number of brain pixels required for valid score computation
# Slices with fewer brain pixels are considered invalid
ECNN_DEFAULT_MIN_BRAIN_PIXELS = 50

# Default threshold experiments to run
# Each tuple: (score_method, threshold_method, threshold_param)
ECNN_DEFAULT_EXPERIMENTS = [
    ("mean", "reference", None),      # Original threshold from training
    ("mean", "fpr", 0.10),            # FPR-controlled at 10%
    ("mean", "fpr", 0.20),            # FPR-controlled at 20%
    ("p95", "fpr", 0.05),             # 95th percentile, FPR 5%
    ("p95", "fpr", 0.10),             # 95th percentile, FPR 10%
    ("p95", "fpr", 0.20),             # 95th percentile, FPR 20%
    ("p90", "fpr", 0.20),             # 90th percentile, FPR 20%
]

# =============================================================================
# PROTOTYPE TESTING DEFAULTS
# =============================================================================

# Default API URL for Flask backend testing
DEFAULT_API_URL = "http://localhost:5000"

# API endpoints
API_HEALTH_ENDPOINT = "/health"
API_PREDICT_ENDPOINT = "/predict"

# Test timeout in seconds
API_TEST_TIMEOUT = 30

# =============================================================================
# VISUALIZATION DEFAULTS
# =============================================================================

# Figure settings
DEFAULT_FIGURE_DPI = 150
DEFAULT_FIGURE_FORMAT = "png"

# Color maps
HEATMAP_CMAP = "hot"
ERROR_CMAP = "jet"
BRAIN_CMAP = "gray"

# Panel display settings
MAX_DISPLAY_SAMPLES = 8

# =============================================================================
# MODEL NAMES AND IDENTIFIERS
# =============================================================================

# Known model identifiers for result JSON matching
KNOWN_MODELS = {
    "cnn_ae": ["cnn_ae", "cnn-ae", "cnn_autoencoder", "baseline"],
    "cnn_ae_large": ["cnn_ae_large", "large_cnn_ae", "cnn_autoencoder_large"],
    "cnn_ae_augmented": ["cnn_ae_augmented", "augmented", "aug"],
    "resnet_feature": ["resnet_feature", "resnet_feat", "feature_distance"],
    "resnet_ae": ["resnet_ae", "resnet_autoencoder"],
    "resnet_finetuned": ["resnet_finetuned", "resnet_ft", "finetuned"],
    "ecnn": ["ecnn", "e2cnn", "equivariant"],
    "ecnn_optimized": ["ecnn_optimized", "ecnn_opt", "ecnn_v2", "ecnn_v3"],
}

# =============================================================================
# DIRECTORY CREATION FUNCTION
# =============================================================================

def ensure_directories_exist():
    """
    Create all required output directories if they do not exist.
    
    This function should be called at the start of any evaluation notebook
    or script to ensure the output directory structure is in place.
    
    Returns:
        dict: Dictionary mapping directory names to their paths and creation status.
    """
    directories = {
        "evaluations_root": EVALUATIONS_ROOT,
        "tables": TABLES_DIR,
        "figures": FIGURES_DIR,
        "json": JSON_DIR,
        "logs": LOGS_DIR,
        # Subdirectories for specific outputs
        "json_ecnn_threshold": JSON_DIR / "ecnn_threshold_experiments",
        "figures_localization": FIGURES_DIR / "localization",
        "figures_tp_fp_fn_tn": FIGURES_DIR / "tp_fp_fn_tn",
        "figures_roc_pr": FIGURES_DIR / "roc_pr_curves",
        "figures_comparisons": FIGURES_DIR / "model_comparisons",
    }
    
    status = {}
    for name, path in directories.items():
        try:
            path.mkdir(parents=True, exist_ok=True)
            status[name] = {"path": str(path), "created": True, "error": None}
        except Exception as e:
            status[name] = {"path": str(path), "created": False, "error": str(e)}
    
    return status


def print_config_summary():
    """
    Print a summary of the current configuration.
    
    Useful for debugging and verifying paths in notebooks.
    """
    print("=" * 60)
    print("SymAD-ECNN Evaluation Configuration Summary")
    print("=" * 60)
    print(f"\nProject Root:     {DRIVE_PROJECT_ROOT}")
    print(f"Evaluations Root: {EVALUATIONS_ROOT}")
    print(f"\nOutput Directories:")
    print(f"  Tables:  {TABLES_DIR}")
    print(f"  Figures: {FIGURES_DIR}")
    print(f"  JSON:    {JSON_DIR}")
    print(f"  Logs:    {LOGS_DIR}")
    print(f"\nData Directories:")
    print(f"  Models:  {MODELS_DIR}")
    print(f"  Results: {RESULTS_DIR}")
    print(f"  Data:    {DATA_DIR}")
    print(f"\nECNN Defaults:")
    print(f"  Score Methods: {ECNN_DEFAULT_SCORE_METHODS}")
    print(f"  Target FPRs:   {ECNN_DEFAULT_FPRS}")
    print(f"  Error Mode:    {ECNN_DEFAULT_ERROR_MODE}")
    print("=" * 60)


# =============================================================================
# COLAB ENVIRONMENT DETECTION
# =============================================================================

def is_running_in_colab():
    """
    Check if the code is running in Google Colab.
    
    Returns:
        bool: True if running in Colab, False otherwise.
    """
    try:
        import google.colab
        return True
    except ImportError:
        return False


def get_local_project_root():
    """
    Get the local project root for VS Code development.
    
    This is used when editing code locally but execution target is Colab.
    
    Returns:
        Path: Local project root path.
    """
    # Attempt to find project root by looking for known files
    current = Path(__file__).resolve().parent
    while current != current.parent:
        if (current / "README.md").exists() and (current / "notebooks").exists():
            return current
        current = current.parent
    return Path(__file__).resolve().parent.parent.parent


if __name__ == "__main__":
    # When run directly, print configuration summary
    print_config_summary()
    print(f"\nRunning in Colab: {is_running_in_colab()}")
