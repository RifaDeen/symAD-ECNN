# SymAD-ECNN Evaluation Package

## Overview

This evaluation package provides comprehensive tools for evaluating the SymAD-ECNN brain MRI anomaly detection system. The code is designed to run in Google Colab with outputs saved to Google Drive.


## Directory Structure

```
notebooks/evals/
├── README.md                           # This file
├── notebook_index.md                   # Guide to all notebooks
├── requirements_eval.txt               # Python dependencies
├── config.py                           # Central configuration
├── path_utils.py                       # File/path discovery utilities
├── metrics_utils.py                    # Metric computation functions
├── plotting_utils.py                   # Visualization functions
├── io_utils.py                         # I/O operations (JSON, CSV, logs)
│
├── model_comparison/                   # Chapter 8.3/8.4: Model Comparison
│   ├── build_master_results_table.py   # Aggregate all model results
│   └── evaluation_model_comparison.ipynb
│
├── ecnn_thresholding/                  # Chapter 8.5: ECNN Analysis
│   ├── ecnn_model_loader.py            # Load ECNN checkpoints
│   ├── run_ecnn_threshold_experiments.py
│   ├── visualize_tp_fp_fn_tn.py        # Classification visualization
│   └── evaluation_ecnn_thresholding.ipynb
│
├── prototype_testing/                  # Chapter 8.3: Functional Testing
│   ├── api_smoke_tests.py              # Flask API tests
│   ├── volume_inference_tests.py       # Multi-slice aggregation
│   └── prototype_testing.ipynb
│
└── localization/                       # Chapter 8.5: Optional Localization
    ├── extract_brats_mask_pairs.py     # Extract tumor masks
    ├── compute_pixel_metrics.py        # Pixel-level metrics
    └── localization_eval_optional.ipynb
```

## Output Directory

All evaluation outputs are saved to:
```
/content/drive/MyDrive/symAD-ECNN/evaluations/
├── json/           # JSON result files
├── figures/        # PNG figures and plots
├── tables/         # CSV and Markdown tables
└── logs/           # Experiment logs
```

## Quick Start

### 1. In Google Colab

```python
# Mount Drive
from google.colab import drive
drive.mount('/content/drive')

# Add to path
import sys
sys.path.insert(0, '/content/drive/MyDrive/symAD-ECNN/notebooks/evals')

# Import and configure
from config import ensure_directories_exist
ensure_directories_exist()
```

### 2. Run a Notebook

1. Open any `.ipynb` file from the `evals/` folder in Colab
2. Run the first cells to mount Drive and import modules
3. Follow the notebook sections to run evaluations

## Core Modules

### config.py
Central configuration for all evaluation scripts:
- `DRIVE_PROJECT_ROOT`: Path to project on Drive
- `EVALUATIONS_ROOT`: Path for evaluation outputs
- `ECNN_DEFAULT_SCORE_METHODS`: Score methods to test
- `ECNN_DEFAULT_FPRS`: False positive rates for thresholds
- `ensure_directories_exist()`: Create output directories

### path_utils.py
File discovery with Drive-aware fallbacks:
- `get_drive_project_root()`: Find project root
- `find_ecnn_checkpoint()`: Locate model checkpoint
- `find_results_jsons()`: Find all result JSON files
- `find_data_paths()`: Find data directories

### metrics_utils.py
Metric computation functions:
- `compute_score()`: Score methods (MSE, MAE, SSIM)
- `threshold_from_normal_scores()`: Compute thresholds
- `compute_binary_metrics()`: Accuracy, precision, recall, F1
- `compute_auroc()`: Area under ROC curve

### plotting_utils.py
Visualization functions:
- `plot_roc_curve()`: ROC curves with AUC
- `plot_confusion_matrix()`: Confusion matrix heatmaps
- `plot_metric_comparison()`: Bar charts for model comparison
- `plot_tp_fp_fn_tn_panels()`: Sample visualization panels
- `save_figure()`: Save to evaluations folder

### io_utils.py
I/O operations:
- `save_json()`, `load_json()`: JSON handling
- `save_csv()`: DataFrame to CSV
- `save_markdown_table()`: Markdown tables
- `log_message()`: Experiment logging

## Evaluation Notebooks

### 1. Model Comparison (`evaluation_model_comparison.ipynb`)
**Chapter 8.3 & 8.4**
- Aggregates results from all trained models (ECNN, CNN-AE, ResNet)
- Generates master comparison tables
- Creates ROC curve overlays and bar charts
- Outputs dissertation-ready figures

### 2. ECNN Thresholding (`evaluation_ecnn_thresholding.ipynb`)
**Chapter 8.5**
- Tests different score methods (MSE, MAE, SSIM)
- Experiments with threshold settings (FPR-based)
- Generates TP/FP/FN/TN visualization panels
- Analyzes threshold sensitivity

### 3. Prototype Testing (`prototype_testing.ipynb`)
**Chapter 8.3**
- Tests Flask backend API endpoints
- Validates health check and prediction endpoints
- Tests volume-level inference aggregation
- Measures throughput and latency

### 4. Localization Evaluation (`localization_eval_optional.ipynb`)
**Chapter 8.5 (Optional)**
- Extracts BraTS tumor masks as ground truth
- Computes pixel-level localization metrics
- Calculates Dice, IoU, AUROC at pixel level
- Requires BraTS raw data with segmentations

## Key Dependencies

```
torch>=1.9.0
e2cnn>=0.2.0
numpy>=1.19.0
pandas>=1.0.0
matplotlib>=3.3.0
scikit-learn>=0.24.0
Pillow>=8.0.0
tqdm>=4.50.0
requests>=2.25.0
nibabel>=3.2.0  # For localization only
```

## Configuration Options

Edit `config.py` to customize:

```python
# Score methods to test
ECNN_DEFAULT_SCORE_METHODS = ["mse", "mae", "ssim"]

# FPR values for threshold calculation
ECNN_DEFAULT_FPRS = [0.05, 0.10, 0.15, 0.20]

# API settings for prototype testing
DEFAULT_API_URL = "http://localhost:5000"
API_TEST_TIMEOUT = 30
```

## Usage Examples

### Compute metrics for a model

```python
from metrics_utils import compute_binary_metrics, compute_auroc

# Binary metrics at a threshold
metrics = compute_binary_metrics(y_true, y_pred)
print(f"Accuracy: {metrics['accuracy']:.4f}")
print(f"F1: {metrics['f1']:.4f}")

# ROC-based metrics
auroc = compute_auroc(y_true, scores)
print(f"AUROC: {auroc:.4f}")
```

### Save results to Drive

```python
from io_utils import save_json, save_csv
from plotting_utils import save_figure

# Save JSON
save_json(results_dict, "experiment_results.json")

# Save DataFrame
save_csv(df, "metrics_table.csv")

# Save figure
save_figure(fig, "roc_comparison.png")
```

### Load ECNN model

```python
from ecnn_thresholding.ecnn_model_loader import load_ecnn_model

model = load_ecnn_model(checkpoint_path)
model.eval()
```

## Troubleshooting

### Drive not mounted
```python
from google.colab import drive
drive.mount('/content/drive', force_remount=True)
```

### Module not found
```python
import sys
sys.path.insert(0, '/content/drive/MyDrive/symAD-ECNN/notebooks/evals')
```

### Missing dependencies
```python
!pip install e2cnn nibabel scikit-learn
```

### Path issues
```python
from path_utils import validate_paths
issues = validate_paths()
for issue in issues:
    print(issue)
```
