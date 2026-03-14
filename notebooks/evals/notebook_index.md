# Evaluation Notebooks Index

## Overview

This document provides a guide to all evaluation notebooks in the SymAD-ECNN project, organized by dissertation chapter alignment.

---

## Chapter 8.3: Model Testing

### Prototype Testing Notebook
**File**: `prototype_testing/prototype_testing.ipynb`

**Purpose**: Functional testing of the Flask backend prototype system.

**Sections**:
1. Environment Setup (Drive mounting, imports)
2. API Configuration
3. API Smoke Tests (health check, prediction endpoint)
4. Volume Inference Tests (multi-slice processing)
5. Aggregation Method Comparison
6. Throughput Analysis (latency, throughput metrics)
7. Summary Report Generation

**Prerequisites**:
- Flask backend must be running (or skip API tests)
- Test images available in data directories

**Key Outputs**:
- `api_smoke_test_results.json`
- `volume_inference_results.json`
- `prototype_test_report.json`
- `inference_latency_analysis.png`

---

## Chapter 8.4: Benchmarking

### Model Comparison Notebook
**File**: `model_comparison/evaluation_model_comparison.ipynb`

**Purpose**: Comprehensive comparison of all trained models (ECNN, CNN-AE, ResNet baseline).

**Sections**:
1. Environment Setup
2. Data Discovery (find all result JSONs)
3. Results Aggregation (build master table)
4. Performance Comparison (bar charts, tables)
5. ROC Curve Analysis (overlaid ROC curves)
6. Statistical Analysis (confidence intervals)
7. Dissertation Table Generation

**Prerequisites**:
- Trained model results in `results/` directory
- Result JSON files from training notebooks

**Key Outputs**:
- `master_results.json`
- `model_comparison_table.csv`
- `model_roc_comparison.png`
- `model_performance_bars.png`
- `chapter8_model_comparison.md`

---

## Chapter 8.5: Further Evaluations

### ECNN Thresholding Notebook
**File**: `ecnn_thresholding/evaluation_ecnn_thresholding.ipynb`

**Purpose**: Detailed analysis of ECNN model behavior under different thresholding strategies.

**Sections**:
1. Environment Setup
2. Model Loading (ECNN checkpoint)
3. Score Method Experiments (MSE, MAE, SSIM)
4. Threshold Experiments (FPR-based thresholds)
5. TP/FP/FN/TN Visualization Panels
6. Threshold Sensitivity Analysis
7. Results Summary

**Prerequisites**:
- Trained ECNN checkpoint
- IXI validation/test data
- BraTS test data

**Key Outputs**:
- `ecnn_threshold_experiments.json`
- `ecnn_score_method_comparison.csv`
- `tp_fp_fn_tn_panels.png`
- `threshold_sensitivity.png`

### Localization Evaluation Notebook (Optional)
**File**: `localization/localization_eval_optional.ipynb`

**Purpose**: Pixel-level anomaly localization evaluation using BraTS tumor masks.

**Sections**:
1. Environment Setup
2. Data Availability Check
3. Tumor Mask Extraction (from BraTS NIfTI)
4. Error Map Generation
5. Pixel-Level Metric Computation (Dice, IoU, AUROC)
6. Visualization of Localization Quality
7. Correlation Analysis (tumor size vs. localization)
8. Final Report

**Prerequisites**:
- BraTS 2021 raw data with segmentation masks
- nibabel library for NIfTI loading

**Key Outputs**:
- `brats_mask_extraction_summary.json`
- `localization_full_results.csv`
- `localization_metric_distributions.png`
- `localization_final_report.json`

---

## Quick Reference Table

| Notebook | Chapter | Primary Focus | Key Metrics |
|----------|---------|---------------|-------------|
| `prototype_testing.ipynb` | 8.3 | API testing, volume inference | Latency, throughput, success rate |
| `evaluation_model_comparison.ipynb` | 8.3, 8.4 | Model benchmarking | AUROC, F1, accuracy, precision |
| `evaluation_ecnn_thresholding.ipynb` | 8.5 | ECNN analysis | Threshold sensitivity, score methods |
| `localization_eval_optional.ipynb` | 8.5 | Pixel-level evaluation | Dice, IoU, pixel AUROC |

---

## Recommended Execution Order

1. **First**: `evaluation_model_comparison.ipynb`
   - Establishes baseline comparisons
   - Generates master results table

2. **Second**: `evaluation_ecnn_thresholding.ipynb`
   - Deep-dive into best model (ECNN)
   - Threshold optimization analysis

3. **Third**: `prototype_testing.ipynb`
   - Tests deployed system
   - Requires running Flask backend

4. **Optional**: `localization_eval_optional.ipynb`
   - Requires BraTS raw data with masks
   - Provides pixel-level analysis

---

## Common Operations

### Run All Notebooks in Sequence

```python
# In Colab, execute notebooks programmatically
import subprocess

notebooks = [
    'model_comparison/evaluation_model_comparison.ipynb',
    'ecnn_thresholding/evaluation_ecnn_thresholding.ipynb',
    'prototype_testing/prototype_testing.ipynb',
    # 'localization/localization_eval_optional.ipynb',  # Optional
]

for nb in notebooks:
    subprocess.run(['jupyter', 'nbconvert', '--execute', nb, '--to', 'notebook'])
```

### Export All Figures

All figures are automatically saved to:
```
/content/drive/MyDrive/symAD-ECNN/evaluations/figures/
```

### Export All Tables

All tables are saved to:
```
/content/drive/MyDrive/symAD-ECNN/evaluations/tables/
```

---

## Dissertation Integration

### Figures for Chapter 8

| Figure | Source Notebook | Output File |
|--------|-----------------|-------------|
| Model ROC Comparison | `evaluation_model_comparison.ipynb` | `model_roc_comparison.png` |
| Performance Bar Chart | `evaluation_model_comparison.ipynb` | `model_performance_bars.png` |
| TP/FP/FN/TN Panels | `evaluation_ecnn_thresholding.ipynb` | `tp_fp_fn_tn_panels.png` |
| Threshold Sensitivity | `evaluation_ecnn_thresholding.ipynb` | `threshold_sensitivity.png` |
| Latency Distribution | `prototype_testing.ipynb` | `inference_latency_analysis.png` |
| Localization Examples | `localization_eval_optional.ipynb` | `localization_best_worst_examples.png` |

### Tables for Chapter 8

| Table | Source Notebook | Output File |
|-------|-----------------|-------------|
| Model Comparison | `evaluation_model_comparison.ipynb` | `chapter8_model_comparison.md` |
| Threshold Experiments | `evaluation_ecnn_thresholding.ipynb` | `ecnn_threshold_experiments.csv` |
| API Test Results | `prototype_testing.ipynb` | `prototype_test_results.md` |
| Localization Metrics | `localization_eval_optional.ipynb` | `localization_results.md` |

---

## Notes

- All notebooks are designed to be run in Google Colab
- Outputs are saved to Google Drive under `evaluations/`
- Each notebook can be run independently
- For local execution, modify paths in `config.py`
