# Results Directory

Training results, metrics, and visualizations are saved here.

## What Goes Here

### JSON Files (Commit to Git ✅)
- `baseline_results.json` - Metrics, loss curves, training time
- `cnn_results.json` - CNN-AE performance
- `ecnn_results.json` - ECNN-AE performance
- `comparison_results.json` - Side-by-side comparison

### Figures (Commit small PNGs ✅)
- `figures/baseline_roc.png` - ROC curves
- `figures/training_curves.png` - Loss over epochs
- `figures/model_comparison.png` - Bar charts
- `figures/rotation_invariance.png` - Equivariance validation

### Large Files (Store in Drive ❌)
- Don't commit large result files
- Use Drive for intermediate results

## File Format

### Example: baseline_results.json
```json
{
  "model": "Baseline Autoencoder",
  "parameters": 8500000,
  "training": {
    "epochs": 50,
    "batch_size": 32,
    "learning_rate": 0.001,
    "time": "25 minutes"
  },
  "performance": {
    "auroc": 0.78,
    "auprc": 0.72,
    "best_val_loss": 0.0234
  }
}
```

## Accessing Results

### In Notebooks
```python
import json

# Load results
with open('results/baseline_results.json', 'r') as f:
    results = json.load(f)

print(f"AUROC: {results['performance']['auroc']}")
```

### From Colab
```python
# Save to Drive during training
import json

results = {'auroc': 0.85, 'auprc': 0.78}
results_path = '/content/drive/MyDrive/symAD-ECNN/results/baseline_results.json'

with open(results_path, 'w') as f:
    json.dump(results, f, indent=2)
```

## Visualization Guidelines

- **Format**: PNG (for Git) or PDF (for paper)
- **DPI**: 300 (publication quality)
- **Size**: Keep under 1MB for Git

See **PROJECT_OVERVIEW.md** for expected results and metrics.
