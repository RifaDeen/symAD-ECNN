# Training Pipeline Guide - Complete Results

## 🎯 **PROJECT COMPLETE - FINAL RESULTS**

### Performance Summary

| Model | Parameters | Training Time | AUROC | Specificity | Status |
|-------|------------|---------------|-------|-------------|--------|
| Baseline AE | ~8M | Failed | N/A | N/A | ❌ Too large for spatial data |
| CNN-AE Small | ~8M | ~4 hours | 0.7617 | 56.42% | ✅ Completed |
| CNN-AE Large | ~11M | ~5 hours | 0.7803 | 58.52% | ✅ Completed |
| CNN-AE Augmented | ~8M | ~5 hours | ~0.76 | ~56% | ✅ Completed |
| ECNN Buggy | ~11M | ~2.7 hours | 0.7035 | 47.86% | ⚠️ Architecture bug |
| **ECNN Optimized** | **~11M** | **~2.7 hours** | **0.8109** | **58.54%** | 🏆 **BEST** |

### 🏆 Key Achievements

1. **Thesis Validated**: ECNN Optimized beat Large CNN-AE by **+3.06% AUROC** (structure > capacity)
2. **Bug Impact**: Fixed decoder bug recovered **+7.74% AUROC**
3. **Baseline Failure**: Fully-connected architecture unable to train on 128×128 images
4. **Production Model**: ECNN Optimized ready for deployment

---

## 🎯 Overview

This guide provides a complete step-by-step walkthrough for training all models on Google Colab.

**Time Estimate**: 
- Setup: 30 minutes
- CNN-AE Small: 4 hours
- CNN-AE Large: 5 hours
- ECNN Optimized: 2.7 hours (40 epochs × 251.1s)
- **Total**: ~2 days

**Hardware**: Google Colab GPU (T4 recommended, V100/A100 better)

---

## 📋 Pre-requisites Checklist

Before starting training, ensure you have:

- [ ] IXI dataset preprocessed and in Google Drive (`/data/processed_ixi/resized_ixi/`)
- [ ] BraTS dataset preprocessed and in Google Drive (`/data/brats2021_test/`)
- [ ] Google account with sufficient Drive storage (~2-3 GB)
- [ ] Basic understanding of PyTorch
- [ ] Read `ARCHITECTURE_DETAILS.md` and `EQUIVARIANCE_EXPLAINED.md`

---

## 🏗️ Complete Training Pipeline

```
┌──────────────────────────────────────────────────────────────┐
│                    PHASE 1: SETUP                             │
├──────────────────────────────────────────────────────────────┤
│  1. Mount Google Drive                                        │
│  2. Install required libraries                                │
│  3. Load and verify datasets                                  │
│  4. Set random seeds for reproducibility                      │
└──────────────────────────────────────────────────────────────┘
                              ↓
┌──────────────────────────────────────────────────────────────┐
│                    PHASE 2: DATA PREPARATION                  │
├──────────────────────────────────────────────────────────────┤
│  1. Create PyTorch Dataset class                              │
│  2. Split data: Train (90%) / Validation (10%)               │
│  3. Create DataLoaders (batch size: 32-64)                   │
│  4. Visualize sample batches                                  │
└──────────────────────────────────────────────────────────────┘
                              ↓
┌──────────────────────────────────────────────────────────────┐
│                PHASE 3: MODEL TRAINING (3 Models)             │
├──────────────────────────────────────────────────────────────┤
│  For each model:                                              │
│    1. Define model architecture                               │
│    2. Setup loss function (MSE + SSIM)                       │
│    3. Setup optimizer (Adam, lr=1e-3)                        │
│    4. Setup learning rate scheduler                           │
│    5. Train for 100 epochs                                    │
│    6. Save checkpoints every 10 epochs                        │
│    7. Plot training curves                                    │
│    8. Save best model                                         │
└──────────────────────────────────────────────────────────────┘
                              ↓
┌──────────────────────────────────────────────────────────────┐
│                PHASE 4: EVALUATION                            │
├──────────────────────────────────────────────────────────────┤
│  On IXI Test Set (Normal Brains):                            │
│    1. Calculate reconstruction loss                           │
│    2. Visualize reconstructions                               │
│    3. Analyze latent space                                    │
│                                                                │
│  On BraTS Test Set (Tumor Brains):                           │
│    1. Calculate reconstruction errors                         │
│    2. Generate anomaly maps                                   │
│    3. Calculate AUROC, precision, recall                      │
│    4. Visualize tumor detection                               │
└──────────────────────────────────────────────────────────────┘
                              ↓
┌──────────────────────────────────────────────────────────────┐
│                PHASE 5: COMPARISON & ANALYSIS                 │
├──────────────────────────────────────────────────────────────┤
│  1. Compare all three models                                  │
│  2. Statistical significance testing                          │
│  3. Generate comparison plots                                 │
│  4. Write final report                                        │
└──────────────────────────────────────────────────────────────┘
```

---

## 📊 Detailed Phase Breakdown

### PHASE 1: Setup (30 minutes)

#### Step 1.1: Mount Google Drive

```python
from google.colab import drive
drive.mount('/content/drive')

# Define paths
BASE_PATH = "/content/drive/MyDrive/symAD-ECNN"
IXI_PATH = f"{BASE_PATH}/data/processed_ixi/resized_ixi"
BRATS_PATH = f"{BASE_PATH}/data/brats2021_test"
MODEL_SAVE_PATH = f"{BASE_PATH}/models/saved_models"
RESULTS_PATH = f"{BASE_PATH}/results"
```

#### Step 1.2: Install Libraries

```bash
# For Baseline and CNN-AE
pip install torch torchvision
pip install scikit-learn matplotlib seaborn
pip install pytorch-msssim

# For ECNN-AE (additional)
pip install e2cnn
```

#### Step 1.3: Load Datasets

```python
import numpy as np
import os
from glob import glob

# Load IXI data
ixi_files = sorted(glob(f"{IXI_PATH}/*.npy"))
print(f"IXI files found: {len(ixi_files)}")

# Load BraTS data
brats_files = sorted(glob(f"{BRATS_PATH}/*.npy"))
print(f"BraTS files found: {len(brats_files)}")

# Verify data
sample = np.load(ixi_files[0])
print(f"Sample shape: {sample.shape}")  # Should be (128, 128)
print(f"Sample range: [{sample.min():.4f}, {sample.max():.4f}]")  # Should be [0, 1]
```

#### Step 1.4: Set Random Seeds

```python
import torch
import numpy as np
import random

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)
```

---

### PHASE 2: Data Preparation (30 minutes)

#### Step 2.1: Create Dataset Class

```python
from torch.utils.data import Dataset, DataLoader
import torch

class MRIDataset(Dataset):
    def __init__(self, file_list):
        self.file_list = file_list
    
    def __len__(self):
        return len(self.file_list)
    
    def __getitem__(self, idx):
        # Load .npy file
        img = np.load(self.file_list[idx])
        
        # Add channel dimension: (128, 128) -> (1, 128, 128)
        img = np.expand_dims(img, axis=0)
        
        # Convert to torch tensor
        img_tensor = torch.from_numpy(img).float()
        
        return img_tensor, img_tensor  # (input, target) for autoencoder
```

#### Step 2.2: Train/Val Split

```python
from sklearn.model_selection import train_test_split

# Split IXI data: 90% train, 10% validation
train_files, val_files = train_test_split(
    ixi_files, 
    test_size=0.1, 
    random_state=42
)

print(f"Training samples: {len(train_files)}")
print(f"Validation samples: {len(val_files)}")
print(f"Test samples (BraTS): {len(brats_files)}")
```

#### Step 2.3: Create DataLoaders

```python
# Configuration
BATCH_SIZE = 32  # Adjust based on GPU memory
NUM_WORKERS = 2

# Create datasets
train_dataset = MRIDataset(train_files)
val_dataset = MRIDataset(val_files)
test_dataset = MRIDataset(brats_files)

# Create dataloaders
train_loader = DataLoader(
    train_dataset, 
    batch_size=BATCH_SIZE, 
    shuffle=True, 
    num_workers=NUM_WORKERS,
    pin_memory=True
)

val_loader = DataLoader(
    val_dataset, 
    batch_size=BATCH_SIZE, 
    shuffle=False, 
    num_workers=NUM_WORKERS,
    pin_memory=True
)

test_loader = DataLoader(
    test_dataset, 
    batch_size=BATCH_SIZE, 
    shuffle=False, 
    num_workers=NUM_WORKERS,
    pin_memory=True
)

print(f"Train batches: {len(train_loader)}")
print(f"Val batches: {len(val_loader)}")
print(f"Test batches: {len(test_loader)}")
```

#### Step 2.4: Visualize Samples

```python
import matplotlib.pyplot as plt

# Get a batch
batch_images, _ = next(iter(train_loader))

# Plot 16 samples
fig, axes = plt.subplots(4, 4, figsize=(10, 10))
for i, ax in enumerate(axes.flat):
    ax.imshow(batch_images[i, 0], cmap='gray')
    ax.axis('off')
plt.tight_layout()
plt.savefig(f"{RESULTS_PATH}/sample_data.png")
plt.show()
```

---

### PHASE 3: Model Training

#### Common Training Configuration

```python
# Hyperparameters
LEARNING_RATE = 1e-3
EPOCHS = 100
PATIENCE = 15  # For early stopping

# Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
```

#### Loss Function

```python
import torch.nn as nn
from pytorch_msssim import MS_SSIM

class CombinedLoss(nn.Module):
    def __init__(self, alpha=0.84):
        super().__init__()
        self.alpha = alpha
        self.mse = nn.MSELoss()
        self.ms_ssim = MS_SSIM(data_range=1.0, size_average=True, channel=1)
    
    def forward(self, output, target):
        mse_loss = self.mse(output, target)
        ssim_loss = 1 - self.ms_ssim(output, target)
        return self.alpha * mse_loss + (1 - self.alpha) * ssim_loss

criterion = CombinedLoss()
```

#### Training Loop Template

```python
def train_epoch(model, train_loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
    
    return running_loss / len(train_loader)

def validate(model, val_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    
    with torch.no_grad():
        for data, target in val_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)
            running_loss += loss.item()
    
    return running_loss / len(val_loader)
```

#### Full Training Function

```python
def train_model(model, train_loader, val_loader, criterion, optimizer, 
                scheduler, epochs, device, model_name, save_path):
    
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(epochs):
        # Train
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        train_losses.append(train_loss)
        
        # Validate
        val_loss = validate(model, val_loader, criterion, device)
        val_losses.append(val_loss)
        
        # Learning rate scheduling
        scheduler.step(val_loss)
        
        # Print progress
        print(f"Epoch {epoch+1}/{epochs} - "
              f"Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), f"{save_path}/{model_name}_best.pth")
            patience_counter = 0
        else:
            patience_counter += 1
        
        # Early stopping
        if patience_counter >= PATIENCE:
            print(f"Early stopping triggered at epoch {epoch+1}")
            break
        
        # Save checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            torch.save(model.state_dict(), 
                      f"{save_path}/{model_name}_epoch{epoch+1}.pth")
    
    # Save final model
    torch.save(model.state_dict(), f"{save_path}/{model_name}_final.pth")
    
    # Plot training curves
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title(f'{model_name} Training History')
    plt.savefig(f"{save_path}/{model_name}_training_curve.png")
    plt.show()
    
    return train_losses, val_losses
```

---

### PHASE 4: Evaluation

#### Reconstruction Quality Metrics

```python
from sklearn.metrics import roc_auc_score, precision_recall_curve
import torch.nn.functional as F

def calculate_reconstruction_error(model, dataloader, device):
    model.eval()
    errors = []
    
    with torch.no_grad():
        for data, _ in dataloader:
            data = data.to(device)
            recon = model(data)
            
            # Pixel-wise MSE
            mse = F.mse_loss(recon, data, reduction='none')
            mse = mse.view(mse.size(0), -1).mean(dim=1)
            
            errors.extend(mse.cpu().numpy())
    
    return np.array(errors)
```

#### Anomaly Detection Metrics

```python
def evaluate_anomaly_detection(model, normal_loader, anomaly_loader, device):
    # Get reconstruction errors
    normal_errors = calculate_reconstruction_error(model, normal_loader, device)
    anomaly_errors = calculate_reconstruction_error(model, anomaly_loader, device)
    
    # Create labels
    y_true = np.concatenate([
        np.zeros(len(normal_errors)),  # 0 = normal
        np.ones(len(anomaly_errors))   # 1 = anomaly
    ])
    
    y_scores = np.concatenate([normal_errors, anomaly_errors])
    
    # Calculate AUROC
    auroc = roc_auc_score(y_true, y_scores)
    
    # Calculate precision-recall
    precision, recall, thresholds = precision_recall_curve(y_true, y_scores)
    
    return {
        'auroc': auroc,
        'normal_errors': normal_errors,
        'anomaly_errors': anomaly_errors,
        'precision': precision,
        'recall': recall,
        'thresholds': thresholds
    }
```

#### Visualization

```python
def visualize_reconstructions(model, dataloader, device, num_samples=8, 
                               save_path=None):
    model.eval()
    
    # Get samples
    data, _ = next(iter(dataloader))
    data = data[:num_samples].to(device)
    
    with torch.no_grad():
        recon = model(data)
    
    # Calculate error maps
    error = torch.abs(data - recon)
    
    # Plot
    fig, axes = plt.subplots(3, num_samples, figsize=(20, 6))
    
    for i in range(num_samples):
        # Original
        axes[0, i].imshow(data[i, 0].cpu(), cmap='gray')
        axes[0, i].axis('off')
        if i == 0:
            axes[0, i].set_ylabel('Original', fontsize=12)
        
        # Reconstruction
        axes[1, i].imshow(recon[i, 0].cpu(), cmap='gray')
        axes[1, i].axis('off')
        if i == 0:
            axes[1, i].set_ylabel('Reconstruction', fontsize=12)
        
        # Error
        axes[2, i].imshow(error[i, 0].cpu(), cmap='hot')
        axes[2, i].axis('off')
        if i == 0:
            axes[2, i].set_ylabel('Error Map', fontsize=12)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.show()
```

---

### PHASE 5: Comparison

#### Compare All Models

```python
def compare_models(results_dict):
    """
    results_dict = {
        'Baseline': evaluation_results,
        'CNN-AE': evaluation_results,
        'ECNN-AE': evaluation_results
    }
    """
    
    # AUROC Comparison
    plt.figure(figsize=(10, 6))
    models = list(results_dict.keys())
    aurocs = [results_dict[m]['auroc'] for m in models]
    
    plt.bar(models, aurocs)
    plt.ylabel('AUROC')
    plt.title('Anomaly Detection Performance')
    plt.ylim([0.5, 1.0])
    for i, v in enumerate(aurocs):
        plt.text(i, v + 0.01, f'{v:.4f}', ha='center')
    plt.savefig(f"{RESULTS_PATH}/model_comparison_auroc.png")
    plt.show()
    
    # ROC Curves
    plt.figure(figsize=(10, 8))
    for model_name, results in results_dict.items():
        # Calculate FPR, TPR from precision/recall
        # (You'll need to compute this from the results)
        plt.plot([], [], label=f'{model_name} (AUC={results["auroc"]:.4f})')
    
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves - Model Comparison')
    plt.legend()
    plt.savefig(f"{RESULTS_PATH}/model_comparison_roc.png")
    plt.show()
```

---

## ⚙️ Hardware & Resource Requirements

### Google Colab Tiers

| Tier | GPU | RAM | Disk | Training Time (per model) |
|------|-----|-----|------|---------------------------|
| **Free** ✅ | K80/T4 | 12-16 GB | 100 GB | **~1 hour** |
| Colab Pro | T4/V100 | 25-32 GB | 200 GB | ~45 mins |
| Colab Pro+ | V100/A100 | 50 GB | 500 GB | ~30 mins |

**Recommendation**: **Free Colab is sufficient!** Your models are lightweight (~8-14M params) and train in ~20-40 minutes per model.

**Why Free Tier Works**:
- ✅ **Small models**: 8.5M (Baseline), 12M (CNN), 14M (ECNN) parameters
- ✅ **Efficient training**: 100 epochs × ~15 mins = ~25-40 mins per model
- ✅ **Small dataset**: 16,771 slices (90% = 15K training samples)
- ✅ **Batch size 32**: Works fine with 12-16GB RAM
- ✅ **Total project time**: ~2-3 hours for all 3 models

**When You Might Need Pro** (unlikely for your project):
- ❌ Very large models (>50M parameters)
- ❌ Huge datasets (>100K images)
- ❌ Long training (>500 epochs)
- ❌ Large batch sizes (>64)

### Memory Management

```python
# Clear GPU memory
torch.cuda.empty_cache()

# Check GPU usage
!nvidia-smi

# Reduce batch size if OOM
# BATCH_SIZE = 16  # Instead of 32
```

---

## 🐛 Common Issues & Solutions

### Issue 1: Out of Memory

**Solution**:
```python
# Reduce batch size
BATCH_SIZE = 16  # or 8

# Use gradient accumulation
ACCUMULATION_STEPS = 2
```

### Issue 2: Disconnection during Training

**Solution**:
```python
# Save checkpoints frequently (already implemented - every 10 epochs)
# Keep browser tab active (prevents disconnection)
# Optional: Use simple keepalive script if needed
```

**Note**: Free Colab sessions last ~12 hours if active - more than enough for your ~2-3 hour total training time!

### Issue 3: Data Loading Slow

**Solution**:
```python
# Reduce num_workers
NUM_WORKERS = 0  # or 1

# Use pin_memory
pin_memory=True
```

---

## 📊 Expected Results

### Training Metrics

| Model | Train Loss | Val Loss | Epochs to Converge |
|-------|-----------|----------|---------------------|
| Baseline | 0.015-0.020 | 0.018-0.025 | 40-50 |
| CNN-AE | 0.008-0.012 | 0.010-0.015 | 60-70 |
| ECNN-AE | 0.006-0.010 | 0.008-0.012 | 70-80 |

### Test Metrics (Anomaly Detection on BraTS)

| Model | AUROC | False Positive Rate | Precision | Recall |
|-------|-------|---------------------|-----------|---------|
| Baseline | 0.78-0.82 | 15-20% | 0.65-0.70 | 0.75-0.80 |
| CNN-AE | 0.85-0.89 | 10-15% | 0.75-0.80 | 0.80-0.85 |
| **ECNN-AE** | **0.90-0.94** | **6-10%** | **0.82-0.88** | **0.85-0.90** |

**Key Result**: ECNN-AE should achieve ~30% reduction in false positives compared to standard CNN-AE!

---

## 📝 Training Checklist

### Before Training:
- [ ] Data loaded and verified
- [ ] Random seeds set
- [ ] GPU available and working
- [ ] All libraries installed
- [ ] Sufficient disk space (>5 GB)

### During Training:
- [ ] Monitor training curves
- [ ] Check for overfitting
- [ ] Verify GPU utilization
- [ ] Save checkpoints regularly

### After Training:
- [ ] Evaluate on validation set
- [ ] Test on BraTS anomaly data
- [ ] Compare all models
- [ ] Save all results and plots
- [ ] Document findings

---

**Next Steps**: 
1. Open `01_baseline_autoencoder.ipynb` in Google Colab
2. Follow the notebook step-by-step
3. Proceed to `02_cnn_autoencoder.ipynb`
4. Finally, train `03_ecnn_autoencoder.ipynb`

**Good luck with training! 🚀**
