# Model Implementation Quick Start Guide

## 🚀 Getting Started

This guide provides the complete code structure for all three models. Copy the relevant sections into your Colab notebooks.

---

## 📁 Notebook Structure

Each notebook follows this structure:

1. **Setup & Data Loading** (15-20 cells)
2. **Model Architecture** (5-10 cells)
3. **Training** (10-15 cells)
4. **Evaluation** (10-15 cells)
5. **Visualization** (5-10 cells)

**Total cells per notebook**: ~45-70 cells

---

## 🔧 Model 1: Baseline Autoencoder

###Complete Model Code

```python
import torch
import torch.nn as nn

class BaselineAutoencoder(nn.Module):
    """
    Simple fully-connected autoencoder
    Input: 128×128×1 = 16,384 pixels
    Latent: 128 dimensions
    """
    
    def __init__(self, input_dim=16384, latent_dim=128):
        super(BaselineAutoencoder, self).__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(True),
            
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(True),
            nn.Dropout(0.2),
            
            nn.Linear(256, latent_dim),
            nn.ReLU(True)
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(True),
            
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(True),
            nn.Dropout(0.2),
            
            nn.Linear(512, input_dim),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        # Flatten input
        batch_size = x.size(0)
        x_flat = x.view(batch_size, -1)
        
        # Encode
        z = self.encoder(x_flat)
        
        # Decode
        x_recon = self.decoder(z)
        
        # Reshape to image
        x_recon = x_recon.view(batch_size, 1, 128, 128)
        
        return x_recon
    
    def get_latent(self, x):
        """Get latent representation"""
        batch_size = x.size(0)
        x_flat = x.view(batch_size, -1)
        return self.encoder(x_flat)

# Create model
model = BaselineAutoencoder().to(device)
print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
```

---

## 🖼️ Model 2: CNN-Autoencoder

### Complete Model Code

```python
import torch
import torch.nn as nn

class CNNAutoencoder(nn.Module):
    """
    Convolutional autoencoder with spatial feature extraction
    Input: 128×128×1
    Latent: 256 dimensions
    """
    
    def __init__(self, latent_dim=256):
        super(CNNAutoencoder, self).__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            # 128×128×1 -> 128×128×32
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),  # 64×64×32
            
            # 64×64×32 -> 64×64×64
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),  # 32×32×64
            
            # 32×32×64 -> 32×32×128
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),  # 16×16×128
            
            # 16×16×128 -> 16×16×256
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2)  # 8×8×256
        )
        
        # Latent space
        self.flatten = nn.Flatten()
        self.fc_encode = nn.Linear(8 * 8 * 256, latent_dim)
        self.fc_decode = nn.Linear(latent_dim, 8 * 8 * 256)
        
        # Decoder
        self.decoder = nn.Sequential(
            # 8×8×256 -> 16×16×256
            nn.ConvTranspose2d(256, 256, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            
            # 16×16×256 -> 32×32×128
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            
            # 32×32×128 -> 64×64×64
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            
            # 64×64×64 -> 128×128×32
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            
            # 128×128×32 -> 128×128×1
            nn.Conv2d(32, 1, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        # Encode
        features = self.encoder(x)
        
        # Latent
        batch_size = features.size(0)
        flat = self.flatten(features)
        z = self.fc_encode(flat)
        
        # Decode
        decoded_flat = self.fc_decode(z)
        decoded_features = decoded_flat.view(batch_size, 256, 8, 8)
        x_recon = self.decoder(decoded_features)
        
        return x_recon
    
    def get_latent(self, x):
        """Get latent representation"""
        features = self.encoder(x)
        flat = self.flatten(features)
        return self.fc_encode(flat)

# Create model
model = CNNAutoencoder().to(device)
print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
```

---

## ⚡ Model 3: E(n)-Equivariant CNN-Autoencoder (MAIN MODEL)

### Installation

```bash
!pip install e2cnn
```

### Complete Model Code

```python
import torch
import torch.nn as nn
from e2cnn import gspaces
from e2cnn import nn as e2nn

class ECNNAutoencoder(nn.Module):
    """
    E(2)-Equivariant CNN Autoencoder
    Uses C4 group (4 rotations: 0°, 90°, 180°, 270°)
    Input: 128×128×1
    Latent: 256 dimensions
    """
    
    def __init__(self, latent_dim=256):
        super(ECNNAutoencoder, self).__init__()
        
        # Define the E(2) group acting on the plane (C4 discrete rotations)
        self.r2_act = gspaces.Rot2dOnR2(N=4)  # C4 group
        
        # Input type: trivial representation (scalar field - grayscale image)
        self.in_type = e2nn.FieldType(self.r2_act, [self.r2_act.trivial_repr])
        
        # Feature types for each layer
        self.feat_type_32 = e2nn.FieldType(self.r2_act, 32*[self.r2_act.regular_repr])
        self.feat_type_64 = e2nn.FieldType(self.r2_act, 64*[self.r2_act.regular_repr])
        self.feat_type_128 = e2nn.FieldType(self.r2_act, 128*[self.r2_act.regular_repr])
        self.feat_type_256 = e2nn.FieldType(self.r2_act, 256*[self.r2_act.regular_repr])
        
        # Encoder (E(2)-Equivariant)
        self.encoder = nn.Sequential(
            # 128×128×1 -> 128×128×32
            e2nn.R2Conv(self.in_type, self.feat_type_32, kernel_size=3, padding=1),
            e2nn.IIDBatchNorm2d(self.feat_type_32),
            e2nn.ReLU(self.feat_type_32),
            e2nn.PointwiseMaxPool(self.feat_type_32, 2),  # 64×64×32
            
            # 64×64×32 -> 64×64×64
            e2nn.R2Conv(self.feat_type_32, self.feat_type_64, kernel_size=3, padding=1),
            e2nn.IIDBatchNorm2d(self.feat_type_64),
            e2nn.ReLU(self.feat_type_64),
            e2nn.PointwiseMaxPool(self.feat_type_64, 2),  # 32×32×64
            
            # 32×32×64 -> 32×32×128
            e2nn.R2Conv(self.feat_type_64, self.feat_type_128, kernel_size=3, padding=1),
            e2nn.IIDBatchNorm2d(self.feat_type_128),
            e2nn.ReLU(self.feat_type_128),
            e2nn.PointwiseMaxPool(self.feat_type_128, 2),  # 16×16×128
            
            # 16×16×128 -> 16×16×256
            e2nn.R2Conv(self.feat_type_128, self.feat_type_256, kernel_size=3, padding=1),
            e2nn.IIDBatchNorm2d(self.feat_type_256),
            e2nn.ReLU(self.feat_type_256),
            e2nn.PointwiseMaxPool(self.feat_type_256, 2)  # 8×8×256
        )
        
        # Group pooling (make invariant)
        self.group_pool = e2nn.GroupPooling(self.feat_type_256)
        
        # Latent space (fully connected, not equivariant)
        self.fc_encode = nn.Linear(8 * 8 * 256, latent_dim)
        self.fc_decode = nn.Linear(latent_dim, 8 * 8 * 256)
        
        # Decoder (E(2)-Equivariant)
        # Note: Transposed convolutions in e2cnn
        self.upconv1 = e2nn.R2Conv(self.feat_type_256, self.feat_type_256, 
                                   kernel_size=3, padding=1)
        self.bn1 = e2nn.IIDBatchNorm2d(self.feat_type_256)
        self.relu1 = e2nn.ReLU(self.feat_type_256)
        
        self.upconv2 = e2nn.R2Conv(self.feat_type_256, self.feat_type_128, 
                                   kernel_size=3, padding=1)
        self.bn2 = e2nn.IIDBatchNorm2d(self.feat_type_128)
        self.relu2 = e2nn.ReLU(self.feat_type_128)
        
        self.upconv3 = e2nn.R2Conv(self.feat_type_128, self.feat_type_64, 
                                   kernel_size=3, padding=1)
        self.bn3 = e2nn.IIDBatchNorm2d(self.feat_type_64)
        self.relu3 = e2nn.ReLU(self.feat_type_64)
        
        self.upconv4 = e2nn.R2Conv(self.feat_type_64, self.feat_type_32, 
                                   kernel_size=3, padding=1)
        self.bn4 = e2nn.IIDBatchNorm2d(self.feat_type_32)
        self.relu4 = e2nn.ReLU(self.feat_type_32)
        
        # Final layer (back to trivial representation)
        self.final_conv = e2nn.R2Conv(self.feat_type_32, self.in_type, 
                                      kernel_size=3, padding=1)
        
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        # Wrap input as GeometricTensor
        x_g = e2nn.GeometricTensor(x, self.in_type)
        
        # Encode (equivariant features)
        features = self.encoder(x_g)
        
        # Group pooling (make invariant)
        invariant_features = self.group_pool(features)
        
        # Latent space
        batch_size = invariant_features.size(0)
        flat = invariant_features.tensor.view(batch_size, -1)
        z = self.fc_encode(flat)
        
        # Decode from latent
        decoded_flat = self.fc_decode(z)
        decoded_features = decoded_flat.view(batch_size, 256, 8, 8)
        
        # Wrap as GeometricTensor for decoder
        decoded_g = e2nn.GeometricTensor(decoded_features, self.feat_type_256)
        
        # Equivariant decoder with upsampling
        x = self.upconv1(decoded_g)
        x = self.bn1(x)
        x = self.relu1(x)
        x = e2nn.GeometricTensor(
            nn.functional.interpolate(x.tensor, scale_factor=2, mode='bilinear'),
            x.type
        )  # 16×16
        
        x = self.upconv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = e2nn.GeometricTensor(
            nn.functional.interpolate(x.tensor, scale_factor=2, mode='bilinear'),
            x.type
        )  # 32×32
        
        x = self.upconv3(x)
        x = self.bn3(x)
        x = self.relu3(x)
        x = e2nn.GeometricTensor(
            nn.functional.interpolate(x.tensor, scale_factor=2, mode='bilinear'),
            x.type
        )  # 64×64
        
        x = self.upconv4(x)
        x = self.bn4(x)
        x = self.relu4(x)
        x = e2nn.GeometricTensor(
            nn.functional.interpolate(x.tensor, scale_factor=2, mode='bilinear'),
            x.type
        )  # 128×128
        
        # Final convolution
        x = self.final_conv(x)
        
        # Extract tensor and apply sigmoid
        x_recon = self.sigmoid(x.tensor)
        
        return x_recon
    
    def get_latent(self, x):
        """Get latent representation"""
        x_g = e2nn.GeometricTensor(x, self.in_type)
        features = self.encoder(x_g)
        invariant_features = self.group_pool(features)
        batch_size = invariant_features.size(0)
        flat = invariant_features.tensor.view(batch_size, -1)
        return self.fc_encode(flat)
    
    def test_equivariance(self, x, angle_deg):
        """Test rotation equivariance property"""
        import torchvision.transforms.functional as TF
        
        # Forward pass on original
        features_original = self.encoder(e2nn.GeometricTensor(x, self.in_type))
        
        # Rotate input
        x_rotated = TF.rotate(x, angle_deg)
        features_rotated = self.encoder(e2nn.GeometricTensor(x_rotated, self.in_type))
        
        # Check if features rotated accordingly
        # (This is a simplified test - proper testing requires geometric comparison)
        return features_original, features_rotated

# Create model
model = ECNNAutoencoder().to(device)
print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
```

---

## 🎓 Training Code (Same for All Models)

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from pytorch_msssim import MS_SSIM
import matplotlib.pyplot as plt

# Combined Loss Function
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

# Training Configuration
criterion = CombinedLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)

# Training Loop
def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    
    for data, target in dataloader:
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
    
    return running_loss / len(dataloader)

def validate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    
    with torch.no_grad():
        for data, target in dataloader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)
            running_loss += loss.item()
    
    return running_loss / len(dataloader)

# Training
EPOCHS = 100
train_losses, val_losses = [], []
best_val_loss = float('inf')

for epoch in range(EPOCHS):
    train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
    val_loss = validate(model, val_loader, criterion, device)
    
    train_losses.append(train_loss)
    val_losses.append(val_loss)
    
    scheduler.step(val_loss)
    
    print(f"Epoch {epoch+1}/{EPOCHS} - Train: {train_loss:.6f}, Val: {val_loss:.6f}")
    
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), 'best_model.pth')

# Plot training curves
plt.figure(figsize=(10, 5))
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Val Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Training History')
plt.show()
```

---

## 📊 Evaluation Code

```python
import numpy as np
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc
import matplotlib.pyplot as plt

def calculate_reconstruction_error(model, dataloader, device):
    """Calculate pixel-wise reconstruction error"""
    model.eval()
    errors = []
    
    with torch.no_grad():
        for data, _ in dataloader:
            data = data.to(device)
            recon = model(data)
            mse = nn.functional.mse_loss(recon, data, reduction='none')
            mse = mse.view(mse.size(0), -1).mean(dim=1)
            errors.extend(mse.cpu().numpy())
    
    return np.array(errors)

# Calculate errors on normal and anomaly data
normal_errors = calculate_reconstruction_error(model, val_loader, device)
anomaly_errors = calculate_reconstruction_error(model, test_loader, device)

# Create labels and scores
y_true = np.concatenate([np.zeros(len(normal_errors)), np.ones(len(anomaly_errors))])
y_scores = np.concatenate([normal_errors, anomaly_errors])

# Calculate AUROC
auroc = roc_auc_score(y_true, y_scores)
print(f"AUROC: {auroc:.4f}")

# Plot distribution of reconstruction errors
plt.figure(figsize=(10, 5))
plt.hist(normal_errors, bins=50, alpha=0.5, label='Normal (IXI)', density=True)
plt.hist(anomaly_errors, bins=50, alpha=0.5, label='Anomaly (BraTS)', density=True)
plt.xlabel('Reconstruction Error')
plt.ylabel('Density')
plt.legend()
plt.title('Reconstruction Error Distribution')
plt.show()

# Visualize reconstructions
def visualize_results(model, dataloader, device, num_samples=8):
    model.eval()
    data, _ = next(iter(dataloader))
    data = data[:num_samples].to(device)
    
    with torch.no_grad():
        recon = model(data)
    
    error = torch.abs(data - recon)
    
    fig, axes = plt.subplots(3, num_samples, figsize=(20, 6))
    for i in range(num_samples):
        axes[0, i].imshow(data[i, 0].cpu(), cmap='gray')
        axes[0, i].axis('off')
        axes[1, i].imshow(recon[i, 0].cpu(), cmap='gray')
        axes[1, i].axis('off')
        axes[2, i].imshow(error[i, 0].cpu(), cmap='hot')
        axes[2, i].axis('off')
    
    axes[0, 0].set_ylabel('Original', fontsize=12)
    axes[1, 0].set_ylabel('Reconstruction', fontsize=12)
    axes[2, 0].set_ylabel('Error Map', fontsize=12)
    plt.tight_layout()
    plt.show()

# Visualize on normal data
print("Normal Brain Reconstructions:")
visualize_results(model, val_loader, device)

# Visualize on anomaly data
print("Tumor Brain Reconstructions:")
visualize_results(model, test_loader, device)
```

---

## 📝 Complete Notebook Outline

### Notebook Structure for Each Model:

1. **Title and Overview** (markdown)
2. **Setup** (4-5 cells)
   - Mount Drive
   - Install libraries
   - Import packages
   - Set random seeds

3. **Data Loading** (6-8 cells)
   - Define paths
   - Load file lists
   - Create Dataset class
   - Train/val split
   - Create DataLoaders
   - Visualize samples

4. **Model Definition** (2-3 cells)
   - Define model class
   - Create model instance
   - Print model summary

5. **Loss & Optimizer** (2-3 cells)
   - Define loss function
   - Create optimizer
   - Setup scheduler

6. **Training** (5-6 cells)
   - Training loop function
   - Validation function
   - Main training loop
   - Save checkpoints
   - Plot training curves

7. **Evaluation** (8-10 cells)
   - Load best model
   - Calculate errors
   - Compute metrics (AUROC, etc.)
   - Visualization functions
   - Visualize reconstructions
   - Generate anomaly maps

8. **Comparison** (3-4 cells)
   - Save results
   - Compare with other models
   - Final summary

**Total**: ~45-55 cells per notebook

---

## 🎯 Key Differences Between Models

| Aspect | Baseline | CNN-AE | ECNN-AE |
|--------|----------|--------|---------|
| **Input Processing** | Flatten | Keep spatial | Keep spatial |
| **Core Layers** | Linear | Conv2D | R2Conv |
| **Library** | torch.nn | torch.nn | e2cnn |
| **Equivariance** | None | Translation only | Rotation + Translation |
| **Wrapper Needed** | No | No | Yes (GeometricTensor) |
| **Group Pooling** | No | No | Yes |

---

## ⚡ Quick Copy-Paste Sections

### 1. Complete Setup Cell (All Models)
```python
# Mount Drive
from google.colab import drive
drive.mount('/content/drive')

# Paths
BASE = "/content/drive/MyDrive/symAD-ECNN"
IXI_PATH = f"{BASE}/data/processed_ixi/resized_ixi"
BRATS_PATH = f"{BASE}/data/brats2021_test"
MODEL_PATH = f"{BASE}/models/saved_models"
RESULTS_PATH = f"{BASE}/results"

# Imports
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from glob import glob
import os

# Seeds
torch.manual_seed(42)
np.random.seed(42)

# Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")
```

### 2. Dataset Class (All Models)
```python
class MRIDataset(Dataset):
    def __init__(self, file_list):
        self.files = file_list
    
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        img = np.load(self.files[idx])
        img = np.expand_dims(img, 0)  # Add channel
        return torch.from_numpy(img).float(), torch.from_numpy(img).float()
```

### 3. Data Loading (All Models)
```python
# Load files
ixi_files = sorted(glob(f"{IXI_PATH}/*.npy"))
brats_files = sorted(glob(f"{BRATS_PATH}/*.npy"))

# Split
train_files, val_files = train_test_split(ixi_files, test_size=0.1, random_state=42)

# Datasets
train_ds = MRIDataset(train_files)
val_ds = MRIDataset(val_files)
test_ds = MRIDataset(brats_files)

# DataLoaders
train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=2)
val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, num_workers=2)
test_loader = DataLoader(test_ds, batch_size=32, shuffle=False, num_workers=2)

print(f"Train: {len(train_ds)}, Val: {len(val_ds)}, Test: {len(test_ds)}")
```

---

**Your notebooks are ready! Copy the relevant model code and training/evaluation sections into your Colab notebooks and start training!** 🚀
