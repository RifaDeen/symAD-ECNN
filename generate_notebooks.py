"""
Generate complete training notebooks for all three models
Run this script to populate the empty notebook files with cells
"""

import json
import os

def create_notebook_cells(title, model_code, model_name, model_desc):
    """Create all cells for a model training notebook"""
    
    cells = []
    
    # Title cell
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [f"# {title}\n\n## 📋 Overview\n\n{model_desc}\n\n---"]
    })
    
    # Setup cells (same for all models)
    setup_cells = [
        {
            "cell_type": "markdown",
            "source": ["## 1️⃣ Setup and Environment Configuration"]
        },
        {
            "cell_type": "code",
            "source": [
                "# Mount Google Drive\n",
                "from google.colab import drive\n",
                "drive.mount('/content/drive')\n",
                "\n",
                "print('✅ Google Drive mounted successfully!')"
            ]
        },
        {
            "cell_type": "code",
            "source": [
                "# Install required packages\n",
                "!pip install pytorch-msssim -q\n",
                f"{'!pip install e2cnn -q' if 'ECNN' in title else ''}\n",
                "\n",
                "print('✅ All packages installed!')"
            ]
        },
        {
            "cell_type": "code",
            "source": [
                "# Import libraries\n",
                "import torch\n",
                "import torch.nn as nn\n",
                "import torch.optim as optim\n",
                "from torch.utils.data import Dataset, DataLoader\n",
                "from torch.optim.lr_scheduler import ReduceLROnPlateau\n",
                "from pytorch_msssim import MS_SSIM\n",
                "" + ("from e2cnn import gspaces\nfrom e2cnn import nn as e2nn\n" if 'ECNN' in title else "") + "\n",
                "import numpy as np\n",
                "import matplotlib.pyplot as plt\n",
                "from sklearn.model_selection import train_test_split\n",
                "from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve, auc\n",
                "from glob import glob\n",
                "import os\n",
                "import time\n",
                "from tqdm import tqdm\n",
                "\n",
                "# Set random seeds\n",
                "torch.manual_seed(42)\n",
                "np.random.seed(42)\n",
                "\n",
                "print('✅ All libraries imported!')"
            ]
        },
        {
            "cell_type": "code",
            "source": [
                "# Setup device\n",
                "device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')\n",
                "print(f'🚀 Using device: {device}')"
            ]
        },
        {
            "cell_type": "code",
            "source": [
                "# Define paths\n",
                "BASE_PATH = '/content/drive/MyDrive/symAD-ECNN'\n",
                "IXI_PATH = f'{BASE_PATH}/data/processed_ixi/resized_ixi'\n",
                "BRATS_PATH = f'{BASE_PATH}/data/brats2021_test'\n",
                "MODEL_PATH = f'{BASE_PATH}/models/saved_models'\n",
                "RESULTS_PATH = f'{BASE_PATH}/results'\n",
                "\n",
                "os.makedirs(MODEL_PATH, exist_ok=True)\n",
                "os.makedirs(RESULTS_PATH, exist_ok=True)\n",
                "\n",
                "print('📁 Paths configured!')"
            ]
        }
    ]
    
    # Add setup cells
    for cell in setup_cells:
        cells.append({
            "cell_type": cell["cell_type"],
            "metadata": {},
            "source": cell["source"]
        })
    
    # Data loading cells (same for all)
    data_cells = [
        {
            "cell_type": "markdown",
            "source": ["## 2️⃣ Data Loading and Preprocessing"]
        },
        {
            "cell_type": "code",
            "source": [
                "class MRIDataset(Dataset):\n",
                "    def __init__(self, file_list):\n",
                "        self.files = file_list\n",
                "    \n",
                "    def __len__(self):\n",
                "        return len(self.files)\n",
                "    \n",
                "    def __getitem__(self, idx):\n",
                "        img = np.load(self.files[idx])\n",
                "        img = np.expand_dims(img, 0)\n",
                "        img_tensor = torch.from_numpy(img).float()\n",
                "        return img_tensor, img_tensor\n",
                "\n",
                "print('✅ Dataset class defined!')"
            ]
        },
        {
            "cell_type": "code",
            "source": [
                "# Load files\n",
                "ixi_files = sorted(glob(f'{IXI_PATH}/*.npy'))\n",
                "brats_files = sorted(glob(f'{BRATS_PATH}/*.npy'))\n",
                "\n",
                "train_files, val_files = train_test_split(ixi_files, test_size=0.1, random_state=42)\n",
                "\n",
                "print(f'Train: {len(train_files)}, Val: {len(val_files)}, Test: {len(brats_files)}')"
            ]
        },
        {
            "cell_type": "code",
            "source": [
                "# Create dataloaders\n",
                "BATCH_SIZE = 32\n",
                "\n",
                "train_loader = DataLoader(MRIDataset(train_files), batch_size=BATCH_SIZE, shuffle=True)\n",
                "val_loader = DataLoader(MRIDataset(val_files), batch_size=BATCH_SIZE, shuffle=False)\n",
                "test_loader = DataLoader(MRIDataset(brats_files), batch_size=BATCH_SIZE, shuffle=False)\n",
                "\n",
                "print('✅ DataLoaders created!')"
            ]
        }
    ]
    
    for cell in data_cells:
        cells.append({
            "cell_type": cell["cell_type"],
            "metadata": {},
            "source": cell["source"]
        })
    
    # Model architecture cell
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": ["## 3️⃣ Model Architecture"]
    })
    
    cells.append({
        "cell_type": "code",
        "metadata": {},
        "source": model_code
    })
    
    cells.append({
        "cell_type": "code",
        "metadata": {},
        "source": [
            f"# Create model\n",
            f"model = {model_name}().to(device)\n",
            "total_params = sum(p.numel() for p in model.parameters())\n",
            f"print(f'🧠 {model_name} Created!')\n",
            "print(f'   Total parameters: {total_params:,}')"
        ]
    })
    
    # Training cells (same for all)
    training_cells = [
        {
            "cell_type": "markdown",
            "source": ["## 4️⃣ Training"]
        },
        {
            "cell_type": "code",
            "source": [
                "class CombinedLoss(nn.Module):\n",
                "    def __init__(self, alpha=0.84):\n",
                "        super().__init__()\n",
                "        self.alpha = alpha\n",
                "        self.mse = nn.MSELoss()\n",
                "        self.ms_ssim = MS_SSIM(data_range=1.0, size_average=True, channel=1)\n",
                "    \n",
                "    def forward(self, output, target):\n",
                "        mse_loss = self.mse(output, target)\n",
                "        ssim_loss = 1 - self.ms_ssim(output, target)\n",
                "        return self.alpha * mse_loss + (1 - self.alpha) * ssim_loss\n",
                "\n",
                "criterion = CombinedLoss()\n",
                "optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)\n",
                "scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=5)\n",
                "\n",
                "print('✅ Loss, optimizer, scheduler ready!')"
            ]
        },
        {
            "cell_type": "code",
            "source": [
                "def train_epoch(model, loader, criterion, optimizer, device):\n",
                "    model.train()\n",
                "    total_loss = 0\n",
                "    for data, target in tqdm(loader):\n",
                "        data, target = data.to(device), target.to(device)\n",
                "        optimizer.zero_grad()\n",
                "        output = model(data)\n",
                "        loss = criterion(output, target)\n",
                "        loss.backward()\n",
                "        optimizer.step()\n",
                "        total_loss += loss.item()\n",
                "    return total_loss / len(loader)\n",
                "\n",
                "def validate(model, loader, criterion, device):\n",
                "    model.eval()\n",
                "    total_loss = 0\n",
                "    with torch.no_grad():\n",
                "        for data, target in loader:\n",
                "            data, target = data.to(device), target.to(device)\n",
                "            output = model(data)\n",
                "            loss = criterion(output, target)\n",
                "            total_loss += loss.item()\n",
                "    return total_loss / len(loader)\n",
                "\n",
                "print('✅ Training functions defined!')"
            ]
        },
        {
            "cell_type": "code",
            "source": [
                "# Training loop\n",
                "NUM_EPOCHS = 100\n",
                "train_losses, val_losses = [], []\n",
                "best_val_loss = float('inf')\n",
                "\n",
                "for epoch in range(NUM_EPOCHS):\n",
                "    train_loss = train_epoch(model, train_loader, criterion, optimizer, device)\n",
                "    val_loss = validate(model, val_loader, criterion, device)\n",
                "    \n",
                "    train_losses.append(train_loss)\n",
                "    val_losses.append(val_loss)\n",
                "    scheduler.step(val_loss)\n",
                "    \n",
                "    print(f'Epoch {epoch+1}/{NUM_EPOCHS} - Train: {train_loss:.6f}, Val: {val_loss:.6f}')\n",
                "    \n",
                "    if val_loss < best_val_loss:\n",
                "        best_val_loss = val_loss\n",
                f"        torch.save(model.state_dict(), f'{{MODEL_PATH}}/{model_name.lower()}_best.pth')\n",
                "\n",
                "print('🎉 Training complete!')"
            ]
        }
    ]
    
    for cell in training_cells:
        cells.append({
            "cell_type": cell["cell_type"],
            "metadata": {},
            "source": cell["source"]
        })
    
    # Evaluation cells
    eval_cells = [
        {
            "cell_type": "markdown",
            "source": ["## 5️⃣ Evaluation"]
        },
        {
            "cell_type": "code",
            "source": [
                "# Calculate reconstruction errors\n",
                "def calc_errors(model, loader):\n",
                "    model.eval()\n",
                "    errors = []\n",
                "    with torch.no_grad():\n",
                "        for data, _ in loader:\n",
                "            data = data.to(device)\n",
                "            recon = model(data)\n",
                "            mse = nn.functional.mse_loss(recon, data, reduction='none')\n",
                "            mse = mse.view(mse.size(0), -1).mean(dim=1)\n",
                "            errors.extend(mse.cpu().numpy())\n",
                "    return np.array(errors)\n",
                "\n",
                "normal_errors = calc_errors(model, val_loader)\n",
                "anomaly_errors = calc_errors(model, test_loader)\n",
                "\n",
                "y_true = np.concatenate([np.zeros(len(normal_errors)), np.ones(len(anomaly_errors))])\n",
                "y_scores = np.concatenate([normal_errors, anomaly_errors])\n",
                "\n",
                "auroc = roc_auc_score(y_true, y_scores)\n",
                "print(f'📈 AUROC: {auroc:.4f}')"
            ]
        },
        {
            "cell_type": "code",
            "source": [
                "# Visualize results\n",
                "plt.figure(figsize=(12, 5))\n",
                "\n",
                "plt.subplot(1, 2, 1)\n",
                "plt.hist(normal_errors, bins=50, alpha=0.7, label='Normal', density=True)\n",
                "plt.hist(anomaly_errors, bins=50, alpha=0.7, label='Anomaly', density=True)\n",
                "plt.xlabel('Reconstruction Error')\n",
                "plt.ylabel('Density')\n",
                "plt.legend()\n",
                "plt.title('Error Distribution')\n",
                "\n",
                "plt.subplot(1, 2, 2)\n",
                "fpr, tpr, _ = roc_curve(y_true, y_scores)\n",
                "plt.plot(fpr, tpr, label=f'AUROC = {auroc:.4f}')\n",
                "plt.plot([0, 1], [0, 1], 'k--')\n",
                "plt.xlabel('FPR')\n",
                "plt.ylabel('TPR')\n",
                "plt.legend()\n",
                "plt.title('ROC Curve')\n",
                "\n",
                "plt.tight_layout()\n",
                f"plt.savefig(f'{{RESULTS_PATH}}/{model_name.lower()}_results.png')\n",
                "plt.show()\n",
                "\n",
                "print('✅ Results saved!')"
            ]
        }
    ]
    
    for cell in eval_cells:
        cells.append({
            "cell_type": cell["cell_type"],
            "metadata": {},
            "source": cell["source"]
        })
    
    # Convert to proper format
    formatted_cells = []
    for cell in cells:
        formatted_cell = {
            "cell_type": cell["cell_type"],
            "metadata": {},
            "source": cell["source"]
        }
        if cell["cell_type"] == "code":
            formatted_cell["outputs"] = []
            formatted_cell["execution_count"] = None
        formatted_cells.append(formatted_cell)
    
    return formatted_cells


# CNN Autoencoder model code
cnn_model_code = [
    "class CNNAutoencoder(nn.Module):\n",
    "    def __init__(self):\n",
    "        super().__init__()\n",
    "        \n",
    "        self.encoder = nn.Sequential(\n",
    "            nn.Conv2d(1, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),\n",
    "            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),\n",
    "            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2),\n",
    "            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(), nn.MaxPool2d(2)\n",
    "        )\n",
    "        \n",
    "        self.decoder = nn.Sequential(\n",
    "            nn.ConvTranspose2d(256, 128, 3, stride=2, padding=1, output_padding=1), nn.BatchNorm2d(128), nn.ReLU(),\n",
    "            nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1), nn.BatchNorm2d(64), nn.ReLU(),\n",
    "            nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1), nn.BatchNorm2d(32), nn.ReLU(),\n",
    "            nn.ConvTranspose2d(32, 1, 3, stride=2, padding=1, output_padding=1), nn.Sigmoid()\n",
    "        )\n",
    "    \n",
    "    def forward(self, x):\n",
    "        x = self.encoder(x)\n",
    "        x = self.decoder(x)\n",
    "        return x\n",
    "\n",
    "print('✅ CNN Autoencoder defined!')"
]

# ECNN Autoencoder model code
ecnn_model_code = [
    "class ECNNAutoencoder(nn.Module):\n",
    "    def __init__(self):\n",
    "        super().__init__()\n",
    "        \n",
    "        self.r2_act = gspaces.Rot2dOnR2(N=4)  # C4 group\n",
    "        self.in_type = e2nn.FieldType(self.r2_act, [self.r2_act.trivial_repr])\n",
    "        self.feat_32 = e2nn.FieldType(self.r2_act, 32*[self.r2_act.regular_repr])\n",
    "        self.feat_64 = e2nn.FieldType(self.r2_act, 64*[self.r2_act.regular_repr])\n",
    "        \n",
    "        self.encoder = e2nn.SequentialModule(\n",
    "            e2nn.R2Conv(self.in_type, self.feat_32, 3, padding=1),\n",
    "            e2nn.ReLU(self.feat_32),\n",
    "            e2nn.PointwiseMaxPool(self.feat_32, 2),\n",
    "            e2nn.R2Conv(self.feat_32, self.feat_64, 3, padding=1),\n",
    "            e2nn.ReLU(self.feat_64),\n",
    "            e2nn.PointwiseMaxPool(self.feat_64, 2)\n",
    "        )\n",
    "        \n",
    "        # Simplified decoder (spatial upsampling)\n",
    "        self.decoder = nn.Sequential(\n",
    "            nn.ConvTranspose2d(256, 128, 3, 2, 1, 1), nn.ReLU(),\n",
    "            nn.ConvTranspose2d(128, 64, 3, 2, 1, 1), nn.ReLU(),\n",
    "            nn.Conv2d(64, 1, 3, 1, 1), nn.Sigmoid()\n",
    "        )\n",
    "    \n",
    "    def forward(self, x):\n",
    "        x_g = e2nn.GeometricTensor(x, self.in_type)\n",
    "        features = self.encoder(x_g)\n",
    "        x = self.decoder(features.tensor)\n",
    "        return x\n",
    "\n",
    "print('✅ ECNN Autoencoder defined!')"
]


# Generate CNN notebook
print("Generating CNN-Autoencoder notebook...")
cnn_cells = create_notebook_cells(
    "🖼️ CNN-Autoencoder for Brain MRI Anomaly Detection",
    cnn_model_code,
    "CNNAutoencoder",
    "Convolutional autoencoder with spatial feature extraction. Uses standard Conv2D layers with encoder-decoder architecture."
)

cnn_notebook = {
    "cells": cnn_cells,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "name": "python",
            "version": "3.10.0"
        },
        "accelerator": "GPU"
    },
    "nbformat": 4,
    "nbformat_minor": 0
}

with open('notebooks/models/02_cnn_autoencoder.ipynb', 'w') as f:
    json.dump(cnn_notebook, f, indent=2)

print("✅ CNN-Autoencoder notebook created!")


# Generate ECNN notebook
print("Generating ECNN-Autoencoder notebook...")
ecnn_cells = create_notebook_cells(
    "⚡ E(n)-Equivariant CNN-Autoencoder for Brain MRI Anomaly Detection",
    ecnn_model_code,
    "ECNNAutoencoder",
    "E(2)-Equivariant CNN using e2cnn library with C4 group. Handles rotations without data augmentation for improved anomaly detection."
)

ecnn_notebook = {
    "cells": ecnn_cells,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "name": "python",
            "version": "3.10.0"
        },
        "accelerator": "GPU"
    },
    "nbformat": 4,
    "nbformat_minor": 0
}

with open('notebooks/models/03_ecnn_autoencoder.ipynb', 'w') as f:
    json.dump(ecnn_notebook, f, indent=2)

print("✅ ECNN-Autoencoder notebook created!")

print("\n🎉 All notebooks generated successfully!")
print("   - 01_baseline_autoencoder.ipynb (already populated)")
print("   - 02_cnn_autoencoder.ipynb (generated)")
print("   - 03_ecnn_autoencoder.ipynb (generated)")
