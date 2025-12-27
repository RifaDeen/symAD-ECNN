# Data Directory

This folder is for data files. **Actual data is stored in Google Drive.**

## Google Drive Structure

All training data is stored in Google Drive due to size constraints:

```
Google Drive/MyDrive/symAD-ECNN/
├── data/
│   ├── processed_ixi/
│   │   ├── train/          # ~15,000 .npy files (~2GB)
│   │   └── val/            # ~1,700 .npy files (~200MB)
│   └── brats2021/          # ~250 patient folders
│       ├── BraTS2021_00000/
│       └── ...
├── models/
│   └── saved_models/       # Trained model checkpoints (.pth)
└── results/                # Training results, figures
```

## Why Not in Git?

- **GitHub file limit**: 100MB per file
- **Dataset size**: 2-3GB total
- **Solution**: Google Drive for data, GitHub for code

## Access in Colab

Mount Google Drive at the start of each notebook:

```python
from google.colab import drive
drive.mount('/content/drive')

# Access your data
DATA_PATH = '/content/drive/MyDrive/symAD-ECNN/data/processed_ixi/train'
```

## Local Development

For local testing without downloading all data:
1. Use a small sample subset
2. Or work directly in Colab (recommended)

See **GOOGLE_DRIVE_SETUP.md** for complete setup instructions.
