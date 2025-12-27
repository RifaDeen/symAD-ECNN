# Saved Models Directory

Trained model checkpoints (`.pth` files) are saved here.

## Storage Locations

### During Training (Colab)
Models are saved to **Google Drive** for persistence:
```
/content/drive/MyDrive/symAD-ECNN/models/saved_models/
```

### For Version Control (GitHub)
- ❌ **Don't commit** `.pth` files to Git (too large: 50-100MB each)
- ✅ **Do commit** small sample models or architecture definitions
- ✅ **Do document** model performance in `results/`

## Expected Models

After training, you'll have:
- `baseline_ae_final.pth` (~56MB) - Baseline Autoencoder
- `cnn_ae_final.pth` (~100MB) - CNN Autoencoder  
- `ecnn_ae_final.pth` (~120MB) - ECNN Autoencoder

## Loading Models

```python
import torch

# Load from Drive in Colab
model_path = '/content/drive/MyDrive/symAD-ECNN/models/saved_models/baseline_ae_final.pth'
checkpoint = torch.load(model_path)
model.load_state_dict(checkpoint['model_state_dict'])
```

See **GOOGLE_DRIVE_SETUP.md** for complete instructions.
