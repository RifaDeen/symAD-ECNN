# 🧠 Brain MRI Anomaly Detection - Streamlit Web Interface

A production-ready web interface for detecting brain anomalies using E(2)-Equivariant CNN Autoencoder.

## 📋 Overview

This Streamlit application provides an intuitive web interface for the symAD-ECNN anomaly detection system. Users can upload brain MRI scans and receive real-time anomaly detection results with interactive visualizations.

### Key Features

- 🚀 **Easy Upload**: Support for NIfTI (.nii, .nii.gz) and image formats (.png, .jpg)
- 🔄 **Automatic Preprocessing**: Matches training pipeline (RAS orientation, resizing, normalization)
- 🧠 **ECNN Inference**: Uses trained E(2)-Equivariant CNN model (AUROC 0.8109)
- 📊 **Interactive Visualization**: Plotly and Matplotlib visualizations
- ⚡ **Real-time Results**: Instant anomaly scores and error maps
- 🎨 **Professional UI**: Clean, intuitive interface with custom styling

## 🏗️ Architecture

### Model: ECNN Optimized
- **Architecture**: E(2)-Equivariant CNN Autoencoder
- **Group**: C4 (90° rotation equivariance)
- **Parameters**: ~11M
- **Performance**: AUROC 0.8109 on BraTS test set
- **Training Data**: IXI (healthy brains)
- **Test Data**: BraTS 2021 (tumors)

### Anomaly Detection Method
- **Approach**: Reconstruction-based anomaly detection
- **Score**: Mean Squared Error (MSE) between input and reconstruction
- **Threshold**: Configurable (default: 0.0035)
- **Interpretation**: Higher scores indicate greater anomaly likelihood

## 🚀 Quick Start

### 1. Installation

```bash
# Clone repository (if needed)
git clone https://github.com/yourusername/symAD-ECNN.git
cd symAD-ECNN

# Install dependencies
pip install -r requirements_streamlit.txt
```

### 2. Prepare Model

Ensure the trained ECNN model is available:
```
models/saved_models/ecnn_optimized_best.pth
```

If you don't have the model, train it using:
```bash
# Run training notebook
jupyter notebook notebooks/models/07_ecnn_autoencoder.ipynb
```

### 3. Run Application

```bash
streamlit run streamlit_app.py
```

The app will open in your browser at `http://localhost:8501`

## 📖 Usage Guide

### Step 1: Launch Application
```bash
streamlit run streamlit_app.py
```

### Step 2: Configure Settings (Sidebar)
- **Model Path**: Path to trained model checkpoint
  - Default: `models/saved_models/ecnn_optimized_best.pth`
- **Anomaly Threshold**: Reconstruction error threshold
  - Default: 0.0035
  - Range: 0.001 - 0.010
  - Lower = more sensitive, Higher = more specific

### Step 3: Upload MRI Scan
- Click "Browse files" or drag & drop
- Supported formats:
  - **NIfTI**: `.nii`, `.nii.gz` (recommended)
  - **Images**: `.png`, `.jpg`, `.jpeg`

### Step 4: View Results
The application displays:
- **Anomaly Status**: NORMAL ✓ or ANOMALY DETECTED ⚠️
- **Anomaly Score**: Reconstruction error (lower = more normal)
- **Detailed Metrics**: Max error, mean error
- **Visualizations**:
  - Original MRI
  - Reconstruction
  - Error map (heatmap showing anomalous regions)

### Step 5: Interpret Results

#### Normal Scan (Score < 0.0035)
```
✓ NORMAL SCAN
Score: 0.002341
Status: No significant anomalies detected
```
- Reconstruction closely matches input
- Error map shows minimal deviation
- Low anomaly score

#### Anomalous Scan (Score > 0.0035)
```
⚠️ ANOMALY DETECTED
Score: 0.005678
Recommendation: Further clinical examination recommended
```
- Reconstruction fails to capture anomalies
- Error map highlights suspicious regions
- High anomaly score

## 🎨 User Interface

### Main Components

1. **Header**
   - Title and project branding
   - Model information

2. **Sidebar**
   - Settings (model path, threshold)
   - Model information card
   - Instructions

3. **Upload Section**
   - File uploader with drag & drop
   - File information display
   - Preprocessing status

4. **Results Section**
   - Anomaly status card (color-coded)
   - Detailed metrics
   - Recommendation

5. **Visualization Section**
   - Interactive Plotly charts
   - Static Matplotlib plots
   - Side-by-side comparison

### Color Coding
- 🟢 **Green**: Normal scan detected
- 🔴 **Red**: Anomaly detected
- 🔵 **Blue**: Model/system information
- ⚪ **Gray**: Neutral content

## 🛠️ Technical Details

### Preprocessing Pipeline

1. **Load NIfTI**
   ```python
   img = nib.load(nifti_path)
   img_canonical = nib.as_closest_canonical(img)  # RAS orientation
   ```

2. **Extract Middle Axial Slice**
   ```python
   mid_slice = data.shape[2] // 2
   slice_2d = data[:, :, mid_slice]
   ```

3. **Normalize to [0, 1]**
   ```python
   slice_2d = (slice_2d - slice_2d.min()) / (slice_2d.max() - slice_2d.min())
   ```

4. **Resize to 128×128**
   ```python
   pil_img.resize((128, 128), Image.BICUBIC)
   ```

### Inference Pipeline

1. **Convert to Tensor**
   ```python
   img_tensor = torch.from_numpy(image).float().unsqueeze(0).unsqueeze(0)
   ```

2. **Forward Pass**
   ```python
   reconstruction = model(img_tensor)
   ```

3. **Compute MSE**
   ```python
   mse = torch.nn.functional.mse_loss(reconstruction, img_tensor)
   anomaly_score = mse.mean().item()
   ```

4. **Generate Error Map**
   ```python
   error_map = np.abs(preprocessed - reconstruction)
   ```

### Model Architecture

```
Input (1×128×128)
    ↓
[Encoder Block 1] → R2Conv + BatchNorm + ReLU → (32×64×64)
    ↓
[Encoder Block 2] → R2Conv + BatchNorm + ReLU → (64×32×32)
    ↓
[Encoder Block 3] → R2Conv + BatchNorm + ReLU → (128×16×16)
    ↓
[Encoder Block 4] → R2Conv + BatchNorm + ReLU → (256×8×8)
    ↓
[Group Pooling] → Rotation Invariance
    ↓
[Latent Space] → FC(256×8×8 → 1024) → FC(1024 → 256×4×8×8)
    ↓
[Decoder Block 1] → R2Conv + BatchNorm + ReLU + Upsample → (128×16×16)
    ↓
[Decoder Block 2] → R2Conv + BatchNorm + ReLU + Upsample → (64×32×32)
    ↓
[Decoder Block 3] → R2Conv + BatchNorm + ReLU + Upsample → (32×64×64)
    ↓
[Decoder Block 4] → R2Conv + Sigmoid + Upsample → (1×128×128)
    ↓
Output (1×128×128)
```

## 📊 Performance Metrics

### Model Performance (on BraTS Test Set)
- **AUROC**: 0.8109
- **Specificity**: 58.54%
- **Training Time**: 2.7 hours (40 epochs, 251.1s/epoch)
- **Inference Time**: ~50ms per slice (CPU), ~10ms (GPU)

### Comparison with Other Models
| Model | AUROC | Training Time | Inference Time |
|-------|-------|---------------|----------------|
| **ECNN Optimized** | **0.8109** | 2.7h | 50ms |
| Large CNN-AE | 0.7803 | 1.23h | 30ms |
| Small CNN-AE | 0.7617 | 4h | 25ms |
| ResNet-AE | 0.8748 | 20min | 15ms |
| ResNet Mahalanobis | **0.9240** | 0min | 5ms |

**Key Insight**: ECNN achieves best **from-scratch** performance with rotation equivariance. ResNet models leverage pretrained features for superior results.

## 🔧 Configuration

### Environment Variables
```bash
# Optional: Set model path
export ECNN_MODEL_PATH="models/saved_models/ecnn_optimized_best.pth"

# Optional: Set default threshold
export ANOMALY_THRESHOLD=0.0035
```

### Streamlit Config
Create `.streamlit/config.toml`:
```toml
[theme]
primaryColor = "#1f77b4"
backgroundColor = "#ffffff"
secondaryBackgroundColor = "#f0f2f6"
textColor = "#262730"
font = "sans serif"

[server]
maxUploadSize = 200
enableCORS = false
```

## 🐛 Troubleshooting

### Issue 1: Model Not Found
```
Error: Model not found at models/saved_models/ecnn_optimized_best.pth
```
**Solution**: Train the model first or update the path in the sidebar.

### Issue 2: e2cnn Import Error
```
Error: No module named 'e2cnn'
```
**Solution**: Install e2cnn library
```bash
pip install e2cnn
```

### Issue 3: CUDA Out of Memory
```
Error: CUDA out of memory
```
**Solution**: The app automatically falls back to CPU. Reduce image size if needed.

### Issue 4: NIfTI Orientation Issues
```
Warning: Slice appears rotated
```
**Solution**: The app uses RAS canonical orientation. If issues persist, check your NIfTI file orientation.

## 🚀 Deployment

### Local Deployment
```bash
streamlit run streamlit_app.py
```

### Cloud Deployment (Streamlit Cloud)

1. **Push to GitHub**
   ```bash
   git add streamlit_app.py requirements_streamlit.txt
   git commit -m "Add Streamlit web interface"
   git push origin main
   ```

2. **Deploy on Streamlit Cloud**
   - Visit [streamlit.io/cloud](https://streamlit.io/cloud)
   - Connect GitHub repository
   - Select `streamlit_app.py`
   - Deploy!

3. **Configure Secrets** (if needed)
   - Add model path in Streamlit Cloud settings
   - Set up Google Drive integration for model loading

### Docker Deployment

Create `Dockerfile`:
```dockerfile
FROM python:3.9-slim

WORKDIR /app

COPY requirements_streamlit.txt .
RUN pip install -r requirements_streamlit.txt

COPY streamlit_app.py .
COPY models/ models/

EXPOSE 8501

CMD ["streamlit", "run", "streamlit_app.py", "--server.address", "0.0.0.0"]
```

Build and run:
```bash
docker build -t ecnn-streamlit .
docker run -p 8501:8501 ecnn-streamlit
```

## 📝 API Documentation

### Key Functions

#### `preprocess_nifti(nifti_path)`
Preprocess NIfTI file to match training pipeline.

**Parameters:**
- `nifti_path` (str): Path to NIfTI file

**Returns:**
- `preprocessed` (np.ndarray): Preprocessed 128×128 image
- `original` (np.ndarray): Original slice for display

#### `load_model(model_path)`
Load trained ECNN model with caching.

**Parameters:**
- `model_path` (str): Path to model checkpoint

**Returns:**
- `model` (nn.Module): Loaded model in eval mode
- `device` (torch.device): Device (CPU/CUDA)

#### `compute_anomaly_score(model, image, device)`
Compute anomaly score using reconstruction error.

**Parameters:**
- `model` (nn.Module): Trained ECNN model
- `image` (np.ndarray): Preprocessed image
- `device` (torch.device): Device for inference

**Returns:**
- `reconstruction` (np.ndarray): Reconstructed image
- `score` (float): Anomaly score (MSE)

## 🎓 Educational Use

This application is ideal for:
- **Thesis Demonstrations**: Live demo during defense
- **Research Presentations**: Interactive showcase of results
- **Educational Workshops**: Teaching anomaly detection
- **Proof of Concept**: Demonstrating model capabilities

## ⚠️ Limitations & Disclaimers

1. **Research Prototype**: Not approved for clinical use
2. **Single Slice**: Analyzes middle axial slice only (not full 3D volume)
3. **Binary Classification**: Detects anomaly presence, not specific diagnosis
4. **Training Data Bias**: Trained on IXI (adults), may not generalize to all populations
5. **Threshold Sensitivity**: Performance depends on threshold tuning
6. **Computational Requirements**: Requires sufficient RAM for NIfTI loading

## 📚 References

### Model Architecture
- Cohen & Welling (2016) - Group Equivariant CNNs
- e2cnn Library: [github.com/QUVA-Lab/e2cnn](https://github.com/QUVA-Lab/e2cnn)

### Datasets
- **IXI Dataset**: [brain-development.org/ixi-dataset](https://brain-development.org/ixi-dataset/)
- **BraTS 2021**: [synapse.org/Synapse:syn25829067](https://www.synapse.org/Synapse:syn25829067)

### Related Work
- Baur et al. (2021) - Autoencoders for Anomaly Detection in Brain MRI
- Schlegl et al. (2017) - Unsupervised Anomaly Detection with GANs

## 🤝 Contributing

Contributions welcome! Areas for improvement:
- [ ] Add 3D volume support
- [ ] Implement DICOM support
- [ ] Add batch processing
- [ ] Integrate grad-CAM visualization
- [ ] Add model comparison mode
- [ ] Export reports (PDF)

## 📄 License

MIT License - See LICENSE file for details

## 👨‍💻 Author

**symAD-ECNN Project**  
For questions or issues, please open a GitHub issue.

---

**Last Updated**: January 2026  
**Version**: 1.0.0  
**Status**: Production Ready ✅
