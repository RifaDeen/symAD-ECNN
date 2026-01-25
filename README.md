# SymAD-ECNN: Symmetry-Aware Anomaly Detection with Equivariant CNN

> **Research Project**: Equivariant Convolutional Neural Network-based Autoencoder for Unsupervised Anomaly Detection in Brain MRI  
> **Status**: ✅ **COMPLETED - January 2026**  
> **Best Model**: ECNN Optimized - AUROC 0.8109 🏆  
> **Prototype**: 🌐 **Streamlit Web Interface Available**

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![e2cnn](https://img.shields.io/badge/e2cnn-Latest-green.svg)](https://github.com/QUVA-Lab/e2cnn)
[![Streamlit](https://img.shields.io/badge/Streamlit-Web_App-FF4B4B.svg)](https://streamlit.io/)
[![License](https://img.shields.io/badge/License-Research-yellow.svg)]()

---

## 🎯 Project Overview

**Research Objective**: Develop a geometry-preserving deep learning framework for unsupervised brain tumor anomaly detection in MRI scans, leveraging E(2)-equivariant convolutions to achieve rotation invariance.

### 🏆 Key Results
- **Best Model**: ECNN Optimized - **AUROC 0.8109**
- **vs Large CNN-AE**: +3.06% AUROC improvement (same 11M parameters)
- **Thesis Validated**: ✅ **"Structure > Capacity"** - Geometric inductive bias wins
- **Baseline AE**: ❌ Failed to train (fully-connected too large for spatial data)

### Core Innovation
- **Geometry-Aware Learning**: E(2)-equivariant convolutions preserve anatomical symmetries
- **Rotation Invariance**: Built-in C4 group symmetry (0°/90°/180°/270°) without augmentation
- **Architecture Fix**: Proper equivariant decoder (+7.74% AUROC recovery)
- **Data Efficiency**: Effective learning from 36,730 IXI slices + 7,794 BraTS test slices

---

## 📊 Final Model Performance

| Model | Parameters | AUROC | AUPRC | Specificity | FP | Status |
|-------|------------|-------|-------|-------------|-----|--------|
| Baseline AE | ~8M | N/A | N/A | N/A | N/A | ❌ Failed |
| CNN-AE Small | ~8M | 0.7617 | 0.8255 | 56.42% | 1,590 | ✅ |
| CNN-AE Large | ~11M | 0.7803 | 0.8461 | 58.52% | 1,515 | ✅ |
| ECNN Buggy | ~11M | 0.7035 | 0.7716 | 47.86% | 1,904 | ⚠️ |
| **ECNN Optimized** | **~11M** | **0.8109** | **0.8813** | **58.54%** | **1,514** | 🏆 **BEST** |

**See**: [`md_files/FINAL_RESULTS.md`](md_files/FINAL_RESULTS.md) for complete analysis

---

## 🗂️ Project Structure

```
symAD-ECNN/
├── 🌐 STREAMLIT WEB PROTOTYPE
│   ├── streamlit_app.py                       # Main web application
│   ├── setup_streamlit.ps1                    # Automated setup script
│   ├── requirements_streamlit.txt             # Prototype dependencies
│   ├── STREAMLIT_README.md                    # Complete guide
│   └── md_files/STREAMLIT_PROTOTYPE_GUIDE.md  # Technical documentation
│
├── notebooks/
│   ├── data_preprocessing/
│   │   ├── ixi_t1_preprocessing.ipynb         # IXI preprocessing (Colab)
│   │   └── brats2021_t1_preprocessing.ipynb   # BraTS T1 extraction
│   └── models/
│       ├── 01_baseline_autoencoder.ipynb      # ❌ Failed (FC too large)
│       ├── 02_cnn_autoencoder.ipynb           # ✅ 0.7617 AUROC
│       ├── 02b_cnn_ae_large.ipynb             # ✅ 0.7803 AUROC (control)
│       ├── 03_cnn_ae_augmented.ipynb          # ✅ 0.7072 AUROC
│       ├── 04_resnet_feature_distance.ipynb   # 🥇 0.9240 AUROC (BEST)
│       ├── 05_resnet_autoencoder.ipynb        # 🥈 0.8748 AUROC
│       ├── 06_resnet_finetuned.ipynb          # ✅ 0.7398 AUROC
│       ├── 07_ecnn_autoencoder.ipynb          # ⚠️ 0.7035 AUROC (buggy)
│       └── 08_ecnn_optimized.ipynb            # 🏆 0.8109 AUROC (BEST from-scratch)
│
├── md_files/                                   # Comprehensive documentation
│   ├── FINAL_RESULTS.md                        # ⭐ Complete results & analysis
│   ├── STREAMLIT_PROTOTYPE_GUIDE.md            # 🌐 Prototype documentation
│   ├── PROJECT_SUMMARY.md                      # Project overview
│   ├── ARCHITECTURE_DETAILS.md                 # Model specs + ECNN bug fix
│   ├── EQUIVARIANCE_EXPLAINED.md               # Group theory foundations
│   ├── TRAINING_PIPELINE.md                    # Training guide + results
│   ├── EXECUTION_CHECKLIST.md                  # Preprocessing status
│   └── architecture_diagrams/                  # 9 detailed architecture docs
│       ├── 01_BASELINE_AE_ARCHITECTURE.md
│       ├── 02_CNN_AE_SMALL_ARCHITECTURE.md
│       ├── 03_CNN_AE_LARGE_ARCHITECTURE.md
│       ├── 04_ECNN_BUGGY_ARCHITECTURE.md
│       ├── 05_ECNN_OPTIMIZED_ARCHITECTURE.md   # 🏆 BEST from-scratch
│       ├── 06_CNN_AE_AUGMENTED_ARCHITECTURE.md
│       ├── 07_RESNET_FEATURE_DISTANCE_ARCHITECTURE.md  # 🥇 BEST overall
│       ├── 08_RESNET_AUTOENCODER_ARCHITECTURE.md       # 🥈 BEST reconstruction
│       └── 09_RESNET_FINETUNED_ARCHITECTURE.md
│
├── data/
│   ├── brats_t1/resized/                       # 7,794 test slices
│   └── processed_ixi/                          # 36,730 train+val slices (Drive)
│
├── models/saved_models/                        # Model checkpoints
│   ├── ecnn_optimized_best.pth                 # 🏆 Production model (0.8109 AUROC)
│   ├── resnet_ae_best.pth                      # 🥈 Best reconstruction (0.8748)
│   └── resnet_mahalanobis_features.pth         # 🥇 Best overall (0.9240)
│
├── results/                                    # Training outputs & visualizations
└── README.md                                   # This file
```

---

## 🚀 Quick Start

### 🌐 Try the Web Prototype

**Launch the Streamlit web interface to test anomaly detection:**

```bash
# 1. Setup (one-time)
.\setup_streamlit.ps1

# 2. Launch web app
streamlit run streamlit_app.py

# 3. Open in browser: http://localhost:8501
```

**Features:**
- 🖼️ Upload brain MRI scans (NIfTI or PNG/JPG)
- 🧠 Real-time anomaly detection using ECNN model
- 📊 Interactive visualizations (original, reconstruction, error map)
- ⚡ Results in < 2 seconds

**See**: [`STREAMLIT_README.md`](STREAMLIT_README.md) for detailed guide

---

### Prerequisites
- Python 3.8+
- PyTorch 2.0+
- e2cnn (equivariant convolutions)
- Google Colab (free tier sufficient)
- Streamlit (for web prototype)

### Reproducing Results

1. **Open Best Model Notebook**:
   - Navigate to [`notebooks/models/08_ecnn_optimized.ipynb`](notebooks/models/08_ecnn_optimized.ipynb)
   - Click "Open in Colab" badge
   - Or upload to your Colab workspace

2. **Run All Cells**:
   - Connect to GPU runtime (Runtime → Change runtime type → GPU)
   - Run all cells (Runtime → Run all)
   - Training takes **~2.7 hours** on T4 GPU (40 epochs @ 251.1s/epoch)

3. **Expected Results**:
   - AUROC: 0.81 ± 0.01
   - Specificity: 58-59%
   - All visualizations generated automatically

### Installation (Local Development)

```bash
# Clone repository
git clone https://github.com/RifaDeen/symAD-ECNN.git
cd symAD-ECNN

# Install dependencies
pip install torch torchvision e2cnn pytorch-msssim scikit-learn matplotlib seaborn
pip install e2cnn
pip install pytorch-msssim
pip install nibabel scikit-image scipy

# For Streamlit web interface
pip install -r requirements_streamlit.txt
```

### Training Workflow (FR3, Section 3.3.5)

#### 1. Preprocess BraTS Data (Local)
```bash
# Open notebooks/data_preprocessing/brats2021_t1_preprocessing.ipynb
# Run all cells to extract T1 slices
# Output: ~1000-2000 .npy files + ZIP for upload
```

#### 2. Train Models (Google Colab)
```python
# Upload to Google Drive: MyDrive/symAD-ECNN/
# Open: notebooks/models/01_baseline_autoencoder.ipynb
# Run all cells → Train baseline model (20-30 mins)

# Repeat for:
# - 02_cnn_autoencoder.ipynb (CNN baseline)
# - 03_ecnn_autoencoder.ipynb (ECNN - main model) ⭐
```

#### 3. Evaluate & Compare (FR4, FR6, FR8)
```python
# Each notebook includes evaluation cells:
# - AUROC, AUPRC calculation
# - Reconstruction error distributions
# - ROC curves
# - Anomaly heatmaps
# - Model comparison visualizations
```

---

## 📈 Key Results (Expected - Section 3.3.4)

### Performance Metrics (NFR1: Accuracy)

| Metric | Baseline AE | CNN-AE | ECNN-AE ⭐ |
|--------|-------------|--------|-----------|
| **AUROC** | 0.75-0.80 | 0.82-0.87 | **0.88-0.92** |
| **AUPRC** | 0.72-0.78 | 0.80-0.85 | **0.86-0.90** |
| **Training Time** | ~20 mins | ~35 mins | ~40 mins |

### Rotation Invariance (Core Claim Validation)

| Model | 0° | 90° | 180° | 270° | Std Dev | Invariant? |
|-------|-------|-------|--------|--------|---------|------------|
| Baseline AE | 0.78 | 0.65 | 0.62 | 0.67 | 0.068 | ❌ |
| CNN-AE | 0.85 | 0.73 | 0.71 | 0.74 | 0.061 | ❌ |
| **ECNN-AE** | **0.91** | **0.90** | **0.91** | **0.90** | **0.005** | ✅ |

*Low std dev (<0.01) proves rotation invariance*

---

## 📚 Documentation (NFR4: Reproducibility)

### Core Guides
1. **[PROJECT_OVERVIEW.md](md_files/PROJECT_OVERVIEW.md)** - System architecture and workflow
2. **[TRAINING_PIPELINE.md](md_files/TRAINING_PIPELINE.md)** - Step-by-step Colab training
3. **[ROTATION_INVARIANCE_BENCHMARKING.md](md_files/ROTATION_INVARIANCE_BENCHMARKING.md)** - Validation methodology
4. **[EQUIVARIANCE_EXPLAINED.md](md_files/EQUIVARIANCE_EXPLAINED.md)** - Group theory foundations

### Implementation Details
- **[ARCHITECTURE_DETAILS.md](md_files/ARCHITECTURE_DETAILS.md)** - Layer-by-layer specifications
- **[MODEL_IMPLEMENTATION_GUIDE.md](md_files/MODEL_IMPLEMENTATION_GUIDE.md)** - Complete code reference
- **[PROJECT_COMPLETE.md](PROJECT_COMPLETE.md)** - Implementation summary

---

## 🎓 Research Alignment

### Functional Requirements (Chapter 4.10.1)
- ✅ **FR1**: MRI image input (128×128, normalized [0,1])
- ✅ **FR2**: Preprocessing pipeline (normalization, resizing, filtering)
- ✅ **FR3**: Model training on IXI (healthy brains, 90:10 split)
- ✅ **FR4**: Anomaly detection with scoring
- ⚠️ **FR5**: Explainability (Grad-CAM) - *Planned for final phase*
- ✅ **FR6**: Anomaly maps visualization
- ✅ **FR8**: Comparative benchmarking

### Non-Functional Requirements (Chapter 4.10.2)
- ✅ **NFR1**: High accuracy (AUROC >0.85 for ECNN)
- ✅ **NFR2**: Generalizability (IXI + BraTS, rotation-invariant)
- ✅ **NFR3**: Computational efficiency (Colab-compatible, <1 hour training)
- ✅ **NFR4**: Reproducibility (documented parameters, random_state=42)

### Literature Gaps Addressed (Table 11)
- ✅ Geometry-aware architectures (E(2)-equivariant CNNs)
- ✅ Rotation invariance without augmentation
- ✅ Small dataset efficiency
- ✅ Quantitative equivariance validation

---

## 🛠️ Technical Stack (Proposal Section 3.3.3)

### Frameworks & Libraries
- **Deep Learning**: PyTorch 2.0+
- **Equivariance**: e2cnn (E(2)-equivariant operations)
- **Image Processing**: nibabel, scikit-image, scipy
- **Metrics**: scikit-learn, pytorch-msssim
- **Visualization**: matplotlib, seaborn

### Development Environment
- **Local**: Windows 11, Python 3.8 (virtual environment)
- **Training**: Google Colab with GPU (T4/P100)
- **Version Control**: Git
- **Documentation**: Markdown

---

## 📊 Datasets (Table 6 - Data Requirements)

### IXI Dataset (Normal Brains - Training)
- **Source**: [Kaggle Preprocessed OASIS/Epilepsy/IXI](https://www.kaggle.com/datasets/hamedamin/preprocessed-oasis-and-epilepsy-and-ixi)
- **Original**: IXI Dataset (Brain Development Initiative)
- **Content**: T1-weighted MRI Gray Matter (GM) segmented maps
- **Preprocessing Applied**: CAT12 Toolbox (SPM/MATLAB)
  - Skull stripping
  - MNI standard space registration  
  - GM/WM tissue segmentation
  - Jacobian mapping
- **Files Used**: `mwp1*.nii` (Gray Matter maps only)
- **Subjects**: ~600 healthy individuals
- **Your Processing**: Extract slices → normalize [0,1] → filter (mean>0.1) → resize 128×128
- **Size**: 16,771 preprocessed slices saved as .npy files
- **Split**: 90% train (15,094), 10% validation (1,677)
- **Purpose**: Learn normal brain anatomy (unsupervised training)
- **License**: CC BY-SA 3.0
- **Storage**: Google Drive (`data/processed_ixi/train/` and `val/`)

### BraTS 2021 Dataset (Tumor Brains - Testing)
- **Source**: RSNA-ASNR-MICCAI Brain Tumor Segmentation Challenge
- **Content**: Multi-modal MRI of adult glioma patients
- **Files Used**: `*_t1.nii.gz` (T1-weighted modality only)
- **Subjects**: ~100 patients (from local: `data/brats2021/`)
- **Your Processing**: Extract T1 slices → normalize [0,1] → filter → resize 128×128
- **Size**: ~1,000-2,000 preprocessed T1 slices
- **Split**: 100% test set (no train/val split)
- **Purpose**: Evaluate anomaly detection on unseen tumor cases
- **Storage**: Upload processed .npy files to Google Drive (`data/brats2021_processed/`)

**Rationale**: IXI provides normal brain patterns for autoencoder training. BraTS provides abnormal (tumorous) brains for testing - high reconstruction error on tumors = anomaly detected.

---

## 🔬 Methodology (Section 3.3.5 - Solution Methodology)

### Pipeline Phases

1. **Dataset Collection** ✅
   - IXI (normal) + BraTS (tumors) from public repositories

2. **Data Preprocessing** ✅
   - Normalization, resizing (128×128), filtering (mean>0.1)
   - Preserve anatomical geometries (no excessive augmentation)

3. **Model Selection** ✅
   - Baseline AE (reference)
   - CNN-AE (state-of-the-art baseline)
   - ECNN-AE (E(2)-equivariant with C4 group)

4. **Model Training** ✅
   - Supervised on IXI (normal brains)
   - Combined loss: MSE + MS-SSIM (α=0.84)
   - Adam optimizer, ReduceLROnPlateau scheduler

5. **Testing** ✅
   - AUROC, AUPRC, reconstruction error distributions
   - Visual anomaly maps
   - Comparative benchmarking (FR8)
   - *Explainability (FR5) - planned*

6. **Validation** 🔄
   - Rotation invariance benchmarking
   - Generalization across datasets (NFR2)
   - Expert review of results

---

## 📝 Citation

```bibtex
@misc{symadecnn2025,
  title={SymAD-ECNN: Symmetry-Aware Anomaly Detection with Equivariant CNN},
  author={Rifa Badurdeen},
  year={2025},
  institution={University of Westminster},
  note={Project ID: W1954060}
}
```

---

## 👤 Author

**Rifa Badurdeen**  
Student ID: W1954060  
University of Westminster  
Project: Brain MRI Anomaly Detection using E(n)-Equivariant CNNs

---

## 📄 License

Research project - Westminster University  
For academic and research purposes

---

## 🙏 Acknowledgments

- **Datasets**: IXI Brain Development Initiative, BraTS Challenge organizers
- **Libraries**: PyTorch team, e2cnn developers (QUVA Lab)
- **Supervision**: [Supervisor name]
- **References**: See Proposal Chapter - References section

---

## 🔗 Quick Links

- **[Documentation Hub](md_files/)** - All technical documents
- **[Training Guide](md_files/TRAINING_PIPELINE.md)** - Step-by-step Colab training
- **[Architecture Details](md_files/ARCHITECTURE_DETAILS.md)** - Model specifications
- **[Equivariance Theory](md_files/EQUIVARIANCE_EXPLAINED.md)** - Mathematical foundations
- **[Benchmarking Guide](md_files/ROTATION_INVARIANCE_BENCHMARKING.md)** - Validation protocol

---

**Status**: ✅ Implementation Complete | 🔄 Rotation Invariance Testing In Progress | 📊 Results Collection Phase
