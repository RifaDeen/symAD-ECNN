# SymAD-ECNN: Geometry Aware Equivariant Learning for Anomaly Detection in Biomedical Imaging

> **Research Project**: Equivariant Convolutional Neural Network-based Autoencoder for Unsupervised Anomaly Detection in Brain MRI  
> **Status**: ✅ **COMPLETED - March 2026**  
> **Best Model**: ECNN Optimized - AUROC 0.8625 🏆  
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
- **Best Model**: ECNN Optimized - **AUROC 0.8600**
- **vs Large CNN-AE**: +7.97% AUROC improvement (same 11M parameters)
- **Thesis Validated**: ✅ **"Structure > Capacity"** - Geometric inductive bias wins
- **Baseline AE**: ❌ Failed to train (fully-connected too large for spatial data)

### Core Innovation
- **Geometry-Aware Learning**: E(2)-equivariant convolutions preserve anatomical symmetries
- **Rotation Invariance**: Built-in C4 group symmetry (0°/90°/180°/270°) without augmentation
- **Architecture Fix**: Proper equivariant decoder (+15.65% AUROC recovery)
- **Data Efficiency**: Effective learning from 36,730 IXI slices + 7,794 BraTS test slices


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
│       ├── 02_cnn_autoencoder.ipynb           # parameters unmatched
│       ├── 02b_cnn_ae_large.ipynb             
│       ├── 03_cnn_ae_augmented.ipynb          
│       ├── 04_resnet_feature_distance.ipynb   
│       ├── 05_resnet_autoencoder.ipynb        
│       ├── 06_resnet_finetuned.ipynb         
│       ├── 07_ecnn_autoencoder.ipynb           (buggy)
│       └── 08_ecnn_optimized.ipynb            
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
│       ├── 07_RESNET_FEATURE_DISTANCE_ARCHITECTURE.md  
│       ├── 08_RESNET_AUTOENCODER_ARCHITECTURE.md      
│       └── 09_RESNET_FINETUNED_ARCHITECTURE.md
│
├── data/
│   ├── brats_t1/resized/                       # 7,794 test slices
│   └── processed_ixi/                          # 36,730 train+val slices (Drive)
│
├── models/saved_models/                        # Model checkpoints
│   ├── ecnn_optimized_best.pth                 # 🏆 Production model (0.8600 AUROC)
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

## 🎓 Research Alignment

### Functional Requirements (Chapter 4.10.1)
- ✅ **FR1**: MRI image input (128×128, normalized [0,1])
- ✅ **FR2**: Preprocessing pipeline (normalization, resizing, filtering)
- ✅ **FR3**: Model training on IXI (healthy brains, 90:10 split)
- ✅ **FR4**: Anomaly detection with scoring
- ✅ **FR5**: Explainability - Reconsttruction error maps
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
- **Size**: 36,730 preprocessed slices saved as .npy files
- **Split**: 90% train (33,078), 10% validation (3,652)
- **Purpose**: Learn normal brain anatomy (unsupervised training)
- **License**: CC BY-SA 3.0
- **Storage**: Google Drive (`data/processed_ixi/train/` and `val/`)

### BraTS 2021 Dataset (Tumor Brains - Testing)
- **Source**: RSNA-ASNR-MICCAI Brain Tumor Segmentation Challenge
- **Content**: Multi-modal MRI of adult glioma patients
- **Files Used**: `*_t1.nii.gz` (T1-weighted modality only)
- **Subjects**: 100 patients (from local: `data/brats2021/`)
- **Your Processing**: Extract T1 slices → normalize [0,1] → filter → resize 128×128
- **Size**: 7,794 preprocessed T1 slices
- **Split**: 100% test set (no train/val split)
- **Purpose**: Evaluate anomaly detection on unseen tumor cases
- **Storage**: Local processed test set at `data/brats_t1/resized/` (and optionally mirrored to Drive)

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
   - Unsupervised reconstruction training on IXI (normal brains)
   - Combined loss: MSE + MS-SSIM (α=0.84)
   - Adam optimizer, ReduceLROnPlateau scheduler

5. **Testing** ✅
   - AUROC, AUPRC, reconstruction error distributions
   - Visual anomaly maps
   - Comparative benchmarking (FR8)
   - *Explainability (FR5) - planned*

6. **Validation** ✅
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
- **Supervision**: Ms. Nethmi Wijesinghe
- **References**: See Proposal Chapter - References section

---

