# app.py
"""
🧠 Brain MRI Anomaly Detection System
======================================
Streamlit web interface for detecting brain anomalies using E(2)-Equivariant CNN Autoencoder

Author: symAD-ECNN Project
Model: ECNN Optimized (AUROC 0.8109)

✅ Preprocessing: matches IXI + BraTS pipelines
- RAS orientation (NIfTI)
- Artifact removal (99th percentile clip on non-zero)
- Normalize to [0,1]
- Middle 50% axial slice selection
- Slice validity check (BraTS-style)
- Resize 128×128 with BICUBIC (order=3)
- Centering via center-of-mass → (64,64)

✅ Outputs:
- Anomaly / Normal decision (threshold)
- Reconstruction
- Error map (abs diff)
- Smoothed map (Gaussian)
- Comparison chart (threshold vs score)
- Interactive explorer (Plotly)
"""

import os
import io
import tempfile
from pathlib import Path

import streamlit as st
import numpy as np
import torch
import torch.nn as nn

import matplotlib.pyplot as plt
import scipy.ndimage as ndimage

from PIL import Image

# NIfTI
try:
    import nibabel as nib
except ImportError:
    nib = None

# Resize (IXI/BraTS used skimage.resize with bicubic order=3)
try:
    from skimage.transform import resize as sk_resize
except ImportError:
    sk_resize = None

import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Import e2cnn for the model
try:
    from e2cnn import gspaces
    from e2cnn import nn as e2nn
except ImportError:
    st.error("⚠️ e2cnn library not found. Please install: `pip install e2cnn`")
    st.stop()

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="Brain MRI Anomaly Detection",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# CSS
# ============================================================
st.markdown(
    """
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }

    .anomaly-detected {
        background-color: #ef5350;
        color: white !important;
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin-bottom: 20px;
    }
    .anomaly-detected h3, .anomaly-detected p { color: white !important; }

    .normal-detected {
        background-color: #66bb6a;
        color: white !important;
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin-bottom: 20px;
    }
    .normal-detected h3, .normal-detected p { color: white !important; }
</style>
""",
    unsafe_allow_html=True
)

# ============================================================
# MODEL (ECNN OPTIMIZED) — must match your saved checkpoint
# ============================================================
class ECNNAutoencoderV3(nn.Module):
    """
    The 'Champion' Architecture (Final Match).
    1. Enforces C4 Rotational Symmetry.
    2. Enforces Information Bottleneck.
    3. Scaled up to ~11M params (Wide Channels + 1024 latent).
    """
    def __init__(self, latent_dim=1024):
        super().__init__()

        # Symmetry group
        self.r2_act = gspaces.Rot2dOnR2(N=4)
        self.in_type = e2nn.FieldType(self.r2_act, [self.r2_act.trivial_repr])

        # Wide feature fields
        self.type_128 = e2nn.FieldType(self.r2_act, 32 * [self.r2_act.regular_repr])
        self.type_256 = e2nn.FieldType(self.r2_act, 64 * [self.r2_act.regular_repr])
        self.type_512 = e2nn.FieldType(self.r2_act, 128 * [self.r2_act.regular_repr])
        self.type_1024 = e2nn.FieldType(self.r2_act, 256 * [self.r2_act.regular_repr])

        # Encoder
        self.encoder = nn.Sequential(
            e2nn.R2Conv(self.in_type, self.type_128, kernel_size=7, padding=3, stride=2),
            e2nn.InnerBatchNorm(self.type_128), e2nn.ReLU(self.type_128),

            e2nn.R2Conv(self.type_128, self.type_256, kernel_size=3, padding=1, stride=2),
            e2nn.InnerBatchNorm(self.type_256), e2nn.ReLU(self.type_256),

            e2nn.R2Conv(self.type_256, self.type_512, kernel_size=3, padding=1, stride=2),
            e2nn.InnerBatchNorm(self.type_512), e2nn.ReLU(self.type_512),

            e2nn.R2Conv(self.type_512, self.type_1024, kernel_size=3, padding=1, stride=2),
            e2nn.InnerBatchNorm(self.type_1024), e2nn.ReLU(self.type_1024),

            e2nn.PointwiseMaxPool(self.type_1024, kernel_size=2, stride=2)
        )

        # Bottleneck
        self.group_pool = e2nn.GroupPooling(self.type_1024)
        self.flat_dim = 256 * 4 * 4
        self.fc_encode = nn.Linear(self.flat_dim, latent_dim)
        self.fc_decode = nn.Linear(latent_dim, self.flat_dim)

        # Decoder blocks
        self.up1 = self._up_block(self.type_1024, self.type_512)
        self.up2 = self._up_block(self.type_512, self.type_256)
        self.up3 = self._up_block(self.type_256, self.type_128)
        self.final_conv = e2nn.R2Conv(self.type_128, self.in_type, kernel_size=3, padding=1)
        self.sigmoid = nn.Sigmoid()

    def _up_block(self, in_type, out_type):
        return nn.Sequential(
            e2nn.R2Conv(in_type, out_type, kernel_size=3, padding=1),
            e2nn.InnerBatchNorm(out_type),
            e2nn.ReLU(out_type)
        )

    def forward(self, x):
        x_geo = e2nn.GeometricTensor(x, self.in_type)

        features = self.encoder(x_geo)
        invariant = self.group_pool(features)

        b = invariant.tensor.size(0)
        flat = invariant.tensor.view(b, -1)

        z = self.fc_encode(flat)
        z_expand = self.fc_decode(z)

        z_view = z_expand.view(-1, 256, 4, 4)

        # Expand back into regular reps (repeat 4)
        x_recon = e2nn.GeometricTensor(z_view.repeat(1, 4, 1, 1), self.type_1024)

        # Upsample chain (matches your notebook style)
        x_recon = nn.functional.interpolate(x_recon.tensor, scale_factor=2, mode="bilinear")
        x_recon = e2nn.GeometricTensor(x_recon, self.type_1024)

        x_recon = nn.functional.interpolate(x_recon.tensor, scale_factor=2, mode="bilinear")
        x_recon = self.up1(e2nn.GeometricTensor(x_recon, self.type_1024))

        x_recon = nn.functional.interpolate(x_recon.tensor, scale_factor=2, mode="bilinear")
        x_recon = self.up2(e2nn.GeometricTensor(x_recon, self.type_512))

        x_recon = nn.functional.interpolate(x_recon.tensor, scale_factor=2, mode="bilinear")
        x_recon = self.up3(e2nn.GeometricTensor(x_recon, self.type_256))

        x_recon = nn.functional.interpolate(x_recon.tensor, scale_factor=2, mode="bilinear")
        x_recon = self.final_conv(e2nn.GeometricTensor(x_recon, self.type_128))

        return self.sigmoid(x_recon.tensor)

# ============================================================
# PREPROCESSING (IXI + BraTS aligned)
# ============================================================
IMG_SIZE = (128, 128)
MIDDLE_PERCENTAGE = 0.5  # middle 50%

def _require_skimage():
    if sk_resize is None:
        st.error("⚠️ scikit-image not found. Install: `pip install scikit-image`")
        st.stop()

def remove_artifacts(volume: np.ndarray, percentile: float = 99.0) -> np.ndarray:
    v = volume.astype(np.float32)
    nz = v[v > 0]
    if nz.size == 0:
        return v
    thr = np.percentile(nz, percentile)
    return np.clip(v, 0, thr)

def normalize_01(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32)
    mn, mx = float(x.min()), float(x.max())
    if mx - mn < 1e-8:
        return np.zeros_like(x, dtype=np.float32)
    return (x - mn) / (mx - mn)

def is_valid_slice_brats_style(slice_array: np.ndarray, min_nonzero_ratio: float = 0.12, min_mean: float = 0.10) -> bool:
    nonzero_ratio = np.count_nonzero(slice_array) / slice_array.size
    if nonzero_ratio < min_nonzero_ratio:
        return False
    s = normalize_01(slice_array)
    if float(s.mean()) < min_mean:
        return False
    return True

def center_slice(slice_img: np.ndarray, target_center=(64, 64)) -> np.ndarray:
    if np.count_nonzero(slice_img) < 100:
        return slice_img.astype(np.float32)
    com = ndimage.center_of_mass(slice_img > 0.1)
    cy, cx = com
    shift_y = target_center[0] - cy
    shift_x = target_center[1] - cx
    centered = ndimage.shift(slice_img, [shift_y, shift_x], order=1, mode="constant", cval=0)
    return centered.astype(np.float32)

def resize_128_bicubic(slice_img: np.ndarray) -> np.ndarray:
    _require_skimage()
    return sk_resize(
        slice_img,
        IMG_SIZE,
        order=3,
        mode="reflect",
        anti_aliasing=True,
        preserve_range=True
    ).astype(np.float32)

def extract_candidate_indices(z_slices: int, middle_percentage: float = 0.5) -> np.ndarray:
    start = int(z_slices * (0.5 - middle_percentage / 2))
    end = int(z_slices * (0.5 + middle_percentage / 2))
    start = max(0, start)
    end = min(z_slices, end)
    if end <= start:
        start, end = 0, z_slices
    return np.arange(start, end, dtype=int)

def preprocess_volume_like_ixi_brats(volume: np.ndarray):
    vol = remove_artifacts(volume, percentile=99.0)
    vol = normalize_01(vol)

    z = vol.shape[2]
    candidates = extract_candidate_indices(z, MIDDLE_PERCENTAGE).tolist()

    valid = []
    for i in candidates:
        sl = vol[:, :, i]
        if is_valid_slice_brats_style(sl):
            valid.append(i)

    if len(valid) == 0:
        valid = [z // 2]

    default_idx = valid[len(valid) // 2]
    return vol, valid, default_idx

def finalize_slice_pipeline(slice_img_2d: np.ndarray, apply_centering: bool = True):
    s = normalize_01(slice_img_2d)
    s_resized = resize_128_bicubic(s)
    s_centered = center_slice(s_resized, target_center=(64, 64)) if apply_centering else s_resized
    s_centered = np.clip(s_centered, 0.0, 1.0)
    return s_centered.astype(np.float32), s_resized.astype(np.float32)

def preprocess_nifti_file(path: str, apply_centering: bool = True, slice_strategy: str = "auto"):
    if nib is None:
        st.error("⚠️ nibabel not found. Install: `pip install nibabel`")
        st.stop()

    img = nib.load(path)
    img = nib.as_closest_canonical(img)
    data = img.get_fdata()

    if data.ndim == 4:
        data = data[..., 0]
    if data.ndim != 3:
        raise ValueError(f"NIfTI must be 3D (or 4D). Got shape {data.shape}")

    vol_norm, valid_indices, default_idx = preprocess_volume_like_ixi_brats(data)

    if slice_strategy == "mid":
        idx = vol_norm.shape[2] // 2
    else:
        idx = default_idx

    slice_2d = vol_norm[:, :, idx]
    pre_128, resized_128 = finalize_slice_pipeline(slice_2d, apply_centering=apply_centering)

    return {
        "preprocessed": pre_128,
        "resized_no_center": resized_128,
        "volume_norm": vol_norm,
        "valid_indices": valid_indices,
        "selected_index": idx,
        "source_type": "nifti"
    }

def preprocess_any_image_bytes(file_bytes: bytes, apply_centering: bool = True):
    img = Image.open(io.BytesIO(file_bytes)).convert("RGB")
    arr = np.array(img)

    if arr.ndim == 3:
        arr = arr.mean(axis=2)

    arr = normalize_01(arr.astype(np.float32))
    pre_128, resized_128 = finalize_slice_pipeline(arr, apply_centering=apply_centering)

    return {"preprocessed": pre_128, "resized_no_center": resized_128, "source_type": "image"}

def preprocess_npy_bytes(file_bytes: bytes, apply_centering: bool = True):
    buf = io.BytesIO(file_bytes)
    arr = np.load(buf, allow_pickle=False)
    arr = np.array(arr)

    if arr.ndim == 3 and arr.shape[-1] in (3, 4):
        arr = arr[..., :3].mean(axis=2)
        pre_128, resized_128 = finalize_slice_pipeline(arr, apply_centering=apply_centering)
        return {"preprocessed": pre_128, "resized_no_center": resized_128, "source_type": "npy(image)"}

    if arr.ndim == 3:
        if arr.shape[0] < 32 and arr.shape[2] >= 32:
            vol = np.transpose(arr, (1, 2, 0))
        else:
            vol = arr

        vol_norm, valid_indices, default_idx = preprocess_volume_like_ixi_brats(vol)
        idx = default_idx
        slice_2d = vol_norm[:, :, idx]
        pre_128, resized_128 = finalize_slice_pipeline(slice_2d, apply_centering=apply_centering)
        return {
            "preprocessed": pre_128,
            "resized_no_center": resized_128,
            "volume_norm": vol_norm,
            "valid_indices": valid_indices,
            "selected_index": idx,
            "source_type": "npy(volume)"
        }

    if arr.ndim == 2:
        pre_128, resized_128 = finalize_slice_pipeline(arr, apply_centering=apply_centering)
        return {"preprocessed": pre_128, "resized_no_center": resized_128, "source_type": "npy(2d)"}

    raise ValueError(f"Unsupported .npy shape: {arr.shape}")

# ============================================================
# MODEL LOADING / INFERENCE
# ============================================================
@st.cache_resource
def load_model(model_path: str, latent_dim: int = 1024):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ECNNAutoencoderV3(latent_dim=latent_dim).to(device)

    if not os.path.exists(model_path):
        return None, device, f"Model file not found: {model_path}"

    try:
        checkpoint = torch.load(model_path, map_location=device)
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            state = checkpoint["model_state_dict"]
        else:
            state = checkpoint

        missing, unexpected = model.load_state_dict(state, strict=False)
        model.eval()

        warn_msg = ""
        if missing or unexpected:
            warn_msg = f"Loaded with strict=False. Missing keys: {len(missing)}, Unexpected keys: {len(unexpected)}"
        return model, device, warn_msg
    except Exception as e:
        return None, device, f"Error loading model: {e}"

def compute_anomaly(model, image_128: np.ndarray, device):
    img_tensor = torch.from_numpy(image_128).float().unsqueeze(0).unsqueeze(0).to(device)
    with torch.no_grad():
        recon = model(img_tensor)

    mse = torch.nn.functional.mse_loss(recon, img_tensor, reduction="none")
    mse = mse.view(mse.size(0), -1).mean(dim=1)

    recon_np = recon.detach().cpu().squeeze().numpy().astype(np.float32)
    score = float(mse.detach().cpu().item())
    return recon_np, score

# ============================================================
# VISUALS
# ============================================================
def plot_results_plotly(original, reconstruction, error_map, smooth_map, score, threshold):
    fig = make_subplots(
        rows=1, cols=4,
        subplot_titles=("Original (128×128)", "Reconstruction", "Error Map", "Smoothed Map"),
        horizontal_spacing=0.03
    )

    fig.add_trace(go.Heatmap(z=original, colorscale="gray", showscale=False), row=1, col=1)
    fig.add_trace(go.Heatmap(z=reconstruction, colorscale="gray", showscale=False), row=1, col=2)
    fig.add_trace(go.Heatmap(z=error_map, colorscale="hot", showscale=True,
                             colorbar=dict(title="Error", x=1.02)), row=1, col=3)
    fig.add_trace(go.Heatmap(z=smooth_map, colorscale="jet", showscale=False), row=1, col=4)

    fig.update_xaxes(showticklabels=False)
    fig.update_yaxes(showticklabels=False)

    status = "(ANOMALY ⚠️)" if score > threshold else "(NORMAL ✓)"
    fig.update_layout(
        height=420,
        title_text=f"Anomaly Score: {score:.6f} {status}",
        title_font_size=16
    )
    return fig

def plot_clinical_dashboard(original, reconstruction, error_map, smooth_map, score, threshold):
    status = "🚨 ANOMALY" if score > threshold else "✅ HEALTHY"
    color = "red" if score > threshold else "green"

    fig, axes = plt.subplots(1, 6, figsize=(20, 5))

    axes[0].imshow(original, cmap="gray")
    axes[0].set_title("1. Patient Scan", fontweight="bold", fontsize=10)
    axes[0].axis("off")

    axes[1].imshow(reconstruction, cmap="gray")
    axes[1].set_title("2. AI Reconstruction", fontsize=10)
    axes[1].axis("off")

    axes[2].imshow(error_map, cmap="hot", vmin=0, vmax=np.max(error_map) * 0.8 + 1e-8)
    axes[2].set_title("3. Raw Difference\n(Noisy)", fontsize=10)
    axes[2].axis("off")

    axes[3].imshow(smooth_map, cmap="jet", vmin=0, vmax=np.max(smooth_map) + 1e-8)
    axes[3].set_title("4. Smoothed Map\n(Clinical Focus)", fontsize=10, fontweight="bold")
    axes[3].axis("off")

    axes[4].imshow(smooth_map, cmap="jet", vmin=0, vmax=max(0.1, float(np.max(smooth_map))))
    axes[4].set_title("5. Localization", fontsize=10, fontweight="bold")
    axes[4].axis("off")

    bars = axes[5].bar(["Limit", "Score"], [threshold, score], color=["gray", color])
    axes[5].axhline(y=threshold, color="black", linestyle="--", alpha=0.5)
    axes[5].set_title(f"6. Diagnosis\n{status}", fontweight="bold", color=color, fontsize=10)
    axes[5].set_ylim(0, max(threshold, score) * 1.3 + 1e-8)

    for bar in bars:
        yval = bar.get_height()
        axes[5].text(bar.get_x() + bar.get_width() / 2, yval, f"{yval:.4f}",
                     ha="center", va="bottom", fontsize=9)

    axes[5].spines["top"].set_visible(False)
    axes[5].spines["right"].set_visible(False)

    plt.tight_layout()
    return fig

# ============================================================
# APP
# ============================================================
def main():
    st.markdown('<p class="main-header">🧠 Brain MRI Anomaly Detection System</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Powered by E(2)-Equivariant CNN Autoencoder (symAD-ECNN)</p>', unsafe_allow_html=True)

    with st.sidebar:
        st.image("https://img.icons8.com/color/96/000000/brain.png", width=80)
        st.title("Settings")

        model_path = st.text_input(
            "Model Path",
            value="saved_models/ecnn_optimized_best.pth",
            help="Path to your trained checkpoint (must match ECNNAutoencoderV3)."
        )

        threshold = st.slider("Anomaly Threshold", 0.0005, 0.0200, 0.0035, 0.0001, format="%.4f")

        st.divider()
        apply_centering = st.toggle("Apply Centering (IXI/BraTS)", value=True)
        slice_strategy = st.selectbox("Slice Strategy (for 3D volumes)", ["auto", "manual", "mid"], index=0)

        st.divider()
        st.markdown("**Model**: ECNN Optimized (≈11M Params)\n\n**Input**: 128×128 (1-channel)")

    col1, col2 = st.columns([1, 1], gap="large")

    prep = None
    original_128 = None

    with col1:
        st.subheader("📤 Upload Scan (Anything)")
        st.caption("Supported: .nii / .nii.gz / .png / .jpg / .jpeg / .bmp / .tif / .tiff / .npy")

        uploaded = st.file_uploader(
            "Choose a file",
            type=["nii", "nii.gz", "png", "jpg", "jpeg", "bmp", "tif", "tiff", "npy"],
        )

        if uploaded is not None:
            st.info(f"**Filename**: {uploaded.name}")

            suffix = Path(uploaded.name).suffix.lower()
            if uploaded.name.lower().endswith(".nii.gz"):
                suffix = ".nii.gz"

            file_bytes = uploaded.getvalue()

            try:
                with st.spinner("🔄 Preprocessing (IXI + BraTS pipeline)..."):
                    if suffix in [".nii", ".nii.gz"]:
                        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                            tmp.write(file_bytes)
                            tmp_path = tmp.name

                        prep = preprocess_nifti_file(
                            tmp_path,
                            apply_centering=apply_centering,
                            slice_strategy=slice_strategy
                        )
                        os.unlink(tmp_path)

                        if slice_strategy == "manual" and "volume_norm" in prep and "valid_indices" in prep:
                            valid = prep["valid_indices"]
                            if len(valid) > 1:
                                chosen = st.select_slider(
                                    "Select Axial Slice Index (from valid middle-50%)",
                                    options=valid,
                                    value=prep["selected_index"]
                                )
                                slice_2d = prep["volume_norm"][:, :, int(chosen)]
                                pre_128, resized_128 = finalize_slice_pipeline(slice_2d, apply_centering=apply_centering)
                                prep["preprocessed"] = pre_128
                                prep["resized_no_center"] = resized_128
                                prep["selected_index"] = int(chosen)

                        st.success("✅ Ready")
                        st.caption(f"Source: NIfTI (RAS-corrected). Selected slice index: {prep.get('selected_index', 'N/A')}")

                    elif suffix == ".npy":
                        prep = preprocess_npy_bytes(file_bytes, apply_centering=apply_centering)
                        st.success("✅ Ready")
                        st.caption(f"Source: {prep.get('source_type')}")

                    else:
                        prep = preprocess_any_image_bytes(file_bytes, apply_centering=apply_centering)
                        st.success("✅ Ready")
                        st.caption("Source: 2D image")

                if prep is not None:
                    original_128 = prep["preprocessed"]
                    st.image(original_128, caption="Preprocessed Input (128×128)", use_column_width=True, clamp=True)

            except Exception as e:
                st.error(f"Preprocessing failed: {e}")

    with col2:
        st.subheader("🔍 Analysis Results")

        if prep is not None and original_128 is not None:
            with st.spinner("📦 Loading model..."):
                model, device, warn = load_model(model_path, latent_dim=1024)

            if warn:
                st.warning(warn)

            if model is None:
                st.error("Model not loaded. Check your path and checkpoint compatibility.")
            else:
                with st.spinner("🧠 Running inference..."):
                    recon_128, score = compute_anomaly(model, original_128, device)

                error_map = np.abs(original_128 - recon_128).astype(np.float32)
                smooth_map = ndimage.gaussian_filter(error_map, sigma=2).astype(np.float32)

                if score > threshold:
                    st.markdown(
                        f"""
                        <div class="anomaly-detected">
                            <h3>⚠️ ANOMALY DETECTED</h3>
                            <p><strong>Score:</strong> {score:.6f}</p>
                            <p><strong>Threshold:</strong> {threshold:.4f}</p>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
                else:
                    st.markdown(
                        f"""
                        <div class="normal-detected">
                            <h3>✓ NORMAL SCAN</h3>
                            <p><strong>Score:</strong> {score:.6f}</p>
                            <p><strong>Threshold:</strong> {threshold:.4f}</p>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                c1, c2, c3 = st.columns(3)
                c1.metric("MSE Score", f"{score:.6f}")
                c2.metric("Max Error", f"{float(error_map.max()):.6f}")
                c3.metric("Mean Error", f"{float(error_map.mean()):.6f}")

                st.divider()
                st.write("### Quick Views")
                v1, v2 = st.columns(2)
                v1.image(recon_128, caption="Reconstruction", use_column_width=True, clamp=True)
                v2.image(error_map, caption="Error Map (abs diff)", use_column_width=True, clamp=True)

                st.caption("Smoothed map shown in dashboard below (Gaussian σ=2).")

    if prep is not None and original_128 is not None:
        with st.spinner("📊 Preparing visualizations..."):
            model, device, _ = load_model(model_path, latent_dim=1024)
            if model is not None:
                recon_128, score = compute_anomaly(model, original_128, device)
                error_map = np.abs(original_128 - recon_128).astype(np.float32)
                smooth_map = ndimage.gaussian_filter(error_map, sigma=2).astype(np.float32)

                st.divider()
                st.subheader("🖼️ Clinical Visualization")

                tab1, tab2 = st.tabs(["🏥 Clinical Dashboard (Notebook-style)", "🔬 Interactive Explorer (Plotly)"])

                with tab1:
                    st.write("### 6-Panel Diagnostic View")
                    st.caption("Includes reconstruction, raw error, smoothed heatmap, localization, and threshold vs score chart.")
                    fig = plot_clinical_dashboard(original_128, recon_128, error_map, smooth_map, score, threshold)
                    st.pyplot(fig)

                with tab2:
                    st.write("### Interactive Pixel Explorer (4 Panels)")
                    figp = plot_results_plotly(original_128, recon_128, error_map, smooth_map, score, threshold)
                    st.plotly_chart(figp, use_column_width=True)

if __name__ == "__main__":
    main()
