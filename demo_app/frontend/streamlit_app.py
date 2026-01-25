import json
from pathlib import Path

import requests
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots

st.set_page_config(page_title="Brain MRI Anomaly Detection", page_icon="🧠", layout="wide")
st.title("🧠 Brain MRI Anomaly Detection (ECNN Optimized V3)")
st.caption("Mode A: raw MRI (preprocess ON) | Mode B: preprocessed validation slice (preprocess OFF)")

# ------------------------------------------------------------
# Paths
# ------------------------------------------------------------
APP_DIR = Path(__file__).resolve().parent
DEMO_APP_DIR = APP_DIR.parent
METRICS_PATH = DEMO_APP_DIR / "backend" / "metrics_ecnn_v3.json"

DEFAULT_OPT_THR = 0.002618970815092325
METRICS = None
if METRICS_PATH.exists():
    try:
        METRICS = json.loads(METRICS_PATH.read_text())
        DEFAULT_OPT_THR = float(METRICS.get("optimal_threshold", DEFAULT_OPT_THR))
    except Exception:
        METRICS = None

# ------------------------------------------------------------
# Sidebar header
# ------------------------------------------------------------
col1, col2 = st.sidebar.columns([1, 3])
with col1:
    st.image("https://img.icons8.com/color/96/000000/brain.png", width=80)
with col2:
    st.markdown("<h1 style='margin-top: 2px; margin-bottom: 5px; font-weight: 700;'>symAD-ECNN</h1>", unsafe_allow_html=True)

st.sidebar.header("Settings")
API_URL = st.sidebar.text_input("Flask API URL", "http://localhost:5000")

mode = st.sidebar.radio(
    "Input Mode",
    ["Raw MRI (apply preprocessing)", "Preprocessed (skip preprocessing)"],
    index=0
)
skip_preprocess = (mode == "Preprocessed (skip preprocessing)")

st.sidebar.divider()
use_opt = st.sidebar.toggle("Use optimal threshold (Youden J)", value=True)
if use_opt:
    threshold = DEFAULT_OPT_THR
    st.sidebar.success(f"Threshold = {threshold:.6f}")
else:
    threshold = st.sidebar.slider("Manual threshold", 0.0005, 0.02, float(DEFAULT_OPT_THR), 0.0001, format="%.6f")

apply_center = True
apply_nyul = True
if not skip_preprocess:
    st.sidebar.divider()
    st.sidebar.subheader("Preprocessing options")
    apply_center = st.sidebar.toggle("Centering (COM → 64,64)", value=True)
    apply_nyul = st.sidebar.toggle("Nyul matching", value=True)

if METRICS:
    st.sidebar.divider()
    st.sidebar.subheader("Reported Evaluation")
    st.sidebar.write(f"AUROC: **{METRICS['auroc']:.4f}**")
    st.sidebar.write(f"AUPRC: **{METRICS['auprc']:.4f}**")
    st.sidebar.write(f"Accuracy: **{METRICS['accuracy']:.4f}**")
    st.sidebar.write(f"Precision: **{METRICS['precision']:.4f}**")
    st.sidebar.write(f"Recall: **{METRICS['recall']:.4f}**")
    st.sidebar.write(f"Specificity: **{METRICS['specificity']:.4f}**")
    st.sidebar.write(f"F1: **{METRICS['f1_score']:.4f}**")

# ------------------------------------------------------------
# Inference strategy (slice aggregation)
# ------------------------------------------------------------
st.sidebar.divider()
st.sidebar.subheader("Inference Strategy")

use_aggregation = False
agg_slices = 7
agg_method = "mean"

# Only meaningful for raw NIfTI mode
if not skip_preprocess:
    use_aggregation = st.sidebar.toggle("Use slice aggregation (recommended)", value=True)
    if use_aggregation:
        agg_slices = st.sidebar.select_slider("Number of slices (odd)", options=[1, 3, 5, 7, 9], value=7)
        agg_method = st.sidebar.selectbox("Aggregation method", ["mean", "median"], index=0)
else:
    st.sidebar.info("Aggregation disabled in preprocessed mode.")

st.divider()

# ------------------------------------------------------------
# Upload section
# ------------------------------------------------------------
if not skip_preprocess:
    st.header("Raw Upload (Preprocessing ON)")
    st.write("Upload `.nii/.nii.gz` or images. Pipeline: clip99 → normalize → resize128 → (center) → (Nyul)")
    upload_types = ["nii", "nii.gz", "png", "jpg", "jpeg", "bmp", "tif", "tiff", "npy"]
else:
    st.header("Preprocessed Upload (Preprocessing OFF)")
    st.write("Upload validation `.npy` slices already `128×128` in `[0,1]`.")
    upload_types = ["npy", "png", "jpg", "jpeg"]

uploaded = st.file_uploader("Upload file", type=upload_types)

# ------------------------------------------------------------
# Visualization helper (Notebook-style)
# ------------------------------------------------------------
def render_notebook_panels(x, recon, err, smooth, score=None):
    """
    Notebook-style visualization:
    - Input: gray
    - Recon: gray
    - Raw Error: hot, vmax = 0.8*max(err)
    - Smoothed: jet, vmax = max(smooth)
    """
    vmax_err = float(err.max()) * 0.8 if float(err.max()) > 0 else 1e-6
    vmax_sm  = float(smooth.max()) if float(smooth.max()) > 0 else 1e-6

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))

    axes[0].imshow(x, cmap="gray")
    axes[0].set_title("Input")
    axes[0].axis("off")

    axes[1].imshow(recon, cmap="gray")
    axes[1].set_title(f"Reconstruction\nMSE: {score:.5f}" if score is not None else "Reconstruction")
    axes[1].axis("off")

    im2 = axes[2].imshow(err, cmap="hot", vmin=0, vmax=vmax_err)
    axes[2].set_title("Raw Error")
    axes[2].axis("off")
    fig.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.02)

    im3 = axes[3].imshow(smooth, cmap="jet", vmin=0, vmax=vmax_sm)
    axes[3].set_title("Smoothed Map (σ=2)")
    axes[3].axis("off")
    fig.colorbar(im3, ax=axes[3], fraction=0.046, pad=0.02)

    plt.tight_layout()
    return fig


if uploaded:
    st.info(f"File: {uploaded.name}")

    files = {"file": (uploaded.name, uploaded.getvalue())}
    data = {
        "threshold": str(threshold),
        "apply_center": str(apply_center).lower(),
        "apply_nyul": str(apply_nyul).lower(),
        "skip_preprocess": str(skip_preprocess).lower(),
        "use_aggregation": str(use_aggregation).lower(),
        "agg_slices": str(int(agg_slices)),
        "agg_method": agg_method,
    }

    with st.spinner("Running inference..."):
        r = requests.post(f"{API_URL}/predict", files=files, data=data, timeout=180)

    if r.status_code != 200:
        st.error(f"API Error ({r.status_code}): {r.text}")
        st.stop()

    out = r.json()
    if "error" in out:
        st.error(out["error"])
        st.stop()

    score = float(out["score"])
    risk = out.get("risk_level", "UNKNOWN")
    agg = out.get("aggregation", {"enabled": False})

    # Risk badge UI
    if risk in ["HIGH", "VERY_HIGH"]:
        st.error(f"⚠️ Risk: {risk} — score={score:.6f} | threshold={threshold:.6f}")
    elif risk == "MEDIUM":
        st.warning(f"⚠️ Risk: MEDIUM — score={score:.6f} | threshold={threshold:.6f}")
    else:
        st.success(f"✓ Risk: LOW — score={score:.6f} | threshold={threshold:.6f}")

    # Aggregation summary
    if agg.get("enabled"):
        st.caption(
            f"Aggregation ON → method={agg.get('method')} | slices={agg.get('k')} | "
            f"indices={agg.get('slice_indices')} | representative slice={agg.get('rep_slice')}"
        )

    arr = out["arrays"]
    x = np.array(arr["input"], dtype=np.float32)
    recon = np.array(arr["reconstruction"], dtype=np.float32)
    err = np.array(arr["error_abs"], dtype=np.float32)
    smooth = np.array(arr["error_smooth"], dtype=np.float32)

    if agg.get("enabled") and "slice_scores" in agg:
        st.subheader("Slice Score Distribution")
        st.line_chart({"score": agg["slice_scores"]}, height=200)

    st.divider()

    nz_ratio = float(np.count_nonzero(x) / x.size)
    if nz_ratio < 0.05:
        st.warning("Sparse input (<5% non-zero). This might be a mask/empty slice.")

    # Notebook-style maps
    st.subheader("Notebook-style Visualization")
    fig = render_notebook_panels(x, recon, err, smooth, score=score)
    st.pyplot(fig, use_container_width=True)

    # # Localization view (fixed vmax)
    # show_localization = st.checkbox("Show Localization View (fixed vmax=0.1)", value=True)
    # if show_localization:
    #     fig2, ax = plt.subplots(1, 1, figsize=(4, 4))
    #     im = ax.imshow(smooth, cmap="jet", vmin=0, vmax=0.1)
    #     ax.set_title("Localization (vmax=0.1)")
    #     ax.axis("off")
    #     fig2.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    #     st.pyplot(fig2, use_container_width=False)

    st.divider()

    # Threshold vs Score chart
    st.subheader("Threshold vs Score")
    fig3, ax3 = plt.subplots(figsize=(4.6, 3.4))
    ax3.bar(["Threshold", "Score"], [threshold, score])
    ax3.axhline(threshold, linestyle="--", alpha=0.7)
    ax3.set_ylim(0, max(threshold, score) * 1.35 + 1e-6)
    st.pyplot(fig3)

    st.divider()

    # Plotly explorer
    st.subheader("Interactive Explorer")
    figp = make_subplots(rows=1, cols=3, subplot_titles=("Input", "Reconstruction", "Error"))
    figp.add_trace(go.Heatmap(z=x, colorscale="gray", showscale=False), row=1, col=1)
    figp.add_trace(go.Heatmap(z=recon, colorscale="gray", showscale=False), row=1, col=2)
    figp.add_trace(go.Heatmap(z=err, colorscale="hot", showscale=True), row=1, col=3)
    figp.update_xaxes(showticklabels=False)
    figp.update_yaxes(showticklabels=False)
    figp.update_layout(height=420, title_text=f"Score={score:.6f} | Thr={threshold:.6f}")
    st.plotly_chart(figp, use_container_width=True)
