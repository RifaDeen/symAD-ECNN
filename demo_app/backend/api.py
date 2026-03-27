import io
import os
import json
import tempfile
from pathlib import Path

import numpy as np
from flask import Flask, request, jsonify

import torch
import torch.nn as nn

import nibabel as nib
from skimage.transform import resize
from scipy import ndimage, interpolate

from e2cnn import gspaces
from e2cnn import nn as e2nn

from domain_models import PredictOptions
from inference_service import InferenceService, RiskScoringService
from prediction_service import PredictionService
from preprocessing_service import PreprocessingService


# ============================================================
# PATHS (relative to this file, so it works anywhere)
# ============================================================
BACKEND_DIR = Path(__file__).resolve().parent
REPO_ROOT = BACKEND_DIR.parents[1]  # demo_app/backend -> demo_app -> repo root

DEFAULT_MODEL_PATH = REPO_ROOT / "models" / "saved_models" / "ecnn_optimized_best.pth"
DEFAULT_METRICS_PATH = BACKEND_DIR / "metrics_ecnn_v3.json"


# ============================================================
# CONFIG
# ============================================================
IMG_SIZE = (128, 128)
MIDDLE_SLICE_RATIO = 0.5  # middle 50%

# Default threshold from metrics JSON (Youden J)
DEFAULT_THRESHOLD = 0.0035
METRICS_JSON_PATH = Path(os.environ.get("SYMAD_METRICS_JSON", str(DEFAULT_METRICS_PATH)))
if METRICS_JSON_PATH.exists():
    try:
        metrics = json.loads(METRICS_JSON_PATH.read_text())
        DEFAULT_THRESHOLD = float(metrics.get("optimal_threshold", DEFAULT_THRESHOLD))
    except Exception:
        pass

# Nyul landmarks (from your notebook)
STANDARD_LANDMARKS = np.array([
    1.11170489e-41, 5.18809652e-25, 3.05301042e-17, 4.70776470e-12,
    6.94896974e-07, 5.00224718e-03, 1.75081036e-01, 5.40189319e-01,
    7.13654902e-01, 8.37299432e-01, 9.57721521e-01
], dtype=np.float32)
PERCENTILES = [1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 99]


# ============================================================
# MODEL (ECNN Optimized V3)
# ============================================================
class ECNNAutoencoderV3(nn.Module):
    def __init__(self, latent_dim=1024):
        super().__init__()
        self.r2_act = gspaces.Rot2dOnR2(N=4)
        self.in_type = e2nn.FieldType(self.r2_act, [self.r2_act.trivial_repr])

        self.type_128 = e2nn.FieldType(self.r2_act, 32 * [self.r2_act.regular_repr])
        self.type_256 = e2nn.FieldType(self.r2_act, 64 * [self.r2_act.regular_repr])
        self.type_512 = e2nn.FieldType(self.r2_act, 128 * [self.r2_act.regular_repr])
        self.type_1024 = e2nn.FieldType(self.r2_act, 256 * [self.r2_act.regular_repr])

        self.encoder = nn.Sequential(
            e2nn.R2Conv(self.in_type, self.type_128, kernel_size=7, padding=3, stride=2),
            e2nn.InnerBatchNorm(self.type_128),
            e2nn.ReLU(self.type_128),

            e2nn.R2Conv(self.type_128, self.type_256, kernel_size=3, padding=1, stride=2),
            e2nn.InnerBatchNorm(self.type_256),
            e2nn.ReLU(self.type_256),

            e2nn.R2Conv(self.type_256, self.type_512, kernel_size=3, padding=1, stride=2),
            e2nn.InnerBatchNorm(self.type_512),
            e2nn.ReLU(self.type_512),

            e2nn.R2Conv(self.type_512, self.type_1024, kernel_size=3, padding=1, stride=2),
            e2nn.InnerBatchNorm(self.type_1024),
            e2nn.ReLU(self.type_1024),

            e2nn.PointwiseMaxPool(self.type_1024, kernel_size=2, stride=2),
        )

        self.group_pool = e2nn.GroupPooling(self.type_1024)
        self.flat_dim = 256 * 4 * 4
        self.fc_encode = nn.Linear(self.flat_dim, latent_dim)
        self.fc_decode = nn.Linear(latent_dim, self.flat_dim)

        self.up1 = self._up_block(self.type_1024, self.type_512)
        self.up2 = self._up_block(self.type_512, self.type_256)
        self.up3 = self._up_block(self.type_256, self.type_128)

        self.final_conv = e2nn.R2Conv(self.type_128, self.in_type, kernel_size=3, padding=1)
        self.sigmoid = nn.Sigmoid()

    def _up_block(self, in_type, out_type):
        return nn.Sequential(
            e2nn.R2Conv(in_type, out_type, kernel_size=3, padding=1),
            e2nn.InnerBatchNorm(out_type),
            e2nn.ReLU(out_type),
        )

    def forward(self, x):
        x_geo = e2nn.GeometricTensor(x, self.in_type)
        feats = self.encoder(x_geo)

        inv = self.group_pool(feats)
        b = inv.tensor.size(0)
        flat = inv.tensor.view(b, -1)

        z = self.fc_encode(flat)
        z_expand = self.fc_decode(z)
        z_view = z_expand.view(-1, 256, 4, 4)

        x_recon = e2nn.GeometricTensor(z_view.repeat(1, 4, 1, 1), self.type_1024)

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
# PREPROCESSING
# ============================================================
def remove_artifacts(volume, percentile=99):
    nz = volume[volume > 0]
    if nz.size == 0:
        return volume
    thr = np.percentile(nz, percentile)
    return np.clip(volume, 0, thr)

def normalize01(x):
    x = x.astype(np.float32)
    mn, mx = float(x.min()), float(x.max())
    if mx > mn:
        return (x - mn) / (mx - mn)
    return x

def resize_128(x2d):
    return resize(
        x2d, IMG_SIZE,
        order=3, mode="reflect",
        anti_aliasing=True,
        preserve_range=True
    ).astype(np.float32)

def center_slice(slice_img, target_center=(64, 64)):
    if np.count_nonzero(slice_img) < 100:
        return slice_img
    cy, cx = ndimage.center_of_mass(slice_img > 0.1)
    shift_y = target_center[0] - cy
    shift_x = target_center[1] - cx
    return ndimage.shift(slice_img, [shift_y, shift_x], order=1, mode="constant", cval=0)

def nyul_normalize(img_01):
    img = img_01.astype(np.float32)
    mask = img > 0
    if np.sum(mask) < 100:
        return img

    curr_landmarks = np.percentile(img[mask], PERCENTILES).astype(np.float32)

    x = np.concatenate(([0.0], curr_landmarks, [float(img.max())])).astype(np.float32)
    y = np.concatenate(([0.0], STANDARD_LANDMARKS, [float(STANDARD_LANDMARKS[-1])])).astype(np.float32)

    x_unique, idx = np.unique(x, return_index=True)
    y_unique = y[idx]
    if x_unique.size < 2:
        return img

    f = interpolate.interp1d(x_unique, y_unique, bounds_error=False, fill_value="extrapolate")
    out = f(img).astype(np.float32)
    out[~mask] = 0.0
    return np.clip(out, 0.0, 1.0)

def pick_middle_slice_index(vol):
    z = vol.shape[2]
    start = int(z * (0.5 - MIDDLE_SLICE_RATIO / 2))
    end = int(z * (0.5 + MIDDLE_SLICE_RATIO / 2))
    start = max(0, start)
    end = min(z, end)
    if end <= start:
        return z // 2

    best_i = (start + end) // 2
    best_score = -1
    for i in range(start, end):
        sl = vol[:, :, i]
        score = int(np.count_nonzero(sl))
        if score > best_score:
            best_score = score
            best_i = i
    return best_i

def preprocess_any(file_bytes: bytes, filename: str, apply_nyul: bool, apply_center: bool):
    name = filename.lower()
    debug = {}

    if name.endswith(".nii") or name.endswith(".nii.gz"):
        suffix = ".nii.gz" if name.endswith(".nii.gz") else ".nii"
        tmp_path = None
        try:
            # Create temp file and CLOSE it before nibabel reads (important on Windows)
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
            tmp_path = tmp.name
            tmp.write(file_bytes)
            tmp.close()

            img_obj = nib.load(tmp_path)
            img_obj = nib.as_closest_canonical(img_obj)
            vol = img_obj.get_fdata()

        finally:
            # Ensure nibabel released file handle before deleting
            if tmp_path is not None:
                try:
                    # This helps Windows release file locks
                    import gc
                    gc.collect()
                    os.unlink(tmp_path)
                except Exception:
                    pass


        if vol.ndim == 4:
            vol = vol[..., 0]
        if vol.ndim != 3:
            raise ValueError(f"Expected 3D NIfTI. Got shape {vol.shape}")

        vol = remove_artifacts(vol, percentile=99)
        vol = normalize01(vol)

        idx = pick_middle_slice_index(vol)
        sl = vol[:, :, idx]

        sl = resize_128(sl)
        if apply_center:
            sl = center_slice(sl)
        sl = np.clip(sl, 0, 1).astype(np.float32)
        if apply_nyul:
            sl = nyul_normalize(sl)

        nz_ratio = float(np.count_nonzero(sl) / sl.size)
        debug.update({
            "type": "nifti",
            "orig_shape": tuple(vol.shape),
            "selected_slice": int(idx),
            "nonzero_ratio": nz_ratio,
            "min": float(sl.min()),
            "max": float(sl.max())
        })
        if nz_ratio < 0.05:
            debug["warning"] = "Sparse slice (<5% non-zero) - might be mask/empty slice."

        return sl, debug

    if name.endswith(".npy"):
        arr = np.load(io.BytesIO(file_bytes), allow_pickle=False)
        arr = np.array(arr)

        if arr.ndim == 3:
            if arr.shape[0] < 32 and arr.shape[2] >= 32:
                arr = np.transpose(arr, (1, 2, 0))
            idx = pick_middle_slice_index(arr)
            sl = arr[:, :, idx]
            debug["selected_slice"] = int(idx)
        elif arr.ndim == 2:
            sl = arr
        else:
            raise ValueError(f"Unsupported npy shape: {arr.shape}")

        sl = sl.astype(np.float32)
        if np.any(sl > 0):
            sl = np.clip(sl, 0, np.percentile(sl[sl > 0], 99))
        sl = normalize01(sl)

        sl = resize_128(sl)
        if apply_center:
            sl = center_slice(sl)
        sl = np.clip(sl, 0, 1).astype(np.float32)
        if apply_nyul:
            sl = nyul_normalize(sl)

        debug.update({
            "type": "npy",
            "orig_shape": tuple(arr.shape),
            "nonzero_ratio": float(np.count_nonzero(sl) / sl.size),
            "min": float(sl.min()),
            "max": float(sl.max())
        })
        return sl, debug

    # images
    from PIL import Image
    img = Image.open(io.BytesIO(file_bytes)).convert("L")
    sl = np.array(img).astype(np.float32)

    if np.any(sl > 0):
        sl = np.clip(sl, 0, np.percentile(sl[sl > 0], 99))
    sl = normalize01(sl)

    sl = resize_128(sl)
    if apply_center:
        sl = center_slice(sl)
    sl = np.clip(sl, 0, 1).astype(np.float32)
    if apply_nyul:
        sl = nyul_normalize(sl)

    debug.update({
        "type": "image",
        "orig_shape": tuple(np.array(img).shape),
        "nonzero_ratio": float(np.count_nonzero(sl) / sl.size),
        "min": float(sl.min()),
        "max": float(sl.max())
    })
    return sl, debug


# ============================================================
# INFERENCE
# ============================================================
def compute_score_and_maps(model, device, x_128):
    inp = torch.from_numpy(x_128).float().unsqueeze(0).unsqueeze(0).to(device)

    with torch.no_grad():
        recon = model(inp)

    mse_map = (recon - inp) ** 2
    score = float(mse_map.view(1, -1).mean().detach().cpu().item())

    recon_np = recon.detach().cpu().squeeze().numpy().astype(np.float32)
    err_abs = np.abs(x_128 - recon_np).astype(np.float32)
    err_smooth = ndimage.gaussian_filter(err_abs, sigma=2).astype(np.float32)

    return recon_np, err_abs, err_smooth, score

def compute_risk_level(score: float, threshold: float, normal_mean: float, anomaly_mean: float) -> str:
    """
    Simple, thesis-safe risk banding using your reported means.
    - Low: below threshold
    - Medium: between threshold and anomaly_mean
    - High: >= anomaly_mean
    - Very High: >= 1.5 * anomaly_mean  (optional but useful)
    """
    if score < threshold:
        return "LOW"
    if score < anomaly_mean:
        return "MEDIUM"
    if score < 1.5 * anomaly_mean:
        return "HIGH"
    return "VERY_HIGH"


def get_slice_indices_around(center_idx: int, depth: int, k: int) -> list:
    """
    Return k slice indices centered around center_idx (clamped to [0, depth-1]).
    k should be odd (5,7,9).
    """
    k = int(k)
    if k <= 1:
        return [int(center_idx)]
    if k % 2 == 0:
        k += 1

    half = k // 2
    idxs = [center_idx + i for i in range(-half, half + 1)]
    idxs = [max(0, min(depth - 1, i)) for i in idxs]

    # remove duplicates if clamped at edges
    idxs_unique = []
    for i in idxs:
        if i not in idxs_unique:
            idxs_unique.append(i)

    return idxs_unique


def preprocess_single_slice_from_volume(vol01: np.ndarray, idx: int, apply_nyul: bool, apply_center: bool):
    """
    vol01 is already normalized and artifact-clipped.
    Returns 128x128 slice in [0,1].
    """
    sl = vol01[:, :, idx]
    sl = resize_128(sl)
    if apply_center:
        sl = center_slice(sl)
    sl = np.clip(sl, 0, 1).astype(np.float32)
    if apply_nyul:
        sl = nyul_normalize(sl)
    return sl


# ============================================================
# LOAD MODEL
# ============================================================
def load_model(model_path: Path):
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found at: {model_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ECNNAutoencoderV3(latent_dim=1024).to(device)

    ckpt = torch.load(str(model_path), map_location=device)
    state = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
    model.load_state_dict(state, strict=False)
    model.eval()
    return model, device


# ============================================================
# FLASK APP
# ============================================================
app = Flask(__name__)

MODEL_PATH = Path(os.environ.get("SYMAD_MODEL_PATH", str(DEFAULT_MODEL_PATH)))
MODEL, DEVICE = load_model(MODEL_PATH)

preprocessing_service = PreprocessingService()
inference_service = InferenceService(MODEL, DEVICE)
risk_service = RiskScoringService()
prediction_service = PredictionService(preprocessing_service, inference_service, risk_service)


@app.get("/health")
def health():
    return jsonify({
        "status": "ok",
        "device": str(DEVICE),
        "model_path": str(MODEL_PATH),
        "default_threshold": DEFAULT_THRESHOLD,
        "metrics_json": str(METRICS_JSON_PATH) if METRICS_JSON_PATH.exists() else None
    })


@app.post("/predict")
def predict():
    if "file" not in request.files:
        return jsonify({"error": "Missing 'file' in form-data"}), 400

    f = request.files["file"]
    filename = f.filename or "upload"
    file_bytes = f.read()

    threshold = float(request.form.get("threshold", DEFAULT_THRESHOLD))
    options = PredictOptions(
        threshold=threshold,
        apply_center=request.form.get("apply_center", "true").lower() == "true",
        apply_nyul=request.form.get("apply_nyul", "true").lower() == "true",
        skip_preprocess=request.form.get("skip_preprocess", "false").lower() == "true",
        use_aggregation=request.form.get("use_aggregation", "false").lower() == "true",
        agg_slices=int(request.form.get("agg_slices", "7")),
        agg_method=request.form.get("agg_method", "mean").lower().strip(),
    )

    normal_mean = float(metrics.get("normal_error_mean", 0.0028592257294803858)) if METRICS_JSON_PATH.exists() else 0.0028592257
    anomaly_mean = float(metrics.get("anomaly_error_mean", 0.00501307612285018)) if METRICS_JSON_PATH.exists() else 0.0050130761

    try:
        result = prediction_service.predict(
            file_bytes=file_bytes,
            filename=filename,
            options=options,
            normal_mean=normal_mean,
            anomaly_mean=anomaly_mean,
        )
        return jsonify(result.to_dict())
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
