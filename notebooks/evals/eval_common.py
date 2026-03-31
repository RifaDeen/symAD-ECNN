from __future__ import annotations

import shutil
import zipfile
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm


DEFAULT_EXTS: Sequence[str] = ("*.npy", "*.png", "*.jpg", "*.jpeg")


def find_files(root_dir: Path, exts: Optional[Sequence[str]] = None) -> List[Path]:
    """Return sorted data files recursively under ``root_dir``."""
    patterns = exts or DEFAULT_EXTS
    files: List[Path] = []
    for ext in patterns:
        files.extend(root_dir.rglob(ext))
    return sorted(p for p in files if p.is_file())


def extract_zip(zip_path: Path, out_dir: Path, clean: bool = True) -> Path:
    """Extract zip to output directory, optionally clearing destination first."""
    if clean and out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(out_dir)
    return out_dir


def get_state_dict(ckpt_obj):
    """Normalize checkpoint formats to a plain state dict."""
    if isinstance(ckpt_obj, dict):
        if "model_state_dict" in ckpt_obj and isinstance(ckpt_obj["model_state_dict"], dict):
            return ckpt_obj["model_state_dict"]
        if "state_dict" in ckpt_obj and isinstance(ckpt_obj["state_dict"], dict):
            return ckpt_obj["state_dict"]
    return ckpt_obj


def resolve_checkpoint_path(
    cfg: Dict,
    models_root: Path,
    dirs_key: str = "checkpoint_dirs",
    patterns_key: str = "checkpoint_patterns",
    name_key: str = "display_name",
) -> Path:
    """Resolve first matching checkpoint from a model-root + config patterns."""
    roots: List[Path] = []
    for sub in cfg[dirs_key]:
        roots.append(models_root if sub == "." else (models_root / sub))

    for root in roots:
        if not root.exists():
            continue
        for pattern in cfg[patterns_key]:
            matches = sorted(root.rglob(pattern))
            if matches:
                return matches[0]

    name = cfg.get(name_key, cfg.get("key", "model"))
    raise FileNotFoundError(f"No checkpoint found for {name} under {models_root}")


def find_best_checkpoint(
    cfg: Dict,
    checkpoint_roots: Iterable[Path],
    subdirs_key: str = "subdirs",
    patterns_key: str = "patterns",
) -> Optional[Path]:
    """Find a best checkpoint across roots with preference for filenames containing 'best'."""
    candidates: List[Path] = []
    for root in checkpoint_roots:
        for sd in cfg[subdirs_key]:
            base = root if sd == "." else root / sd
            if not base.exists():
                continue
            for pat in cfg[patterns_key]:
                candidates.extend(p for p in base.rglob(pat) if p.is_file())

    candidates = sorted(set(candidates))
    if not candidates:
        return None

    candidates = sorted(
        candidates,
        key=lambda p: (("best" not in p.name.lower()), -p.stat().st_mtime),
    )
    return candidates[0]


@torch.no_grad()
def compute_reconstruction_errors(
    model: nn.Module,
    dataloader: DataLoader,
    device: str,
    desc: str = "Computing reconstruction errors",
) -> np.ndarray:
    """Compute per-sample reconstruction errors.

    By default this returns the mean squared-error per sample. Optional
    brain-masked scoring is supported by setting ``use_brain_mask=True``.

    Args:
        model: model to run inference with
        dataloader: dataloader yielding (input, target) pairs
        device: device string
        desc: progress description shown by tqdm
        error_mode: 'squared' (MSE) or 'abs' (L1)
        use_brain_mask: if True, reduce error only over pixels where input > 0.01
        min_brain_pixels: minimum number of brain pixels required; slices with
            fewer pixels are skipped

    Returns:
        numpy array of per-sample scalar scores (order preserved for samples
        that were not skipped).
    """
    from typing import List

    # Backwards-compatible parameter defaults
    error_mode = getattr(compute_reconstruction_errors, "_error_mode_default", "squared")
    use_brain_mask = getattr(compute_reconstruction_errors, "_use_brain_mask_default", False)
    min_brain_pixels = getattr(compute_reconstruction_errors, "_min_brain_pixels_default", 50)

    model.eval()
    errors: List[float] = []
    for x, y in tqdm(dataloader, desc=desc, leave=False):
        x = x.to(device)
        y = y.to(device)
        recon = model(x)

        if error_mode == "abs":
            per_pixel_error = torch.abs(recon - y)
        else:
            per_pixel_error = (recon - y) ** 2

        # per_pixel_error shape: (B, C, H, W) -- reduce per-sample
        b = per_pixel_error.size(0)
        for i in range(b):
            err_map = per_pixel_error[i]
            # If multi-channel, collapse channels by mean
            if err_map.dim() == 3 and err_map.size(0) > 1:
                err_map_reduced = err_map.view(err_map.size(0), -1).mean(dim=0)
            else:
                err_map_reduced = err_map.view(-1)

            if use_brain_mask:
                inp = x[i]
                if inp.dim() == 3:
                    inp_ch = inp[0]
                else:
                    inp_ch = inp

                # Determine spatial shape of the error map
                if err_map.dim() == 3:
                    h_err, w_err = int(err_map.size(-2)), int(err_map.size(-1))
                elif err_map.dim() == 2:
                    h_err, w_err = int(err_map.size(-2)), int(err_map.size(-1))
                else:
                    # Fallback: infer from flattened reduced map
                    flat_len = err_map_reduced.numel()
                    # assume square if we can't infer
                    h_err = w_err = int(int(flat_len ** 0.5))

                # Build mask from input channel and resize to match error map spatial dims
                mask_np = inp_ch.cpu().numpy() > 0.01
                if mask_np.sum() < min_brain_pixels:
                    # skip this slice if insufficient brain pixels
                    continue

                if mask_np.shape != (h_err, w_err):
                    # Resize mask to error-map resolution using bilinear interpolation
                    mask_t = torch.from_numpy(mask_np.astype(np.float32)).unsqueeze(0).unsqueeze(0)
                    mask_resized = F.interpolate(mask_t, size=(h_err, w_err), mode='bilinear', align_corners=False)
                    mask_bool = (mask_resized.squeeze().cpu().numpy() > 0.5)
                else:
                    mask_bool = mask_np

                vals = err_map_reduced.cpu().numpy()[mask_bool.flatten()]
                if vals.size == 0:
                    continue
                score = float(vals.mean())
            else:
                score = float(err_map_reduced.mean().cpu().item())

            errors.append(score)

    return np.asarray(errors, dtype=np.float32)


# Provide configurable defaults that callers may override via attributes
compute_reconstruction_errors._error_mode_default = "squared"
compute_reconstruction_errors._use_brain_mask_default = False
compute_reconstruction_errors._min_brain_pixels_default = 50
