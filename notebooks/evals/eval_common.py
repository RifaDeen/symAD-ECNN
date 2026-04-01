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
    """Compute per-sample reconstruction errors with optional brain-mask scoring.

    Runtime defaults can be overridden by setting function attributes:
    - ``compute_reconstruction_errors._error_mode_default`` in {"squared", "abs"}
    - ``compute_reconstruction_errors._use_brain_mask_default`` as bool
    - ``compute_reconstruction_errors._min_brain_pixels_default`` as int
    """
    model.eval()

    error_mode = str(getattr(compute_reconstruction_errors, "_error_mode_default", "squared")).lower()
    use_brain_mask = bool(getattr(compute_reconstruction_errors, "_use_brain_mask_default", False))
    min_brain_pixels = int(getattr(compute_reconstruction_errors, "_min_brain_pixels_default", 50))

    if error_mode not in {"squared", "abs"}:
        raise ValueError(f"Unsupported error mode: {error_mode}")

    errors: List[float] = []
    for x, y in tqdm(dataloader, desc=desc, leave=False):
        x = x.to(device)
        y = y.to(device)
        recon = model(x)

        diff = recon - y
        err_map = diff.abs() if error_mode == "abs" else diff.pow(2)

        if not use_brain_mask:
            batch_scores = err_map.view(err_map.size(0), -1).mean(dim=1)
            errors.extend(batch_scores.detach().cpu().numpy().tolist())
            continue

        # Mask-aware scoring: use non-zero target region (brain) and skip tiny-mask slices.
        # Keep behavior consistent across shape variants by resizing mask if needed.
        if y.dim() == 4:
            y_for_mask = y[:, 0, :, :] if y.size(1) >= 1 else y.mean(dim=1)
        else:
            y_for_mask = y

        for i in range(err_map.size(0)):
            sample_err = err_map[i]
            if sample_err.dim() == 3:
                sample_err = sample_err[0]

            mask = (y_for_mask[i] > 0.01).float().unsqueeze(0).unsqueeze(0)
            target_hw = sample_err.shape[-2:]
            if tuple(mask.shape[-2:]) != tuple(target_hw):
                mask = F.interpolate(mask, size=target_hw, mode="nearest")

            mask_bool = mask.squeeze().bool()
            if int(mask_bool.sum().item()) < min_brain_pixels:
                continue

            score = sample_err[mask_bool].mean()
            errors.append(float(score.detach().cpu().item()))

    return np.asarray(errors, dtype=np.float32)
