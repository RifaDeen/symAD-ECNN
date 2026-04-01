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
    """Compute per-sample mean MSE reconstruction errors."""
    model.eval()
    errors: List[float] = []
    for x, y in tqdm(dataloader, desc=desc, leave=False):
        x = x.to(device)
        y = y.to(device)
        recon = model(x)
        mse = F.mse_loss(recon, y, reduction="none").view(x.size(0), -1).mean(dim=1)
        errors.extend(mse.detach().cpu().numpy().tolist())
    return np.asarray(errors, dtype=np.float32)
