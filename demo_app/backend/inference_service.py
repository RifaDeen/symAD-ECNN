from __future__ import annotations

import numpy as np
import torch
from scipy import ndimage

from domain_models import InferenceMaps


class InferenceService:
    def __init__(self, model, device):
        self.model = model
        self.device = device

    def compute_score_and_maps(
        self, x_128: np.ndarray, use_brain_mask: bool = True, min_brain_pixels: int = 50
    ) -> InferenceMaps:
        """
        Compute reconstruction score and error maps.

        CRITICAL: use_brain_mask=True matches training/evaluation pipeline.
        The model was evaluated with brain-mask scoring, so inference must use the same method.

        Args:
            x_128: Input slice (128x128)
            use_brain_mask: If True, score only brain region (pixels > 0.01). MANDATORY for matching eval.
            min_brain_pixels: Skip slices with fewer brain pixels. Default 50 matches evaluation.
        """
        inp = torch.from_numpy(x_128).float().unsqueeze(0).unsqueeze(0).to(self.device)

        with torch.no_grad():
            recon = self.model(inp)

        mse_map = (recon - inp) ** 2

        # CRITICAL: Brain-mask scoring to match evaluation pipeline
        if use_brain_mask:
            # Create brain mask (same threshold as evaluation: 0.01)
            mask = (inp > 0.01).float()
            brain_pixels = mask.sum()

            if brain_pixels < min_brain_pixels:
                # Skip slices with too few brain pixels (matches evaluation behavior)
                score = 0.0
            else:
                # Score ONLY brain region (matches evaluation exactly)
                score = float((mse_map * mask).sum() / brain_pixels)
        else:
            # Legacy method: entire image (NOT recommended, doesn't match evaluation)
            score = float(mse_map.view(1, -1).mean().detach().cpu().item())

        recon_np = recon.detach().cpu().squeeze().numpy().astype(np.float32)
        err_abs = np.abs(x_128 - recon_np).astype(np.float32)
        err_smooth = ndimage.gaussian_filter(err_abs, sigma=2).astype(np.float32)

        return InferenceMaps(
            input_slice=x_128,
            reconstruction=recon_np,
            error_abs=err_abs,
            error_smooth=err_smooth,
            score=score,
        )


class RiskScoringService:
    def compute_risk_level(self, score: float, threshold: float, anomaly_mean: float) -> str:
        if score < threshold:
            return "LOW"
        if score < anomaly_mean:
            return "MEDIUM"
        if score < 1.5 * anomaly_mean:
            return "HIGH"
        return "VERY_HIGH"
