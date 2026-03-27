from __future__ import annotations

import numpy as np
import torch
from scipy import ndimage

from domain_models import InferenceMaps


class InferenceService:
    def __init__(self, model, device):
        self.model = model
        self.device = device

    def compute_score_and_maps(self, x_128: np.ndarray) -> InferenceMaps:
        inp = torch.from_numpy(x_128).float().unsqueeze(0).unsqueeze(0).to(self.device)

        with torch.no_grad():
            recon = self.model(inp)

        mse_map = (recon - inp) ** 2
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
