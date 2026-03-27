from __future__ import annotations

from pathlib import Path

import torch

from model_architecture import ECNNAutoencoderV3


class ModelLoaderService:
    def __init__(self, model_path: Path):
        self.model_path = model_path
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None

    def load(self):
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found at: {self.model_path}")

        model = ECNNAutoencoderV3(latent_dim=1024).to(self.device)
        ckpt = torch.load(str(self.model_path), map_location=self.device)
        state = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
        model.load_state_dict(state, strict=False)
        model.eval()

        self.model = model
        return self.model, self.device

    def get_model_and_device(self):
        if self.model is None:
            return self.load()
        return self.model, self.device
