"""
ECNN Model Loader for Evaluation.

This module provides functions to load the trained ECNN model checkpoint
and define the model architecture for inference during evaluation.

Since importing from the original training notebooks may be complex,
this module defines the ECNN architecture locally for evaluation purposes.

Purpose: Load ECNN model for threshold experiments and evaluation
"""

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from typing import Optional, Tuple, Dict
import sys

# Try to import e2cnn for equivariant layers
try:
    from e2cnn import gspaces
    from e2cnn import nn as e2nn
    E2CNN_AVAILABLE = True
except ImportError:
    E2CNN_AVAILABLE = False
    print("Warning: e2cnn not available. Install with: pip install e2cnn")


# =============================================================================
# ECNN ARCHITECTURE DEFINITION
# =============================================================================

class ECNNAutoencoder(nn.Module):
    """
    Equivariant CNN Autoencoder for brain MRI anomaly detection.
    
    This architecture uses E(2)-equivariant convolutional layers that are
    invariant to rotations, which is beneficial for brain MRI analysis
    where the orientation should not affect the anomaly detection.
    
    The model follows an encoder-decoder structure:
    - Encoder: Progressively downsamples the input while increasing channels
    - Decoder: Progressively upsamples to reconstruct the input
    - The reconstruction error serves as the anomaly score
    """
    
    def __init__(
        self,
        in_channels: int = 1,
        base_channels: int = 32,
        latent_dim: int = 256,
        n_rotations: int = 8,
        input_size: int = 128
    ):
        """
        Initialize the ECNN Autoencoder.
        
        Args:
            in_channels: Number of input channels (1 for grayscale MRI).
            base_channels: Base number of feature channels.
            latent_dim: Dimension of the latent space.
            n_rotations: Number of discrete rotations for equivariance.
            input_size: Expected input image size.
        """
        super().__init__()
        
        if not E2CNN_AVAILABLE:
            raise ImportError("e2cnn is required for ECNNAutoencoder")
        
        self.in_channels = in_channels
        self.base_channels = base_channels
        self.latent_dim = latent_dim
        self.n_rotations = n_rotations
        self.input_size = input_size
        
        # Define the rotation group
        self.r2_act = gspaces.Rot2dOnR2(N=n_rotations)
        
        # Input type: regular scalar field
        self.input_type = e2nn.FieldType(self.r2_act, in_channels * [self.r2_act.trivial_repr])
        
        # Build encoder
        self.encoder = self._build_encoder()
        
        # Build decoder
        self.decoder = self._build_decoder()
        
        # Output projection to get back to regular image
        self.output_type = e2nn.FieldType(self.r2_act, in_channels * [self.r2_act.trivial_repr])
    
    def _build_encoder(self):
        """Build the encoder network."""
        layers = []
        
        # Define channel progression
        channels = [self.base_channels, self.base_channels * 2, self.base_channels * 4, self.base_channels * 8]
        
        in_type = self.input_type
        
        for i, out_ch in enumerate(channels):
            out_type = e2nn.FieldType(self.r2_act, out_ch * [self.r2_act.regular_repr])
            
            layers.append(e2nn.R2Conv(in_type, out_type, kernel_size=3, padding=1, bias=False))
            layers.append(e2nn.InnerBatchNorm(out_type))
            layers.append(e2nn.ReLU(out_type, inplace=True))
            layers.append(e2nn.PointwiseMaxPool(out_type, kernel_size=2, stride=2))
            
            in_type = out_type
        
        self.encoder_output_type = out_type
        return e2nn.SequentialModule(*layers)
    
    def _build_decoder(self):
        """Build the decoder network."""
        layers = []
        
        # Reverse channel progression
        channels = [self.base_channels * 8, self.base_channels * 4, self.base_channels * 2, self.base_channels]
        
        in_type = self.encoder_output_type
        
        for i, out_ch in enumerate(channels[1:]):
            out_type = e2nn.FieldType(self.r2_act, out_ch * [self.r2_act.regular_repr])
            
            layers.append(e2nn.R2Upsampling(in_type, scale_factor=2))
            layers.append(e2nn.R2Conv(in_type, out_type, kernel_size=3, padding=1, bias=False))
            layers.append(e2nn.InnerBatchNorm(out_type))
            layers.append(e2nn.ReLU(out_type, inplace=True))
            
            in_type = out_type
        
        # Final layer back to input type
        layers.append(e2nn.R2Upsampling(in_type, scale_factor=2))
        layers.append(e2nn.R2Conv(in_type, self.output_type, kernel_size=3, padding=1, bias=True))
        
        return e2nn.SequentialModule(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the autoencoder.
        
        Args:
            x: Input tensor of shape (batch, channels, height, width).
            
        Returns:
            Reconstructed tensor of same shape as input.
        """
        # Wrap input as geometric tensor
        x = e2nn.GeometricTensor(x, self.input_type)
        
        # Encode
        z = self.encoder(x)
        
        # Decode
        out = self.decoder(z)
        
        # Extract tensor from geometric wrapper
        return out.tensor
    
    def compute_reconstruction_error(
        self,
        x: torch.Tensor,
        mode: str = "abs"
    ) -> torch.Tensor:
        """
        Compute reconstruction error map.
        
        Args:
            x: Input tensor.
            mode: Error mode - "abs" for absolute error, "squared" for MSE.
            
        Returns:
            Error map tensor.
        """
        recon = self.forward(x)
        
        if mode == "abs":
            error = torch.abs(x - recon)
        elif mode == "squared":
            error = (x - recon) ** 2
        else:
            raise ValueError(f"Unknown error mode: {mode}")
        
        return error


# =============================================================================
# SIMPLIFIED ECNN FOR FALLBACK
# =============================================================================

class SimplifiedECNN(nn.Module):
    """
    Simplified CNN Autoencoder as fallback when e2cnn is not available.
    
    This is a standard CNN autoencoder that can be used for evaluation
    when the e2cnn library is not installed.
    """
    
    def __init__(
        self,
        in_channels: int = 1,
        base_channels: int = 32,
        latent_dim: int = 256,
        input_size: int = 128
    ):
        super().__init__()
        
        self.in_channels = in_channels
        self.base_channels = base_channels
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, 3, padding=1),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            nn.Conv2d(base_channels, base_channels * 2, 3, padding=1),
            nn.BatchNorm2d(base_channels * 2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            nn.Conv2d(base_channels * 2, base_channels * 4, 3, padding=1),
            nn.BatchNorm2d(base_channels * 4),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            nn.Conv2d(base_channels * 4, base_channels * 8, 3, padding=1),
            nn.BatchNorm2d(base_channels * 8),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(base_channels * 8, base_channels * 4, 3, padding=1),
            nn.BatchNorm2d(base_channels * 4),
            nn.ReLU(inplace=True),
            
            nn.Upsample(scale_factor=2),
            nn.Conv2d(base_channels * 4, base_channels * 2, 3, padding=1),
            nn.BatchNorm2d(base_channels * 2),
            nn.ReLU(inplace=True),
            
            nn.Upsample(scale_factor=2),
            nn.Conv2d(base_channels * 2, base_channels, 3, padding=1),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),
            
            nn.Upsample(scale_factor=2),
            nn.Conv2d(base_channels, in_channels, 3, padding=1),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encoder(x)
        return self.decoder(z)
    
    def compute_reconstruction_error(
        self,
        x: torch.Tensor,
        mode: str = "abs"
    ) -> torch.Tensor:
        recon = self.forward(x)
        
        if mode == "abs":
            error = torch.abs(x - recon)
        elif mode == "squared":
            error = (x - recon) ** 2
        else:
            raise ValueError(f"Unknown error mode: {mode}")
        
        return error


# =============================================================================
# MODEL LOADING FUNCTIONS
# =============================================================================

def load_ecnn_model(
    checkpoint_path: Path,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    use_simplified: bool = False
) -> Tuple[nn.Module, Dict]:
    """
    Load the ECNN model from checkpoint.
    
    Args:
        checkpoint_path: Path to the model checkpoint (.pth file).
        device: Device to load model on.
        use_simplified: If True, use simplified CNN instead of ECNN.
        
    Returns:
        Tuple of (model, checkpoint_info dict).
    """
    checkpoint_path = Path(checkpoint_path)
    
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Extract model configuration from checkpoint if available
    config = checkpoint.get("config", {})
    in_channels = config.get("in_channels", 1)
    base_channels = config.get("base_channels", 32)
    latent_dim = config.get("latent_dim", 256)
    input_size = config.get("input_size", 128)
    
    # Create model
    if use_simplified or not E2CNN_AVAILABLE:
        print("Using simplified CNN autoencoder.")
        model = SimplifiedECNN(
            in_channels=in_channels,
            base_channels=base_channels,
            latent_dim=latent_dim,
            input_size=input_size
        )
    else:
        n_rotations = config.get("n_rotations", 8)
        model = ECNNAutoencoder(
            in_channels=in_channels,
            base_channels=base_channels,
            latent_dim=latent_dim,
            n_rotations=n_rotations,
            input_size=input_size
        )
    
    # Load state dict
    state_dict_key = "model_state_dict" if "model_state_dict" in checkpoint else "state_dict"
    if state_dict_key in checkpoint:
        try:
            model.load_state_dict(checkpoint[state_dict_key])
            print(f"Model weights loaded from: {checkpoint_path}")
        except RuntimeError as e:
            print(f"Warning: Could not load state dict directly: {e}")
            print("Attempting to load with strict=False...")
            model.load_state_dict(checkpoint[state_dict_key], strict=False)
    else:
        # Assume checkpoint is just the state dict
        try:
            model.load_state_dict(checkpoint)
            print(f"Model weights loaded from: {checkpoint_path}")
        except RuntimeError as e:
            print(f"Warning: Could not load weights: {e}")
    
    model = model.to(device)
    model.eval()
    
    # Extract checkpoint info
    checkpoint_info = {
        "path": str(checkpoint_path),
        "config": config,
        "epoch": checkpoint.get("epoch"),
        "best_loss": checkpoint.get("best_loss"),
        "device": device,
    }
    
    return model, checkpoint_info


def get_model_for_inference(
    checkpoint_path: Optional[Path] = None,
    device: str = None
) -> Tuple[nn.Module, Dict]:
    """
    Get ECNN model ready for inference.
    
    This is a convenience function that handles finding the checkpoint
    if not specified and loads the model.
    
    Args:
        checkpoint_path: Optional explicit path to checkpoint.
        device: Device to use (auto-detected if None).
        
    Returns:
        Tuple of (model, info dict).
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # If no path specified, try to find it
    if checkpoint_path is None:
        sys.path.insert(0, str(Path(__file__).parent.parent))
        from path_utils import find_ecnn_checkpoint, require_file
        
        path, candidates = find_ecnn_checkpoint()
        checkpoint_path = require_file(path, "ECNN checkpoint", candidates)
    
    return load_ecnn_model(checkpoint_path, device)


# =============================================================================
# INFERENCE UTILITIES
# =============================================================================

@torch.no_grad()
def compute_error_maps(
    model: nn.Module,
    images: torch.Tensor,
    error_mode: str = "abs"
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute reconstruction and error maps for a batch of images.
    
    Args:
        model: The ECNN model.
        images: Input tensor of shape (batch, channels, height, width).
        error_mode: "abs" for absolute error, "squared" for squared error.
        
    Returns:
        Tuple of (reconstructions, error_maps).
    """
    model.eval()
    
    reconstructions = model(images)
    
    if error_mode == "abs":
        error_maps = torch.abs(images - reconstructions)
    elif error_mode == "squared":
        error_maps = (images - reconstructions) ** 2
    else:
        raise ValueError(f"Unknown error mode: {error_mode}")
    
    return reconstructions, error_maps


@torch.no_grad()
def process_single_image(
    model: nn.Module,
    image: np.ndarray,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    error_mode: str = "abs"
) -> Dict[str, np.ndarray]:
    """
    Process a single image through the model.
    
    Args:
        model: The ECNN model.
        image: Input image as numpy array (H, W) or (1, H, W).
        device: Device to run inference on.
        error_mode: Error computation mode.
        
    Returns:
        Dictionary with 'input', 'reconstruction', 'error_map'.
    """
    model.eval()
    
    # Prepare input
    if image.ndim == 2:
        image = image[np.newaxis, np.newaxis, :, :]  # Add batch and channel dims
    elif image.ndim == 3:
        image = image[np.newaxis, :, :, :]  # Add batch dim
    
    # Convert to tensor
    x = torch.from_numpy(image).float().to(device)
    
    # Forward pass
    recon = model(x)
    
    if error_mode == "abs":
        error = torch.abs(x - recon)
    else:
        error = (x - recon) ** 2
    
    return {
        "input": x.squeeze().cpu().numpy(),
        "reconstruction": recon.squeeze().cpu().numpy(),
        "error_map": error.squeeze().cpu().numpy(),
    }


if __name__ == "__main__":
    print("ECNN Model Loader")
    print("=" * 40)
    print(f"e2cnn available: {E2CNN_AVAILABLE}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    # Test model creation
    if E2CNN_AVAILABLE:
        print("\nTesting ECNN model creation...")
        model = ECNNAutoencoder()
        print(f"Model created successfully.")
        
        # Test forward pass
        x = torch.randn(1, 1, 128, 128)
        with torch.no_grad():
            out = model(x)
        print(f"Input shape: {x.shape}")
        print(f"Output shape: {out.shape}")
    else:
        print("\nTesting simplified CNN model...")
        model = SimplifiedECNN()
        x = torch.randn(1, 1, 128, 128)
        with torch.no_grad():
            out = model(x)
        print(f"Input shape: {x.shape}")
        print(f"Output shape: {out.shape}")
