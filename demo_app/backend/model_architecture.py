from __future__ import annotations

import torch
import torch.nn as nn

from e2cnn import gspaces
from e2cnn import nn as e2nn


class ECNNAutoencoderV3(nn.Module):
    def __init__(self, latent_dim: int = 1024):
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
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
