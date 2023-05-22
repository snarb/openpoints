import torch.nn as nn
import torch
from .build import LOSS

# matrices batch*3*3
# both matrix are orthogonal rotation matrices
# out theta between 0 to 180 degree batch
# Adapted version of geodesic loss from the https://arxiv.org/pdf/2202.12555.pdf
# to homogeneous transform case
@LOSS.register_module()
class HomogeneousGeodesicLoss(nn.Module):
    def __init__(self, alpha=1.0, eps=1e-7):
        super().__init__()
        self.eps = float(eps)
        self.alpha = alpha  # Weighting factor for translation loss

    def forward(self, T1, T2):
        # T1 and T2 are 4x4 transformation matrices
        # Extract rotation parts (top-left 3x3 matrices)
        R1 = T1[:, :3, :3]
        R2 = T2[:, :3, :3]

        # Compute geodesic loss for rotation
        m = torch.bmm(R1, R2.transpose(1, 2))
        cos = (m[:, 0, 0] + m[:, 1, 1] + m[:, 2, 2] - 1) / 2
        theta = torch.acos(torch.clamp(cos, -1 + self.eps, 1 - self.eps))
        geodesic_loss = torch.mean(theta)

        # Extract translation parts (last column except the final element)
        t1 = T1[:, :3, 3]
        t2 = T2[:, :3, 3]

        # Compute L2 loss for translation
        translation_loss = torch.mean(torch.norm(t1 - t2, dim=1))

        # Combine the rotation and translation loss
        return geodesic_loss + self.alpha * translation_loss
