import torch.nn as nn
import torch
from .build import LOSS
from openpoints.utils.utils_3shape import  rotation_matrix_to_normal

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
        self.mse_loss = nn.MSELoss()  # Initialize the MSE loss function


    def forward(self, preds, gt, normals, ret_full = False):

        # T1 and T2 are 4x4 transformation matrices
        # Extract rotation parts (top-left 3x3 matrices)
        R_pred = preds[:, :3, :3]
        R_gt = gt[:, :3, :3]

        # Compute geodesic loss for rotation
        m = torch.bmm(R_pred, R_gt.transpose(1, 2))
        cos = (m[:, 0, 0] + m[:, 1, 1] + m[:, 2, 2] - 1) / 2
        theta = torch.acos(torch.clamp(cos, -1 + self.eps, 1 - self.eps))
        geodesic_loss = torch.mean(theta)

        # Extract translation parts (last column except the final element)
        trans_pred = preds[:, :3, 3]
        trans_gt = gt[:, :3, 3]

        # Gives us the projection of the vector from the predicted origin to the target point along the direction
        # of the normal vector. This is equivalent to the shortest distance from the target point
        # to the plane defined by the predicted origin and the normal vector.
        ##normals = rotation_matrix_to_normal(R_pred)
        #translation_loss = self.mse_loss(trans_gt, trans_pred )

        dot_product = torch.sum(normals * (trans_gt - trans_pred), dim=1)
        dot_sum = torch.abs(dot_product)
        translation_loss = torch.mean(dot_sum)
        #translation_loss = torch.mean(torch.norm(trans_gt - trans_pred, dim=1))
        #return translation_loss
        # Combine the rotation and translation loss
        total_loss = geodesic_loss + self.alpha * translation_loss
        if ret_full:
            return total_loss,  theta + self.alpha * dot_sum
        else:
            return geodesic_loss + self.alpha * translation_loss
