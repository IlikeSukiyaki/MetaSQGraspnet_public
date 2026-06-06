import torch
import torch.nn as nn
from functions.utils_torch import quaternions_to_rotation_matrices_torch

class DeformableSuperquadricLoss(nn.Module):
    def __init__(self, **kargs):
        super(DeformableSuperquadricLoss, self).__init__()
        self.weight = kargs["weight"]

    def forward(self, x, l_gt, output):
        """
        Args:
            x (b x 3 x n): ground-truth point cloud.
            l_gt (b x 3): orientation prior.
            output (b x 16): deformable superquadric parameters.
                             (network output)

        Returns:
            loss (float): Gross and Boult loss
        """

        # network output processing
        position = output[:, :3]
        orientation = output[:, 3:7]
        rotation = quaternions_to_rotation_matrices_torch(orientation)
        parameters = output[:, 7:]

        # ground-truth point cloud
        x_position = x[:, :3, :]
        rotation_t = rotation.permute(0,2,1)
        x_transformed = -rotation_t@position.unsqueeze(2) + rotation_t@x_position

        # Gross and Boult superquadric loss
        loss_dist = torch.mean(
            self.dsq_distance(x_transformed, parameters)**2, 
            dim=1
        )

        # orientation loss
        l_output = rotation[:, :3, 2]
        loss_z = torch.norm(torch.cross(l_output, l_gt, dim=1), dim=1)

        # total loss
        loss = torch.mean(loss_dist) + self.weight * torch.mean(loss_z)
        if torch.isnan(loss) or torch.isinf(loss):
            raise ValueError('The loss function is inf or nan.')

        return loss  

    def dsq_distance(self, x, parameters):
        
        # parameter decomposition
        a1 = parameters[:,0:1]
        a2 = parameters[:,1:2]
        a3 = parameters[:,2:3]
        e1 = parameters[:,3:4]
        e2 = parameters[:,4:5]
        k = parameters[:,5:6]
        b = parameters[:,6:7]
        cos_alpha = parameters[:,7:8]
        sin_alpha = parameters[:,8:9]
        b = b / torch.max(a1, a2)

        # epsilon for numerical stability
        eps = 1e-4

        # inverse deformation
        beta = torch.atan2(x[:, 1, :], x[:, 0, :])
        R = (cos_alpha * torch.cos(beta) + sin_alpha * torch.sin(beta)) * (x[:, 0, :] ** 2 + x[:, 1, :] ** 2) ** (1/2)
        r = 1 / b  - (x[:, 2, :] ** 2 + (1 / b - R) ** 2) ** (1/2)
        gamma = torch.atan2(x[:, 2, :], 1 / b - R)
        X = x[:, 0, :] - cos_alpha * (R - r)
        Y = x[:, 1, :] - sin_alpha * (R - r)
        Z = gamma * 1 / b
        f = k / a3 * Z + 1 

        beta = (
            (
            torch.abs(X/(torch.abs(a1 * f) + eps))**(2/e2) 
            + torch.abs(Y/(torch.abs(a2 * f) + eps))**(2/e2)
            )**(e2/e1) + torch.abs(Z/a3)**(2/e1)
        + eps)**(-e1/2)

        F = torch.norm(x, dim=1) * (1 - beta)

        return F 