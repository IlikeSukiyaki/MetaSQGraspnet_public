import torch
import torch.nn as nn
from functions.utils_torch import quaternions_to_rotation_matrices_torch

class SuperquadricLoss(nn.Module):
    def __init__(self, **kargs):
        super(SuperquadricLoss, self).__init__()
        self.weight = kargs['weight']

    def forward(self, x, l_gt, output):
        """
        Args:
            x (b x 3 x n): ground-truth point cloud.
            l_gt (b x 3): orientation prior.
            output (b x 12): deformable superquadric parameters.
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
            self.sq_distance(x_transformed, parameters)**2, 
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

    def sq_distance(self, x, parameters):
        
        # parameters
        a1 = parameters[:,0:1]
        a2 = parameters[:,1:2]
        a3 = parameters[:,2:3]
        e1 = parameters[:,3:4]
        e2 = parameters[:,4:5]

        # epsilon for numerical stability
        eps = 1e-4

        beta = (
            (
            torch.abs(x[:, 0, :]/a1)**(2/e2) 
            + torch.abs(x[:, 1, :]/a2)**(2/e2)
            )**(e2/e1) + torch.abs(x[:, 2, :]/a3)**(2/e1)
        )**(-e1/2)

        F = torch.norm(x, dim=1) * (1 - beta)

        return F 