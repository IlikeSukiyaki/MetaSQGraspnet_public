import torch
import torch.nn as nn

from geometry.utils_torch import quaternions_to_rotation_matrices_torch


class SqEstiLoss(nn.Module):
    def __init__(self, **kargs):
        super().__init__()
        self.weight = kargs["weight"]

    def compute_terms(self, x, l_gt, output):
        """
        Args:
            x (b x 3 x n): ground-truth point cloud.
            l_gt (b x 3): orientation prior.
            output (b x 12): superquadric parameters.

        Returns:
            dict with loss_total, loss_dist, loss_z, loss_z_weighted (batch means).
        """
        position = output[:, :3]
        orientation = output[:, 3:7]
        rotation = quaternions_to_rotation_matrices_torch(orientation)
        parameters = output[:, 7:]

        x_position = x[:, :3, :]
        rotation_t = rotation.permute(0, 2, 1)
        x_transformed = -rotation_t @ position.unsqueeze(2) + rotation_t @ x_position

        loss_dist = torch.mean(self.sq_distance(x_transformed, parameters) ** 2, dim=1)
        l_output = rotation[:, :3, 2]
        loss_z = torch.norm(torch.cross(l_output, l_gt, dim=1), dim=1)

        loss_dist_mean = torch.mean(loss_dist)
        loss_z_mean = torch.mean(loss_z)
        loss_z_weighted = self.weight * loss_z_mean
        loss_total = loss_dist_mean + loss_z_weighted

        if torch.isnan(loss_total) or torch.isinf(loss_total):
            raise ValueError("The loss function is inf or nan.")

        return {
            "loss_total": loss_total,
            "loss_dist": loss_dist_mean,
            "loss_z": loss_z_mean,
            "loss_z_weighted": loss_z_weighted,
        }

    def forward(self, x, l_gt, output):
        return self.compute_terms(x, l_gt, output)["loss_total"]

    def sq_distance(self, x, parameters):
        a1 = parameters[:, 0:1]
        a2 = parameters[:, 1:2]
        a3 = parameters[:, 2:3]
        e1 = parameters[:, 3:4]
        e2 = parameters[:, 4:5]

        beta = (
            (torch.abs(x[:, 0, :] / a1) ** (2 / e2) + torch.abs(x[:, 1, :] / a2) ** (2 / e2))
            ** (e2 / e1)
            + torch.abs(x[:, 2, :] / a3) ** (2 / e1)
        ) ** (-e1 / 2)

        return torch.norm(x, dim=1) * (1 - beta)
