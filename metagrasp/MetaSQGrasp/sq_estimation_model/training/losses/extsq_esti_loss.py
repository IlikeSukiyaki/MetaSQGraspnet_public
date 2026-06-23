import torch
import torch.nn as nn

from geometry.utils_torch import quaternions_to_rotation_matrices_torch


class ExtSqEstiLoss(nn.Module):
    def __init__(self, **kargs):
        super().__init__()
        self.weight = kargs["weight"]

    def compute_terms(self, x, l_gt, output):
        """
        Args:
            x (b x 3 x n): ground-truth point cloud.
            l_gt (b x 3): orientation prior.
            output (b x 16): extended superquadric parameters.

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

        loss_dist = torch.mean(self.extsq_distance(x_transformed, parameters) ** 2, dim=1)
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

    def extsq_distance(self, x, parameters):
        a1 = parameters[:, 0:1]
        a2 = parameters[:, 1:2]
        a3 = parameters[:, 2:3]
        e1 = parameters[:, 3:4]
        e2 = parameters[:, 4:5]
        k = parameters[:, 5:6]
        b = parameters[:, 6:7]
        cos_alpha = parameters[:, 7:8]
        sin_alpha = parameters[:, 8:9]
        b = b / torch.max(a1, a2)

        eps = 1e-4

        beta = torch.atan2(x[:, 1, :], x[:, 0, :])
        r = (cos_alpha * torch.cos(beta) + sin_alpha * torch.sin(beta)) * (
            x[:, 0, :] ** 2 + x[:, 1, :] ** 2
        ) ** (1 / 2)
        r_val = 1 / b - (x[:, 2, :] ** 2 + (1 / b - r) ** 2) ** (1 / 2)
        gamma = torch.atan2(x[:, 2, :], 1 / b - r)
        x_val = x[:, 0, :] - cos_alpha * (r - r_val)
        y_val = x[:, 1, :] - sin_alpha * (r - r_val)
        z_val = gamma * 1 / b
        f = k / a3 * z_val + 1

        beta_sq = (
            (
                torch.abs(x_val / (torch.abs(a1 * f) + eps)) ** (2 / e2)
                + torch.abs(y_val / (torch.abs(a2 * f) + eps)) ** (2 / e2)
            )
            ** (e2 / e1)
            + torch.abs(z_val / a3) ** (2 / e1)
            + eps
        ) ** (-e1 / 2)

        return torch.norm(x, dim=1) * (1 - beta_sq)
