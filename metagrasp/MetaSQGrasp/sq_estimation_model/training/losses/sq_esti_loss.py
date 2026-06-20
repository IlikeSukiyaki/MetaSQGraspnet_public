import torch
import torch.nn as nn

from geometry.utils_torch import quaternions_to_rotation_matrices_torch


class SqEstiLoss(nn.Module):
    def __init__(self, **kargs):
        super().__init__()
        self.weight = kargs["weight"]

    def forward(self, x, l_gt, output):
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

        loss = torch.mean(loss_dist) + self.weight * torch.mean(loss_z)
        if torch.isnan(loss) or torch.isinf(loss):
            raise ValueError("The loss function is inf or nan.")
        return loss

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
