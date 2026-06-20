import torch
import torch.nn as nn
import torch.nn.functional as F


class ExtendedSqEstiModel(nn.Module):
    """Deformable superquadric estimation network (extended SQ estimation model)."""

    def __init__(self, backbone, **args):
        super().__init__()
        self.backbone = backbone
        backbone_out_channel = self.backbone.global_feature_dim
        dict_input_dim = {"input_dim": backbone_out_channel}

        self.net_position = MLP(**args["position"], **dict_input_dim)
        self.net_orientation = MLP(**args["orientation"], **dict_input_dim)
        self.net_size = MLP(**args["size"], **dict_input_dim)
        self.net_shape = MLP(**args["shape"], **dict_input_dim)
        self.net_taper = MLP(**args["taper"], **dict_input_dim)
        self.net_bend = MLP(**args["bending"], **dict_input_dim)
        self.net_bend_angle = MLP(**args["bending_angle"], **dict_input_dim)

    def forward(self, x):
        x = self.backbone.global_feature_map(x)
        sigmoid = nn.Sigmoid()

        x_pos = self.net_position(x)
        x_ori = F.normalize(self.net_orientation(x), p=2, dim=1)
        x_size = 0.5 * sigmoid(self.net_size(x)) + 0.03
        x_shape = 1.5 * sigmoid(self.net_shape(x)) + 0.2
        x_taper = 1.8 * sigmoid(self.net_taper(x)) - 0.9
        x_bend_k = 0.01 + 0.74 * sigmoid(self.net_bend(x))
        x_bend_a = F.normalize(self.net_bend_angle(x), p=2, dim=1)

        return torch.cat([x_pos, x_ori, x_size, x_shape, x_taper, x_bend_k, x_bend_a], dim=1)

    def train_step(self, x, y, optimizer, loss_function, clip_grad=1, x_gt=None, l_gt=None, **kwargs):
        optimizer.zero_grad()
        loss = loss_function(x_gt, l_gt, self(x))
        loss.backward(retain_graph=True)
        if clip_grad is not None:
            torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=clip_grad)
        optimizer.step()

        pc = x.detach().cpu().permute([0, 2, 1]).numpy()
        pc_gt = x_gt[:, :3, :].detach().cpu().permute([0, 2, 1]).numpy()
        mesh = self(x).detach().cpu().numpy()
        return {
            "loss": loss.item(),
            "pc": pc,
            "pc_gt": pc_gt,
            "mesh": mesh,
            "mesh_gt": y,
        }

    def validation_step(self, x, y, loss_function, x_gt=None, l_gt=None, **kwargs):
        loss = loss_function(x_gt, l_gt, self(x))
        pc = x.detach().cpu().permute([0, 2, 1]).numpy()
        pc_gt = x_gt[:, :3, :].detach().cpu().permute([0, 2, 1]).numpy()
        mesh = self(x).detach().cpu().numpy()
        return {
            "loss": loss.item(),
            "pc": pc,
            "pc_gt": pc_gt,
            "mesh": mesh,
            "mesh_gt": y,
        }


class MLP(nn.Module):
    def __init__(self, **args):
        super().__init__()
        self.l_hidden = args["l_hidden"]
        self.output_dim = args["output_dim"]
        self.input_dim = args["input_dim"]
        l_neurons = self.l_hidden + [self.output_dim]

        l_layer = []
        prev_dim = self.input_dim
        for i, n_hidden in enumerate(l_neurons):
            l_layer.append(nn.Linear(prev_dim, n_hidden))
            if i < len(l_neurons) - 1:
                l_layer.append(nn.LeakyReLU(0.2))
            prev_dim = n_hidden
        self.net = nn.Sequential(*l_layer)

    def forward(self, x):
        return self.net(x)
