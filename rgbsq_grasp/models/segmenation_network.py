import torch
import torch.nn as nn

class SegmentationNetwork(nn.Module):
    def __init__(self, backbone, **args):
        super(SegmentationNetwork, self).__init__()
        self.backbone = backbone

        # backbone_out_channel = self.backbone.local_global_feature_map(
        #     torch.rand(1, self.backbone.input_dim, 100)
        # ).shape[1]
        backbone_out_channel = self.backbone.fusion_feature_dim
        l_hidden = list(args["l_hidden"])
        in_channels = [backbone_out_channel] + l_hidden
        out_channels = l_hidden + [args["num_primitives"]]

        l_layer = []
        for in_channel, out_channel in zip(in_channels, out_channels):
            block = [
                nn.Conv1d(in_channel, out_channel, kernel_size=1, bias=False),
                nn.LeakyReLU(0.2),
            ]
            l_layer.extend(block)
        l_layer.append(nn.Softmax(dim=1))

        self.pointwise_mlp = nn.Sequential(*l_layer)

    def forward(self, x):
        x = self.backbone.local_global_feature_map(x)
        x = self.pointwise_mlp(x)
        return x.transpose(1, 2)

    def train_step(self, x, y, optimizer, loss_function, clip_grad=None, **kwargs):
        optimizer.zero_grad()

        pred = self(x)

        loss = loss_function(y, pred)
        loss.backward()

        if clip_grad is not None:
            torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=clip_grad)

        optimizer.step()

        # input point cloud
        pc = x.detach().cpu().permute([0, 2, 1]).numpy()

        # ground truth segmentation label
        gt = y.detach().cpu().numpy()

        # predicted segmentation label
        pred = pred.detach().cpu().numpy()

        return {
            "loss": loss.item(),
            "pc": pc,
            "gt": gt,
            "pred": pred,
        }

    def validation_step(self, x, y, loss_function, **kwargs):
        pred = self(x)

        loss = loss_function(y, pred)

        # input point cloud
        pc = x.detach().cpu().permute([0, 2, 1]).numpy()

        # ground truth segmentation label
        gt = y.detach().cpu().numpy()

        # predicted segmentation label
        pred = pred.detach().cpu().numpy()

        return {
            "loss": loss.item(),
            "pc": pc,
            "gt": gt,
            "pred": pred,
        }
