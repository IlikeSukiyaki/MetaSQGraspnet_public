"""PointNet backbone for SQ estimation.

Adapted from the MIT-licensed fxia22/pointnet.pytorch implementation vendored
next to this repository. The wrapper exposes the same API as the existing
DGCNN backbone: ``global_feature_dim`` and ``global_feature_map(x)``.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def _norm1d(num_features, use_batch_norm):
    if use_batch_norm:
        return nn.BatchNorm1d(num_features)
    return nn.Identity()


class InputTransform(nn.Module):
    def __init__(self, input_dim=3, use_batch_norm=True):
        super().__init__()
        self.input_dim = input_dim
        self.conv1 = nn.Conv1d(input_dim, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, input_dim * input_dim)

        self.bn1 = _norm1d(64, use_batch_norm)
        self.bn2 = _norm1d(128, use_batch_norm)
        self.bn3 = _norm1d(1024, use_batch_norm)
        self.bn4 = _norm1d(512, use_batch_norm)
        self.bn5 = _norm1d(256, use_batch_norm)

        nn.init.zeros_(self.fc3.weight)
        nn.init.zeros_(self.fc3.bias)

    def forward(self, x):
        batch_size = x.size(0)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = torch.max(x, dim=2, keepdim=False)[0]

        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        identity = torch.eye(self.input_dim, dtype=x.dtype, device=x.device).view(1, -1)
        x = x + identity.repeat(batch_size, 1)
        return x.view(batch_size, self.input_dim, self.input_dim)


class FeatureTransform(nn.Module):
    def __init__(self, feature_dim=64, use_batch_norm=True):
        super().__init__()
        self.feature_dim = feature_dim
        self.conv1 = nn.Conv1d(feature_dim, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, feature_dim * feature_dim)

        self.bn1 = _norm1d(64, use_batch_norm)
        self.bn2 = _norm1d(128, use_batch_norm)
        self.bn3 = _norm1d(1024, use_batch_norm)
        self.bn4 = _norm1d(512, use_batch_norm)
        self.bn5 = _norm1d(256, use_batch_norm)

        nn.init.zeros_(self.fc3.weight)
        nn.init.zeros_(self.fc3.bias)

    def forward(self, x):
        batch_size = x.size(0)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = torch.max(x, dim=2, keepdim=False)[0]

        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        identity = torch.eye(self.feature_dim, dtype=x.dtype, device=x.device).view(1, -1)
        x = x + identity.repeat(batch_size, 1)
        return x.view(batch_size, self.feature_dim, self.feature_dim)


class PointNetBackbone(nn.Module):
    """PointNet global feature encoder for 3D point clouds."""

    def __init__(self, **kwargs):
        super().__init__()
        self.input_dim = kwargs.get("input_dim", 3)
        self.global_feature_dim = kwargs.get("global_feature_dim", 1024)
        self.use_input_transform = kwargs.get("use_input_transform", True)
        self.use_feature_transform = kwargs.get("use_feature_transform", False)
        self.use_batch_norm = kwargs.get("use_batch_norm", True)

        if self.input_dim != 3:
            raise ValueError("PointNetBackbone currently expects input_dim: 3.")
        if self.global_feature_dim != 1024:
            raise ValueError("PointNetBackbone currently uses a fixed 1024-d global feature.")

        self.input_transform = InputTransform(
            input_dim=self.input_dim,
            use_batch_norm=self.use_batch_norm,
        )
        self.conv1 = nn.Conv1d(self.input_dim, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, self.global_feature_dim, 1)
        self.bn1 = _norm1d(64, self.use_batch_norm)
        self.bn2 = _norm1d(128, self.use_batch_norm)
        self.bn3 = _norm1d(self.global_feature_dim, self.use_batch_norm)

        if self.use_feature_transform:
            self.feature_transform = FeatureTransform(
                feature_dim=64,
                use_batch_norm=self.use_batch_norm,
            )
        else:
            self.feature_transform = None

        self.last_input_transform = None
        self.last_feature_transform = None

    def global_feature_map(self, x):
        if self.use_input_transform:
            trans = self.input_transform(x)
            x = torch.bmm(x.transpose(2, 1), trans).transpose(2, 1)
            self.last_input_transform = trans
        else:
            self.last_input_transform = None

        x = F.relu(self.bn1(self.conv1(x)))
        if self.feature_transform is not None:
            trans_feat = self.feature_transform(x)
            x = torch.bmm(x.transpose(2, 1), trans_feat).transpose(2, 1)
            self.last_feature_transform = trans_feat
        else:
            self.last_feature_transform = None

        x = F.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        x = torch.max(x, dim=2, keepdim=False)[0]
        return x
