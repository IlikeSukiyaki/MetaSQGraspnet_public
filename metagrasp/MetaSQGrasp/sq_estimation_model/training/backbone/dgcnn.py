#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author: Yue Wang
@Contact: yuewangx@mit.edu
@File: model.py
@Time: 2018/10/13 6:35 PM
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

def knn(x, k):
    inner = -2 * torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x ** 2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)

    idx = pairwise_distance.topk(k=k, dim=-1)[1]  # (batch_size, num_points, k)
    return idx

def get_graph_feature(x, k=20, idx=None):
    batch_size = x.size(0)
    num_points = x.size(2)
    x = x.view(batch_size, -1, num_points)
    if idx is None:
        idx = knn(x, k=k)  # (batch_size, num_points, k)

    idx_base = (
        torch.arange(0, batch_size).to(x).to(torch.long).view(-1, 1, 1) * num_points
    )

    idx = idx + idx_base

    idx = idx.view(-1)

    _, num_dims, _ = x.size()

    x = x.transpose(
        2, 1
    ).contiguous()  # (batch_size, num_points, num_dims)  -> (batch_size*num_points, num_dims) #   batch_size * num_points * k + range(0, batch_size*num_points)
    feature = x.view(batch_size * num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims)
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)

    feature = torch.cat((feature - x, x), dim=3).permute(0, 3, 1, 2).contiguous()

    return feature

class Transform_Net(nn.Module):
    def __init__(self, args):
        super(Transform_Net, self).__init__()
        self.args = args
        self.k = 3

        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(128)
        self.bn3 = nn.BatchNorm1d(1024)

        self.conv1 = nn.Sequential(
            nn.Conv2d(6, 64, kernel_size=1, bias=False),
            self.bn1,
            nn.LeakyReLU(negative_slope=0.2),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=1, bias=False),
            self.bn2,
            nn.LeakyReLU(negative_slope=0.2),
        )
        self.conv3 = nn.Sequential(
            nn.Conv1d(128, 1024, kernel_size=1, bias=False),
            self.bn3,
            nn.LeakyReLU(negative_slope=0.2),
        )

        self.linear1 = nn.Linear(1024, 512, bias=False)
        self.bn3 = nn.BatchNorm1d(512)
        self.linear2 = nn.Linear(512, 256, bias=False)
        self.bn4 = nn.BatchNorm1d(256)

        self.transform = nn.Linear(256, 3 * 3)
        nn.init.constant_(self.transform.weight, 0)
        nn.init.eye_(self.transform.bias.view(3, 3))

    def forward(self, x):
        batch_size = x.size(0)

        x = self.conv1(x)
        x = self.conv2(x)
        x = x.max(dim=-1, keepdim=False)[0]

        x = self.conv3(x)
        x = x.max(dim=-1, keepdim=False)[0]

        x = F.leaky_relu(self.bn3(self.linear1(x)), negative_slope=0.2)
        x = F.leaky_relu(self.bn4(self.linear2(x)), negative_slope=0.2)

        x = self.transform(x)
        x = x.view(batch_size, 3, 3)

        return x

class GCM_Block(nn.Module):
    def __init__(
        self,
        k=10,
        input_dim=64,
        output_dim=64,
        leakyrelu_slope=0.01,
        use_batch_norm=False,
    ):
        super(GCM_Block, self).__init__()
        self.k = k
        if use_batch_norm:
            self.conv2d = nn.Sequential(
                nn.Conv2d(2 * input_dim, output_dim, kernel_size=1, bias=False),
                nn.BatchNorm2d(output_dim),
                nn.LeakyReLU(negative_slope=leakyrelu_slope),
            )
        else:
            self.conv2d = nn.Sequential(
                nn.Conv2d(2 * input_dim, output_dim, kernel_size=1, bias=False),
                nn.LeakyReLU(negative_slope=leakyrelu_slope),
            )

    def forward(self, x):
        x = get_graph_feature(x, k=self.k)
        x = self.conv2d(x)
        x = x.max(dim=-1, keepdim=False)[0]
        return x


class DGCNN(nn.Module):
    def __init__(self, **kargs):
        super(DGCNN, self).__init__()
        self.kargs = kargs
        self.k = kargs["k"]
        self.leakyrelu_slope = kargs["leakyrelu_slope"]
        self.l_hidden_local = kargs["l_hidden_local"]
        self.global_feature_dim = kargs["global_feature_dim"]
        self.input_dim = kargs["input_dim"]
        self.local_feature_dim = sum(self.l_hidden_local)
        self.fusion_feature_dim = self.local_feature_dim + self.global_feature_dim
        self.use_spatial_transform = kargs["use_spatial_transform"]
        self.use_mean_global_feature = kargs["use_mean_global_feature"]
        if "use_batch_norm" in kargs:
            self.use_batch_norm = kargs["use_batch_norm"]
        else:
            self.use_batch_norm = False

        # spatial transformation
        self.layers_spatial_transformation = Transform_Net(kargs)

        # local features
        block_layers = []
        prev_dim = self.input_dim
        for n_hidden in self.l_hidden_local:
            block_layers.append(
                GCM_Block(
                    k=self.k,
                    input_dim=prev_dim,
                    output_dim=n_hidden,
                    leakyrelu_slope=self.leakyrelu_slope,
                    use_batch_norm=self.use_batch_norm,
                )
            )
            prev_dim = n_hidden

        self.layers_local = nn.Sequential(*block_layers)

        # global features
        if self.use_batch_norm:
            self.layers_global = nn.Sequential(
                nn.Conv1d(
                    self.local_feature_dim,
                    self.global_feature_dim,
                    kernel_size=1,
                    bias=False,
                ),
                nn.BatchNorm1d(self.global_feature_dim),
                nn.LeakyReLU(negative_slope=self.leakyrelu_slope),
            )
        else:
            self.layers_global = nn.Sequential(
                nn.Conv1d(
                    self.local_feature_dim,
                    self.global_feature_dim,
                    kernel_size=1,
                    bias=False,
                ),
                nn.LeakyReLU(negative_slope=self.leakyrelu_slope),
            )

    def local_feature_map(self, x):
        if self.use_spatial_transform:
            x0 = get_graph_feature(x, k=self.k)
            t = self.layers_spatial_transformation(x0)
            x = x.transpose(2, 1)
            x = torch.bmm(x, t)
            x = x.transpose(2, 1)

        for index, layer in enumerate(self.layers_local):
            x = layer(x)
            if index == 0:
                x_cat = x
            else:
                x_cat = torch.cat((x_cat, x), dim=1)

        return x_cat

    def global_feature_map(self, x):
        batch_size = x.size(0)
        x_local = self.local_feature_map(x)
        x = self.layers_global(x_local)
        x_global1 = F.adaptive_max_pool1d(x, 1).view(batch_size, -1)

        if self.use_mean_global_feature is False:
            return x_global1
        elif self.use_mean_global_feature is True:
            x_global2 = F.adaptive_avg_pool1d(x, 1).view(batch_size, -1)
            return torch.cat((x_global1, x_global2), 1)
        else:
            raise ("Specify use_mean_global_feature: True or False")

    def local_global_feature_map(self, x):
        num_points = x.size(2)
        x_local = self.local_feature_map(x)
        x = self.layers_global(x_local)
        x_global = x.max(dim=-1, keepdim=False)[0]
        num_dims = x_global.shape[1]
        x_global = x_global.view(-1, num_dims, 1).repeat(1, 1, num_points)

        x_fusion = torch.cat((x_local, x_global), dim=1)

        return x_fusion
