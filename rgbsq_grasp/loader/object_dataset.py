import os
from tqdm import trange

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import normalize

class ObjectDataset(torch.utils.data.Dataset):
    def __init__(self, data_cfg):
        self.num_primitives = data_cfg["num_primitives"]
        self.num_points = data_cfg["num_points"]
        self.num_sampled_points = data_cfg['num_sampled_points']
        self.noise_augment = data_cfg.get("noise_augment", True)
        self.noise_std = data_cfg.get("noise_std", 0)

        # load data name list
        csv_path = os.path.join(data_cfg["path"], data_cfg["csv_name"])
        csv_file = pd.read_csv(csv_path, delimiter=",", header=None)
        self.data_namelist = list(csv_file[0])

        self.pc_list, self.label_list = self.load_data(data_cfg)

    def __len__(self):
        return len(self.pc_list)

    def __getitem__(self, idx):

        # random idx
        perm = torch.randperm(self.num_points)
        permidx = perm[:self.num_sampled_points]
        tempidx = torch.zeros(self.num_points)
        tempidx[permidx] = 1
        tempidx = tempidx.type(torch.bool)
        pc = self.pc_list[idx][:, tempidx]

        if self.noise_augment:
            pc = self.noise_augmentation(pc)

        pc, diagonal_len = self.normalize_pointcloud(pc)
        pc = torch.Tensor(pc)
        label = torch.Tensor(self.label_list[idx][tempidx])

        return pc, label, diagonal_len

    def load_data(self, data_cfg):
        pc_list = [None] * len(self.data_namelist)
        label_list = [None] * len(self.data_namelist)
        for file_idx in trange(
            len(self.data_namelist), desc=f'{data_cfg["csv_name"]} loading...'
        ):
            file = self.data_namelist[file_idx]
            file_prefix = "_".join(file.split("_")[:-3])

            # data load
            data = np.load(
                os.path.join(data_cfg["path"], file_prefix, file), allow_pickle=True
            ).item()

            assert (
                data["partial_pc"].shape[0] == self.num_points
            ), f"number of points should be {self.num_points}, but is {data['partial_pc'].shape[0]}."

            # point cloud data
            pc = data["partial_pc"].transpose()
            pc_list[file_idx] = pc

            # segmenation label
            segmentation_label = data["membership"]
            segmentation_label -= min(segmentation_label)

            assert (
                max(segmentation_label) <= self.num_primitives
            ), "too many primitives, more then expected."

            # one hot encode segmentation label
            segmentation_label_1hot = np.zeros(
                (segmentation_label.shape[0], self.num_primitives)
            )
            for point_idx in range(segmentation_label.shape[0]):
                segmentation_label_1hot[point_idx, segmentation_label[point_idx]] = 1

            label_list[file_idx] = segmentation_label_1hot

        return pc_list, label_list

    def noise_augmentation(self, pc):
        noise = np.random.uniform(-1, 1, size=pc.shape)
        noise = normalize(noise, axis=0, norm="l2")

        scale = np.random.normal(loc=0, scale=self.noise_std, size=(1, pc.shape[1]))
        scale = scale.repeat(pc.shape[0], axis=0)

        pc = pc + noise * scale

        return pc

    def normalize_pointcloud(self, pc):
        max_xyz = np.max(pc, axis=1)
        min_xyz = np.min(pc, axis=1)
        diagonal_len = np.linalg.norm(max_xyz - min_xyz, axis=0)
        pc = pc / diagonal_len

        return pc, diagonal_len
