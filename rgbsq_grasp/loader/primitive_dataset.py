import os
from tqdm import trange

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import normalize

class PrimitiveDataset(torch.utils.data.Dataset): 

    def __init__(self, data_cfg):
        self.object_types = data_cfg['object_types']
        self.num_points = data_cfg['num_points']
        self.num_gt_points = data_cfg['num_gt_points']
        self.num_sampled_points = data_cfg['num_sampled_points']
        self.noise_augment = data_cfg.get("noise_augment", True)
        self.noise_std = data_cfg.get("noise_std", 0)               
        
        # load data name list
        csv_path = os.path.join(data_cfg["path"], data_cfg["csv_name"])
        csv_file = pd.read_csv(csv_path, delimiter=',', header=None)
        self.data_namelist = list(csv_file[0])

        # only load interested objects and top-down view point exclusion
        self.data_namelist = [file for file in self.data_namelist if (
                                ("_".join(file.split("_")[:-3]) in self.object_types) and 
                                (file.split("_")[-1].split(".")[0] != '00') and
                                (file.split("_")[-1].split(".")[0] != '15')
                            )
                        ]

        self.pc_list, self.pc_gt_list, self.z_gt_list, self.shape_info_list = self.load_data(data_cfg)

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
        pc_gt = self.pc_gt_list[idx]
        shape_info = self.shape_info_list[idx]

        if self.noise_augment:
            pc = self.noise_augmentation(pc)

        # normalization
        pc, diagonal_len = self.normalize_pointcloud(pc)
        pc_gt = pc_gt / diagonal_len
        shape_info = self.normalize_shape_info(shape_info, diagonal_len)

        pc = torch.Tensor(pc)
        pc_gt = torch.Tensor(pc_gt)
        z_gt = torch.Tensor(self.z_gt_list[idx])
        shape_info = self.shape_info_list[idx]
        return pc, pc_gt, z_gt, shape_info

    def load_data(self, data_cfg):

        pc_list = [None] * len(self.data_namelist)
        pc_gt_list = [None] * len(self.data_namelist)
        z_gt_list = [None] * len(self.data_namelist)
        shape_info_list = [None] * len(self.data_namelist)
        for file_idx in trange(
            len(self.data_namelist), desc=f'{data_cfg["csv_name"]} loading...'
        ):
            # data name processing
            file = self.data_namelist[file_idx]
            file_prefix = "_".join(file.split("_")[:-3])

            # data load
            data = np.load(
                os.path.join(data_cfg["path"], file_prefix, file), allow_pickle=True
            ).item()

            assert ( 
                data["partial_pc"].shape[0] == self.num_points
            ), f"number of points should be {self.num_points}, but is {data['partial_pc'].shape[0]}."
            assert ( 
                data["full_pc"].shape[0] == self.num_gt_points
            ), f"number of points should be {self.num_gt_points}, but is {data['full_pc'].shape[0]}."
            assert (
                len(data["primitives"]) == 1
            ), "number of primitives should be one, but it is more than one."

            # point cloud data
            pc = data["partial_pc"].transpose()
            pc_list[file_idx] = pc

            # ground truth point cloud data
            pc_gt = data["full_pc"].transpose()
            pc_gt_list[file_idx] = pc_gt

            # load ground-truth shape
            shape_info = data["primitives"][0]
            shape_pose = shape_info["SE3"]

            # ground-truth z axis data
            z_gt = self.orientation_prior(file_prefix, shape_pose)
            z_gt_list[file_idx] = z_gt

            # ground-truth shape info
            shape_info_ = self.preprocess_shape_info(shape_info)
            shape_info_list[file_idx] = shape_info_

        return pc_list, pc_gt_list, z_gt_list, shape_info_list

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

    def orientation_prior(self, shape_type, pose):
        if shape_type == "truncated_torus":
            z = pose[:3,1].tolist()
        else:
            z = pose[:3,2].tolist()
        return z

    def preprocess_shape_info(self, shape_info):
        shape_parameters = shape_info["parameters"]
        shape_parameters_new = dict.fromkeys([f"param{i+1}" for i in range(5)], 0)
        for idx, (_, value) in enumerate(shape_parameters.items()):
            shape_parameters_new[f"param{idx+1}"] = value
        shape_info["parameters"] = shape_parameters_new

        return shape_info

    def normalize_shape_info(self, shape_info, diagonal_len):
        # position normalize
        shape_pose = shape_info["SE3"]
        shape_pose[:3, 3] = shape_pose[:3, 3] / diagonal_len

        # parameter normalize
        shape_type = shape_info["type"]
        shape_parameters = shape_info["parameters"]
        for idx, (key, value) in enumerate(shape_parameters.items()):
            if shape_type == "torus" and key == "param3":
                pass
            elif shape_type == "superquadric" and (key == "param4" or key == "param5"):
                pass
            else:
                shape_parameters[key] = value / diagonal_len

        shape_info["parameters"] = shape_parameters

        return shape_info