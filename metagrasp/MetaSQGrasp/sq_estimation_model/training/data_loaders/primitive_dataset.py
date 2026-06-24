import os
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
from sklearn.preprocessing import normalize
from tqdm import trange

from data_loaders.manifest import filter_entries, get_split_entries, load_manifest


class PrimitiveDataset(torch.utils.data.Dataset):
    def __init__(self, data_cfg):
        self.num_points = data_cfg["num_points"]
        self.num_gt_points = data_cfg["num_gt_points"]
        self.num_sampled_points = data_cfg["num_sampled_points"]
        self.noise_augment = data_cfg.get("noise_augment", True)
        self.noise_std = data_cfg.get("noise_std", 0)

        dataset_root = Path(data_cfg["path"])
        manifest_path = dataset_root / data_cfg["manifest"]
        manifest, _, _ = load_manifest(manifest_path)

        split = data_cfg["split"]
        class_ids = data_cfg.get("class_ids")
        self.samples = filter_entries(
            get_split_entries(manifest, split),
            class_ids=class_ids,
        )
        max_samples = data_cfg.get("max_samples")
        if max_samples is not None:
            self.samples = self.samples[: int(max_samples)]
        self.dataset_root = dataset_root

        self.pc_list, self.pc_gt_list, self.z_gt_list, self.shape_info_list = self.load_data()

    def __len__(self):
        return len(self.pc_list)

    def __getitem__(self, idx):
        perm = torch.randperm(self.num_points)
        permidx = perm[: self.num_sampled_points]
        tempidx = torch.zeros(self.num_points)
        tempidx[permidx] = 1
        tempidx = tempidx.type(torch.bool)

        pc = self.pc_list[idx][:, tempidx]
        pc_gt = self.pc_gt_list[idx]
        shape_info = deepcopy(self.shape_info_list[idx])

        if self.noise_augment:
            pc = self.noise_augmentation(pc)

        pc, diagonal_len = self.normalize_pointcloud(pc)
        pc_gt = pc_gt / diagonal_len
        shape_info = self.normalize_shape_info(shape_info, diagonal_len)

        pc = torch.Tensor(pc)
        pc_gt = torch.Tensor(pc_gt)
        z_gt = torch.Tensor(self.z_gt_list[idx])
        return pc, pc_gt, z_gt, shape_info

    def load_data(self):
        pc_list = [None] * len(self.samples)
        pc_gt_list = [None] * len(self.samples)
        z_gt_list = [None] * len(self.samples)
        shape_info_list = [None] * len(self.samples)

        for file_idx in trange(len(self.samples), desc="primitive manifest loading..."):
            sample = self.samples[file_idx]
            sample_path = self.dataset_root / sample["path"]
            data = np.load(sample_path, allow_pickle=True).item()

            assert data["partial_pc"].shape[0] == self.num_points
            assert data["full_pc"].shape[0] == self.num_gt_points
            assert len(data["primitives"]) == 1

            pc_list[file_idx] = data["partial_pc"].transpose()
            pc_gt_list[file_idx] = data["full_pc"].transpose()

            shape_info = data["primitives"][0]
            z_gt_list[file_idx] = self.orientation_prior(sample["source_type"], shape_info["SE3"])
            shape_info_list[file_idx] = self.preprocess_shape_info(shape_info)

        return pc_list, pc_gt_list, z_gt_list, shape_info_list

    def noise_augmentation(self, pc):
        noise = np.random.uniform(-1, 1, size=pc.shape)
        noise = normalize(noise, axis=0, norm="l2")
        scale = np.random.normal(loc=0, scale=self.noise_std, size=(1, pc.shape[1]))
        scale = scale.repeat(pc.shape[0], axis=0)
        return pc + noise * scale

    def normalize_pointcloud(self, pc):
        max_xyz = np.max(pc, axis=1)
        min_xyz = np.min(pc, axis=1)
        diagonal_len = np.linalg.norm(max_xyz - min_xyz, axis=0)
        return pc / diagonal_len, diagonal_len

    def orientation_prior(self, source_type, pose):
        if source_type == "truncated_torus":
            return pose[:3, 1].tolist()
        return pose[:3, 2].tolist()

    def preprocess_shape_info(self, shape_info):
        shape_parameters = shape_info["parameters"]
        n_params = len(shape_parameters)
        shape_parameters_new = dict.fromkeys([f"param{i+1}" for i in range(max(n_params, 5))], 0)
        for idx, (_, value) in enumerate(shape_parameters.items()):
            shape_parameters_new[f"param{idx+1}"] = value
        shape_info["parameters"] = shape_parameters_new
        return shape_info

    def normalize_shape_info(self, shape_info, diagonal_len):
        shape_pose = shape_info["SE3"]
        shape_pose[:3, 3] = shape_pose[:3, 3] / diagonal_len

        shape_type = shape_info["type"]
        shape_parameters = shape_info["parameters"]
        dimensionless_keys = {
            "torus": {"param3"},
            "superquadric": {"param4", "param5"},
            "extended_sq": {"param4", "param5", "param6", "param8", "param9"},
            "deformable_superquadric": {"param4", "param5", "param6", "param8", "param9"},  # legacy alias
        }
        skip_keys = dimensionless_keys.get(shape_type, set())
        for idx, (key, value) in enumerate(shape_parameters.items()):
            if key in skip_keys:
                pass
            else:
                shape_parameters[key] = value / diagonal_len

        shape_info["parameters"] = shape_parameters
        return shape_info
