from pathlib import Path

import numpy as np
import torch
from sklearn.preprocessing import normalize
from tqdm import trange

from data_loaders.manifest import filter_entries, get_split_entries, load_manifest


class ObjectDataset(torch.utils.data.Dataset):
    def __init__(self, data_cfg):
        self.num_primitives = data_cfg["num_primitives"]
        self.num_points = data_cfg["num_points"]
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
        self.dataset_root = dataset_root

        self.pc_list, self.label_list = self.load_data()

    def __len__(self):
        return len(self.pc_list)

    def __getitem__(self, idx):
        perm = torch.randperm(self.num_points)
        permidx = perm[: self.num_sampled_points]
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

    def load_data(self):
        pc_list = [None] * len(self.samples)
        label_list = [None] * len(self.samples)

        for file_idx in trange(len(self.samples), desc="object manifest loading..."):
            sample = self.samples[file_idx]
            sample_path = self.dataset_root / sample["path"]
            data = np.load(sample_path, allow_pickle=True).item()

            assert data["partial_pc"].shape[0] == self.num_points

            pc_list[file_idx] = data["partial_pc"].transpose()
            segmentation_label = data["membership"]
            segmentation_label -= min(segmentation_label)
            assert max(segmentation_label) <= self.num_primitives

            segmentation_label_1hot = np.zeros((segmentation_label.shape[0], self.num_primitives))
            for point_idx in range(segmentation_label.shape[0]):
                segmentation_label_1hot[point_idx, segmentation_label[point_idx]] = 1
            label_list[file_idx] = segmentation_label_1hot

        return pc_list, label_list

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
