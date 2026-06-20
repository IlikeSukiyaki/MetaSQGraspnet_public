import torch

from data_loaders.object_dataset import ObjectDataset
from data_loaders.primitive_dataset import PrimitiveDataset


def get_dataloader(data_cfg):
    dataset = get_dataset(data_cfg)
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=data_cfg["batch_size"],
        num_workers=data_cfg["num_workers"],
        shuffle=True,
    )


def get_dataset(data_cfg):
    name = data_cfg["loader"]
    dataset_instance = get_dataset_instance(name)
    return dataset_instance(data_cfg)


def get_dataset_instance(name):
    try:
        return {
            "primitive": PrimitiveDataset,
            "object": ObjectDataset,
        }[name]
    except KeyError as exc:
        raise ValueError(f"Dataset {name} not available") from exc
