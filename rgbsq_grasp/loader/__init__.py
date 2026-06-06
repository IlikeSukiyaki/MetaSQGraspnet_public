import torch
from .primitive_dataset import PrimitiveDataset
from .object_dataset import ObjectDataset

def get_dataloader(data_cfg):

    # dataset
    dataset = get_dataset(data_cfg)

    # dataloader   
    loader = torch.utils.data.DataLoader(
        dataset, 
        batch_size = data_cfg["batch_size"], 
        num_workers = data_cfg["num_workers"], 
        shuffle = True)
    return loader

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
    except:
        raise (f"Dataset {name} not available")
