import torch
import os 
from omegaconf import OmegaConf

from .dgcnn import DGCNN
from .sqnet import SuperquadricNetwork
from .dsqnet import DeformableSuperquadricNetwork
from .segmenation_network import SegmentationNetwork

def get_model(cfg, *args, **kwargs):
    cfg_model = OmegaConf.create(OmegaConf.to_container(cfg["model"], resolve=True))
    name = cfg_model["arch"]

    cfg_backbone = cfg_model.pop("backbone")
    backbone = get_backbone_instance(cfg_backbone["arch"])(**cfg_backbone)

    model = get_model_instance(name)
    return model(backbone, **cfg_model)

def get_model_instance(name):
    try:
        return {
            "sqnet": SuperquadricNetwork,
            "dsqnet": DeformableSuperquadricNetwork,
            "segnet": SegmentationNetwork,
        }[name]
    except KeyError as exc:
        raise ValueError("Model {} not available".format(name)) from exc

def get_backbone_instance(name):
    try:
        return {
            "dgcnn": DGCNN,
        }[name]
    except KeyError as exc:
        raise ValueError("backbone {} not available".format(name)) from exc

def safe_torch_load(path, map_location='cpu'):
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)

def load_pretrained(identifier, config_file, ckpt_file, root='pretrained', **kwargs):
    """
    load pre-trained model.
    identifier: '<model name>/<run name>'. e.g. 'ae_mnist/z16'
    config_file: name of a config file. e.g. 'ae.yml'
    ckpt_file: name of a model checkpoint file. e.g. 'model_best.pth'
    root: path to pretrained directory
    """
    config_path = os.path.join(root, identifier, config_file)
    ckpt_path = os.path.join(root, identifier, ckpt_file)
    missing = [path for path in (config_path, ckpt_path) if not os.path.exists(path)]
    if missing:
        missing_list = "\n".join(f"  - {path}" for path in missing)
        raise FileNotFoundError(
            "Missing pretrained DSQNet asset(s):\n"
            f"{missing_list}\n"
            "Download/copy the pretrained files into the expected pretrained/ layout."
        )
    cfg = OmegaConf.load(config_path)
    if "model" in cfg:
        model_name = cfg["model"]['arch']
    else:
        model_name = cfg['arch']
    
    model = get_model(cfg)
    ckpt = safe_torch_load(ckpt_path, map_location='cpu')
    if 'model_state' in ckpt:
        ckpt = ckpt['model_state']
    model.load_state_dict(ckpt)

    return model, cfg