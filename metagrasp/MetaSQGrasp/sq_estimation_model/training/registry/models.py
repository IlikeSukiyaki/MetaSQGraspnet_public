import os

import torch
from omegaconf import OmegaConf

from backbone.dgcnn import DGCNN
from extsq_esti_model.network import ExtSqEstiModel
from sq_esti_model.network import SqEstiModel


def get_model(cfg, *args, **kwargs):
    cfg_model = OmegaConf.create(OmegaConf.to_container(cfg["model"], resolve=True))
    name = cfg_model["arch"]

    cfg_backbone = cfg_model.pop("backbone")
    backbone = get_backbone_instance(cfg_backbone["arch"])(**cfg_backbone)

    model_cls = get_model_instance(name)
    return model_cls(backbone, **cfg_model)


def get_model_instance(name):
    try:
        return {
            "sq_esti_model": SqEstiModel,
            "extsq_esti_model": ExtSqEstiModel,
        }[name]
    except KeyError as exc:
        raise ValueError(f"Model {name} not available") from exc


def get_backbone_instance(name):
    try:
        return {
            "dgcnn": DGCNN,
        }[name]
    except KeyError as exc:
        raise ValueError(f"backbone {name} not available") from exc


def safe_torch_load(path, map_location="cpu"):
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def load_pretrained(identifier, config_file, ckpt_file, root="pretrained", **kwargs):
    config_path = os.path.join(root, identifier, config_file)
    ckpt_path = os.path.join(root, identifier, ckpt_file)
    missing = [path for path in (config_path, ckpt_path) if not os.path.exists(path)]
    if missing:
        missing_list = "\n".join(f"  - {path}" for path in missing)
        raise FileNotFoundError(
            "Missing pretrained SQ estimation asset(s):\n"
            f"{missing_list}\n"
            "Copy trained checkpoints into the expected pretrained/ layout."
        )

    cfg = OmegaConf.load(config_path)
    model = get_model(cfg)
    ckpt = safe_torch_load(ckpt_path, map_location="cpu")
    if "model_state" in ckpt:
        ckpt = ckpt["model_state"]
    model.load_state_dict(ckpt)
    return model, cfg
