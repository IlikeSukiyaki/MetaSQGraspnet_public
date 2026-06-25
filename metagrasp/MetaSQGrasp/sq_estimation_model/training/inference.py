from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np
import torch
from omegaconf import OmegaConf

from registry.models import get_model, safe_torch_load

ArrayLike = Union[np.ndarray, torch.Tensor]


def resolve_device(device: Union[str, int, torch.device]) -> str:
    device_str = str(device)
    if device_str == "cpu" or not torch.cuda.is_available():
        return "cpu"
    if device_str.startswith("cuda"):
        return device_str
    return f"cuda:{device_str}"


def load_recognition_model(
    config_path: Union[str, Path],
    checkpoint_path: Union[str, Path],
    device: Union[str, int, torch.device] = "cuda:0",
) -> Tuple[torch.nn.Module, object]:
    device = resolve_device(device)

    cfg = OmegaConf.load(config_path)
    model = get_model(cfg)
    ckpt = safe_torch_load(str(checkpoint_path), map_location="cpu")
    state_dict = ckpt.get("model_state", ckpt)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model, cfg


def prepare_point_cloud(
    point_cloud: ArrayLike,
    *,
    num_points: Optional[int] = None,
    normalize: bool = True,
    device: Union[str, torch.device] = "cuda:0",
) -> torch.Tensor:
    if isinstance(point_cloud, torch.Tensor):
        tensor = point_cloud.detach().clone().float()
    else:
        tensor = torch.as_tensor(point_cloud, dtype=torch.float32)

    if tensor.ndim != 2:
        raise ValueError(f"Expected a 2D point cloud, got shape {tuple(tensor.shape)}.")
    if tensor.shape[1] == 3:
        tensor = tensor.transpose(0, 1)
    elif tensor.shape[0] != 3:
        raise ValueError(f"Expected point cloud shape N x 3 or 3 x N, got {tuple(tensor.shape)}.")

    if num_points is not None and tensor.shape[1] != num_points:
        if tensor.shape[1] < num_points:
            repeats = int(np.ceil(num_points / tensor.shape[1]))
            tensor = tensor.repeat(1, repeats)[:, :num_points]
        else:
            tensor = tensor[:, torch.randperm(tensor.shape[1])[:num_points]]

    if normalize:
        diagonal = torch.linalg.norm(tensor.max(dim=1).values - tensor.min(dim=1).values)
        if diagonal > 0:
            tensor = tensor / diagonal

    if str(device).startswith("cuda") and not torch.cuda.is_available():
        device = "cpu"
    return tensor.unsqueeze(0).to(device)


@torch.no_grad()
def predict_parameters(
    model: torch.nn.Module,
    point_cloud: ArrayLike,
    *,
    num_points: Optional[int] = 300,
    normalize: bool = True,
) -> np.ndarray:
    device = next(model.parameters()).device
    inputs = prepare_point_cloud(
        point_cloud, num_points=num_points, normalize=normalize, device=device
    )
    return model(inputs).squeeze(0).detach().cpu().numpy()
