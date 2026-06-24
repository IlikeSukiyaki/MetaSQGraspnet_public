"""ExtSQ estimation inference helpers for one partial point cloud."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional, Union

import numpy as np
import open3d as o3d
import torch

from geometry.pc_preprocess import normalize_numpy_pc, process_pc
from geometry.utils_numpy import define_SE3, quaternions_to_rotation_matrices
from inference import load_recognition_model

ArrayLike = Union[np.ndarray, list, tuple]
PathLike = Union[str, Path]

DEFAULT_CONFIG = "pretrained/extsq_esti_model/extsq_esti_model_config.yml"
DEFAULT_CHECKPOINT = "pretrained/extsq_esti_model/extsq_esti_model_best.pkl"
REQUIRED_PARAM_KEYS = ("a1", "a2", "a3", "e1", "e2", "k", "b", "cos_alpha", "sin_alpha")


def resolve_device(device: Union[str, int]) -> str:
    device_str = str(device)
    if device_str == "cpu" or not torch.cuda.is_available():
        return "cpu"
    if device_str.startswith("cuda"):
        return device_str
    return f"cuda:{device_str}"


def as_point_cloud_nx3(point_cloud: ArrayLike) -> np.ndarray:
    points = np.asarray(point_cloud, dtype=np.float64)
    if points.ndim != 2:
        raise ValueError(f"Expected a 2D point cloud, got shape {tuple(points.shape)}.")
    if points.shape[1] == 3:
        return points
    if points.shape[0] == 3:
        return points.T
    raise ValueError(f"Expected point cloud shape N x 3 or 3 x N, got {tuple(points.shape)}.")


def pred_to_extsq_parameters(
    pred_fit: np.ndarray, normalizer: float, segment_frame: np.ndarray
) -> tuple[np.ndarray, dict[str, float]]:
    pred_fit = np.asarray(pred_fit, dtype=np.float64).reshape(-1)
    if pred_fit.shape[0] < 16:
        raise ValueError(
            f"ExtSQ estimation expects 16 output parameters, got {pred_fit.shape[0]}."
        )

    se3 = define_SE3(
        np.squeeze(quaternions_to_rotation_matrices(np.expand_dims(pred_fit[3:7], axis=0))),
        normalizer * pred_fit[:3],
    )
    se3 = np.dot(segment_frame, se3)
    parameters = {
        "a1": float(pred_fit[7] * normalizer),
        "a2": float(pred_fit[8] * normalizer),
        "a3": float(pred_fit[9] * normalizer),
        "e1": float(pred_fit[10]),
        "e2": float(pred_fit[11]),
        "k": float(pred_fit[12]),
        "b": float(pred_fit[13]),
        "cos_alpha": float(pred_fit[14]),
        "sin_alpha": float(pred_fit[15]),
    }
    return se3, parameters


def load_sample(path: PathLike) -> tuple[np.ndarray, Optional[dict[str, Any]], Optional[dict[str, Any]]]:
    path = Path(path)
    data = np.load(path, allow_pickle=True)
    if getattr(data, "dtype", None) == object:
        payload = data.item()
        if isinstance(payload, dict):
            if "partial_pc" not in payload:
                raise KeyError(f"{path} dict must contain 'partial_pc'.")
            points = as_point_cloud_nx3(payload["partial_pc"])
            gt = payload["primitives"][0] if payload.get("primitives") else None
            return points, gt, payload.get("generation_metadata")
        return as_point_cloud_nx3(payload), None, None
    return as_point_cloud_nx3(data), None, None


@torch.no_grad()
def recognize_extsq(
    point_cloud: ArrayLike,
    model: torch.nn.Module,
    *,
    num_fitting_points: int = 300,
    parent_frame: Optional[np.ndarray] = None,
) -> dict[str, Any]:
    points = as_point_cloud_nx3(point_cloud)
    if points.shape[0] < 3:
        raise ValueError("Need at least 3 points to fit an ExtSQ primitive.")

    segment_pcd = o3d.geometry.PointCloud(points=o3d.utility.Vector3dVector(points))
    segment_pcd, segment_frame = process_pc(segment_pcd, num_processed_pc_points=num_fitting_points)
    if segment_pcd is None or segment_frame is None:
        raise ValueError("Could not compute a stable point-cloud frame for ExtSQ inference.")

    normalized_points, normalizer = normalize_numpy_pc(np.asarray(segment_pcd.points))
    device = next(model.parameters()).device
    inputs = torch.tensor(
        np.expand_dims(normalized_points.transpose(), axis=0),
        dtype=torch.float32,
        device=device,
    )
    pred_fit = np.squeeze(model(inputs).detach().cpu().numpy())
    se3, parameters = pred_to_extsq_parameters(pred_fit, normalizer, segment_frame)

    if parent_frame is not None:
        parent_frame = np.asarray(parent_frame, dtype=np.float64)
        if parent_frame.shape != (4, 4):
            raise ValueError(f"parent_frame must be 4 x 4, got {parent_frame.shape}.")
        se3 = np.dot(parent_frame, se3)

    return {
        "SE3": se3,
        "parameters": parameters,
        "pred_fit": pred_fit,
        "normalizer": float(normalizer),
        "segment_frame": segment_frame,
        "input_points": points,
        "processed_points": normalized_points,
    }


def load_extsq_model(
    config_path: PathLike = DEFAULT_CONFIG,
    checkpoint_path: PathLike = DEFAULT_CHECKPOINT,
    device: Union[str, int] = "cuda:0",
) -> tuple[torch.nn.Module, object]:
    resolved_checkpoint = Path(checkpoint_path).expanduser().resolve()
    return load_recognition_model(config_path, resolved_checkpoint, device=resolve_device(device))
