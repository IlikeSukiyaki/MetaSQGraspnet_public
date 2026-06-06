"""SQNet-only superquadric shape recognition for a single partial point cloud segment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Optional, Union

import numpy as np
import open3d as o3d
import torch

from functions.pc_preprocess import normalize_numpy_pc, process_pc
from functions.primitives import Superquadric
from functions.utils_numpy import define_SE3, quaternions_to_rotation_matrices
from inference import load_recognition_model

ArrayLike = Union[np.ndarray, list, tuple]
PathLike = Union[str, Path]

DEFAULT_CONFIG = "pretrained/sqnet_config/example/sqnet_config.yml"
DEFAULT_CHECKPOINT = "pretrained/sqnet_config/example/model_best.pkl"


def resolve_device(device: Union[str, int]) -> str:
    """Accept cuda:0, cpu, or a bare GPU index like 0 (same as evaluation.py)."""
    device_str = str(device)
    if device_str == "cpu" or not torch.cuda.is_available():
        return "cpu"
    if device_str.startswith("cuda"):
        return device_str
    return f"cuda:{device_str}"


def pred_to_parameters(pred_fit: np.ndarray, normalizer: float, segment_frame: np.ndarray):
    """Decode SQNet output into world-frame SE3 and superquadric parameters."""
    pred_fit = np.asarray(pred_fit, dtype=np.float64).reshape(-1)
    SE3 = define_SE3(
        np.squeeze(quaternions_to_rotation_matrices(np.expand_dims(pred_fit[3:7], axis=0))),
        normalizer * pred_fit[:3],
    )
    SE3 = np.dot(segment_frame, SE3)

    parameters: Dict[str, float] = {
        "a1": float(pred_fit[7] * normalizer),
        "a2": float(pred_fit[8] * normalizer),
        "a3": float(pred_fit[9] * normalizer),
        "e1": float(pred_fit[10]),
        "e2": float(pred_fit[11]),
    }
    if len(pred_fit) > 12:
        parameters["k"] = float(pred_fit[12])
        parameters["b"] = float(pred_fit[13])
        parameters["cos_alpha"] = float(pred_fit[14])
        parameters["sin_alpha"] = float(pred_fit[15])

    return SE3, parameters


def _as_point_cloud_nx3(point_cloud: ArrayLike) -> np.ndarray:
    points = np.asarray(point_cloud, dtype=np.float64)
    if points.ndim != 2:
        raise ValueError(f"Expected a 2D point cloud, got shape {tuple(points.shape)}.")
    if points.shape[1] == 3:
        return points
    if points.shape[0] == 3:
        return points.T
    raise ValueError(f"Expected point cloud shape N x 3 or 3 x N, got {tuple(points.shape)}.")


def load_point_cloud(path: PathLike) -> np.ndarray:
    """Load an N x 3 point cloud from a .npy file (raw array or dict with partial_pc)."""
    data = np.load(path, allow_pickle=True)
    if getattr(data, "dtype", None) == object:
        payload = data.item()
        if isinstance(payload, dict):
            if "partial_pc" not in payload:
                raise KeyError(f"{path} dict must contain 'partial_pc'.")
            return _as_point_cloud_nx3(payload["partial_pc"])
        return _as_point_cloud_nx3(payload)
    return _as_point_cloud_nx3(data)


def load_evaluation_segment(
    object_name: str,
    index: int,
    *,
    viewpoint: Optional[int] = None,
    segment_id: Optional[int] = None,
    data_root: PathLike = "datasets/evaluation_dataset",
) -> tuple[np.ndarray, str]:
    """Load one partial segment (or whole cloud) from the evaluation dataset."""
    data_root = Path(data_root)
    file_number = f"{index + 90:04}"
    if viewpoint is None:
        for vp in range(1, 15):
            candidate = data_root / object_name / f"{object_name}_{file_number}_viewpoint_{vp:02}.npy"
            if candidate.exists():
                viewpoint = vp
                break
        if viewpoint is None:
            raise FileNotFoundError(
                f"No evaluation sample found for object={object_name!r}, index={index} under {data_root}."
            )

    sample_path = data_root / object_name / f"{object_name}_{file_number}_viewpoint_{viewpoint:02}.npy"
    if not sample_path.exists():
        raise FileNotFoundError(sample_path)

    data = np.load(sample_path, allow_pickle=True).item()
    points = _as_point_cloud_nx3(data["partial_pc"])
    if segment_id is not None:
        membership = np.asarray(data["membership"], dtype=np.int64)
        mask = membership == segment_id
        if not np.any(mask):
            raise ValueError(
                f"segment_id={segment_id} not found in membership for {sample_path.name} "
                f"(valid ids: {np.unique(membership).tolist()})."
            )
        points = points[mask]

    label = sample_path.name
    if segment_id is not None:
        label = f"{label}#segment_{segment_id}"
    return points, label


@torch.no_grad()
def recognize_superquadric(
    point_cloud: ArrayLike,
    model: torch.nn.Module,
    *,
    num_fitting_points: int = 300,
    parent_frame: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """
    Fit one superquadric to a single partial point cloud segment with SQNet.

    The input segment should already be the primitive patch you want to fit
    (for example, one SegNet segment or a manually cropped partial view).
    """
    points = _as_point_cloud_nx3(point_cloud)
    if points.shape[0] < 3:
        raise ValueError("Need at least 3 points to fit a superquadric.")

    segment_pcd = o3d.geometry.PointCloud(points=o3d.utility.Vector3dVector(points))
    segment_pcd, segment_frame = process_pc(segment_pcd, num_processed_pc_points=num_fitting_points)

    normalized_points, normalizer = normalize_numpy_pc(np.asarray(segment_pcd.points))
    device = next(model.parameters()).device
    inputs = torch.tensor(
        np.expand_dims(normalized_points.transpose(), axis=0),
        dtype=torch.float32,
        device=device,
    )
    pred_fit = np.squeeze(model(inputs).detach().cpu().numpy())
    sq_SE3, sq_parameters = pred_to_parameters(pred_fit, normalizer, segment_frame)

    if parent_frame is not None:
        parent_frame = np.asarray(parent_frame, dtype=np.float64)
        if parent_frame.shape != (4, 4):
            raise ValueError(f"parent_frame must be 4 x 4, got {parent_frame.shape}.")
        sq_SE3 = np.dot(parent_frame, sq_SE3)

    return {
        "SE3": sq_SE3,
        "parameters": sq_parameters,
        "pred_fit": pred_fit,
        "normalizer": float(normalizer),
        "segment_frame": segment_frame,
        "input_points": points,
        "processed_points": normalized_points,
    }


def _result_to_jsonable(result: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "SE3": np.asarray(result["SE3"]).tolist(),
        "parameters": result["parameters"],
        "normalizer": result["normalizer"],
        "segment_frame": np.asarray(result["segment_frame"]).tolist(),
        "pred_fit": np.asarray(result["pred_fit"]).tolist(),
        "num_input_points": int(np.asarray(result["input_points"]).shape[0]),
        "num_processed_points": int(np.asarray(result["processed_points"]).shape[0]),
    }


def visualize_result(result: Dict[str, Any], title: str = "SQNet superquadric fit") -> None:
    input_pcd = o3d.geometry.PointCloud(
        points=o3d.utility.Vector3dVector(np.asarray(result["input_points"]))
    )
    input_pcd.paint_uniform_color([0.55, 0.55, 0.55])

    fit = Superquadric(result["SE3"], result["parameters"], color=[0.9, 0.35, 0.2])
    fit.mesh.compute_vertex_normals()

    wire = o3d.geometry.LineSet.create_from_triangle_mesh(fit.mesh)
    wire.paint_uniform_color([0.1, 0.45, 0.9])

    o3d.visualization.draw_geometries(
        [input_pcd, fit.mesh, wire],
        window_name=title,
        width=1280,
        height=720,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run SQNet on one partial point cloud segment (no SegNet)."
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--input",
        type=str,
        help="Path to .npy file containing N x 3 points or a dict with partial_pc.",
    )
    source.add_argument(
        "--object",
        type=str,
        help="Evaluation dataset object name (e.g. box, cylinder). Use with --index.",
    )

    parser.add_argument("--index", type=int, default=0, help="Evaluation sample index (0-9).")
    parser.add_argument(
        "--viewpoint",
        type=int,
        default=None,
        help="Evaluation viewpoint id (1-14). Defaults to the first existing sample.",
    )
    parser.add_argument(
        "--segment-id",
        type=int,
        default=None,
        help="Optional GT segment id from evaluation membership (extract one primitive patch).",
    )
    parser.add_argument("--config", type=str, default=DEFAULT_CONFIG, help="SQNet config YAML.")
    parser.add_argument("--checkpoint", type=str, default=DEFAULT_CHECKPOINT, help="SQNet checkpoint.")
    parser.add_argument(
        "--device",
        default=0,
        help="Torch device: GPU index (e.g. 0), cuda:0, or cpu.",
    )
    parser.add_argument(
        "--num-fitting-points",
        type=int,
        default=300,
        help="Resampled point count passed to SQNet (matches evaluation per segment).",
    )
    parser.add_argument("--output", type=str, default=None, help="Optional JSON path for predicted parameters.")
    parser.add_argument("--visualize", action="store_true", help="Open an Open3D viewer.")
    parser.add_argument("--no-visualize", action="store_true", help="Disable visualization.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.input:
        points = load_point_cloud(args.input)
        label = Path(args.input).name
    else:
        if args.object is None:
            raise SystemExit("--object is required when --input is not provided.")
        points, label = load_evaluation_segment(
            args.object,
            args.index,
            viewpoint=args.viewpoint,
            segment_id=args.segment_id,
        )

    device = resolve_device(args.device)
    if str(args.device) != "cpu" and device == "cpu":
        print("CUDA unavailable; falling back to CPU.")

    model, _ = load_recognition_model(args.config, args.checkpoint, device=device)
    result = recognize_superquadric(
        points,
        model,
        num_fitting_points=args.num_fitting_points,
    )

    print(f"Sample: {label}")
    print(f"Input points: {result['input_points'].shape[0]}")
    print("Predicted superquadric parameters:")
    for key, value in result["parameters"].items():
        print(f"  {key}: {value:.6f}")

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as handle:
            json.dump(_result_to_jsonable(result), handle, indent=2)
        print(f"Wrote {output_path}")

    if args.visualize and not args.no_visualize:
        visualize_result(result, title=f"SQNet fit: {label}")


if __name__ == "__main__":
    main()
