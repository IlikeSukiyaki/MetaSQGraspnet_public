"""DSQNet-only deformable superquadric fitting for a single partial point cloud segment."""

from __future__ import annotations

import argparse
import json
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import open3d as o3d
import torch

from functions.pc_preprocess import normalize_numpy_pc, process_pc
from functions.primitives import DeformableSuperquadric
from functions.utils_numpy import define_SE3, quaternions_to_rotation_matrices
from inference import load_recognition_model

ArrayLike = Union[np.ndarray, list, tuple]
PathLike = Union[str, Path]

DEFAULT_CONFIG = (
    "pretrained/dsqnet_config/example/dsqnet_extended_sq_config.yml"
)
DEFAULT_CHECKPOINT = (
    "train_results_tensorboard/dsqnet_extended_sq_config/extended_sq_tb/model_best.pkl"
)

REQUIRED_PARAM_KEYS = ("a1", "a2", "a3", "e1", "e2", "k", "b", "cos_alpha", "sin_alpha")


def resolve_device(device: Union[str, int]) -> str:
    device_str = str(device)
    if device_str == "cpu" or not torch.cuda.is_available():
        return "cpu"
    if device_str.startswith("cuda"):
        return device_str
    return f"cuda:{device_str}"


def _as_point_cloud_nx3(point_cloud: ArrayLike) -> np.ndarray:
    points = np.asarray(point_cloud, dtype=np.float64)
    if points.ndim != 2:
        raise ValueError(f"Expected a 2D point cloud, got shape {tuple(points.shape)}.")
    if points.shape[1] == 3:
        return points
    if points.shape[0] == 3:
        return points.T
    raise ValueError(f"Expected point cloud shape N x 3 or 3 x N, got {tuple(points.shape)}.")


def pred_to_parameters(
    pred_fit: np.ndarray, normalizer: float, segment_frame: np.ndarray
) -> Tuple[np.ndarray, Dict[str, float]]:
    """Decode DSQNet output (16-d) into world-frame SE3 and deformable SQ parameters."""
    pred_fit = np.asarray(pred_fit, dtype=np.float64).reshape(-1)
    if pred_fit.shape[0] < 16:
        raise ValueError(
            f"DSQNet expects 16 output parameters, got {pred_fit.shape[0]}. "
            "Check that the checkpoint is a DSQNet model."
        )

    SE3 = define_SE3(
        np.squeeze(quaternions_to_rotation_matrices(np.expand_dims(pred_fit[3:7], axis=0))),
        normalizer * pred_fit[:3],
    )
    SE3 = np.dot(segment_frame, SE3)

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
    return SE3, parameters


def load_sample(path: PathLike) -> Tuple[np.ndarray, Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
    """
    Load partial_pc from a .npy file.

    Returns:
        points (N x 3), optional ground-truth primitive dict, optional generation_metadata.
    """
    path = Path(path)
    data = np.load(path, allow_pickle=True)
    if getattr(data, "dtype", None) == object:
        payload = data.item()
        if isinstance(payload, dict):
            if "partial_pc" not in payload:
                raise KeyError(f"{path} dict must contain 'partial_pc'.")
            points = _as_point_cloud_nx3(payload["partial_pc"])
            gt = None
            if payload.get("primitives") and len(payload["primitives"]) == 1:
                gt = payload["primitives"][0]
            metadata = payload.get("generation_metadata")
            return points, gt, metadata
        return _as_point_cloud_nx3(payload), None, None
    return _as_point_cloud_nx3(data), None, None


def load_evaluation_segment(
    object_name: str,
    index: int,
    *,
    viewpoint: Optional[int] = None,
    segment_id: Optional[int] = None,
    data_root: PathLike = "datasets/evaluation_dataset",
) -> Tuple[np.ndarray, str, Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
    """Load one partial segment from the evaluation dataset."""
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
    gt = data["primitives"][0] if len(data.get("primitives", [])) == 1 else None

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
    return points, label, gt, None


@torch.no_grad()
def recognize_deformable_superquadric(
    point_cloud: ArrayLike,
    model: torch.nn.Module,
    *,
    num_fitting_points: int = 300,
    parent_frame: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """Fit one deformable superquadric to a partial point cloud with DSQNet."""
    points = _as_point_cloud_nx3(point_cloud)
    if points.shape[0] < 3:
        raise ValueError("Need at least 3 points to fit a deformable superquadric.")

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
    dsq_SE3, dsq_parameters = pred_to_parameters(pred_fit, normalizer, segment_frame)

    if parent_frame is not None:
        parent_frame = np.asarray(parent_frame, dtype=np.float64)
        if parent_frame.shape != (4, 4):
            raise ValueError(f"parent_frame must be 4 x 4, got {parent_frame.shape}.")
        dsq_SE3 = np.dot(parent_frame, dsq_SE3)

    return {
        "SE3": dsq_SE3,
        "parameters": dsq_parameters,
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


# Publication-friendly palette (RGB in [0, 1]).
PAPER_COLORS = {
    "partial_pc": [0.42, 0.45, 0.50],
    "prediction": [0.86, 0.37, 0.24],
    "prediction_wire": [0.55, 0.18, 0.10],
    "ground_truth": [0.23, 0.45, 0.72],
    "ground_truth_wire": [0.12, 0.28, 0.48],
}


def _mesh_from_primitive(primitive: Dict[str, Any], color: list[float]) -> o3d.geometry.TriangleMesh:
    mesh = DeformableSuperquadric(primitive["SE3"], primitive["parameters"]).mesh
    mesh.paint_uniform_color(color)
    mesh.compute_vertex_normals()
    return mesh


def _point_cloud_from_mesh(
    mesh: o3d.geometry.TriangleMesh,
    *,
    color: list[float],
    num_points: int,
) -> o3d.geometry.PointCloud:
    pcd = mesh.sample_points_poisson_disk(number_of_points=num_points, init_factor=5)
    pcd.paint_uniform_color(color)
    return pcd


def _surface_from_primitive(
    primitive: Dict[str, Any],
    color: list[float],
    *,
    surface_style: str = "point_cloud",
    num_points: int = 3500,
) -> o3d.geometry.Geometry:
    mesh = DeformableSuperquadric(primitive["SE3"], primitive["parameters"]).mesh
    if surface_style == "mesh":
        mesh.paint_uniform_color(color)
        mesh.compute_vertex_normals()
        return mesh
    if surface_style == "point_cloud":
        return _point_cloud_from_mesh(mesh, color=color, num_points=num_points)
    raise ValueError(f"Unknown surface_style: {surface_style}")


def _wireframe_from_mesh(
    mesh: o3d.geometry.TriangleMesh,
    color: list[float],
) -> o3d.geometry.LineSet:
    wire = o3d.geometry.LineSet.create_from_triangle_mesh(deepcopy(mesh))
    wire.paint_uniform_color(color)
    return wire


def _point_cloud_for_display(
    points: np.ndarray,
    *,
    color: list[float],
    max_points: int = 2500,
) -> o3d.geometry.PointCloud:
    points = np.asarray(points, dtype=np.float64)
    if points.shape[0] > max_points:
        rng = np.random.default_rng(0)
        indices = rng.choice(points.shape[0], size=max_points, replace=False)
        points = points[indices]

    pcd = o3d.geometry.PointCloud(points=o3d.utility.Vector3dVector(points))
    pcd.paint_uniform_color(color)
    return pcd


def _combined_extent(geometries: List[o3d.geometry.Geometry]) -> np.ndarray:
    min_bound = np.min([g.get_min_bound() for g in geometries], axis=0)
    max_bound = np.max([g.get_max_bound() for g in geometries], axis=0)
    return max_bound - min_bound


def _offset_geometry(geometry: o3d.geometry.Geometry, translation: np.ndarray) -> o3d.geometry.Geometry:
    moved = deepcopy(geometry)
    moved.translate(np.asarray(translation, dtype=np.float64), relative=True)
    return moved


def _build_overlay_scene(
    result: Dict[str, Any],
    *,
    gt_primitive: Optional[Dict[str, Any]] = None,
    surface_style: str = "point_cloud",
    fit_surface_points: int = 3500,
) -> List[o3d.geometry.Geometry]:
    """Partial point cloud with predicted surface; optional GT surface in the same style."""
    geometries: List[o3d.geometry.Geometry] = [
        _point_cloud_for_display(result["input_points"], color=PAPER_COLORS["partial_pc"]),
    ]

    pred_surface = _surface_from_primitive(
        {"SE3": result["SE3"], "parameters": result["parameters"]},
        PAPER_COLORS["prediction"],
        surface_style=surface_style,
        num_points=fit_surface_points,
    )
    geometries.append(pred_surface)

    if gt_primitive is not None:
        if surface_style == "mesh":
            gt_mesh = _mesh_from_primitive(gt_primitive, color=PAPER_COLORS["ground_truth"])
            geometries.append(_wireframe_from_mesh(gt_mesh, PAPER_COLORS["ground_truth_wire"]))
        else:
            geometries.append(
                _surface_from_primitive(
                    gt_primitive,
                    PAPER_COLORS["ground_truth"],
                    surface_style="point_cloud",
                    num_points=fit_surface_points,
                )
            )

    return geometries


def _build_side_by_side_scene(
    result: Dict[str, Any],
    *,
    gt_primitive: Optional[Dict[str, Any]] = None,
    surface_style: str = "point_cloud",
    fit_surface_points: int = 3500,
) -> List[o3d.geometry.Geometry]:
    """Input + prediction on the right, optional GT surface on the left."""
    pred_surface = _surface_from_primitive(
        {"SE3": result["SE3"], "parameters": result["parameters"]},
        PAPER_COLORS["prediction"],
        surface_style=surface_style,
        num_points=fit_surface_points,
    )
    input_pcd = _point_cloud_for_display(result["input_points"], color=PAPER_COLORS["partial_pc"])
    right_group: List[o3d.geometry.Geometry] = [input_pcd, pred_surface]

    if gt_primitive is None:
        return right_group

    gt_surface = _surface_from_primitive(
        gt_primitive,
        PAPER_COLORS["ground_truth"],
        surface_style=surface_style,
        num_points=fit_surface_points,
    )
    extent = _combined_extent([gt_surface, pred_surface, input_pcd])
    gap = 1.35 * extent[0]
    left_shift = np.array([-gap / 2, 0.0, 0.0])
    right_shift = np.array([gap / 2, 0.0, 0.0])

    geometries = [_offset_geometry(gt_surface, left_shift)]
    geometries.extend(_offset_geometry(item, right_shift) for item in right_group)
    return geometries


def _build_pred_only_scene(
    result: Dict[str, Any],
    *,
    surface_style: str = "point_cloud",
    fit_surface_points: int = 3500,
) -> List[o3d.geometry.Geometry]:
    pred_surface = _surface_from_primitive(
        {"SE3": result["SE3"], "parameters": result["parameters"]},
        PAPER_COLORS["prediction"],
        surface_style=surface_style,
        num_points=fit_surface_points,
    )
    return [
        _point_cloud_for_display(result["input_points"], color=PAPER_COLORS["partial_pc"]),
        pred_surface,
    ]


def _configure_render_options(
    vis: o3d.visualization.Visualizer,
    *,
    surface_style: str = "point_cloud",
) -> None:
    opt = vis.get_render_option()
    opt.background_color = np.asarray([1.0, 1.0, 1.0])
    opt.point_size = 2.8 if surface_style == "point_cloud" else 2.2
    opt.line_width = 1.0
    opt.show_coordinate_frame = False
    opt.mesh_show_back_face = True
    opt.light_on = surface_style == "mesh"
    if surface_style == "mesh":
        opt.mesh_color_option = o3d.visualization.MeshColorOption.Color
        if hasattr(o3d.visualization.MeshShadeOption, "Default"):
            opt.mesh_shade_option = o3d.visualization.MeshShadeOption.Default


def _open3d_show(
    geometries: List[o3d.geometry.Geometry],
    *,
    title: str,
    width: int = 1600,
    height: int = 1000,
    save_image: Optional[Path] = None,
    surface_style: str = "point_cloud",
) -> None:
    vis = o3d.visualization.Visualizer()
    vis.create_window(
        window_name=title,
        width=width,
        height=height,
        visible=save_image is None,
    )
    for geometry in geometries:
        vis.add_geometry(geometry)

    _configure_render_options(vis, surface_style=surface_style)
    vis.reset_view_point(True)

    if save_image is not None:
        save_image.parent.mkdir(parents=True, exist_ok=True)
        vis.poll_events()
        vis.update_renderer()
        vis.capture_screen_image(str(save_image), do_render=True)
        print(f"Saved figure to {save_image}")
    else:
        vis.run()
    vis.destroy_window()


def visualize_result(
    result: Dict[str, Any],
    *,
    title: str = "DSQNet deformable superquadric fit",
    gt_primitive: Optional[Dict[str, Any]] = None,
    viz_mode: str = "overlay",
    surface_style: str = "point_cloud",
    fit_surface_points: int = 3500,
    save_image: Optional[Path] = None,
) -> None:
    if viz_mode == "overlay":
        geometries = _build_overlay_scene(
            result,
            gt_primitive=gt_primitive,
            surface_style=surface_style,
            fit_surface_points=fit_surface_points,
        )
    elif viz_mode == "side_by_side":
        geometries = _build_side_by_side_scene(
            result,
            gt_primitive=gt_primitive,
            surface_style=surface_style,
            fit_surface_points=fit_surface_points,
        )
    elif viz_mode == "pred_only":
        geometries = _build_pred_only_scene(
            result,
            surface_style=surface_style,
            fit_surface_points=fit_surface_points,
        )
    else:
        raise ValueError(f"Unknown viz_mode: {viz_mode}")

    _open3d_show(
        geometries,
        title=title,
        save_image=save_image,
        surface_style=surface_style,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run DSQNet on one partial point cloud segment (no SegNet)."
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
    parser.add_argument("--config", type=str, default=DEFAULT_CONFIG, help="DSQNet config YAML.")
    parser.add_argument(
        "--checkpoint", type=str, default=DEFAULT_CHECKPOINT, help="DSQNet checkpoint."
    )
    parser.add_argument(
        "--device",
        default=0,
        help="Torch device: GPU index (e.g. 0), cuda:0, or cpu.",
    )
    parser.add_argument(
        "--num-fitting-points",
        type=int,
        default=300,
        help="Resampled point count passed to DSQNet (matches training).",
    )
    parser.add_argument("--output", type=str, default=None, help="Optional JSON path for predictions.")
    parser.add_argument(
        "--show-gt",
        action="store_true",
        help="Overlay ground-truth mesh from .npy primitives (when available).",
    )
    parser.add_argument("--visualize", action="store_true", help="Open an Open3D viewer.")
    parser.add_argument("--no-visualize", action="store_true", help="Disable visualization.")
    parser.add_argument(
        "--viz-mode",
        type=str,
        default="overlay",
        choices=["overlay", "side_by_side", "pred_only"],
        help=(
            "Visualization style: overlay (input + pred + optional GT), "
            "side_by_side (GT left, input+pred right), or pred_only."
        ),
    )
    parser.add_argument(
        "--surface-style",
        type=str,
        default="point_cloud",
        choices=["point_cloud", "mesh"],
        help="Render fitted superquadrics as a sampled point cloud (default) or triangle mesh.",
    )
    parser.add_argument(
        "--fit-surface-points",
        type=int,
        default=3500,
        help="Number of Poisson-disk surface points when --surface-style point_cloud.",
    )
    parser.add_argument(
        "--save-image",
        type=str,
        default=None,
        help="Save a publication-ready PNG (1600x1000, white background) without opening the viewer.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    gt_primitive = None
    metadata = None

    if args.input:
        points, gt_primitive, metadata = load_sample(args.input)
        label = Path(args.input).name
    else:
        if args.object is None:
            raise SystemExit("--object is required when --input is not provided.")
        points, label, gt_primitive, metadata = load_evaluation_segment(
            args.object,
            args.index,
            viewpoint=args.viewpoint,
            segment_id=args.segment_id,
        )

    device = resolve_device(args.device)
    if str(args.device) != "cpu" and device == "cpu":
        print("CUDA unavailable; falling back to CPU.")

    model, _ = load_recognition_model(args.config, args.checkpoint, device=device)
    result = recognize_deformable_superquadric(
        points,
        model,
        num_fitting_points=args.num_fitting_points,
    )

    print(f"Sample: {label}")
    print(f"Input points: {result['input_points'].shape[0]}")
    if metadata:
        print(f"Profile: {metadata.get('profile', 'unknown')}")
    print("Predicted deformable superquadric parameters:")
    for key in REQUIRED_PARAM_KEYS:
        print(f"  {key}: {result['parameters'][key]:.6f}")

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        payload = _result_to_jsonable(result)
        if metadata:
            payload["generation_metadata"] = metadata
        with output_path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
        print(f"Wrote {output_path}")

    should_visualize = (args.visualize or args.save_image) and not args.no_visualize
    if should_visualize:
        visualize_result(
            result,
            title=f"DSQNet fit: {label}",
            gt_primitive=gt_primitive if args.show_gt else None,
            viz_mode=args.viz_mode,
            surface_style=args.surface_style,
            fit_surface_points=args.fit_surface_points,
            save_image=Path(args.save_image) if args.save_image else None,
        )


if __name__ == "__main__":
    main()
