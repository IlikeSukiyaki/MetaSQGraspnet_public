#!/usr/bin/env python
"""Extract object-only partial point clouds from SQ-v0-Data samples."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from sq_partial_pc import (
    CAMERA_ALIASES,
    COLOR_MODES,
    MASK_MODES,
    extract_object_partial_pc,
    resolve_camera_names,
)

try:
    import open3d as o3d
except ImportError:  # pragma: no cover - optional for ply/show
    o3d = None

EXAMPLES = """
Examples:
  # Recommended default: mesh fallback + instance-seg colors
  python tools/extract_partial_pc.py \\
    --sample datasets/supersq_grasp_data/001/epi_000001 \\
    --camera topdown --mask auto --color instance --show

  # Force instance segmentation mask only
  python tools/extract_partial_pc.py \\
    --sample datasets/supersq_grasp_data/001/epi_000001 \\
    --camera onhand --mask instance --color instance

  # Force mesh bounding-box mask (works on current single-object data)
  python tools/extract_partial_pc.py \\
    --sample datasets/supersq_grasp_data/001/epi_000001 \\
    --camera topdown --mask mesh --color rgb --ply

  # Batch both cameras for all episodes
  python tools/extract_partial_pc.py \\
    --input datasets/supersq_grasp_data/001 \\
    --camera both --mask auto --color instance
"""


def _default_output_path(sample_dir: Path, camera_name: str, mask_mode: str, color_mode: str) -> Path:
    return sample_dir / "partial_pc" / f"{camera_name}_{mask_mode}_{color_mode}.npy"


def _require_open3d(feature: str) -> None:
    if o3d is None:
        raise SystemExit(f"Open3D is required for --{feature}. Install open3d or use .npy output only.")


def _save_payload(payload: dict, output_path: Path, save_ply: bool) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_dict = {key: value for key, value in payload.items() if key != "instance_mask"}
    np.save(output_path, save_dict)

    if save_ply:
        _require_open3d("ply")
        ply_path = output_path.with_suffix(".ply")
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(payload["partial_pc"])
        if "normals" in payload:
            pcd.normals = o3d.utility.Vector3dVector(payload["normals"])
        if "colors" in payload:
            pcd.colors = o3d.utility.Vector3dVector(payload["colors"].astype(np.float64) / 255.0)
        o3d.io.write_point_cloud(str(ply_path), pcd)
    return output_path


def _show_payload(payload: dict) -> None:
    _require_open3d("show")
    if payload["num_points"] == 0:
        print("No object points to visualize.")
        return
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(payload["partial_pc"])
    if "normals" in payload:
        pcd.normals = o3d.utility.Vector3dVector(payload["normals"])
    if "colors" in payload:
        pcd.colors = o3d.utility.Vector3dVector(payload["colors"].astype(np.float64) / 255.0)
    axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
    o3d.visualization.draw_geometries([pcd, axis])


def _resolve_sample_dirs(sample: Path | None, input_dir: Path | None) -> list[Path]:
    if sample is not None:
        return [sample]
    assert input_dir is not None
    return sorted(path for path in input_dir.glob("epi_*") if path.is_dir())


def extract_and_save(
    sample_dir: Path,
    camera: str,
    output_dir: Path | None,
    model_root: Path | None,
    margin: float,
    mask_mode: str,
    color_mode: str,
    save_ply: bool,
    show: bool,
) -> list[Path]:
    saved_paths: list[Path] = []
    for camera_name in resolve_camera_names(camera):
        payload = extract_object_partial_pc(
            sample_dir,
            camera_name,
            model_root=model_root,
            margin=margin,
            mask_mode=mask_mode,
            color_mode=color_mode,
        )
        if output_dir is None:
            output_path = _default_output_path(sample_dir, camera_name, mask_mode, color_mode)
        else:
            output_path = (
                output_dir
                / sample_dir.name
                / f"{camera_name}_{mask_mode}_{color_mode}.npy"
            )
        saved_paths.append(_save_payload(payload, output_path, save_ply))
        color_info = payload.get("color_source", "none")
        print(
            f"{sample_dir.name}/{camera_name}: "
            f"{payload['num_points']} points | mask={payload['mask_method']} | "
            f"color={color_info} -> {output_path}"
        )
        if show:
            _show_payload(payload)
    return saved_paths


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=EXAMPLES,
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--sample", type=Path, help="One episode directory, e.g. datasets/supersq_grasp_data/001/epi_000001.")
    group.add_argument("--input", type=Path, help="Object directory with epi_* samples, e.g. datasets/supersq_grasp_data/001.")

    parser.add_argument(
        "--camera",
        choices=[*CAMERA_ALIASES, "both"],
        default="topdown",
        help="Camera view: onhand, topdown, or both.",
    )
    parser.add_argument(
        "--mask",
        choices=MASK_MODES,
        default="auto",
        help=(
            "How to select object pixels. "
            "auto=instance mask with mesh-bbox fallback (recommended), "
            "instance=instance segmentation only, "
            "mesh=mesh bounding box only."
        ),
    )
    parser.add_argument(
        "--color",
        choices=COLOR_MODES,
        default="instance",
        help="Point color source: instance-seg colormap, camera RGB, or none.",
    )
    parser.add_argument("--output", type=Path, default=None, help="Output directory. Default: <sample>/partial_pc/.")
    parser.add_argument("--model_root", type=Path, default=None, help="Root containing <object_id>/textured.obj.")
    parser.add_argument("--margin", type=float, default=0.015, help="Mesh bbox margin in meters when --mask mesh/auto.")
    parser.add_argument("--ply", action="store_true", help="Also save an Open3D .ply next to each .npy.")
    parser.add_argument("--show", action="store_true", help="Open an Open3D viewer after extraction.")
    args = parser.parse_args()

    sample_dirs = _resolve_sample_dirs(args.sample, args.input)
    if not sample_dirs:
        raise SystemExit("No epi_* sample directories found.")

    all_paths: list[Path] = []
    for sample_dir in sample_dirs:
        all_paths.extend(
            extract_and_save(
                sample_dir,
                args.camera,
                args.output,
                args.model_root,
                args.margin,
                args.mask,
                args.color,
                args.ply,
                args.show,
            )
        )
    print(f"Saved {len(all_paths)} object partial point cloud file(s).")


if __name__ == "__main__":
    main()
