#!/usr/bin/env python
"""Visualize the selected SuperSQ grasp from one debug attempt folder."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from grasp_sampler.visualize_grasp_object import (  # noqa: E402
    convert_sampler_to_ee_frame,
    create_easy_gripper,
)
from tools.supersq_debug_paths import add_debug_dir_arguments, resolve_debug_dir_from_args  # noqa: E402

try:
    import open3d as o3d
    import trimesh
except ImportError as exc:  # pragma: no cover - optional debug tool
    raise SystemExit("Open3D and trimesh are required. Install open3d and trimesh.") from exc

# Vivid green for the selected Hand-E grasp pose (Open3D RGB in [0, 1]).
GRASP_POSE_COLOR = (0.0, 0.85, 0.25)


def _load_json(debug_dir: Path) -> dict:
    meta_path = debug_dir / "sq_fit.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"Missing sq_fit.json in {debug_dir}")
    with meta_path.open("r", encoding="utf-8") as file:
        return json.load(file)


def _load_point_cloud(path: Path):
    if not path.exists():
        return None
    cloud = o3d.io.read_point_cloud(str(path))
    return cloud if cloud.has_points() else None


def _load_triangle_mesh(path: Path):
    if not path.exists():
        return None
    mesh = o3d.io.read_triangle_mesh(str(path))
    if not mesh.has_triangles():
        return None
    mesh.compute_vertex_normals()
    return mesh


def _pose_cm_to_m(pose_cm: np.ndarray) -> np.ndarray:
    pose_m = np.asarray(pose_cm, dtype=np.float64).copy()
    pose_m[:3, 3] /= 100.0
    return pose_m


def _resolve_grasp_pose_cm(selection: dict, frame: str) -> np.ndarray:
    pose_cm = np.asarray(selection["gripper_pose_cm"], dtype=np.float64)
    if frame == "ee":
        return convert_sampler_to_ee_frame(pose_cm)
    if frame != "sampler":
        raise ValueError(f"Unsupported frame '{frame}'. Choose 'sampler' or 'ee'.")
    return pose_cm


def _trimesh_to_open3d(
    mesh: trimesh.Trimesh,
    *,
    uniform_color: tuple[float, float, float] | None = None,
) -> o3d.geometry.TriangleMesh:
    o3d_mesh = o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(np.asarray(mesh.vertices, dtype=np.float64)),
        o3d.utility.Vector3iVector(np.asarray(mesh.faces, dtype=np.int32)),
    )
    if uniform_color is not None:
        o3d_mesh.paint_uniform_color(uniform_color)
    elif mesh.visual.face_colors is not None and len(mesh.visual.face_colors):
        face_colors = np.asarray(mesh.visual.face_colors[:, :3], dtype=np.float64) / 255.0
        vertex_colors = np.zeros((len(mesh.vertices), 3), dtype=np.float64)
        counts = np.zeros(len(mesh.vertices), dtype=np.int32)
        for face_idx, face in enumerate(mesh.faces):
            for vert_idx in face:
                vertex_colors[vert_idx] += face_colors[face_idx]
                counts[vert_idx] += 1
        valid = counts > 0
        vertex_colors[valid] /= counts[valid, None]
        o3d_mesh.vertex_colors = o3d.utility.Vector3dVector(vertex_colors)
    o3d_mesh.compute_vertex_normals()
    return o3d_mesh


def _make_gripper_mesh(
    selection: dict,
    *,
    frame: str,
    show_axis: bool,
) -> o3d.geometry.TriangleMesh | None:
    width_cm = float(selection.get("width_cm", 0.0))
    pose_cm = _resolve_grasp_pose_cm(selection, frame)
    gripper = create_easy_gripper(
        color=[40, 200, 80, 180],
        sections=6,
        show_axis=show_axis,
        width=width_cm,
        frame=frame,
    )
    gripper.apply_scale(0.01)
    gripper.apply_transform(_pose_cm_to_m(pose_cm))
    return _trimesh_to_open3d(gripper, uniform_color=GRASP_POSE_COLOR)


def _make_contact_frame(selection: dict) -> o3d.geometry.TriangleMesh | None:
    contact_pose_cm = selection.get("contact_pose_cm")
    if contact_pose_cm is None:
        return None
    frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.04)
    frame.transform(_pose_cm_to_m(np.asarray(contact_pose_cm, dtype=np.float64)))
    return frame


def _make_approach_arrow(pose_cm: np.ndarray, length_m: float = 0.08) -> o3d.geometry.TriangleMesh:
    origin = _pose_cm_to_m(pose_cm)[:3, 3]
    approach = np.asarray(pose_cm[:3, 2], dtype=np.float64)
    approach = approach / (np.linalg.norm(approach) + 1e-12)
    end = origin + approach * length_m
    direction = end - origin
    height = np.linalg.norm(direction)
    if height <= 1e-9:
        return o3d.geometry.TriangleMesh()

    cylinder = o3d.geometry.TriangleMesh.create_cylinder(radius=0.004, height=height)
    cylinder.paint_uniform_color((0.9, 0.2, 0.2))
    cylinder.compute_vertex_normals()

    z_axis = np.array([0.0, 0.0, 1.0])
    rotation_axis = np.cross(z_axis, direction)
    if np.linalg.norm(rotation_axis) > 1e-9:
        rotation_axis /= np.linalg.norm(rotation_axis)
        angle = np.arccos(np.clip(np.dot(z_axis, direction), -1.0, 1.0))
        rotation = o3d.geometry.get_rotation_matrix_from_axis_angle(rotation_axis * angle)
        cylinder.rotate(rotation, center=(0.0, 0.0, 0.0))
    elif np.dot(z_axis, direction) < 0.0:
        cylinder.rotate(o3d.geometry.get_rotation_matrix_from_axis_angle(np.array([1.0, 0.0, 0.0]) * np.pi), center=(0.0, 0.0, 0.0))

    cylinder.translate((origin + end) / 2.0)
    return cylinder


def inspect_grasps(
    debug_dir: Path,
    *,
    frame: str = "sampler",
    show_context: bool = True,
    show_contact: bool = True,
    show_approach: bool = True,
    show_axis: bool = False,
) -> None:
    debug_dir = debug_dir.expanduser().resolve()
    meta = _load_json(debug_dir)
    selection = meta.get("selection")
    if not selection:
        raise ValueError(f"No selected grasp found in {debug_dir / 'sq_fit.json'}")

    print(f"Debug dir: {debug_dir}")
    print(f"Object: {meta.get('object_id')}")
    print(
        "Selected grasp: "
        f"score={selection.get('score')} "
        f"width_cm={selection.get('width_cm')} "
        f"alignment={selection.get('approach_alignment')} "
        f"candidates={selection.get('candidate_count')}"
    )
    print(f"Visual frame: {frame}")

    geometries: list[o3d.geometry.Geometry3D] = []

    if show_context:
        partial_pc = _load_point_cloud(debug_dir / "partial_pc.ply")
        if partial_pc is not None:
            partial_pc.paint_uniform_color((0.0, 0.45, 1.0))
            geometries.append(partial_pc)

        sq_mesh = _load_triangle_mesh(debug_dir / "sq_mesh_robot.ply")
        if sq_mesh is not None:
            sq_mesh.paint_uniform_color((1.0, 0.55, 0.0))
            geometries.append(sq_mesh)

    gripper_mesh = _make_gripper_mesh(selection, frame=frame, show_axis=show_axis)
    if gripper_mesh is not None:
        geometries.append(gripper_mesh)

    if show_contact:
        contact_frame = _make_contact_frame(selection)
        if contact_frame is not None:
            geometries.append(contact_frame)

    if show_approach:
        pose_cm = _resolve_grasp_pose_cm(selection, frame)
        approach_arrow = _make_approach_arrow(pose_cm)
        if approach_arrow.has_triangles():
            geometries.append(approach_arrow)

    geometries.append(o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1))

    if len(geometries) == 1:
        raise RuntimeError("No grasp geometry could be built from sq_fit.json.")
    o3d.visualization.draw_geometries(geometries)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_debug_dir_arguments(parser)
    parser.add_argument(
        "--frame",
        choices=["sampler", "ee"],
        default="sampler",
        help="Draw the stored sampler-frame grasp or convert to executed EE frame.",
    )
    parser.add_argument(
        "--no-context",
        action="store_true",
        help="Hide partial point cloud and SQ mesh.",
    )
    parser.add_argument(
        "--no-contact",
        action="store_true",
        help="Hide the contact coordinate frame.",
    )
    parser.add_argument(
        "--no-approach",
        action="store_true",
        help="Hide the red approach arrow.",
    )
    parser.add_argument(
        "--show-axis",
        action="store_true",
        help="Embed axes inside the gripper mesh.",
    )
    args = parser.parse_args()
    inspect_grasps(
        resolve_debug_dir_from_args(args),
        frame=args.frame,
        show_context=not args.no_context,
        show_contact=not args.no_contact,
        show_approach=not args.no_approach,
        show_axis=args.show_axis,
    )


if __name__ == "__main__":
    main()
