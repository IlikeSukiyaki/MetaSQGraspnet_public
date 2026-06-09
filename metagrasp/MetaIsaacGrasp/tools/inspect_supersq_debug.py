#!/usr/bin/env python
"""Visualize one supersq_grasp-v0 debug attempt folder."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

try:
    import open3d as o3d
except ImportError as exc:  # pragma: no cover - optional debug tool
    raise SystemExit("Open3D is required. Install open3d in the active environment.") from exc


def _load_point_cloud(path: Path):
    if not path.exists():
        return None
    cloud = o3d.io.read_point_cloud(str(path))
    return cloud if cloud.has_points() else None


def _load_mesh(path: Path):
    if not path.exists():
        return None
    mesh = o3d.io.read_triangle_mesh(str(path))
    if mesh.has_triangles():
        mesh.compute_vertex_normals()
        return mesh
    cloud = o3d.io.read_point_cloud(str(path))
    return cloud if cloud.has_points() else None


def inspect_debug_dir(debug_dir: Path) -> None:
    debug_dir = debug_dir.expanduser().resolve()
    meta_path = debug_dir / "sq_fit.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"Missing sq_fit.json in {debug_dir}")

    with meta_path.open("r", encoding="utf-8") as file:
        meta = json.load(file)

    print(f"Debug dir: {debug_dir}")
    print(f"Object: {meta.get('object_id')}")
    print(f"Camera index: {meta.get('camera_index')}")
    print(f"Points: {meta.get('num_points')} mask={meta.get('mask_method')}")
    selection = meta.get("selection")
    if selection:
        print(
            "Grasp: "
            f"score={selection.get('score')} width_cm={selection.get('width_cm')} "
            f"alignment={selection.get('approach_alignment')}"
        )
    else:
        print("Grasp: none")

    geometries = []
    partial_pc = _load_point_cloud(debug_dir / "partial_pc.ply")
    if partial_pc is not None:
        partial_pc.paint_uniform_color((0.0, 0.45, 1.0))
        geometries.append(partial_pc)

    sq_mesh = _load_mesh(debug_dir / "sq_mesh_robot.ply")
    if sq_mesh is not None:
        if isinstance(sq_mesh, o3d.geometry.TriangleMesh):
            sq_mesh.paint_uniform_color((1.0, 0.55, 0.0))
        geometries.append(sq_mesh)

    grasp_axes = _load_mesh(debug_dir / "grasp_axes.ply")
    if grasp_axes is not None:
        geometries.append(grasp_axes)

    world_axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
    geometries.append(world_axes)

    if len(geometries) == 1:
        print("No geometry artifacts found to display.")
        return
    o3d.visualization.draw_geometries(geometries)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dir", type=Path, required=True, help="Path to one env*_epi*_attempt*_obj* debug folder.")
    args = parser.parse_args()
    inspect_debug_dir(args.dir)


if __name__ == "__main__":
    main()
