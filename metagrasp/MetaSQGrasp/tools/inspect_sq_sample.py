#!/usr/bin/env python
"""Inspect one SQ-v0-Data raw sample."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import open3d as o3d


def _load_meta(sample_dir: Path) -> dict:
    with (sample_dir / "meta.json").open("r", encoding="utf-8") as file:
        return json.load(file)


def _object_mask(
    instance: np.ndarray,
    id_to_labels: dict[str, str],
    object_id: str,
    *,
    strict_target: bool = False,
) -> np.ndarray:
    object_ids = [int(raw_id) for raw_id, label in id_to_labels.items() if label == object_id]
    if object_ids:
        return np.isin(instance, object_ids)
    if strict_target:
        return np.zeros(instance.shape, dtype=bool)
    return instance > 1


def inspect_camera(sample_dir: Path, camera_name: str, show_pcd: bool, show_images: bool) -> None:
    meta = _load_meta(sample_dir)
    camera_dir = sample_dir / camera_name
    depth = np.squeeze(np.load(camera_dir / "depth.npy"))
    instance = np.squeeze(np.load(camera_dir / "instance.npy"))
    pcd_map = np.load(camera_dir / "pcd_map.npy")
    normals_path = camera_dir / "normals.npy"
    normal = np.load(normals_path) if normals_path.exists() else None

    camera_meta = meta["cameras"][camera_name]
    id_to_labels = camera_meta.get("id_to_labels") or {}
    scene_objects = int(meta.get("scene_objects", 1))
    target_object_id = meta.get("target_object_id", meta["object_id"])
    scene_object_ids = meta.get("scene_object_ids", [target_object_id])
    distractor_ids = meta.get("distractor_ids", [])
    strict_target = scene_objects > 1

    valid = np.isfinite(depth) & (depth > 0.0) & (depth < 2.0)
    valid &= np.isfinite(pcd_map).all(axis=-1)

    print(f"Sample: {sample_dir}")
    print(f"Camera: {camera_name}")
    print(f"Target object: {target_object_id}")
    print(f"Scene objects ({scene_objects}): {scene_object_ids}")
    if distractor_ids:
        print(f"Distractors: {distractor_ids}")
    print(f"Depth: shape={depth.shape}, min={float(np.nanmin(depth)):.4f}, max={float(np.nanmax(depth)):.4f}")
    print(f"Instance IDs: {np.unique(instance).tolist()}")
    if normal is not None:
        print(f"Normals: shape={normal.shape}")

    for object_id in scene_object_ids:
        mask = _object_mask(
            instance,
            id_to_labels,
            object_id,
            strict_target=strict_target or object_id == target_object_id,
        )
        object_points = pcd_map[valid & mask].reshape(-1, 3)
        print(f"Valid points [{object_id}]: {object_points.shape[0]}")

    target_mask = _object_mask(
        instance,
        id_to_labels,
        target_object_id,
        strict_target=strict_target,
    )
    object_points = pcd_map[valid & target_mask].reshape(-1, 3)
    valid_pixels = int(valid.sum())
    target_pixels = int((valid & target_mask).sum())
    if valid_pixels > 0:
        print(
            f"Target visible fraction: {target_pixels / valid_pixels:.4f} "
            f"({target_pixels}/{valid_pixels} pixels)"
        )
    print(f"Intrinsics:\n{np.asarray(camera_meta['intrinsics_3x3'])}")
    print(f"Pose 7D: {camera_meta['pose_7d']}")

    if show_images:
        rgb_path = camera_dir / "rgb.png"
        if rgb_path.exists():
            cv2.imshow(f"{camera_name} rgb", cv2.imread(str(rgb_path)))
        cv2.imshow(f"{camera_name} depth", cv2.imread(str(camera_dir / "depth.png"), cv2.IMREAD_GRAYSCALE))
        cv2.imshow(f"{camera_name} instance", cv2.imread(str(camera_dir / "instance.png")))
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    if show_pcd and object_points.size:
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(object_points)
        axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
        o3d.visualization.draw_geometries([pcd, axis])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample", type=Path, required=True, help="Sample directory, e.g. datasets/supersq_grasp_data/001/epi_000001.")
    parser.add_argument("--camera", choices=["camera_onhand", "camera_topdown"], default="camera_onhand")
    parser.add_argument("--show_pcd", action="store_true", help="Open an Open3D viewer for object points.")
    parser.add_argument("--show_images", action="store_true", help="Open OpenCV windows for RGB/depth/instance PNGs.")
    args = parser.parse_args()

    inspect_camera(args.sample, args.camera, args.show_pcd, args.show_images)


if __name__ == "__main__":
    main()
