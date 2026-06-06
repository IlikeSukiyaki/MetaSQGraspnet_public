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


def _object_mask(instance: np.ndarray, id_to_labels: dict[str, str], object_id: str) -> np.ndarray:
    object_ids = [int(raw_id) for raw_id, label in id_to_labels.items() if label == object_id]
    if object_ids:
        return np.isin(instance, object_ids)
    return instance > 1


def inspect_camera(sample_dir: Path, camera_name: str, show_pcd: bool, show_images: bool) -> None:
    meta = _load_meta(sample_dir)
    camera_dir = sample_dir / camera_name
    depth = np.squeeze(np.load(camera_dir / "depth.npy"))
    instance = np.squeeze(np.load(camera_dir / "instance.npy"))
    pcd_map = np.load(camera_dir / "pcd_map.npy")
    normal = np.load(camera_dir / "normals.npy")

    camera_meta = meta["cameras"][camera_name]
    mask = _object_mask(instance, camera_meta.get("id_to_labels") or {}, meta["object_id"])
    valid = np.isfinite(depth) & (depth > 0.0) & (depth < 2.0)
    valid &= np.isfinite(pcd_map).all(axis=-1)
    object_points = pcd_map[valid & mask].reshape(-1, 3)

    print(f"Sample: {sample_dir}")
    print(f"Camera: {camera_name}")
    print(f"Depth: shape={depth.shape}, min={float(np.nanmin(depth)):.4f}, max={float(np.nanmax(depth)):.4f}")
    print(f"Instance IDs: {np.unique(instance).tolist()}")
    print(f"Normals: shape={normal.shape}")
    print(f"Valid object points: {object_points.shape[0]}")
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
    parser.add_argument("--sample", type=Path, required=True, help="Sample directory, e.g. datasets/sqnet_isaac/001/epi_000001.")
    parser.add_argument("--camera", choices=["camera_onhand", "camera_topdown"], default="camera_onhand")
    parser.add_argument("--show_pcd", action="store_true", help="Open an Open3D viewer for object points.")
    parser.add_argument("--show_images", action="store_true", help="Open OpenCV windows for RGB/depth/instance PNGs.")
    args = parser.parse_args()

    inspect_camera(args.sample, args.camera, args.show_pcd, args.show_images)


if __name__ == "__main__":
    main()
