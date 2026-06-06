#!/usr/bin/env python
"""Convert SQ-v0-Data raw samples to rgbsq_grasp .npy point-cloud samples."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import open3d as o3d

REPO_ROOT = Path(__file__).resolve().parents[3]
RGBSQ_ROOT = REPO_ROOT / "rgbsq_grasp"
if str(RGBSQ_ROOT) not in sys.path:
    sys.path.insert(0, str(RGBSQ_ROOT))

from functions.pc_preprocess import process_pc  # noqa: E402

TOOLS_ROOT = Path(__file__).resolve().parent
if str(TOOLS_ROOT) not in sys.path:
    sys.path.insert(0, str(TOOLS_ROOT))
from sq_partial_pc import MASK_MODES, extract_object_partial_pc  # noqa: E402


CAMERA_CHOICES = {
    "onhand": ["camera_onhand"],
    "topdown": ["camera_topdown"],
    "both": ["camera_onhand", "camera_topdown"],
}


def _load_meta(sample_dir: Path) -> dict:
    with (sample_dir / "meta.json").open("r", encoding="utf-8") as file:
        return json.load(file)


def _extract_object_points(
    sample_dir: Path,
    camera_name: str,
    meta: dict,
    mask_mode: str,
) -> np.ndarray:
    payload = extract_object_partial_pc(
        sample_dir,
        camera_name,
        object_id=meta["object_id"],
        mask_mode=mask_mode,
        color_mode="none",
    )
    return payload["partial_pc"].astype(np.float64)


def _process_points(points: np.ndarray, num_points: int) -> tuple[np.ndarray, np.ndarray]:
    if points.shape[0] < 3:
        raise ValueError(f"Need at least 3 object points, got {points.shape[0]}.")

    pcd = o3d.geometry.PointCloud(points=o3d.utility.Vector3dVector(points))
    processed, pcd_frame = process_pc(pcd, num_processed_pc_points=num_points)
    if processed is None or pcd_frame is None:
        raise ValueError("process_pc failed to build a canonical frame.")
    return np.asarray(processed.points, dtype=np.float64), np.asarray(pcd_frame, dtype=np.float64)


def _box_primitive_from_points(points: np.ndarray) -> dict:
    extents = np.maximum(points.max(axis=0) - points.min(axis=0), 1e-6)
    return {
        "type": "box",
        "SE3": np.eye(4, dtype=np.float64),
        "parameters": {
            "width": float(extents[0]),
            "depth": float(extents[1]),
            "height": float(extents[2]),
        },
    }


def convert_samples(
    input_dir: Path,
    output_dir: Path,
    camera_mode: str,
    num_points: int,
    mask_mode: str,
) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    sample_dirs = sorted(path for path in input_dir.glob("epi_*") if path.is_dir())
    camera_names = CAMERA_CHOICES[camera_mode]
    saved_paths: list[Path] = []

    for sample_index, sample_dir in enumerate(sample_dirs):
        meta = _load_meta(sample_dir)
        for camera_index, camera_name in enumerate(camera_names):
            points = _extract_object_points(sample_dir, camera_name, meta, mask_mode)
            processed_points, pcd_frame = _process_points(points, num_points)
            membership = np.zeros((processed_points.shape[0],), dtype=np.int64)
            file_index = sample_index
            viewpoint = camera_index if camera_mode == "both" else 0
            filename = f"{meta['object_id']}_{file_index:04d}_viewpoint_{viewpoint:02d}.npy"
            output_path = output_dir / filename
            payload = {
                "partial_pc": processed_points,
                "membership": membership,
                "primitives": [_box_primitive_from_points(processed_points)],
                "pcd_frame": pcd_frame,
                "source_camera": camera_name,
                "source_sample": str(sample_dir),
                "source_frame": "robot_base_meters",
                "object_id": meta["object_id"],
            }
            np.save(output_path, payload)
            saved_paths.append(output_path)

    datalist_path = output_dir / "datalist.csv"
    with datalist_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        for path in saved_paths:
            writer.writerow([path.name])

    return saved_paths


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Raw SQ object directory, e.g. datasets/sqnet_isaac/001.")
    parser.add_argument("--output", type=Path, required=True, help="Output directory for rgbsq_grasp .npy samples.")
    parser.add_argument("--camera", choices=CAMERA_CHOICES, default="both", help="Which camera views to export.")
    parser.add_argument("--num_points", type=int, default=3000, help="Number of points in each partial_pc.")
    parser.add_argument(
        "--mask",
        choices=MASK_MODES,
        default="auto",
        help="Object pixel selection mode passed to sq_partial_pc.",
    )
    args = parser.parse_args()

    saved_paths = convert_samples(
        args.input, args.output, args.camera, args.num_points, args.mask
    )
    print(f"Saved {len(saved_paths)} SQNet samples to {args.output}")


if __name__ == "__main__":
    main()
