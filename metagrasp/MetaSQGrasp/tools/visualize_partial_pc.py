#!/usr/bin/env python
"""Visualize partial point clouds from SQ-v0-Data samples in one Open3D viewer."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

TOOLS_ROOT = Path(__file__).resolve().parent
if str(TOOLS_ROOT) not in sys.path:
    sys.path.insert(0, str(TOOLS_ROOT))

from sq_partial_pc import extract_object_partial_pc, resolve_camera_names  # noqa: E402

try:
    import open3d as o3d
except ImportError:  # pragma: no cover
    o3d = None

DEFAULT_PALETTE = {
    "000": (1.0, 0.2, 0.2),
    "001": (0.2, 0.8, 1.0),
    "004": (1.0, 0.8, 0.2),
    "011": (0.8, 0.2, 1.0),
}
CAMERA_TINT = {
    "camera_onhand": 0.55,
    "camera_topdown": 0.85,
}

EXAMPLES = """
Examples:
  # Lite export: show saved partial PCs (target object, both cameras)
  python tools/visualize_partial_pc.py \\
    --sample datasets/supersq_grasp_data/multi_obj/000/epi_000009

  # Full export: extract and show all scene objects from top-down view
  python tools/visualize_partial_pc.py \\
    --sample datasets/supersq_grasp_data/multi_obj_full/011/epi_000000 \\
    --source extract --objects all --camera topdown

  # Full export: one object, both cameras
  python tools/visualize_partial_pc.py \\
    --sample datasets/supersq_grasp_data/multi_obj_full/011/epi_000000 \\
    --source extract --objects 000 --camera both
"""


def _load_meta(sample_dir: Path) -> dict:
    with (sample_dir / "meta.json").open("r", encoding="utf-8") as file:
        return json.load(file)


def _color_for(obj_id: str, camera: str | None, palette: dict[str, tuple[float, float, float]]) -> list[float]:
    base = np.array(palette.get(obj_id, (0.7, 0.7, 0.7)), dtype=np.float64)
    if camera is not None:
        base *= CAMERA_TINT.get(camera, 1.0)
    return base.tolist()


def _payload_to_pcd(payload: dict, color: list[float]) -> o3d.geometry.PointCloud:
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(payload["partial_pc"])
    if "colors" in payload and payload.get("color_mode") != "none":
        pcd.colors = o3d.utility.Vector3dVector(payload["colors"].astype(np.float64) / 255.0)
    else:
        pcd.paint_uniform_color(color)
    return pcd


def _resolve_object_ids(meta: dict, objects_arg: str) -> list[str]:
    scene_ids = meta.get("scene_object_ids", [meta["object_id"]])
    if objects_arg == "all":
        return list(scene_ids)
    if objects_arg == "target":
        return [meta.get("target_object_id", meta["object_id"])]
    return [obj_id.strip() for obj_id in objects_arg.split(",") if obj_id.strip()]


def _has_full_tensors(sample_dir: Path, camera_name: str) -> bool:
    camera_dir = sample_dir / camera_name
    return (camera_dir / "depth.npy").exists() and (camera_dir / "pcd_map.npy").exists()


def _load_saved_payloads(sample_dir: Path, meta: dict, cameras: list[str]) -> list[tuple[str, str, dict]]:
    partial_meta = meta.get("partial_pc", {})
    payloads: list[tuple[str, str, dict]] = []
    partial_dir = sample_dir / "partial_pc"

    for camera_name in cameras:
        camera_entry = partial_meta.get(camera_name)
        if camera_entry is None:
            matches = sorted(partial_dir.glob(f"{camera_name}_*.npy"))
            if not matches:
                print(f"[warn] No saved partial_pc for {camera_name}")
                continue
            payload = np.load(matches[0], allow_pickle=True).item()
            payloads.append((camera_name, payload["object_id"], payload))
            continue

        path = partial_dir / camera_entry["file"]
        if not path.exists():
            raise FileNotFoundError(f"Missing saved partial_pc file: {path}")
        payload = np.load(path, allow_pickle=True).item()
        payloads.append((camera_name, payload["object_id"], payload))
    return payloads


def _extract_payloads(
    sample_dir: Path,
    meta: dict,
    cameras: list[str],
    object_ids: list[str],
    mask_mode: str,
) -> list[tuple[str, str, dict]]:
    payloads: list[tuple[str, str, dict]] = []
    for camera_name in cameras:
        if not _has_full_tensors(sample_dir, camera_name):
            raise SystemExit(
                f"{camera_name} is missing depth.npy/pcd_map.npy. "
                "Use --source saved for lite exports, or re-collect with --export_profile full."
            )
        for obj_id in object_ids:
            payload = extract_object_partial_pc(
                sample_dir,
                camera_name,
                object_id=obj_id,
                mask_mode=mask_mode,
                color_mode="none",
            )
            if payload["num_points"] == 0:
                print(f"[warn] {camera_name} / {obj_id}: 0 points")
                continue
            payloads.append((camera_name, obj_id, payload))
    return payloads


def visualize_sample(
    sample_dir: Path,
    source: str,
    camera: str,
    objects_arg: str,
    mask_mode: str,
    palette: dict[str, tuple[float, float, float]],
) -> None:
    if o3d is None:
        raise SystemExit("Open3D is required. Install open3d in the metagrasp environment.")

    sample_dir = Path(sample_dir)
    meta = _load_meta(sample_dir)
    cameras = resolve_camera_names(camera)
    object_ids = _resolve_object_ids(meta, objects_arg)

    if source == "saved":
        if objects_arg not in {"target", "all"}:
            print("[warn] --source saved only loads pre-exported target partial PCs; ignoring --objects.")
        payloads = _load_saved_payloads(sample_dir, meta, cameras)
    else:
        payloads = _extract_payloads(sample_dir, meta, cameras, object_ids, mask_mode)

    if not payloads:
        raise SystemExit("No partial point clouds to visualize.")

    geometries: list[o3d.geometry.Geometry] = []
    for camera_name, obj_id, payload in payloads:
        color = _color_for(obj_id, camera_name if len(cameras) > 1 else None, palette)
        geometries.append(_payload_to_pcd(payload, color))
        print(f"{camera_name} / {obj_id}: {payload['num_points']} points")

    geometries.append(o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1))
    o3d.visualization.draw_geometries(geometries)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=EXAMPLES,
    )
    parser.add_argument(
        "--sample",
        type=Path,
        required=True,
        help="Episode directory, e.g. datasets/supersq_grasp_data/multi_obj/000/epi_000009.",
    )
    parser.add_argument(
        "--source",
        choices=("saved", "extract"),
        default="saved",
        help="saved=load partial_pc/*.npy (lite export); extract=re-run extraction from full tensors.",
    )
    parser.add_argument(
        "--camera",
        choices=("onhand", "topdown", "both"),
        default="both",
        help="Camera view to visualize.",
    )
    parser.add_argument(
        "--objects",
        default="target",
        help="target, all, or comma-separated object ids (extract mode only).",
    )
    parser.add_argument(
        "--mask",
        choices=("auto", "instance", "mesh"),
        default="instance",
        help="Mask mode for --source extract.",
    )
    args = parser.parse_args()

    visualize_sample(
        sample_dir=args.sample,
        source=args.source,
        camera=args.camera,
        objects_arg=args.objects,
        mask_mode=args.mask,
        palette=DEFAULT_PALETTE,
    )


if __name__ == "__main__":
    main()
