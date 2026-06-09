"""Extract object-only partial point clouds from SQ-v0-Data samples."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

import numpy as np

try:
    import cv2
except ImportError:  # pragma: no cover - optional for instance colorization
    cv2 = None

METAISAAC_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = METAISAAC_ROOT.parent.parent
MODEL_ROOT_CANDIDATES = (
    METAISAAC_ROOT / "models" / "models_ifl",
    REPO_ROOT / "models_ifl",
)

MASK_MODES = ("auto", "instance", "mesh")
COLOR_MODES = ("instance", "rgb", "none")
CAMERA_ALIASES = {
    "onhand": "camera_onhand",
    "topdown": "camera_topdown",
}

BACKGROUND_LABELS = {"-1", "0", "background", "BACKGROUND", "unlabelled"}


def resolve_camera_name(camera: str) -> str:
    if camera in CAMERA_ALIASES:
        return CAMERA_ALIASES[camera]
    if camera in CAMERA_ALIASES.values():
        return camera
    raise ValueError(
        f"Unknown camera '{camera}'. Choose from: {', '.join(CAMERA_ALIASES)}."
    )


def resolve_camera_names(camera: str) -> list[str]:
    if camera == "both":
        return [CAMERA_ALIASES["onhand"], CAMERA_ALIASES["topdown"]]
    return [resolve_camera_name(camera)]


def _load_meta(sample_dir: Path) -> dict:
    with (sample_dir / "meta.json").open("r", encoding="utf-8") as file:
        return json.load(file)


def _resolve_model_root(model_root: Path | None) -> Path:
    if model_root is not None:
        return Path(model_root)
    for candidate in MODEL_ROOT_CANDIDATES:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        "Could not find models_ifl. Pass --model_root or place meshes under models/models_ifl."
    )


def _load_obj_vertices(mesh_path: Path) -> np.ndarray:
    vertices = []
    with mesh_path.open("r", encoding="utf-8") as file:
        for line in file:
            if line.startswith("v "):
                vertices.append([float(value) for value in line.split()[1:4]])
    if not vertices:
        raise ValueError(f"No vertices found in {mesh_path}")
    return np.asarray(vertices, dtype=np.float64)


def _object_pose_meters(meta: dict, object_id: str) -> tuple[np.ndarray, np.ndarray]:
    obj_index = meta["obj_id"].index(object_id)
    transform_cm = np.asarray(meta["obj_poses_robot"][obj_index], dtype=np.float64)
    rotation = transform_cm[:3, :3]
    translation = transform_cm[:3, 3] / 100.0
    return rotation, translation


def _background_instance_ids(id_to_labels: dict[str, str]) -> set[int]:
    background_ids = {0}
    for raw_id, label in id_to_labels.items():
        if str(label) in BACKGROUND_LABELS:
            background_ids.add(int(raw_id))
    return background_ids


def _instance_object_mask(
    instance: np.ndarray,
    id_to_labels: dict[str, str],
    object_id: str,
) -> tuple[np.ndarray, str]:
    """Build a pixel mask from instance segmentation labels."""
    id_to_labels = id_to_labels or {}
    background_ids = _background_instance_ids(id_to_labels)
    object_ids = [int(raw_id) for raw_id, label in id_to_labels.items() if label == object_id]

    if object_ids:
        mask = np.isin(instance, object_ids)
        if mask.any():
            return mask, "instance_label"

    multi_object_mask = ~np.isin(instance, list(background_ids)) & (instance > 1)
    if multi_object_mask.any():
        return multi_object_mask, "instance_gt1"

    return np.zeros(instance.shape, dtype=bool), "instance_empty"


def _mask_is_degenerate(mask: np.ndarray, valid_mask: np.ndarray, max_fraction: float = 0.85) -> bool:
    valid_count = int(valid_mask.sum())
    if valid_count == 0:
        return True
    return (mask & valid_mask).sum() / valid_count >= max_fraction


def _instance_segmentation_color_image(instance: np.ndarray) -> np.ndarray:
    """Match the JET colormap used in export.py instance.png previews."""
    labels = np.squeeze(instance).astype(np.uint8)
    if cv2 is not None:
        bgr = cv2.applyColorMap((labels * 40) % 255, cv2.COLORMAP_JET)
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    palette = {}
    colors = np.zeros((*labels.shape, 3), dtype=np.uint8)
    for instance_id in np.unique(labels):
        hue = (int(instance_id) * 97) % 255
        palette[int(instance_id)] = np.array(
            [(hue * 3) % 255, (hue * 5) % 255, (hue * 7) % 255],
            dtype=np.uint8,
        )
    for instance_id, color in palette.items():
        colors[labels == instance_id] = color
    return colors


def _colors_at_mask(color_image: np.ndarray, pixel_mask: np.ndarray) -> np.ndarray:
    return color_image[pixel_mask].reshape(-1, 3).astype(np.uint8)


def _mesh_object_mask(
    points: np.ndarray,
    object_id: str,
    meta: dict,
    model_root: Path | None,
    margin: float,
) -> tuple[np.ndarray, dict]:
    rotation, translation = _object_pose_meters(meta, object_id)
    mesh_path = _resolve_model_root(model_root) / object_id / "textured.obj"
    vertices_local = _load_obj_vertices(mesh_path)
    mins = vertices_local.min(axis=0) - margin
    maxs = vertices_local.max(axis=0) + margin

    points_local = (points - translation) @ rotation
    mask = np.all((points_local >= mins) & (points_local <= maxs), axis=1)
    debug = {
        "mesh_path": str(mesh_path),
        "mesh_bounds_local_m": [mins.tolist(), maxs.tolist()],
        "object_translation_m": translation.tolist(),
    }
    return mask, debug


def _mesh_object_mask_from_pose(
    points: np.ndarray,
    mesh_path: Path,
    object_pose_robot_m: np.ndarray,
    margin: float,
) -> tuple[np.ndarray, dict]:
    transform = np.asarray(object_pose_robot_m, dtype=np.float64)
    if transform.shape != (4, 4):
        raise ValueError(f"object_pose_robot_m must be 4 x 4, got {transform.shape}.")

    rotation = transform[:3, :3]
    translation = transform[:3, 3]
    vertices_local = _load_obj_vertices(mesh_path)
    mins = vertices_local.min(axis=0) - margin
    maxs = vertices_local.max(axis=0) + margin

    points_local = (points - translation) @ rotation
    mask = np.all((points_local >= mins) & (points_local <= maxs), axis=1)
    debug = {
        "mesh_path": str(mesh_path),
        "mesh_bounds_local_m": [mins.tolist(), maxs.tolist()],
        "object_translation_m": translation.tolist(),
    }
    return mask, debug


def _select_pixel_mask(
    *,
    mask_mode: Literal["auto", "instance", "mesh"],
    instance: np.ndarray,
    id_to_labels: dict[str, str],
    object_id: str,
    valid_mask: np.ndarray,
    pcd_map: np.ndarray,
    meta: dict,
    model_root: Path | None,
    margin: float,
) -> tuple[np.ndarray, str, dict]:
    instance_mask, instance_method = _instance_object_mask(instance, id_to_labels, object_id)
    flat_points = pcd_map[valid_mask].reshape(-1, 3)

    if mask_mode == "mesh":
        select_flat, mesh_debug = _mesh_object_mask(
            flat_points, object_id, meta, model_root, margin
        )
        return select_flat, "mesh_bbox", {"instance_method": instance_method, **mesh_debug}

    if mask_mode == "instance":
        select_flat = instance_mask[valid_mask]
        if not select_flat.any():
            raise ValueError(
                "mask=instance found zero object pixels. "
                "Try --mask auto or --mask mesh for this sample."
            )
        return select_flat, instance_method, {"instance_method": instance_method}

    if _mask_is_degenerate(instance_mask, valid_mask):
        select_flat, mesh_debug = _mesh_object_mask(
            flat_points, object_id, meta, model_root, margin
        )
        return select_flat, "mesh_bbox", {"instance_method": instance_method, **mesh_debug}

    select_flat = instance_mask[valid_mask]
    return select_flat, instance_method, {"instance_method": instance_method}


def _select_live_pixel_mask(
    *,
    mask_mode: Literal["auto", "instance", "mesh"],
    instance: np.ndarray,
    id_to_labels: dict[str, str],
    object_id: str,
    valid_mask: np.ndarray,
    pcd_map: np.ndarray,
    object_pose_robot_m: np.ndarray | None,
    mesh_path: Path | None,
    margin: float,
) -> tuple[np.ndarray, str, dict]:
    instance_mask, instance_method = _instance_object_mask(instance, id_to_labels, object_id)
    flat_points = pcd_map[valid_mask].reshape(-1, 3)

    def _mesh_fallback() -> tuple[np.ndarray, str, dict]:
        if object_pose_robot_m is None or mesh_path is None:
            raise ValueError(
                "mesh mask requires object_pose_robot_m and mesh_path for live observations."
            )
        select_flat, mesh_debug = _mesh_object_mask_from_pose(
            flat_points,
            Path(mesh_path),
            object_pose_robot_m,
            margin,
        )
        return select_flat, "mesh_bbox", {"instance_method": instance_method, **mesh_debug}

    if mask_mode == "mesh":
        return _mesh_fallback()

    if mask_mode == "instance":
        select_flat = instance_mask[valid_mask]
        if not select_flat.any():
            raise ValueError(
                "mask=instance found zero object pixels. "
                "Try mask_mode='auto' or 'mesh' for this observation."
            )
        return select_flat, instance_method, {"instance_method": instance_method}

    if _mask_is_degenerate(instance_mask, valid_mask):
        return _mesh_fallback()

    select_flat = instance_mask[valid_mask]
    return select_flat, instance_method, {"instance_method": instance_method}


def _attach_point_colors(
    payload: dict,
    *,
    color_mode: Literal["instance", "rgb", "none"],
    sample_dir: Path,
    camera_name: str,
    instance: np.ndarray,
    pixel_mask: np.ndarray,
) -> None:
    if color_mode == "none":
        return

    if color_mode == "instance":
        instance_color_image = _instance_segmentation_color_image(instance)
        payload["instance_ids"] = instance[pixel_mask].reshape(-1).astype(np.int32)
        payload["colors"] = _colors_at_mask(instance_color_image, pixel_mask)
        payload["color_source"] = "instance_segmentation"
        return

    if cv2 is None:
        raise RuntimeError("OpenCV is required for color=rgb.")
    rgb_path = sample_dir / camera_name / "rgb.png"
    if not rgb_path.exists():
        raise FileNotFoundError(f"RGB image not found for color=rgb: {rgb_path}")
    rgb = cv2.cvtColor(cv2.imread(str(rgb_path)), cv2.COLOR_BGR2RGB)
    payload["colors"] = rgb[pixel_mask].reshape(-1, 3).astype(np.uint8)
    payload["color_source"] = "rgb"


def extract_partial_pc_from_obs(
    depth: np.ndarray,
    instance: np.ndarray,
    pcd_map: np.ndarray,
    *,
    id_to_labels: dict[str, str] | None,
    object_id: str,
    object_pose_robot_m: np.ndarray | None = None,
    mesh_path: Path | str | None = None,
    margin: float = 0.015,
    depth_max: float = 2.0,
    mask_mode: Literal["auto", "instance", "mesh"] = "auto",
    color_mode: Literal["instance", "none"] = "none",
) -> dict:
    """Return an object-only partial point cloud from live camera tensors."""
    if mask_mode not in MASK_MODES:
        raise ValueError(f"mask_mode must be one of {MASK_MODES}, got '{mask_mode}'.")
    if color_mode not in ("instance", "none"):
        raise ValueError("live extraction supports color_mode='instance' or 'none'.")

    depth = np.squeeze(np.asarray(depth))
    instance = np.squeeze(np.asarray(instance))
    pcd_map = np.asarray(pcd_map)
    if pcd_map.ndim != 3 or pcd_map.shape[-1] != 3:
        raise ValueError(f"pcd_map must have shape H x W x 3, got {pcd_map.shape}.")

    valid_mask = np.isfinite(depth) & (depth > 0.0) & (depth < depth_max)
    valid_mask &= np.isfinite(pcd_map).all(axis=-1)

    select_flat, mask_method, mask_debug = _select_live_pixel_mask(
        mask_mode=mask_mode,
        instance=instance,
        id_to_labels=id_to_labels or {},
        object_id=object_id,
        valid_mask=valid_mask,
        pcd_map=pcd_map,
        object_pose_robot_m=object_pose_robot_m,
        mesh_path=Path(mesh_path) if mesh_path is not None else None,
        margin=margin,
    )

    flat_points = pcd_map[valid_mask].reshape(-1, 3)
    object_points = flat_points[select_flat].astype(np.float32)
    pixel_mask = np.zeros(instance.shape, dtype=bool)
    pixel_mask[valid_mask] = select_flat

    payload: dict = {
        "partial_pc": object_points,
        "object_id": object_id,
        "camera": "live",
        "frame": "robot_base",
        "num_points": int(object_points.shape[0]),
        "mask_mode": mask_mode,
        "mask_method": mask_method,
        "mask_debug": mask_debug,
        "color_mode": color_mode,
        "instance_mask": pixel_mask,
    }
    if color_mode == "instance":
        instance_color_image = _instance_segmentation_color_image(instance)
        payload["instance_ids"] = instance[pixel_mask].reshape(-1).astype(np.int32)
        payload["colors"] = _colors_at_mask(instance_color_image, pixel_mask)
        payload["color_source"] = "instance_segmentation"
    return payload


def extract_object_partial_pc(
    sample_dir: Path,
    camera_name: str,
    object_id: str | None = None,
    model_root: Path | None = None,
    margin: float = 0.015,
    depth_max: float = 2.0,
    mask_mode: Literal["auto", "instance", "mesh"] = "auto",
    color_mode: Literal["instance", "rgb", "none"] = "instance",
) -> dict:
    """Return an object-only partial point cloud for one camera view."""
    if mask_mode not in MASK_MODES:
        raise ValueError(f"mask_mode must be one of {MASK_MODES}, got '{mask_mode}'.")
    if color_mode not in COLOR_MODES:
        raise ValueError(f"color_mode must be one of {COLOR_MODES}, got '{color_mode}'.")

    sample_dir = Path(sample_dir)
    camera_name = resolve_camera_name(camera_name)
    meta = _load_meta(sample_dir)
    object_id = object_id or meta["object_id"]

    camera_dir = sample_dir / camera_name
    depth = np.squeeze(np.load(camera_dir / "depth.npy"))
    instance = np.squeeze(np.load(camera_dir / "instance.npy"))
    pcd_map = np.load(camera_dir / "pcd_map.npy")
    normals_path = camera_dir / "normals.npy"
    normals_map = np.load(normals_path) if normals_path.exists() else None

    camera_meta = meta["cameras"][camera_name]
    valid_mask = np.isfinite(depth) & (depth > 0.0) & (depth < depth_max)
    valid_mask &= np.isfinite(pcd_map).all(axis=-1)

    select_flat, mask_method, mask_debug = _select_pixel_mask(
        mask_mode=mask_mode,
        instance=instance,
        id_to_labels=camera_meta.get("id_to_labels") or {},
        object_id=object_id,
        valid_mask=valid_mask,
        pcd_map=pcd_map,
        meta=meta,
        model_root=model_root,
        margin=margin,
    )

    flat_points = pcd_map[valid_mask].reshape(-1, 3)
    object_points = flat_points[select_flat].astype(np.float32)

    pixel_mask = np.zeros(instance.shape, dtype=bool)
    pixel_mask[valid_mask] = select_flat

    payload: dict = {
        "partial_pc": object_points,
        "object_id": object_id,
        "camera": camera_name,
        "frame": meta["frames"]["pcd_map"],
        "num_points": int(object_points.shape[0]),
        "mask_mode": mask_mode,
        "mask_method": mask_method,
        "mask_debug": mask_debug,
        "color_mode": color_mode,
        "source_sample": str(sample_dir),
        "camera_pose_7d": np.asarray(camera_meta["pose_7d"], dtype=np.float64),
        "intrinsics_3x3": np.asarray(camera_meta["intrinsics_3x3"], dtype=np.float64),
        "instance_mask": pixel_mask,
    }

    if normals_map is not None:
        payload["normals"] = normals_map[pixel_mask].reshape(-1, 3).astype(np.float32)

    _attach_point_colors(
        payload,
        color_mode=color_mode,
        sample_dir=sample_dir,
        camera_name=camera_name,
        instance=instance,
        pixel_mask=pixel_mask,
    )
    return payload
