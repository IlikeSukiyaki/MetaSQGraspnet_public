"""Dataset export helpers for SQ-v0-Data."""

from __future__ import annotations

import importlib.util
import json
import re
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch

from isaac_env.utils import pose_vector_to_transformation_matrix

from .element_cfg import (
    CAM_HEIGHT,
    CAM_WIDTH,
    DEPTH_MAX,
    N_MULTIPLE_CAM,
    OBJ_LABLE,
    ROBOT_NAME,
    TOPDOWN_CAMERA_HEIGHT,
    TOPDOWN_CAMERA_ROLE_NAMES,
    focal_length_cm,
    horizontal_aperture_mm,
)


def _to_cpu_tensor(value: torch.Tensor | None) -> torch.Tensor | None:
    if value is None:
        return None
    return value.detach().cpu()


def _to_numpy(value: torch.Tensor | np.ndarray | None) -> np.ndarray | None:
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _jsonable(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return _jsonable(value.detach().cpu().numpy())
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def _remap_id_to_labels(id_to_label: dict[Any, Any] | None) -> dict[str, str] | None:
    if not id_to_label:
        return None

    labels: dict[str, str] = {}
    for key, value in id_to_label.items():
        label = str(value)
        match = re.search(r"obj_(\d+)", label)
        if match:
            try:
                obj_index = int(match.group(1))
                label = OBJ_LABLE[obj_index]
            except (ValueError, IndexError):
                label = "-1"
        else:
            label = "-1"
        labels[str(key)] = label
    return labels


def _camera_id_to_labels(env, cam_id: int, env_id: int) -> dict[str, str] | None:
    try:
        info = env.camera_info[cam_id][env_id]["instance_segmentation_fast"]
        return _remap_id_to_labels(info["idToLabels"])
    except (AttributeError, KeyError, IndexError, TypeError):
        return None


def _save_rgb(path: Path, rgb: np.ndarray | None) -> None:
    if rgb is None:
        return
    image = np.clip(rgb, 0, 255).astype(np.uint8)
    cv2.imwrite(str(path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))


def _save_depth_png(path: Path, depth: np.ndarray | None) -> None:
    if depth is None:
        return
    depth_2d = np.squeeze(depth)
    depth_2d = np.clip(depth_2d, 0.0, DEPTH_MAX)
    image = (depth_2d / DEPTH_MAX * 255.0).astype(np.uint8)
    cv2.imwrite(str(path), image)


def _save_instance_png(path: Path, instance: np.ndarray | None) -> None:
    if instance is None:
        return
    labels = np.squeeze(instance).astype(np.uint8)
    image = cv2.applyColorMap((labels * 40) % 255, cv2.COLORMAP_JET)
    cv2.imwrite(str(path), image)


def _active_object_id(env, env_id: int, scene_obj_id: list[str]) -> str:
    chosen = int(env.obj_chosen[env_id].detach().cpu().item())
    if 0 <= chosen < len(OBJ_LABLE):
        return OBJ_LABLE[chosen]
    if scene_obj_id:
        return scene_obj_id[0]
    return OBJ_LABLE[0]


def _collect_object_poses(env, env_id: int, active_object_id: str | None = None) -> tuple[torch.Tensor | None, list[str]]:
    poses = []
    scene_obj_id = []
    for obj_id in range(env.num_objs):
        label = OBJ_LABLE[obj_id]
        if active_object_id is not None and label != active_object_id:
            continue
        pose = env._get_obj_pose(obj_id, env_id).detach().cpu()
        if pose[2] > -5e-2:
            pose_cm = pose.clone()
            pose_cm[:3] *= 100
            poses.append(pose_vector_to_transformation_matrix(pose_cm))
            scene_obj_id.append(label)

    if not poses:
        return None, scene_obj_id
    return torch.stack(poses), scene_obj_id


def _load_partial_pc_extractor():
    module_path = Path(__file__).resolve().parents[2] / "tools" / "sq_partial_pc.py"
    spec = importlib.util.spec_from_file_location("sq_partial_pc", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load partial point cloud helper from {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules.setdefault("sq_partial_pc", module)
    spec.loader.exec_module(module)
    return module.extract_object_partial_pc


def _save_partial_point_clouds(sample_dir: Path, object_id: str) -> dict[str, Any]:
    extract_object_partial_pc = _load_partial_pc_extractor()
    partial_dir = sample_dir / "partial_pc"
    partial_dir.mkdir(parents=True, exist_ok=True)
    partial_meta: dict[str, Any] = {}

    for role_name in TOPDOWN_CAMERA_ROLE_NAMES.values():
        payload = extract_object_partial_pc(
            sample_dir,
            role_name,
            object_id=object_id,
            mask_mode="auto",
            color_mode="instance",
        )
        save_payload = {key: value for key, value in payload.items() if key != "instance_mask"}
        output_name = f"{role_name}_auto_instance.npy"
        np.save(partial_dir / output_name, save_payload)
        partial_meta[role_name] = {
            "file": output_name,
            "num_points": payload["num_points"],
            "mask_mode": payload["mask_mode"],
            "mask_method": payload["mask_method"],
            "color_mode": payload["color_mode"],
        }

    return partial_meta


def save_scene_snapshot(env, env_id: int, obs_buf: dict[str, Any], output_dir: str | Path) -> Path:
    """Save one stable SQ scene with all camera tensors and metadata."""
    policy = obs_buf["policy"] if "policy" in obs_buf else obs_buf
    episode = int(env.epi_step_count[env_id, 0].detach().cpu().item())

    obj_poses_robot, scene_obj_id = _collect_object_poses(env, env_id)
    object_id = _active_object_id(env, env_id, scene_obj_id)
    obj_poses_robot, scene_obj_id = _collect_object_poses(env, env_id, active_object_id=object_id)

    sample_dir = Path(output_dir) / object_id / f"epi_{episode:06d}"
    sample_dir.mkdir(parents=True, exist_ok=True)
    scene_bundle: dict[str, Any] = {}
    camera_meta: dict[str, Any] = {}
    valid_pixel_counts: dict[str, int] = {}

    for cam_id in range(N_MULTIPLE_CAM):
        camera_key = f"camera_{cam_id}"
        role_name = TOPDOWN_CAMERA_ROLE_NAMES.get(cam_id, camera_key)
        camera_dir = sample_dir / role_name
        camera_dir.mkdir(parents=True, exist_ok=True)

        rgb = policy.get("rgb", None)
        depth = policy.get("distance_to_image_plane", None)
        normal = policy.get("normals", None)
        instance = policy.get("instance_segmentation_fast", None)
        pcd = policy.get("pcd", None)

        rgb_cam = _to_cpu_tensor(rgb[env_id, cam_id]) if rgb is not None else None
        depth_cam = _to_cpu_tensor(depth[env_id, cam_id]) if depth is not None else None
        normal_cam = _to_cpu_tensor(normal[env_id, cam_id]) if normal is not None else None
        instance_cam = _to_cpu_tensor(instance[env_id, cam_id]) if instance is not None else None
        pcd_cam = _to_cpu_tensor(pcd[env_id, cam_id]) if pcd is not None else None
        intrinsics = _to_cpu_tensor(env.camera[cam_id].data.intrinsic_matrices[env_id])
        camera_pose = _to_cpu_tensor(env.get_camera_pose(cam_id, env_id))
        id_to_labels = _camera_id_to_labels(env, cam_id, env_id)

        np.save(camera_dir / "depth.npy", np.squeeze(_to_numpy(depth_cam)).astype(np.float32))
        np.save(camera_dir / "instance.npy", np.squeeze(_to_numpy(instance_cam)).astype(np.int32))
        np.save(camera_dir / "normals.npy", _to_numpy(normal_cam).astype(np.float32))
        np.save(camera_dir / "pcd_map.npy", _to_numpy(pcd_cam).astype(np.float32))
        _save_rgb(camera_dir / "rgb.png", _to_numpy(rgb_cam))
        _save_depth_png(camera_dir / "depth.png", _to_numpy(depth_cam))
        _save_instance_png(camera_dir / "instance.png", _to_numpy(instance_cam))

        depth_np = np.squeeze(_to_numpy(depth_cam))
        pcd_np = _to_numpy(pcd_cam)
        valid_mask = np.isfinite(depth_np) & (depth_np > 0.0) & (depth_np < DEPTH_MAX)
        if pcd_np is not None:
            valid_mask &= np.isfinite(pcd_np).all(axis=-1)
        valid_pixel_counts[role_name] = int(valid_mask.sum())

        scene_bundle[camera_key] = {
            "camera_intrinsics": intrinsics,
            "camera_pose": camera_pose,
            "rgb": rgb_cam,
            "depth": depth_cam,
            "normal": normal_cam,
            "instance": instance_cam,
            "id_to_labels": id_to_labels,
            "pcd": pcd_cam,
        }
        camera_meta[role_name] = {
            "index": cam_id,
            "role": "onhand" if cam_id == 0 else "topdown",
            "height_m": float(getattr(env, "topdown_height", TOPDOWN_CAMERA_HEIGHT))
            if cam_id == 1
            else None,
            "intrinsics_3x3": _jsonable(intrinsics),
            "pose_7d": _jsonable(camera_pose),
            "pinhole_config": {
                "focal_length_cm": float(focal_length_cm),
                "aperture_mm": float(horizontal_aperture_mm),
                "width": int(CAM_WIDTH),
                "height": int(CAM_HEIGHT),
            },
            "id_to_labels": id_to_labels,
            "valid_pixel_count": valid_pixel_counts[role_name],
        }

    scene_bundle["obj_poses_robot"] = obj_poses_robot
    scene_bundle["obj_id"] = scene_obj_id
    torch.save(scene_bundle, sample_dir / "scene.pt")

    meta = {
        "task": "SQ-v0-Data",
        "object_id": object_id,
        "episode": episode,
        "frames": {
            "robot": ROBOT_NAME,
            "world": "IsaacLab",
            "camera_pose": "robot_base",
            "pcd_map": "robot_base_meters",
            "obj_poses_robot": "robot_base_translation_cm",
        },
        "cameras": camera_meta,
        "obj_poses_robot": _jsonable(obj_poses_robot),
        "obj_id": scene_obj_id,
        "valid_pixel_counts": valid_pixel_counts,
    }
    with (sample_dir / "meta.json").open("w", encoding="utf-8") as file:
        json.dump(meta, file, indent=2)

    partial_meta = _save_partial_point_clouds(sample_dir, object_id)
    meta["partial_pc"] = partial_meta
    with (sample_dir / "meta.json").open("w", encoding="utf-8") as file:
        json.dump(meta, file, indent=2)

    manifest_path = Path(output_dir) / "manifest.jsonl"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_row = {
        "sample_dir": str(sample_dir),
        "object_id": object_id,
        "episode": episode,
        "scene_pt": str(sample_dir / "scene.pt"),
        "meta_json": str(sample_dir / "meta.json"),
        "valid_pixel_counts": valid_pixel_counts,
    }
    with manifest_path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(_jsonable(manifest_row)) + "\n")

    print(f"{sample_dir}: saved SQ scene snapshot")
    return sample_dir
