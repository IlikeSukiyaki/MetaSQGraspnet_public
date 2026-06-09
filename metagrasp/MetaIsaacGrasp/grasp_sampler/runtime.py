"""Runtime helpers for classical antipodal grasp selection.

The offline sampler stores grasps in centimeters as
``[approach(3), baseline(3), contact_point(3), width]``.  This module loads
those HDF5 grasps, checks them against the current Isaac scene in Trimesh, and
returns an end-effector command pose in meters.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Iterable

import h5py
import numpy as np
import torch
import trimesh

from grasp_sampler.database_paths import legacy_models_ifl_roots, resolve_grasp_database_root
from grasp_sampler.gripper_profiles import ParallelJawProfile, get_profile


@dataclass(frozen=True)
class GraspSelection:
    """Selected grasp in both Isaac command and debug forms."""

    pose: torch.Tensor
    score: float
    candidate_count: int
    contact_pose_cm: np.ndarray
    gripper_pose_cm: np.ndarray
    width_cm: float
    approach_alignment: float
    grasp_config: np.ndarray


def resolve_mesh_root(mesh_root: str | Path | None = None) -> Path:
    """Resolve the object asset root used by the sampler and Isaac task."""
    if mesh_root is not None:
        return Path(mesh_root).expanduser().resolve()

    database_root = resolve_grasp_database_root()
    if database_root.exists():
        return database_root

    for legacy_root in legacy_models_ifl_roots():
        if legacy_root.exists():
            return legacy_root

    return database_root


def object_mesh_path(obj_id: str, mesh_root: str | Path | None = None) -> Path:
    return resolve_mesh_root(mesh_root) / obj_id / "textured.obj"


def object_hdf5_path(
    obj_id: str,
    mesh_root: str | Path | None = None,
    hdf5_suffix: str = "",
) -> Path:
    return resolve_mesh_root(mesh_root) / obj_id / f"textured.obj{hdf5_suffix}.hdf5"


@lru_cache(maxsize=32)
def load_mesh_grasps(
    obj_id: str,
    mesh_root: str | Path | None = None,
    hdf5_suffix: str = "",
) -> tuple[trimesh.Trimesh, dict]:
    """Load a scaled object mesh and its HDF5 antipodal grasps."""
    mesh_path = object_mesh_path(obj_id, mesh_root)
    hdf5_path = object_hdf5_path(obj_id, mesh_root, hdf5_suffix=hdf5_suffix)
    if not mesh_path.exists():
        raise FileNotFoundError(f"Missing object mesh: {mesh_path}")
    if not hdf5_path.exists():
        raise FileNotFoundError(
            f"Missing antipodal grasp file: {hdf5_path}. "
            "Run tools/generate_object_grasps.py first."
        )

    mesh = trimesh.load(str(mesh_path))
    mesh = mesh.apply_scale(100)  # meters -> centimeters

    with h5py.File(str(hdf5_path), "r") as f:
        grasps = np.array(f["grasps"]["paralleljaw"]["pregrasp_transform"])
        scores = np.array(f["grasps"]["paralleljaw"]["quality_score"])

    return mesh, {
        "paralleljaw_pregrasp_transform": grasps,
        "paralleljaw_pregrasp_score": scores,
    }


def convert_to_gripper_6d(
    vec_a: np.ndarray,
    vec_b: np.ndarray,
    contact_pt: np.ndarray,
    width: float,
    profile: str | ParallelJawProfile | None = None,
) -> np.ndarray:
    """Convert a contact representation to the active gripper frame."""
    return get_profile(profile).convert_to_gripper_6d(vec_a, vec_b, contact_pt, width)


def convert_to_franka_6d(vec_a: np.ndarray, vec_b: np.ndarray, contact_pt: np.ndarray, width: float) -> np.ndarray:
    """Legacy wrapper for callers that still expect the Franka profile."""
    return get_profile("franka").convert_to_gripper_6d(vec_a, vec_b, contact_pt, width)


def convert_to_contact_6d(vec_a: np.ndarray, vec_b: np.ndarray, contact_pt: np.ndarray) -> np.ndarray:
    c_vec = np.cross(vec_a, vec_b)
    transform = np.eye(4, dtype=float)
    transform[:3, :3] = np.array([vec_b, c_vec, vec_a], dtype=float).T
    transform[:3, 3] = contact_pt
    return transform


def get_parallel_gripper_collision_mesh(
    collision_root: str | Path | None,
    grasp_width: float,
    transform: np.ndarray,
    profile: str | ParallelJawProfile | None = None,
):
    return get_profile(profile, mesh_root=collision_root).build_collision_manager(grasp_width, transform)


def quaternion_wxyz_to_matrix(quat: np.ndarray) -> np.ndarray:
    quat = np.asarray(quat, dtype=float)
    quat = quat / (np.linalg.norm(quat) + 1e-12)
    w, x, y, z = quat
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=float,
    )


def matrix_to_quaternion_wxyz(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=float)
    trace = np.trace(matrix)
    if trace > 0:
        scale = np.sqrt(trace + 1.0) * 2
        w = 0.25 * scale
        x = (matrix[2, 1] - matrix[1, 2]) / scale
        y = (matrix[0, 2] - matrix[2, 0]) / scale
        z = (matrix[1, 0] - matrix[0, 1]) / scale
    else:
        major = int(np.argmax(np.diag(matrix)))
        if major == 0:
            scale = np.sqrt(1.0 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2]) * 2
            w = (matrix[2, 1] - matrix[1, 2]) / scale
            x = 0.25 * scale
            y = (matrix[0, 1] + matrix[1, 0]) / scale
            z = (matrix[0, 2] + matrix[2, 0]) / scale
        elif major == 1:
            scale = np.sqrt(1.0 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2]) * 2
            w = (matrix[0, 2] - matrix[2, 0]) / scale
            x = (matrix[0, 1] + matrix[1, 0]) / scale
            y = 0.25 * scale
            z = (matrix[1, 2] + matrix[2, 1]) / scale
        else:
            scale = np.sqrt(1.0 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1]) * 2
            w = (matrix[1, 0] - matrix[0, 1]) / scale
            x = (matrix[0, 2] + matrix[2, 0]) / scale
            y = (matrix[1, 2] + matrix[2, 1]) / scale
            z = 0.25 * scale

    quat = np.array([w, x, y, z], dtype=float)
    return quat / (np.linalg.norm(quat) + 1e-12)


def pose_vector_to_matrix_cm(pose: torch.Tensor | np.ndarray) -> np.ndarray:
    """Build a 4x4 object pose matrix (cm) from a robot-base pose vector."""
    pose_np = pose.detach().cpu().numpy() if isinstance(pose, torch.Tensor) else np.asarray(pose)
    transform = np.eye(4, dtype=float)
    transform[:3, :3] = quaternion_wxyz_to_matrix(pose_np[3:7])
    transform[:3, 3] = pose_np[:3] * 100.0
    return transform


def build_scene_collision_manager(
    object_mesh_cm: trimesh.Trimesh,
    object_pose_cm: np.ndarray,
    desk_center_m: Iterable[float],
    table_extents_cm: tuple[float, float, float] = (120.0, 80.0, 4.0),
    robot_root_pos_m: Iterable[float] | None = None,
):
    """Build a collision scene in the same frame as ``object_pose_cm``.

    Isaac runtime object poses are expressed in the robot-base frame.  The task
    configuration's ``desk_center`` is already in that same frame, so
    ``robot_root_pos_m`` is accepted only for API compatibility.
    """
    manager = trimesh.collision.CollisionManager()

    desk_center_m = np.asarray(tuple(desk_center_m), dtype=float)
    desk_center_cm = desk_center_m * 100.0
    table_top_z_cm = desk_center_cm[2]
    ground_center_z_cm = table_top_z_cm - 5.0

    ground = trimesh.creation.box(extents=(300.0, 300.0, 2.0))
    ground_tf = np.eye(4)
    ground_tf[2, 3] = ground_center_z_cm
    manager.add_object("ground", ground, ground_tf)

    table = trimesh.creation.box(extents=table_extents_cm)
    table_tf = np.eye(4)
    table_tf[:3, 3] = [
        desk_center_cm[0],
        desk_center_cm[1],
        table_top_z_cm - table_extents_cm[2] / 2,
    ]
    manager.add_object("table", table, table_tf)

    obj_mesh = object_mesh_cm.copy()
    obj_mesh.apply_transform(object_pose_cm)
    manager.add_object("target_object", obj_mesh, np.eye(4))
    return manager


def _check_grasp_collision(
    grasp_config: np.ndarray,
    score: float,
    object_pose_cm: np.ndarray,
    collision_manager,
    collision_root: str | Path | None,
    profile: str | ParallelJawProfile | None = None,
    approach_distances_cm: tuple[float, ...] = (0.0, 10.0),
) -> tuple[bool, np.ndarray, np.ndarray, float]:
    gripper_profile = get_profile(profile, mesh_root=collision_root)
    vec_a = np.asarray(grasp_config[0:3], dtype=float)
    vec_b = np.asarray(grasp_config[3:6], dtype=float)
    contact_pt = np.asarray(grasp_config[6:9], dtype=float)
    width = float(grasp_config[9])
    if width <= 0 or width > gripper_profile.width_max_cm:
        return False, np.eye(4), np.eye(4), width

    gripper_pose = object_pose_cm @ gripper_profile.convert_to_gripper_6d(vec_a, vec_b, contact_pt, width)
    contact_pose = object_pose_cm @ convert_to_contact_6d(vec_a, vec_b, contact_pt)
    pregrasp_width = min(width + gripper_profile.pregrasp_extra_cm, gripper_profile.width_max_cm)

    for distance in approach_distances_cm:
        artificial_contact = contact_pt + vec_a * (-1) * distance
        approach_pose = object_pose_cm @ gripper_profile.convert_to_gripper_6d(
            vec_a,
            vec_b,
            artificial_contact,
            width,
        )
        gripper = gripper_profile.build_collision_manager(pregrasp_width, approach_pose)
        if collision_manager.in_collision_other(gripper, return_names=False, return_data=False):
            return False, gripper_pose, contact_pose, width

    for curr_width in np.linspace(pregrasp_width, min(width + gripper_profile.pregrasp_extra_cm, gripper_profile.width_max_cm), 2):
        gripper = gripper_profile.build_collision_manager(curr_width, gripper_pose)
        if collision_manager.in_collision_other(gripper, return_names=False, return_data=False):
            return False, gripper_pose, contact_pose, width

    return True, gripper_pose, contact_pose, width


def filter_grasps_for_pose(
    object_mesh_cm: trimesh.Trimesh,
    grasp_dict: dict,
    object_pose_cm: np.ndarray,
    desk_center_m: Iterable[float],
    collision_root: str | Path | None = None,
    max_candidates: int = 500,
    profile: str | ParallelJawProfile | None = None,
    robot_root_pos_m: Iterable[float] | None = None,
) -> list[tuple[float, np.ndarray, np.ndarray, float, np.ndarray]]:
    gripper_profile = get_profile(profile, mesh_root=collision_root)
    grasps = np.asarray(grasp_dict["paralleljaw_pregrasp_transform"])
    scores = np.asarray(grasp_dict["paralleljaw_pregrasp_score"], dtype=float)
    if len(grasps) == 0:
        return []

    valid_idx = np.where(
        (grasps[:, 9] > 0.0)
        & (grasps[:, 9] <= gripper_profile.width_max_cm)
        & (scores > 0.0)
    )[0]
    if len(valid_idx) == 0:
        return []

    order = valid_idx[np.argsort(scores[valid_idx])[::-1]][:max_candidates]
    collision_manager = build_scene_collision_manager(
        object_mesh_cm,
        object_pose_cm,
        desk_center_m,
        robot_root_pos_m=robot_root_pos_m,
    )

    selections = []
    for idx in order:
        raw_score = float(scores[idx])
        is_free, gripper_pose, contact_pose, width = _check_grasp_collision(
            grasps[idx],
            raw_score,
            object_pose_cm,
            collision_manager,
            collision_root,
            profile=gripper_profile,
        )
        if is_free:
            selections.append((raw_score, gripper_pose, contact_pose, width, grasps[idx]))

    return selections


def gripper_pose_cm_to_ee_pose(
    gripper_pose_cm: np.ndarray,
    device: torch.device | str = "cpu",
    profile: str | ParallelJawProfile | None = None,
) -> torch.Tensor:
    """Convert sampler frame to the active profile's Isaac command pose."""
    return get_profile(profile).gripper_pose_cm_to_ee_pose(gripper_pose_cm, device=device)


def topdown_alignment(
    gripper_pose_cm: np.ndarray,
    down_axis: Iterable[float] = (0.0, 0.0, -1.0),
) -> float:
    """Return 1.0 for a straight top-down approach, 0.0 for side approaches."""
    approach = gripper_pose_cm[:3, 2]
    approach = approach / (np.linalg.norm(approach) + 1e-12)
    down = np.asarray(tuple(down_axis), dtype=float)
    down = down / (np.linalg.norm(down) + 1e-12)
    return float(np.dot(approach, down))


def select_best_grasp(
    obj_id: str,
    object_pose_cm: np.ndarray,
    desk_center_m: Iterable[float],
    mesh_root: str | Path | None = None,
    collision_root: str | Path | None = None,
    max_candidates: int = 500,
    device: torch.device | str = "cpu",
    profile: str | ParallelJawProfile | None = None,
    robot_root_pos_m: Iterable[float] | None = None,
    rank: int = 0,
    hdf5_suffix: str | None = None,
    min_width_cm: float = 0.0,
    prefer_topdown: bool = True,
    topdown_weight: float = 0.5,
    min_topdown_alignment: float = 0.0,
    down_axis: Iterable[float] = (0.0, 0.0, -1.0),
) -> GraspSelection | None:
    gripper_profile = get_profile(profile, mesh_root=collision_root)
    resolved_hdf5_suffix = gripper_profile.hdf5_suffix if hdf5_suffix is None else hdf5_suffix
    mesh, grasp_dict = load_mesh_grasps(
        obj_id,
        str(resolve_mesh_root(mesh_root)),
        resolved_hdf5_suffix,
    )
    selections = filter_grasps_for_pose(
        mesh,
        grasp_dict,
        object_pose_cm,
        desk_center_m,
        collision_root=collision_root,
        max_candidates=max_candidates,
        profile=gripper_profile,
        robot_root_pos_m=robot_root_pos_m,
    )
    if not selections:
        return None

    pool = [item for item in selections if item[3] >= min_width_cm]
    if not pool:
        pool = list(selections)

    if prefer_topdown and min_topdown_alignment > 0.0:
        topdown_pool = [
            item for item in pool
            if topdown_alignment(item[1], down_axis) >= min_topdown_alignment
        ]
        if topdown_pool:
            pool = topdown_pool

    def _rank_key(item: tuple[float, np.ndarray, np.ndarray, float, np.ndarray]) -> tuple[float, float]:
        raw_score, gripper_pose_cm, _, width, _ = item
        alignment = topdown_alignment(gripper_pose_cm, down_axis)
        combined_score = raw_score
        if prefer_topdown:
            combined_score += topdown_weight * alignment
        return combined_score, width

    ranked = sorted(pool, key=_rank_key, reverse=True)

    deduped = []
    seen_pose_keys = set()
    for item in ranked:
        _, gripper_pose_cm, _, _, _ = item
        pose_key = tuple(np.round(gripper_pose_cm[:3, 3], 2))
        if pose_key in seen_pose_keys:
            continue
        seen_pose_keys.add(pose_key)
        deduped.append(item)

    if rank >= len(deduped):
        return None

    raw_score, gripper_pose_cm, contact_pose_cm, width_cm, grasp_config = deduped[rank]
    score_max = max(item[0] for item in selections) + 1e-12
    alignment = topdown_alignment(gripper_pose_cm, down_axis)
    return GraspSelection(
        pose=gripper_profile.gripper_pose_cm_to_ee_pose(gripper_pose_cm, device=device),
        score=float(raw_score / score_max),
        candidate_count=len(deduped),
        contact_pose_cm=contact_pose_cm,
        gripper_pose_cm=gripper_pose_cm,
        width_cm=width_cm,
        approach_alignment=alignment,
        grasp_config=grasp_config,
    )
