"""Superquadric mesh and in-memory antipodal grasp helpers."""

from __future__ import annotations

from pathlib import Path
import random
import sys

import numpy as np
import trimesh
import transforms3d

from grasp_sampler.gripper_profiles import ParallelJawProfile, get_profile


def _training_root() -> Path:
    return Path(__file__).resolve().parents[1] / "sq_estimation_model" / "training"


def _ensure_training_path() -> None:
    root = _training_root()
    if not root.exists():
        raise FileNotFoundError(f"Missing SQ estimation training package: {root}")
    root_str = str(root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)


def superquadric_trimesh_cm(
    parameters: dict[str, float],
    *,
    resolution: int = 12,
) -> trimesh.Trimesh:
    """Build a local-frame superquadric mesh in centimeters."""
    _ensure_training_path()
    from geometry.primitives import mesh_superquadric

    mesh_o3d = mesh_superquadric(parameters, np.eye(4, dtype=float), resolution=resolution)
    vertices = np.asarray(mesh_o3d.vertices, dtype=np.float64) * 100.0
    faces = np.asarray(mesh_o3d.triangles, dtype=np.int64)
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=True)
    if mesh.face_normals.shape[0] == 0:
        raise ValueError("Generated superquadric mesh has no face normals.")
    return mesh


def extsq_trimesh_cm(
    parameters: dict[str, float],
    *,
    resolution: int = 12,
) -> trimesh.Trimesh:
    """Build a local-frame extended superquadric mesh in centimeters."""
    _ensure_training_path()
    from geometry.primitives import mesh_extended_superquadric

    mesh_o3d = mesh_extended_superquadric(parameters, np.eye(4, dtype=float), resolution=resolution)
    vertices = np.asarray(mesh_o3d.vertices, dtype=np.float64) * 100.0
    faces = np.asarray(mesh_o3d.triangles, dtype=np.int64)
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=True)
    if mesh.face_normals.shape[0] == 0:
        raise ValueError("Generated ExtSQ mesh has no face normals.")
    return mesh


def estimation_trimesh_cm(
    parameters: dict[str, float],
    *,
    resolution: int = 12,
) -> trimesh.Trimesh:
    if {"k", "b", "cos_alpha", "sin_alpha"}.issubset(parameters):
        return extsq_trimesh_cm(parameters, resolution=resolution)
    return superquadric_trimesh_cm(parameters, resolution=resolution)


def _rodrigues_rotation(axis: np.ndarray, angle: float) -> np.ndarray:
    axis = axis / (np.linalg.norm(axis) + 1e-12)
    k = np.array(
        [
            [0.0, -axis[2], axis[1]],
            [axis[2], 0.0, -axis[0]],
            [-axis[1], axis[0], 0.0],
        ],
        dtype=float,
    )
    return np.eye(3) + np.sin(angle) * k + (1.0 - np.cos(angle)) * np.dot(k, k)


def _random_approach(max_rotation: float, max_translation: float) -> tuple[np.ndarray, np.ndarray]:
    rotation = transforms3d.euler.euler2mat(
        ai=random.uniform(-max_rotation, max_rotation),
        aj=random.uniform(-max_rotation, max_rotation),
        ak=random.uniform(-max_rotation, max_rotation),
    )
    translation = np.array(
        [
            random.uniform(-max_translation, max_translation),
            random.uniform(-max_translation, max_translation),
            random.uniform(-max_translation, max_translation),
        ],
        dtype=float,
    )
    return np.dot(rotation, np.array([0.0, 0.0, -1.0])), translation


def sample_antipodal_grasps_on_mesh(
    mesh: trimesh.Trimesh,
    *,
    profile: str | ParallelJawProfile | None = None,
    collision_root: str | Path | None = None,
    num_surface_points: int = 120,
    max_grasps: int = 250,
    samples_per_point: int = 8,
    collision_samples: int = 8,
    max_angle_sampling: float = np.pi / 4,
    max_angle_friction: float = np.pi / 4,
    max_translation: float = 0.02,
    distance_gripper: float = 0.0,
) -> dict[str, np.ndarray]:
    """Sample local-frame antipodal grasps from a trimesh object mesh in cm."""
    gripper_profile = get_profile(profile, mesh_root=collision_root)
    contact_points, contact_indexes = trimesh.sample.sample_surface(mesh, num_surface_points)
    collision_manager = trimesh.collision.CollisionManager()
    collision_manager.add_object("mesh_object", mesh, np.eye(4))

    grasp_configs = []
    grasp_scores = []
    for contact_point, contact_index in zip(contact_points, contact_indexes, strict=False):
        if len(grasp_configs) >= max_grasps:
            break

        successful_candidates = []
        positives = 0
        for _ in range(samples_per_point):
            approach_vec, center = _random_approach(max_angle_sampling, max_translation)
            surface_normal = mesh.face_normals[contact_index]
            align, _ = trimesh.geometry.align_vectors(surface_normal, [0, 0, 1], True)
            approach_vec_rot = np.dot(align.T[:3, :3], approach_vec)
            center_rot = np.dot(align.T[:3, :3], center)
            center_pregrasp = center_rot + contact_point - approach_vec_rot
            faces_idx, _, projected = mesh.ray.intersects_id(
                ray_origins=[center_pregrasp],
                ray_directions=[approach_vec_rot],
                multiple_hits=True,
                return_locations=True,
            )
            if len(projected) < 2:
                continue

            p1 = projected[0]
            p1_idx = faces_idx[0]
            random_idx = random.randint(1, len(projected) - 1)
            p2 = projected[random_idx]
            p2_idx = faces_idx[random_idx]
            contact_axis = p2 - p1
            width = np.linalg.norm(contact_axis)
            if width <= 0.0 or width >= gripper_profile.width_max_cm:
                continue

            contact_axis = contact_axis / width
            p1_angle = trimesh.geometry.vector_angle([-contact_axis, mesh.face_normals[p1_idx]])
            p2_angle = trimesh.geometry.vector_angle([contact_axis, mesh.face_normals[p2_idx]])
            if p1_angle <= max_angle_friction and p2_angle <= max_angle_friction:
                successful_candidates.append((p1, p2, contact_axis))
                positives += 1

        random.shuffle(successful_candidates)
        for p1, p2, contact_axis in successful_candidates:
            if len(grasp_configs) >= max_grasps:
                break
            width = np.linalg.norm(p2 - p1)
            for angle in np.linspace(0.0, 2.0 * np.pi, collision_samples):
                rotation = _rodrigues_rotation(contact_axis, angle)
                if contact_axis[0] == 0 and contact_axis[1] == 0:
                    orthogonal = np.array([1.0, 0.0, 0.0])
                else:
                    orthogonal = np.cross(contact_axis, [0.0, 0.0, 1.0])
                orthogonal = orthogonal / (np.linalg.norm(orthogonal) + 1e-12)
                approach_axis = np.dot(rotation, orthogonal)
                approach_axis = approach_axis / (np.linalg.norm(approach_axis) + 1e-12)
                cross_axis = np.cross(approach_axis, contact_axis)
                grasp_center = p1 + width / 2.0 * contact_axis - distance_gripper * approach_axis
                transform = np.eye(4, dtype=float)
                transform[:3, :3] = np.array([contact_axis, cross_axis, approach_axis]).T
                transform[:3, 3] = grasp_center

                grasp_width = min(
                    width + gripper_profile.pregrasp_extra_cm,
                    gripper_profile.width_max_cm,
                )
                gripper_collision = gripper_profile.build_collision_manager(grasp_width, transform)
                if collision_manager.in_collision_other(gripper_collision, return_names=False, return_data=False):
                    continue

                grasp_configs.append(
                    np.array(
                        [
                            approach_axis[0],
                            approach_axis[1],
                            approach_axis[2],
                            contact_axis[0],
                            contact_axis[1],
                            contact_axis[2],
                            p1[0],
                            p1[1],
                            p1[2],
                            width,
                        ],
                        dtype=np.float32,
                    )
                )
                grasp_scores.append(max(positives / max(samples_per_point, 1), 1e-6))
                break

    return {
        "paralleljaw_pregrasp_transform": np.asarray(grasp_configs, dtype=np.float32).reshape(-1, 10),
        "paralleljaw_pregrasp_score": np.asarray(grasp_scores, dtype=np.float32),
    }
