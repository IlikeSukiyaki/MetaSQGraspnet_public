"""Parallel-jaw gripper profiles for antipodal grasp generation.

The sampler stores a hardware-agnostic contact representation.  A profile
supplies the gripper width limits, the contact-to-command-frame offset, and
the collision geometry used to validate candidate grasps.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import os
from pathlib import Path

import numpy as np
import torch
import trimesh
import transforms3d.euler


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _normalize(vec: np.ndarray) -> np.ndarray:
    return vec / (np.linalg.norm(vec) + 1e-12)


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


def _transform(translation_cm, rpy=(0.0, 0.0, 0.0)) -> np.ndarray:
    mat = np.eye(4, dtype=float)
    mat[:3, :3] = transforms3d.euler.euler2mat(*rpy)
    mat[:3, 3] = np.asarray(translation_cm, dtype=float)
    return mat


@lru_cache(maxsize=16)
def _load_mesh(path: str, scale_cm: bool = True) -> trimesh.Trimesh:
    mesh = trimesh.load(path, force="mesh")
    if scale_cm:
        mesh = mesh.apply_scale(100.0)
    return mesh


@dataclass(frozen=True)
class ParallelJawProfile:
    name: str
    width_max_cm: float
    pregrasp_extra_cm: float
    contact_to_ee_offset_cm: float
    mesh_root: Path | None = None

    def convert_to_gripper_6d(
        self,
        vec_a: np.ndarray,
        vec_b: np.ndarray,
        contact_pt: np.ndarray,
        width: float,
    ) -> np.ndarray:
        """Convert one contact-width grasp into the profile command frame."""
        vec_a = _normalize(np.asarray(vec_a, dtype=float))
        vec_b = _normalize(np.asarray(vec_b, dtype=float))
        contact_pt = np.asarray(contact_pt, dtype=float)
        c_vec = _normalize(np.cross(vec_a, vec_b))
        rotation = np.array([vec_b, c_vec, vec_a], dtype=float).T
        midpoint = contact_pt + width / 2 * vec_b
        translation = midpoint - self.contact_to_ee_offset_cm * vec_a

        transform = np.eye(4, dtype=float)
        transform[:3, :3] = rotation
        transform[:3, 3] = translation
        return transform

    def gripper_pose_cm_to_ee_pose(
        self,
        gripper_pose_cm: np.ndarray,
        device: torch.device | str = "cpu",
    ) -> torch.Tensor:
        """Convert sampler frame to the Isaac command frame.

        The state machine backs away along EE local +Y.  The sampler uses local
        +Z as the approach direction, so this remaps columns to
        [baseline, approach, baseline x approach].
        """
        baseline = _normalize(gripper_pose_cm[:3, 0])
        approach = _normalize(gripper_pose_cm[:3, 2])
        up = _normalize(np.cross(baseline, approach))
        rotation = np.stack([baseline, approach, up], axis=1)
        quat = matrix_to_quaternion_wxyz(rotation)
        translation_m = gripper_pose_cm[:3, 3] / 100.0
        pose = np.concatenate([translation_m, quat], axis=0)
        return torch.tensor(pose, dtype=torch.float32, device=device)

    def build_collision_manager(self, grasp_width_cm: float, transform: np.ndarray):
        return _procedural_parallel_jaw(
            grasp_width_cm=grasp_width_cm,
            transform=transform,
            width_max_cm=self.width_max_cm,
            contact_to_ee_offset_cm=self.contact_to_ee_offset_cm,
        )


@dataclass(frozen=True)
class HandEProfile(ParallelJawProfile):
    """Robotiq Hand-E profile for the UR10e `hande_end` command frame."""

    def __init__(self, mesh_root: str | Path | None = None):
        root = (
            Path(mesh_root).expanduser().resolve()
            if mesh_root is not None
            else _repo_root() / "models" / "Gripper" / "hande" / "robotiq_hande_description" / "meshes"
        )
        super().__init__(
            name="hand_e",
            width_max_cm=5.0,
            pregrasp_extra_cm=1.0,
            # `hande_end` is the fingertip command frame in the shipped URDF.
            contact_to_ee_offset_cm=0.0,
            mesh_root=root,
        )

    def build_collision_manager(self, grasp_width_cm: float, transform: np.ndarray):
        if os.environ.get("HAND_E_USE_STL_COLLISION") != "1":
            return _procedural_hande(grasp_width_cm, transform)
        try:
            return _hande_mesh_collision_manager(self.mesh_root, grasp_width_cm, transform)
        except BaseException:
            return _procedural_hande(grasp_width_cm, transform)


@dataclass(frozen=True)
class FrankaProfile(ParallelJawProfile):
    """Legacy Franka-style profile kept for old generated datasets."""

    def __init__(self, mesh_root: str | Path | None = None):
        root = (
            Path(mesh_root).expanduser().resolve()
            if mesh_root is not None
            else _repo_root() / "models"
        )
        super().__init__(
            name="franka",
            width_max_cm=8.0,
            pregrasp_extra_cm=1.0,
            contact_to_ee_offset_cm=10.41,
            mesh_root=root,
        )

    def build_collision_manager(self, grasp_width_cm: float, transform: np.ndarray):
        try:
            return _franka_collision_manager(self.mesh_root, grasp_width_cm, transform)
        except BaseException:
            return _procedural_parallel_jaw(
                grasp_width_cm=grasp_width_cm,
                transform=transform,
                width_max_cm=self.width_max_cm,
                contact_to_ee_offset_cm=self.contact_to_ee_offset_cm,
            )


def get_profile(name: str | ParallelJawProfile | None = None, mesh_root: str | Path | None = None) -> ParallelJawProfile:
    if isinstance(name, ParallelJawProfile):
        return name
    profile_name = (name or "hand_e").lower().replace("-", "_")
    if profile_name in {"hand_e", "hande", "robotiq_hande"}:
        return HandEProfile(mesh_root=mesh_root)
    if profile_name in {"franka", "panda", "franka_panda"}:
        return FrankaProfile(mesh_root=mesh_root)
    raise ValueError(f"Unknown gripper profile: {name}")


def _hande_mesh_collision_manager(root: Path | None, grasp_width_cm: float, transform: np.ndarray):
    if root is None:
        raise FileNotFoundError("Hand-E mesh root is not configured.")
    root = Path(root)
    body_mesh = _load_mesh(str(root / "hande.stl")).copy()
    finger_mesh = _load_mesh(str(root / "finger_collision.stl")).copy()

    manager = trimesh.collision.CollisionManager()
    # The command frame is `hande_end`; the body link is 9.5 cm behind it.
    body_tf = _transform((0.0, 0.0, -9.66), (0.0, -np.pi / 2, -np.pi / 2))
    manager.add_object("hande_body", body_mesh, transform @ body_tf)

    finger_z_cm = -6.05
    left_tf = _transform((-grasp_width_cm / 2.0, 0.0, finger_z_cm), (0.0, 0.0, np.pi))
    right_tf = _transform((grasp_width_cm / 2.0, 0.0, finger_z_cm))
    manager.add_object("finger_left", finger_mesh, transform @ left_tf)
    manager.add_object("finger_right", finger_mesh, transform @ right_tf)
    return manager


def _franka_collision_manager(root: Path | None, grasp_width_cm: float, transform: np.ndarray):
    if root is None:
        raise FileNotFoundError("Franka mesh root is not configured.")
    root = Path(root)
    hand_mesh = _load_mesh(str(root / "hand_collision.stl")).copy()
    finger_left_mesh = _load_mesh(str(root / "finger_collision_left.stl")).copy()
    finger_right_mesh = _load_mesh(str(root / "finger_collision_right.stl")).copy()

    manager = trimesh.collision.CollisionManager()
    manager.add_object("hand", hand_mesh, transform)
    left_tf = _transform((-grasp_width_cm / 2.0, 0.0, 5.81))
    right_tf = _transform((grasp_width_cm / 2.0, 0.0, 5.81))
    manager.add_object("finger_left", finger_left_mesh, transform @ left_tf)
    manager.add_object("finger_right", finger_right_mesh, transform @ right_tf)
    return manager


def _procedural_hande(grasp_width_cm: float, transform: np.ndarray):
    manager = trimesh.collision.CollisionManager()
    body = trimesh.creation.cylinder(radius=3.75, height=9.88, sections=24)
    finger_left = trimesh.creation.box(extents=(0.7, 1.0, 6.0))
    finger_right = trimesh.creation.box(extents=(0.7, 1.0, 6.0))

    body_tf = _transform((0.0, 0.0, -9.5))
    left_tf = _transform((-grasp_width_cm / 2.0, 0.0, -3.0))
    right_tf = _transform((grasp_width_cm / 2.0, 0.0, -3.0))
    manager.add_object("hande_body", body, transform @ body_tf)
    manager.add_object("finger_left", finger_left, transform @ left_tf)
    manager.add_object("finger_right", finger_right, transform @ right_tf)
    return manager


def _procedural_parallel_jaw(
    grasp_width_cm: float,
    transform: np.ndarray,
    width_max_cm: float,
    contact_to_ee_offset_cm: float,
):
    manager = trimesh.collision.CollisionManager()
    palm = trimesh.creation.box(extents=(width_max_cm + 2.0, 1.0, 1.2))
    finger_left = trimesh.creation.box(extents=(0.8, 0.8, max(contact_to_ee_offset_cm, 5.5)))
    finger_right = trimesh.creation.box(extents=(0.8, 0.8, max(contact_to_ee_offset_cm, 5.5)))

    palm_tf = _transform((0.0, 0.0, -contact_to_ee_offset_cm - 0.6))
    finger_z = -max(contact_to_ee_offset_cm / 2.0, 2.75)
    left_tf = _transform((-grasp_width_cm / 2.0, 0.0, finger_z))
    right_tf = _transform((grasp_width_cm / 2.0, 0.0, finger_z))
    manager.add_object("palm", palm, transform @ palm_tf)
    manager.add_object("finger_left", finger_left, transform @ left_tf)
    manager.add_object("finger_right", finger_right, transform @ right_tf)
    return manager
