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

from grasp_sampler.gripper_profile_defaults import (
    FRANKA_HDF5_SUFFIX,
    FRANKA_PREGRASP_EXTRA_CM,
    FRANKA_WIDTH_MAX_CM,
    HAND_E_HDF5_SUFFIX,
    HAND_E_PREGRASP_EXTRA_CM,
    HAND_E_WIDTH_MAX_CM,
    ROBOTIQ_2F85_HDF5_SUFFIX,
    ROBOTIQ_2F85_PREGRASP_EXTRA_CM,
    ROBOTIQ_2F85_WIDTH_MAX_CM,
)


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


def _rotation_z(theta: float) -> np.ndarray:
    c = np.cos(theta)
    s = np.sin(theta)
    return np.array(
        [
            [c, -s, 0.0],
            [s, c, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=float,
    )


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
    hdf5_suffix: str = ""

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
        """Convert sampler frame to the legacy Isaac command frame.

        Legacy environments back away along EE local +Y.  The sampler uses
        local +Z as the approach direction, so this remaps columns to
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

    def __init__(
        self,
        mesh_root: str | Path | None = None,
        width_max_cm: float | None = None,
    ):
        root = (
            Path(mesh_root).expanduser().resolve()
            if mesh_root is not None
            else _repo_root() / "models" / "Gripper" / "hande" / "robotiq_hande_description" / "meshes"
        )
        super().__init__(
            name="hand_e",
            width_max_cm=HAND_E_WIDTH_MAX_CM if width_max_cm is None else float(width_max_cm),
            pregrasp_extra_cm=HAND_E_PREGRASP_EXTRA_CM,
            # `hande_end` is the fingertip command frame in the shipped URDF.
            contact_to_ee_offset_cm=0.0,
            mesh_root=root,
            hdf5_suffix=HAND_E_HDF5_SUFFIX,
        )

    def gripper_pose_cm_to_ee_pose(
        self,
        gripper_pose_cm: np.ndarray,
        device: torch.device | str = "cpu",
    ) -> torch.Tensor:
        """Convert sampler frame to the Hand-E ``hande_end`` command frame.

        The sampler frame stores baseline as local X and approach as local Z.
        The loaded Isaac USD exposes the Hand-E jaw opening on ``hande_end`` +X
        and the palm-to-fingertip approach direction on ``hande_end`` +Y.
        """
        baseline = _normalize(gripper_pose_cm[:3, 0])
        approach = _normalize(gripper_pose_cm[:3, 2])
        side = _normalize(np.cross(baseline, approach))
        rotation = np.stack([baseline, approach, side], axis=1)
        quat = matrix_to_quaternion_wxyz(rotation)
        translation_m = gripper_pose_cm[:3, 3] / 100.0
        pose = np.concatenate([translation_m, quat], axis=0)
        return torch.tensor(pose, dtype=torch.float32, device=device)

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
            width_max_cm=FRANKA_WIDTH_MAX_CM,
            pregrasp_extra_cm=FRANKA_PREGRASP_EXTRA_CM,
            contact_to_ee_offset_cm=10.41,
            mesh_root=root,
            hdf5_suffix=FRANKA_HDF5_SUFFIX,
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


@dataclass(frozen=True)
class Robotiq2F85Profile(ParallelJawProfile):
    """Robotiq 2F85 profile for the UR10e ``tool0`` command frame.

    The initial ``tool0`` offset follows the nominal flange-to-fingertip depth
    for a 2F85-style gripper. Use ``tools/measure_2f85_tool0_frame.py`` to
    confirm this value against the loaded Isaac USD before large-scale runs.
    """

    def __init__(self, mesh_root: str | Path | None = None):
        root = Path(mesh_root).expanduser().resolve() if mesh_root is not None else None
        super().__init__(
            name="2f85",
            width_max_cm=ROBOTIQ_2F85_WIDTH_MAX_CM,
            pregrasp_extra_cm=ROBOTIQ_2F85_PREGRASP_EXTRA_CM,
            contact_to_ee_offset_cm=16.0,
            mesh_root=root,
            hdf5_suffix=ROBOTIQ_2F85_HDF5_SUFFIX,
        )

    def build_collision_manager(self, grasp_width_cm: float, transform: np.ndarray):
        return _procedural_robotiq_2f85(
            grasp_width_cm=grasp_width_cm,
            transform=transform,
            contact_to_ee_offset_cm=self.contact_to_ee_offset_cm,
        )


def get_profile(
    name: str | ParallelJawProfile | None = None,
    mesh_root: str | Path | None = None,
    width_max_cm: float | None = None,
) -> ParallelJawProfile:
    if isinstance(name, ParallelJawProfile):
        return name
    profile_name = (name or "hand_e").lower().replace("-", "_")
    if profile_name in {"hand_e", "hande", "robotiq_hande"}:
        return HandEProfile(mesh_root=mesh_root, width_max_cm=width_max_cm)
    if profile_name in {"2f85", "robotiq_2f85", "robotiq_2f_85"}:
        return Robotiq2F85Profile(mesh_root=mesh_root)
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
    # Collision checks receive the sampler frame (X=jaw baseline, Z=approach).
    # The STL meshes are authored in the URDF Hand-E frame (Y=jaw baseline,
    # Z=approach), so rotate the local mesh frame by -90 deg around Z first.
    hande_frame = transform @ _transform((0.0, 0.0, 0.0), (0.0, 0.0, -np.pi / 2))
    # The command frame is `hande_end`; the body link is 9.5 cm behind it.
    body_tf = _transform((0.0, 0.0, -9.66), (0.0, -np.pi / 2, -np.pi / 2))
    manager.add_object("hande_body", body_mesh, hande_frame @ body_tf)

    finger_z_cm = -6.05
    left_tf = _transform((0.0, -grasp_width_cm / 2.0, finger_z_cm), (0.0, 0.0, np.pi))
    right_tf = _transform((0.0, grasp_width_cm / 2.0, finger_z_cm))
    manager.add_object("finger_left", finger_mesh, hande_frame @ left_tf)
    manager.add_object("finger_right", finger_mesh, hande_frame @ right_tf)
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
    # Procedural collision geometry is intentionally authored in the sampler
    # frame: local X is jaw opening and local Z is the approach direction.
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


def _procedural_robotiq_2f85(
    grasp_width_cm: float,
    transform: np.ndarray,
    contact_to_ee_offset_cm: float,
):
    manager = trimesh.collision.CollisionManager()
    palm = trimesh.creation.box(extents=(10.5, 3.2, 3.0))
    knuckle_left = trimesh.creation.box(extents=(1.5, 2.4, 5.0))
    knuckle_right = trimesh.creation.box(extents=(1.5, 2.4, 5.0))
    finger_left = trimesh.creation.box(extents=(1.0, 1.2, 8.0))
    finger_right = trimesh.creation.box(extents=(1.0, 1.2, 8.0))

    palm_tf = _transform((0.0, 0.0, -contact_to_ee_offset_cm))
    knuckle_z = -contact_to_ee_offset_cm + 3.0
    finger_z = -4.0
    left_x = -grasp_width_cm / 2.0
    right_x = grasp_width_cm / 2.0

    manager.add_object("2f85_palm", palm, transform @ palm_tf)
    manager.add_object("2f85_left_knuckle", knuckle_left, transform @ _transform((left_x, 0.0, knuckle_z)))
    manager.add_object("2f85_right_knuckle", knuckle_right, transform @ _transform((right_x, 0.0, knuckle_z)))
    manager.add_object("2f85_left_finger", finger_left, transform @ _transform((left_x, 0.0, finger_z)))
    manager.add_object("2f85_right_finger", finger_right, transform @ _transform((right_x, 0.0, finger_z)))
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
