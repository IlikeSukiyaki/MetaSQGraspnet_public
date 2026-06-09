"""Classical task constants for UR10e + Robotiq Hand-E."""

from __future__ import annotations

import os

from grasp_sampler.gripper_profile_defaults import (
    HAND_E_HDF5_SUFFIX,
    HAND_E_WIDTH_MAX_CM,
)


HAND_E_TASK_CONFIG = {
    "task_id": "object-v0-Classical",
    "robot_name": "ur10e_hand_e",
    "gripper_profile": "hand_e",
    "hdf5_suffix": HAND_E_HDF5_SUFFIX,
    "width_max_cm": HAND_E_WIDTH_MAX_CM,
    "approach_axis_ee": (0.0, 1.0, 0.0),
    "on_hand": True,
    "n_multiple_cam": 1,
}


def _apply_approach_axis_scalars(namespace: dict, axis: tuple[float, float, float]) -> None:
    namespace["CLASSICAL_APPROACH_AXIS_EE"] = axis
    namespace["CLASSICAL_APPROACH_AXIS_EE_X"] = float(axis[0])
    namespace["CLASSICAL_APPROACH_AXIS_EE_Y"] = float(axis[1])
    namespace["CLASSICAL_APPROACH_AXIS_EE_Z"] = float(axis[2])


def apply_to_element_cfg(namespace: dict) -> None:
    namespace["CLASSICAL_TASK_ID"] = HAND_E_TASK_CONFIG["task_id"]
    namespace["ROBOT_NAME"] = HAND_E_TASK_CONFIG["robot_name"]
    namespace["CLASSICAL_GRIPPER_PROFILE"] = HAND_E_TASK_CONFIG["gripper_profile"]
    namespace["CLASSICAL_HDF5_SUFFIX"] = HAND_E_TASK_CONFIG["hdf5_suffix"]
    namespace["CLASSICAL_GRIPPER_WIDTH_MAX_CM"] = HAND_E_TASK_CONFIG["width_max_cm"]
    _apply_approach_axis_scalars(namespace, HAND_E_TASK_CONFIG["approach_axis_ee"])
    namespace["ON_HAND"] = HAND_E_TASK_CONFIG["on_hand"]
    namespace["N_MULTIPLE_CAM"] = HAND_E_TASK_CONFIG["n_multiple_cam"]
    namespace["CLASSICAL_GRIPPER_COLLISION_ROOT"] = os.path.join(
        namespace["HOME_PATH"],
        "models",
        "Gripper",
        "hande",
        "robotiq_hande_description",
        "meshes",
    )
