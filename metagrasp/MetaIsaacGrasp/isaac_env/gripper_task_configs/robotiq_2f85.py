"""Classical task constants for UR10e + Robotiq 2F85."""

from __future__ import annotations

from isaac_env.gripper_task_configs.hand_e import _apply_approach_axis_scalars


ROBOTIQ_2F85_TASK_CONFIG = {
    "task_id": "object-v1-Classical-2f85",
    "robot_name": "ur10e_2f85",
    "gripper_profile": "2f85",
    "hdf5_suffix": ".2f85",
    # Initial convention mirrors the base profile; measure_2f85_tool0_frame.py
    # should be used to confirm this against the loaded USD.
    "approach_axis_ee": (0.0, 1.0, 0.0),
    "on_hand": False,
    "n_multiple_cam": 0,
}


def apply_to_element_cfg(namespace: dict) -> None:
    robot_name = ROBOTIQ_2F85_TASK_CONFIG["robot_name"]
    namespace["CLASSICAL_TASK_ID"] = ROBOTIQ_2F85_TASK_CONFIG["task_id"]
    namespace["ROBOT_NAME"] = robot_name
    namespace["CLASSICAL_GRIPPER_PROFILE"] = ROBOTIQ_2F85_TASK_CONFIG["gripper_profile"]
    namespace["CLASSICAL_HDF5_SUFFIX"] = ROBOTIQ_2F85_TASK_CONFIG["hdf5_suffix"]
    _apply_approach_axis_scalars(namespace, ROBOTIQ_2F85_TASK_CONFIG["approach_axis_ee"])
    namespace["ON_HAND"] = ROBOTIQ_2F85_TASK_CONFIG["on_hand"]
    namespace["N_MULTIPLE_CAM"] = ROBOTIQ_2F85_TASK_CONFIG["n_multiple_cam"]
    namespace["CAMERA_NAME"] = ""
    namespace["CLASSICAL_GRIPPER_COLLISION_ROOT"] = None

    joint_setup = namespace["ARM_JOINT"].copy()
    joint_setup.update(namespace["gripper_joint_cfg"][robot_name]["default"])
    namespace["JOINT_SETUP"] = joint_setup
    namespace["ee_name"] = namespace["end_effector_frame_name"][robot_name]

    # The base module builds UR10e_2F85_CFG before v1 overrides ROBOT_NAME, so
    # rebuild only the initial joint state for the selected gripper here.
    robot_cfgs = namespace["ROBOT_CFGs"].copy()
    robot_cfgs[robot_name] = namespace["UR10e_2F85_CFG"].replace(
        init_state=namespace["ArticulationCfg"].InitialStateCfg(
            joint_pos=joint_setup,
            pos=namespace["ROBOT_POS"],
        )
    )
    namespace["ROBOT_CFGs"] = robot_cfgs
