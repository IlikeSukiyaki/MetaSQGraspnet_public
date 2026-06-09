"""Shared gripper defaults for sampling, runtime, and Isaac task configs.

Edit these constants once; ``gripper_profiles``, ``tools/generate_object_grasps.py``,
and ``isaac_env/gripper_task_configs/*`` all read from here.
"""

from __future__ import annotations

# Robotiq Hand-E (UR10e + hande_end)
HAND_E_WIDTH_MAX_CM = 7.0
HAND_E_HDF5_SUFFIX = ".hand_e"
HAND_E_PREGRASP_EXTRA_CM = 1.0

# Legacy Franka / Panda datasets
FRANKA_WIDTH_MAX_CM = 8.0
FRANKA_HDF5_SUFFIX = ".franka"
FRANKA_PREGRASP_EXTRA_CM = 1.0

# Robotiq 2F85 (UR10e + tool0)
ROBOTIQ_2F85_WIDTH_MAX_CM = 8.5
ROBOTIQ_2F85_HDF5_SUFFIX = ".2f85"
ROBOTIQ_2F85_PREGRASP_EXTRA_CM = 1.0
