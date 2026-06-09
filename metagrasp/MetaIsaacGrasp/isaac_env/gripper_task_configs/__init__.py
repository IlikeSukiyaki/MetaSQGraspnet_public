"""Gripper-specific bundles for classical grasp tasks."""

from .hand_e import HAND_E_TASK_CONFIG
from .robotiq_2f85 import ROBOTIQ_2F85_TASK_CONFIG

TASK_CONFIGS = {
    HAND_E_TASK_CONFIG["task_id"]: HAND_E_TASK_CONFIG,
    ROBOTIQ_2F85_TASK_CONFIG["task_id"]: ROBOTIQ_2F85_TASK_CONFIG,
}


def get_task_config(task_id: str) -> dict:
    return TASK_CONFIGS[task_id]
