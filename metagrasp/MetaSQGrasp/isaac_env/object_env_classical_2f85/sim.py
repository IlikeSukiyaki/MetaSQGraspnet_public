"""Simulator wrapper for the 2F85 classical antipodal grasp task."""

from isaac_env.classical_shared.sim import ClassicalPickSmBase

from . import element_cfg
from . import env as env_module
from . import env_cfg


class ObjectPickSmClassical2F85(ClassicalPickSmBase):
    element_cfg = element_cfg
    env_module = env_module
    env_cfg_module = env_cfg
