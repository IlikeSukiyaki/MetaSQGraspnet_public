"""2F85 classical antipodal grasp environment wrapper."""

from isaac_env.classical_shared.env import ObjectEnvClassicalBase

from . import element_cfg, env_cfg, wp_cfg


class ObjectEnvClassical2F85(ObjectEnvClassicalBase):
    element_cfg = element_cfg
    env_cfg_module = env_cfg
    wp_cfg_module = wp_cfg
