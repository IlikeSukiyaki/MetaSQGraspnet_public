"""Simulator wrapper for the classical antipodal grasp task."""

from isaac_env.air_env_base.sim import AIRPickSm

from . import element_cfg
from . import env as env_module
from . import env_cfg


def _sync_classical_object_config() -> None:
    env_cfg.NUM_OBJS = element_cfg.NUM_OBJS
    env_cfg.OBJ_LABLE = element_cfg.OBJ_LABLE
    env_cfg.MGN_PATHS = element_cfg.MGN_PATHS
    env_cfg.OBJ_PATH = element_cfg.OBJ_PATH
    env_cfg.MGN_CFGs = element_cfg.MGN_CFGs
    env_cfg.OBJ_CFGs = element_cfg.OBJ_CFGs
    env_module.OBJ_LABLE = element_cfg.OBJ_LABLE


class ObjectPickSmClassical(AIRPickSm):
    def __init__(self, args_cli):
        cli_obj_ids = getattr(args_cli, "obj_ids", None)
        if cli_obj_ids:
            element_cfg.configure_classical_objects(
                [obj_id.strip() for obj_id in cli_obj_ids.split(",") if obj_id.strip()]
            )
        _sync_classical_object_config()
        element_cfg.validate_classical_assets()
        super().__init__(args_cli)
        self.obj_label = element_cfg.OBJ_LABLE
        self.env_unwrapped.obj_label = element_cfg.OBJ_LABLE
