"""Simulator wrapper for the classical antipodal grasp task."""

from isaac_env.air_env_base.sim import AIRPickSm

from .element_cfg import OBJ_LABLE, validate_classical_assets


class ObjectPickSmClassical(AIRPickSm):
    def __init__(self, args_cli):
        validate_classical_assets()
        super().__init__(args_cli)
        self.obj_label = OBJ_LABLE
