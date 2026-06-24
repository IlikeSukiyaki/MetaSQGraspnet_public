"""Shared simulator wrapper for classical antipodal grasp tasks."""

from isaac_env.env_base.sim import PickSm


class ClassicalPickSmBase(PickSm):
    """Base simulator wrapper parameterized by task-specific modules."""

    element_cfg = None
    env_module = None
    env_cfg_module = None

    def __init__(self, args_cli):
        if self.element_cfg is None or self.env_module is None or self.env_cfg_module is None:
            raise RuntimeError("Classical simulator wrappers must set element/env config modules.")

        cli_obj_ids = getattr(args_cli, "obj_ids", None)
        if cli_obj_ids:
            self.element_cfg.configure_classical_objects(
                [obj_id.strip() for obj_id in cli_obj_ids.split(",") if obj_id.strip()]
            )
        self._sync_classical_object_config()
        self.element_cfg.validate_classical_assets()
        super().__init__(args_cli)
        self.obj_label = self.element_cfg.OBJ_LABLE
        self.env_unwrapped.obj_label = self.element_cfg.OBJ_LABLE

    def _sync_classical_object_config(self) -> None:
        for name in ("NUM_OBJS", "OBJ_LABLE", "MGN_PATHS", "OBJ_PATH", "MGN_CFGs", "OBJ_CFGs"):
            setattr(self.env_cfg_module, name, getattr(self.element_cfg, name))
        self.env_module.OBJ_LABLE = self.element_cfg.OBJ_LABLE
